#if defined(DEEPEP_BF16_REPLICA)
#include "mega_moe_replica_descriptor.h"
#endif
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * DeepEP MegaMoe BF16 training backward kernel.
 *
 * Baseline stage pipeline (reuse-do moves each expert dW2 immediately before P2):
 *
 *   P0-A  route gather           grad_y -> peermem.ptrA0 (AIV)
 *   P0-B  restore count table + Dispatch dO from peers -> ptrDO (AIV0)
 *   P0-C  GMM1 recompute         Z = Xpacked @ W1.T -> gate/up planes (AIC)
 *   P1    GMM2 Grad || early-topk epilogue: dHp, Hp, dHp*H, dH=p*dHp, dZ
 *   P2    GMM1 Grad || Combine: dX = dZ @ W1
 *   P3    GMM1 dW                dW1 = dZ.T @ Xpacked -> grad_weight1 (AIC)
 *   P4    GMM2 dW                dW2 = dO.T @ Hp -> grad_weight2 (AIC)
 *   P5    Unpermute              grad_x = sum_slot(ptrD) (AIV)
 *   P6    Reduce/route/unpermute  grad_topk = sum_F(dHp*H) (AIV)
 *
 * Saved-mode multi-M experts consume per-producer row progress in P1;
 * single-M experts and k=0 retain the complete-expert dispatch flag. Every
 * dispatch band still publishes bwFlagDispatch_[expert]. The P0 staging (gather slots,
 * count-table restore tile, dispatch ping-pong ring) lives in a fixed static
 * UB window above the P1/P2 tile region (see StageUb), so no whole-stage UB
 * ownership handover exists between P0 and P1: dispatch of later experts
 * proceeds while the AIC already fixpipes early experts' dHp tiles.
 *
 * Saved forward context (routing metadata, packed X, count-table snapshot and
 * debug progress) and backward-only scratch are passed as separate buffers.
 */
#ifndef MEGA_MOE_BF16_BACKWARD_H
#define MEGA_MOE_BF16_BACKWARD_H

#ifndef MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
#define MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY 0
#endif
#ifndef MEGAFLUX_BACKWARD_NONE_GM
#define MEGAFLUX_BACKWARD_NONE_GM 0
#endif
#ifndef MEGAFLUX_SHARED_EXPERT_BACKWARD
#define MEGAFLUX_SHARED_EXPERT_BACKWARD 0
#endif
#ifndef MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES
#define MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES 0
#endif
#ifndef MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES
#define MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES 0
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/mc2_kernel_utils.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "mega_moe_combine_send.h"
#include "block_epilogue_swiglu_backward_bf16.h"
#include "mega_moe_bf16_backward_catlass_impl.h"
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
#include "mega_moe_shared_expert_backward.h"
#include "mega_moe_topk_weight_gather.h"
#include "backward_x_redispatch_layout.h"
#endif
#include "mega_moe_bf16_timer.h"
#if MEGAFLUX_BACKWARD_RESTORE
#include "mega_moe_backward_restore.h"
#endif

using namespace AscendC;

namespace MegaMoeImpl {

using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

// P1 alternates between both paired AIVs: AIC event 0 targets AIV0 and event
// FLAG_ID_MAX targets AIV1; each AIV consumes its local event 0. P2 targets
// AIV0 with local ping-pong events 0/1.
constexpr uint16_t BW_P1_EVENT = 0;          // P1 AIC <-> AIV0/AIV1 SwiGLU Grad
constexpr uint32_t BW_TOPK_ROUTE_BATCH_ROWS = 1024;
constexpr uint32_t BW_TOPK_PARTIAL_UB_BYTES = 128 * 1024;
// Phase progress markers written to bwProgress_ (indexed by GetBlockIdx()%256)
// so a hung kernel can be localized from the last written value.
constexpr int32_t BW_PROG_START = 0;
constexpr int32_t BW_PROG_P0A = 1;
constexpr int32_t BW_PROG_P0B = 2;
constexpr int32_t BW_PROG_P0C = 3;
constexpr int32_t BW_PROG_P1 = 4;
constexpr int32_t BW_PROG_P2 = 5;
constexpr int32_t BW_PROG_P3 = 6;
constexpr int32_t BW_PROG_P4 = 7;
constexpr int32_t BW_PROG_P5 = 8;
constexpr int32_t BW_PROG_P6 = 9;
constexpr int32_t BW_PROG_SPIN_HDONE = 12;

enum BackwardPhase {
    BW_PHASE_RECOMPUTE = 0,
    BW_PHASE_GMM2GRAD = 1,
    BW_PHASE_GMM1GRAD = 2,
    BW_PHASE_DW1 = 3,
    BW_PHASE_DW2 = 4
};

// Per-core fixed timing slots for the backward pipeline.  Mirrors the forward
// AscendTimerV2 layout: each slot holds a [start, end] cycle pair, and every
// AIC/AIV records its own view of the same stage.  These are the whole-stage
// bars that tile each lane; the per-expert GEMM/epilogue/dispatch stages use
// the shared dynamic slots (see the comment below the enum).
enum BackwardFixedTiming {
    BW_KERNEL_TIMING_IDX = 0,   // whole kernel (timerAll_)
    AIV_GATHER = 1,             // P0-A gather grad_y -> peermem ptrA0 (all AIV)
    AIV_XSYNC_GATHER = 2,       // cross-rank barrier after the gather
    AIV_RESTORE_TABLE = 3,      // P0-B restore saved token-count table + SyncAll
    // Slots 4 (AIC_SPIN_AIV1), 6 (AIC_SPIN_AIV0) and 15 (AIV_P0_TAIL) are
    // RETIRED: the static P0 staging region (StageUb) removed the P0->P1 UB
    // ownership handover, so there is no whole-stage spin or pipe-teardown
    // tail any more.  The enum values stay reserved so the timer layout —
    // and with it every existing raw CSV and parser — remains stable; the
    // slots simply stay zero and vanish from traces/summaries.
    AIC_SPIN_DZ = 5,            // AIC waits last dZ tiles + both AIVs done (P1 exit)
    AIC_SPIN_HP = 7,            // AIC waits Hp/dpProducts published (P4 entry)
    AIV_XSYNC_UNPERMUTE = 8,    // cross-rank barrier before P5 reads ptrD
    AIV_UNPERMUTE = 9,          // P5 grad_x = sum over topk slots of ptrD
    AIV_XSYNC_DP = 10,          // cross-rank barrier: ptrD readers drained
    AIV_DP_REDUCE = 11,         // P6a dp = sum_F(dHp*H), route to source ranks
    AIV_XSYNC_TOPK = 12,        // cross-rank barrier: remote dp writes visible
    AIV_TOPK_UNPERMUTE = 13,    // P6b grad_topk from routed dp
    AIV_TABLE_RESET = 14,       // re-zero the symmetric token table
    AIV_RECOMPUTE_PASS = 16,    // AIV bookkeeping walk of the recompute phase
    // Accumulated WaitForCube time of every P1 tile this AIV lane consumed
    // (Accumulate mode: end-start = sum of waits).  The first wait carries
    // the pipeline ramp (P0-C recompute tail + first-tile GEMM) before the
    // lane's first dHp tile, so AIV_SWIGLU_GRAD_e decomposes into wait
    // (this slot) vs epilogue compute.  Ticked in
    // GMM2GradSwiGLUGradDualAiv (mega_moe_bf16_backward_catlass_impl.h).
    AIV_P1_WAIT = 17,           // P1 per-lane accumulated dHp-tile wait
    AIC_P2_TAIL = 18,           // AIC waits AIV0 consumed every dX tile (P2 exit)
    AIC_INIT = 19,              // AIC wakeup-to-first-stage window
    AIV_INIT = 20,              // AIC wakeup-to-first-stage window
    AIC_DZ_WAIT_EXPERT_BASE = 21,
    AIV_DX_WAIT_EXPERT_BASE = 85,
    AIC_DO_REUSE_WAIT_EXPERT_BASE = 149,
    BW_BACKWARD_TIMING_COUNT = AIC_DO_REUSE_WAIT_EXPERT_BASE + 64
};
// The per-expert stages (dispatch / recompute / GEMMs / epilogue / dW) use
// the shared dynamic slots AIC_BW_* / AIV_BW_* in
// AscendTimer::DynamicTimingType and render as AIC_GMM2GRAD_0..N blocks,
// mirroring the forward's AIC_GMM1_0..N layout.
#if defined(ENABLE_TIMER)
static_assert(AIV_DX_WAIT_EXPERT_BASE - AIC_DZ_WAIT_EXPERT_BASE ==
    AscendTimer::MAX_DYNAMIC_ITER, "expert wait slots must match timer layout");
static_assert(BW_BACKWARD_TIMING_COUNT < AscendTimer::N_TIMING_ITEM_PER_CORE,
    "backward timing slots must fit in the AscendTimerV2 per-core item space");
#endif

class MegaMoeBF16Backward {
public:
    using QuantOutType = bfloat16_t;
    __aicore__ inline MegaMoeBF16Backward() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR gradX, GM_ADDR gradTopkWeights,
        GM_ADDR gradWeight1, GM_ADDR gradWeight2,
        GM_ADDR trainingSavedState, GM_ADDR dispatchOutput,
        GM_ADDR backwardWorkspace,
        MegaMoeTilingData *tilingData, uint32_t splitExperts, GM_ADDR timerAddr = nullptr
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
        , const MegaMoeSharedExpert::SharedExpertBackwardArgs* shared = nullptr
#endif
        );
    __aicore__ inline void Process();
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
    __aicore__ inline void EnableSharedExpertV2(
        const MegaMoeSharedExpert::SharedExpertBackwardArgsV2& shared) {
        sharedArgsV2_ = shared;
        sharedV2Enabled_ = true;
    }
    __aicore__ inline void EnableSharedExpertV3(
        const MegaMoeSharedExpert::SharedExpertBackwardArgsV3& shared) {
        sharedArgsV3_ = shared;
        sharedArgsV2_ = shared.shared;
        sharedV2Enabled_ = true;
        sharedXRedispatchEnabled_ = true;
    }
    __aicore__ inline void EnableRoutedRedispatch(
        const MegaMoeSharedExpert::SharedExpertBackwardArgsV3& args) {
        EnableSharedExpertV3(args);
        sharedEnabled_ = false;
        sharedV2Enabled_ = false;
    }
#endif
#if MEGAFLUX_BACKWARD_RESTORE
    __aicore__ inline void EnableBackwardRestore(uint64_t ready,int epoch,int mode) {
        restoreReady_=ready;restoreEpoch_=epoch;restoreMode_=mode;
    }
#endif
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
    __aicore__ inline void EnableLifetime(uint64_t side) {
        if(!side)return;
        lifetimeCache_=MegaFluxWorker::Field(side,6);
        // ABI2: exporting dW is independent from compute-order eligibility.
        // Keep mode1 for a matched legacy control in the very same binary.
        if(MegaFluxWorker::Field(side,7)==1)gradient_.ready=0;
        rawGradientOutput_=MegaFluxWorker::Field(side,7)==2;
    }
    __aicore__ inline void EnableGradientPipe(uint64_t ready,uint64_t trace,int epoch) {
        gradient_.replica=replica_;gradient_.ready=ready;gradient_.trace=trace;gradient_.epoch=epoch;
        for(int e=0;e<expertPerRank_;++e)gradientOrder_[e]=e;
        // Same global logical-expert priority on every participating rank.
        for(int i=1;i<expertPerRank_;++i) {
            int v=gradientOrder_[i],j=i-1;
            while(j>=0 && GradientKey(gradientOrder_[j])>GradientKey(v)) {
                gradientOrder_[j+1]=gradientOrder_[j];--j;
            }
            gradientOrder_[j+1]=v;
        }
    }
#endif
#if defined(DEEPEP_BF16_REPLICA)
    __aicore__ inline void EnableReplica(const MegaMoeReplicaDescriptor& d) {replica_=d;}
    __aicore__ inline void ReplicaBarrier() {
        if constexpr(g_coreType==AscendC::AIV) {
            AscendC::PipeBarrier<PIPE_ALL>();AscendC::SyncAll<true>();
            CrossRankSyncInWorldSize();
        }
    }
#endif


private:
    bool sharedSavedSourcesPrepared_ = false;
    __aicore__ inline bool XRedispatchEnabled() const {
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
        return sharedXRedispatchEnabled_;
#else
        return false;
#endif
    }
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
    __aicore__ inline void PrepareSharedSavedSources();
    struct XRedispatchHooks {
        MegaMoeBF16Backward* op;
        __aicore__ inline void PrepareSources() { op->PrepareXRedispatchSources(); }
        __aicore__ inline void Transport() { op->TransportXRedispatch(); }
        __aicore__ inline void RetireSourceReads() { op->RetireXRedispatchSources(); }
    };
    __aicore__ inline void XRedispatchStamp(uint32_t phase, uint32_t edge);
    __aicore__ inline void PrepareXRedispatchSources();
    __aicore__ inline void TransportXRedispatch();
    __aicore__ inline void RetireXRedispatchSources();
    __aicore__ inline void DispatchXProbBackward(uint32_t expert, int64_t expertRowBase);
    __aicore__ inline void CopyRedispatchProb(
        GlobalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src, int32_t rows);
#endif
#if MEGAFLUX_BACKWARD_RESTORE
    uint64_t restoreReady_=0;
    int restoreEpoch_=0;
    int restoreMode_=0;
#endif
#if defined(DEEPEP_BF16_REPLICA)
    MegaMoeReplicaDescriptor replica_{};
    uint64_t lifetimeCache_=0;
    bool rawGradientOutput_=false;
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
    MegaFluxGradient::Context gradient_{};
#if defined(DEEPEP_BF16_EARLY_REDUCE)
    bool gradientReducedEarly_=false;
#endif
    int gradientOrder_[MegaMoeBF16Training::kMaxLocalExperts];
    uint32_t dispatchOrdinal_[MegaMoeBF16Training::kMaxLocalExperts]{};
#if MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
    bool earlyDW1Enabled_ = false;
    uint32_t earlyDW1Cursor_ = 0;
    uint32_t earlyDW1Done_ = 0;
    template<bool TransW, class Params>
    __aicore__ inline void GMM1DWExpert(Params& params, uint32_t expertIdx,
        int64_t rows, uint32_t& cursor);
#endif
#if MEGAFLUX_WORKER_BACKWARD
    int workerExpert_=-1;
#if MEGAFLUX_WORKER_SHARED_PREFIX
    bool workerBulk_=false;
    uint32_t workerBeginPos_=0;
#endif
    __aicore__ inline bool WorkerCursorActive() {
        return workerExpert_>=0
#if MEGAFLUX_WORKER_SHARED_PREFIX
            || workerBulk_
#endif
            ;
    }
    bool cooperativeBackward_=false;
    uint32_t workerDH_=0,workerTarget_=0,workerDX_=0,workerDW2_=0,workerDW1_=0;
    MegaFluxGradient::CooperativeOwner ownerService_;
    template<bool TransW> __aicore__ inline void WorkerBackward();
#endif
    __aicore__ inline int GradientKey(int e) {
        const auto& d=replica_;
        const int logical=e<d.home?d.rank*d.home+e:MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+d.rank*d.slots+e-d.home);
        if(logical<0)return 100000+e;
        bool shared=false;for(int i=0;i<d.ranks*d.slots;++i)if(MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+i)==logical)shared=true;
        return (shared?0:10000)+logical;
    }
    __aicore__ inline int ExpertAt(int pos) {return gradient_.ready?gradientOrder_[pos]:pos;}
#else
    __aicore__ inline int ExpertAt(int pos) {return pos;}
#endif
#endif
#if !defined(DEEPEP_BF16_REPLICA)
    __aicore__ inline int ExpertAt(int pos) {return pos;}
#endif
    __aicore__ inline void InitStageLayout();
    __aicore__ inline AscendC::LocalTensor<QuantOutType> StageUb(uint32_t bytes);
    __aicore__ inline void UnpermuteBuffInit(TPipe *tPipe);
    __aicore__ inline void TopkGradBuffInit(TPipe *tPipe);
    __aicore__ inline void GatherRoutes();
    __aicore__ inline void RestoreTokenPerExpert();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void ClearEmptyExpertGradWeights();
    __aicore__ inline void ClearGradWeightPlane(
        GM_ADDR base, uint64_t planeElements, uint32_t expertIdx);
    __aicore__ inline void DispatchTokensBackward(GM_ADDR dstBase, uint32_t expertIdx, uint32_t ordinal);
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                GlobalTensor<QuantOutType> src, int32_t rows, int32_t hiddenSize,
                int32_t& pingpongId);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t phase,
                uint32_t expertIdx);
    __aicore__ inline void UnpermuteBackward();
    __aicore__ inline void ReduceDpPartialBatch(
        uint16_t batchRows, uint16_t partialTiles, uint16_t partialStride);
    __aicore__ inline void ReduceAndRouteTopkGrad();
    __aicore__ inline void UnpermuteTopkGrad();
    __aicore__ inline void CrossRankSyncInWorldSize();

    __aicore__ inline void WriteBwProgress(int32_t phase)
    {
        if (bwProgress_ != nullptr) {
            WriteGmByPassDCache(bwProgress_ + (GetBlockIdx() % 256), phase);
        }
    }

    // Debug build only: print one representative AIC/AIV0/AIV1 per rank.
    // stage uses BW_PROG_* above; edge=0 means enter and edge=1 means leave.
    __aicore__ inline void DebugBwStage(int32_t stage, int32_t edge)
    {
#if defined(MEGA_MOE_BF16_DEBUG_PRINT) && MEGA_MOE_BF16_DEBUG_PRINT
        if (blockIdx_ == 0) {
            if constexpr(g_coreType == AscendC::AIC) {
                AscendC::PRINTF(
                    "[MegaMoeBF16BW] rank=%u core=AIC rawBlock=%u logicalBlock=%u "
                    "stage=%d edge=%d\n",
                    rankId_, GetBlockIdx(), blockIdx_, stage, edge);
            } else {
                AscendC::PRINTF(
                    "[MegaMoeBF16BW] rank=%u core=AIV rawBlock=%u logicalBlock=%u "
                    "subBlock=%u stage=%d edge=%d\n",
                    rankId_, GetBlockIdx(), blockIdx_, subBlockIdx_, stage, edge);
            }
        }
#else
        (void)stage;
        (void)edge;
#endif
    }

    // phase loops (all cores call them; wrappers dispatch on g_coreType)
    template<bool TransW> __aicore__ inline void GMM1RecomputePhase();
    template<bool TransW, bool GmOutput> __aicore__ inline void GMM2GradPhase();
    template<bool TransW, bool ExpertReady> __aicore__ inline void GMM1GradPhase();
    template<bool TransW> __aicore__ inline void GMM1GradReuseDOPhase();
    // AIC-only weight-gradient stages
    template<bool TransW> __aicore__ inline void GMM1DWPhase();
    template<bool TransW> __aicore__ inline void GMM2DWPhase();

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,
        int32_t& coreOffset, int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2;
        int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
        int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
        coreLen = lenPerCoreAlign;
        coreOffset = coreIdx * lenPerCoreAlign;
        if (coreOffset + coreLen >= totalLen) {
            coreLen = totalLen - coreOffset;
        }
        if (coreOffset >= totalLen) {
            coreLen = 0;
        }
    }

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
#if defined(DEEPEP_BF16_REPLICA)
            if (ReadGmByPassDCache(sig_addr) >= cmp_val) {
#else
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
#endif
                return;
            }
        } while (true);
    }

    // ---- Static P0 staging layout -------------------------------
    // One BF16 L1 tile is 128*256*2B; the epilogue binds its three tiles at
    // raw UB offsets 0/1/2 tile bytes (block_epilogue_swiglu_backward_bf16.h)
    // and P1/P2 fixpipe writes the paired AIV's UB at raw offsets 0/64KB, so
    // [0, 3*tileBytes) is the fixed P1/P2 tile region.  Everything P0 needs
    // on the vector side (gather ping-pong slots, count-table restore tile,
    // dispatch copy ring) lives in the static window above it; the three P0
    // users are sequential in time on one lane, so they share one region.
    static constexpr uint32_t BW_TILE_BYTES =
        MegaMoeBF16BackwardCatlassImpl::L1_TILE_M *
        MegaMoeBF16BackwardCatlassImpl::L1_TILE_N *
        sizeof(bfloat16_t);
    static constexpr uint32_t BW_STAGE_BASE_BYTES = 3 * BW_TILE_BYTES;
    static constexpr uint32_t BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES =
        MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes;
    static_assert(
        BW_STAGE_BASE_BYTES + BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES <=
            MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE,
        "the static staging window must hold the gradient-zero tile");

    TBuf<> DataResBuf_;
    TBuf<> DataResFp32Buf_;
    TBuf<> TopkGradRouteBuf_;
    TBuf<> TopkGradPartialBuf_;

#if defined(ENABLE_TIMER)
    AscendTimerDevice timer_;
    AscendTimerDevice timerAll_;
#endif

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    BackwardSavedStateInfo savedState_{};
    BackwardWorkspaceInfo bwWs_{};
    uint32_t splitExperts_ = 0;
    TupleShape problemShape_{};
    using BlockEpilogue = MegaMoeBF16Impl::BlockEpilogueSwigluBackwardBF16<
            MegaMoeBF16BackwardCatlassImpl::L1TileShape>;
    BlockEpilogue epilogueOp_;

    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> savedTokenPerExpert_;
    __gm__ int32_t *bwFlagDispatch_{nullptr};
    __gm__ int32_t *bwFlagAiv1Ready_{nullptr};
    __gm__ int32_t *bwFlagHDone_{nullptr};
    __gm__ int32_t *bwProgress_{nullptr};

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t n_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint32_t savedGateUpMode_ = MegaMoeBF16Training::kSaveNone;
    bool noneGmEnabled_ = false;
    bool backwardPolicyValid_ = true;
    uint64_t maxOutputSize_ = 0;
    int32_t dispatchRows_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 0;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    int32_t dispatchCopyElementsPerPing_ = 0;
    // Per-ping row count of the static dispatch ring: tiling's dispatchRows_
    // clamped to the staging budget (see InitStageLayout).
    uint32_t dispatchPingRows_ = 0;
    __aicore__ inline uint32_t DispatchOrdinal(uint32_t expert, uint32_t traversalOrdinal) const {
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
        return dispatchOrdinal_[expert];
#else
        return traversalOrdinal;
#endif
    }
    __aicore__ inline bool UseDispatchRowReady(uint32_t ordinal) const {
        return MegaMoeBF16Training::UseBackwardDispatchRowReady(
            savedGateUpMode_, ordinal, splitExperts_,
            static_cast<uint64_t>(Get<M_VALUE>(problemShape_)),
            MegaMoeBF16BackwardCatlassImpl::L1_TILE_M, noneGmEnabled_);
    }
    __aicore__ inline MegaMoeBF16BackwardCatlassImpl::BackwardDispatchReadyView
    MakeDispatchRowReady(uint32_t expert, uint32_t ordinal) const {
        MegaMoeBF16BackwardCatlassImpl::BackwardDispatchReadyView v;
        v.progressBase = reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwDispatchProgress);
        v.cumsum = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrcumsumMM);
        v.sources = worldSize_; v.producersPerSource = dispatchBlockNumPerEP_;
        v.experts = expertPerRank_; v.expert = expert; v.progressSlot = ordinal;
        v.rowLimit = static_cast<uint32_t>(maxOutputSize_);
        v.enabled = UseDispatchRowReady(ordinal);
        return v;
    }

    int32_t vecSetSyncCom_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    int64_t cumMRows_ = 0;
    int64_t preOffset_ = 0;
    GM_ADDR gradWeight1GmAddr_{nullptr};
    GM_ADDR gradWeight2GmAddr_{nullptr};
    GM_ADDR gradTopkWeightsGmAddr_{nullptr};
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
    MegaMoeSharedExpert::SharedExpertBackwardArgs sharedArgs_{};
    MegaMoeSharedExpert::SharedExpertBackwardArgsV2 sharedArgsV2_{};
    MegaMoeSharedExpert::SharedExpertBackwardArgsV3 sharedArgsV3_{};
    bool sharedEnabled_{false};
    bool sharedV2Enabled_{false};
    bool sharedXRedispatchEnabled_{false};
#endif
};

__aicore__ inline void MegaMoeBF16Backward::Init(
    GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR gradX, GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
    GM_ADDR trainingSavedState, GM_ADDR dispatchOutput,
    GM_ADDR backwardWorkspace,
    MegaMoeTilingData *tilingData, uint32_t splitExperts, GM_ADDR timerAddr
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
    , const MegaMoeSharedExpert::SharedExpertBackwardArgs* shared
#endif
    )
{
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
    sharedEnabled_ = shared != nullptr;
    if (sharedEnabled_) {
        sharedArgs_ = *shared;
    }
#endif
    splitExperts_ = MegaMoeBF16Training::BackwardSplitExperts(splitExperts);
    m_ = tilingData->m;
    k_ = tilingData->k;
    n_ = tilingData->n;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    rankId_ = tilingData->rankId;
    savedGateUpMode_ = MegaMoeBF16Training::ResolveSavedGateUpMode(
        tilingData->savedGateUpMode, tilingData->swigluLimit);
    // The request bit is a Host/SO handshake, never a change to saved-state
    // or tiling ABI. Old Hosts and non-S9 libraries keep none on UB.
#if defined(DEEPEP_BF16_WORKER_PIPELINE) && MEGAFLUX_BACKWARD_NONE_GM
    constexpr bool noneGmSupported = true;
#else
    constexpr bool noneGmSupported = false;
#endif
    // An inactive replica plan selects the smaller home tiling while Host's
    // split limit still describes home+slots. Validate its fixed ABI bound,
    // not the runtime home-only expert count.
    backwardPolicyValid_ = MegaMoeBF16Training::IsBackwardDispatchPolicyValid(
        splitExperts, savedGateUpMode_, MegaMoeBF16Training::kMaxLocalExperts,
        noneGmSupported);
    if (!backwardPolicyValid_) return;
    noneGmEnabled_ = MegaMoeBF16Training::BackwardNoneGmRequested(splitExperts);

#if defined(ENABLE_TIMER)
    timer_.Init(timerAddr);
    timerAll_.Init(timerAddr);
    timerAll_.Tik();
    timer_.Tik();
    // Publish the per-expert iteration counts for the trace parsers.
    timer_.setDynamicActualIter(AscendTimer::AIV_BW_DISPATCH, expertPerRank_);
    timer_.setDynamicActualIter(
        AscendTimer::AIC_BW_RECOMPUTE,
        savedGateUpMode_ == MegaMoeBF16Training::kSaveNone
            ? expertPerRank_ : 0);
    timer_.setDynamicActualIter(AscendTimer::AIC_BW_DO_WAIT, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_BW_GMM2GRAD, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIV_BW_SWIGLU_GRAD, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_BW_GMM1GRAD, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIV_BW_COMBINE, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_BW_DW1, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_BW_DW2, expertPerRank_);
#endif

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    params_.aGmAddr = gradY;
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.probsGmAddr = topkWeights;
    params_.y2GmAddr = gradX;
    savedState_ = BackwardSavedStateInfo(
        trainingSavedState, dispatchOutput, tilingData);
    params_.workspaceInfo = savedState_;
    params_.peermemInfo = PeermemInfo(
        winRankAddr_[rankId_], tilingData, true);
    params_.tilingData = tilingData;
    gradWeight1GmAddr_ = gradWeight1;
    gradWeight2GmAddr_ = gradWeight2;
    gradTopkWeightsGmAddr_ = gradTopkWeights;
    bwWs_ = BackwardWorkspaceInfo(backwardWorkspace, tilingData, splitExperts_, noneGmEnabled_);
    bwFlagDispatch_ = (__gm__ int32_t*)bwWs_.ptrBwFlagDispatchToGmm2Grad;
    bwFlagAiv1Ready_ = (__gm__ int32_t*)bwWs_.ptrBwFlagAiv1Ready;
    bwFlagHDone_ = (__gm__ int32_t*)bwWs_.ptrBwFlagHDone;
    bwProgress_ = (__gm__ int32_t*)savedState_.ptrBwProgress;

    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
    // Prepare once before S9 chooses its shared-expert order. Progress slots
    // retain the natural nonempty ordinal used by static metadata preparation.
    MegaMoeBF16BackwardCatlassImpl::PreparedExpertCursor dispatchCursor;
    for (uint32_t expert=0; expert<expertPerRank_; ++expert) {
        uint32_t rows=0, ordinal=0;
        const bool nonempty=dispatchCursor.Next(static_cast<int32_t>(groupList_.GetValue(expert)),
            groupListType_,maxOutputSize_,rows,ordinal);
        dispatchOrdinal_[expert]=nonempty?ordinal:splitExperts_;
    }
#endif
    savedTokenPerExpert_.SetGlobalBuffer((__gm__ int32_t*)savedState_.ptrSavedTokenPerExpert);

    GM_ADDR gatePlane =
        savedGateUpMode_ == MegaMoeBF16Training::kSaveNone
            ? bwWs_.ptrRecomputeGate : savedState_.ptrSavedGate;
    GM_ADDR upPlane =
        savedGateUpMode_ == MegaMoeBF16Training::kSaveNone
            ? bwWs_.ptrRecomputeUp : savedState_.ptrSavedUp;
    epilogueOp_.Init({gatePlane, upPlane,
        savedState_.ptrSavedGateScale, savedState_.ptrSavedUpScale,
        savedState_.ptrSavedGateHigh, savedState_.ptrSavedUpLow,
        savedState_.ptrSavedUpHigh,
        bwWs_.ptrGradZ, bwWs_.ptrHp, bwWs_.ptrDpProducts,
        savedState_.ptrDispatchedTopkWeights, 0, 0,
        tilingData->swigluLimit, savedGateUpMode_});
    epilogueOp_.UpdateNextProblem({0, n_, 0, 0});
}

// Fixed-position P0 staging window at the top of UB, above the P1/P2 tile
// region [0, BW_STAGE_BASE_BYTES).  P0-A gather slots, the count-table
// restore tile and the P0-B dispatch ring all bind here; they are
// sequential in time on one lane, so one shared region serves all three.
// Nothing in P0 goes through TPipe, so the staging region never overlaps
// the CATLASS-managed tile area: P1's fixpipe may target the paired AIV's
// tiles while dispatch is still copying later experts, and the per-expert
// bwFlagDispatch_[e] / DO_WAIT_e handshake is the only P1 entry gate.
// A tile fixpiped into a lane that has not entered P1 yet simply sits in
// the (statically bound) tile region until that lane's epilogue consumes
// it; the per-phase CATLASS Resource TBufs that may overlap the region
// are allocation-only on the vector side (never written, never zeroed).
// Raw fixed-position binding, same pattern as the epilogue's own tile
// members (VECIN raw tiles at fixed byte offsets, fed by plain DataCopy in
// both directions).  The return type spells bfloat16_t, NOT the class alias
// QuantOutType: in an out-of-line member definition the return type sits
// before the qualified MegaMoeBF16Backward:: name, where class-scope names
// are not visible yet (parameters and the body are fine).
__aicore__ inline AscendC::LocalTensor<bfloat16_t>
MegaMoeBF16Backward::StageUb(uint32_t bytes)
{
    return AscendC::LocalTensor<bfloat16_t>(
        AscendC::TPosition::VECIN, BW_STAGE_BASE_BYTES,
        bytes / sizeof(bfloat16_t));
}

__aicore__ inline void MegaMoeBF16Backward::InitStageLayout()
{
    // The two-slot dispatch ring must fit the static staging budget
    // [BW_STAGE_BASE_BYTES, UB_SIZE).  Tiling's dispatchRows_ was sized for
    // the old whole-UB firstPipe layout, so clamp the per-ping row count;
    // one row is the floor (K classes that cannot fit a single row twice
    // are outside this layout's contract).  The 32-element alignment of a
    // ping can add up to 64B, hence the margin.
    const uint32_t stageBytes =
        MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE - BW_STAGE_BASE_BYTES;
    const uint32_t maxPingRows =
        stageBytes > 256 ? (stageBytes - 256) / 2 / (k_ * sizeof(bfloat16_t))
                         : 0;
    dispatchPingRows_ = static_cast<uint32_t>(dispatchRows_) < maxPingRows
        ? static_cast<uint32_t>(dispatchRows_)
        : (maxPingRows > 0 ? maxPingRows : 1);
    if (MegaMoeBF16Training::BackwardUsesGm(savedGateUpMode_, noneGmEnabled_))
        dispatchPingRows_ = 1; // Preserve the measured saved-mode single-row path, including k=0.
    dispatchCopyElementsPerPing_ = Ops::Base::CeilAlign(
        dispatchPingRows_ * k_, static_cast<uint32_t>(ALIGN_32));
}

__aicore__ inline void MegaMoeBF16Backward::UnpermuteBuffInit(TPipe *tPipe)
{
    uint32_t unpermuteBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(DataResBuf_, unpermuteBufAlign);
    tPipe->InitBuffer(DataResFp32Buf_, unpermuteBufAlign * 2);
}

__aicore__ inline void MegaMoeBF16Backward::TopkGradBuffInit(TPipe *tPipe)
{
    tPipe->InitBuffer(
        TopkGradRouteBuf_,
        Ops::Base::CeilAlign(
            BW_TOPK_ROUTE_BATCH_ROWS * static_cast<uint32_t>(sizeof(bfloat16_t)),
            static_cast<uint32_t>(ALIGN_32)));
    tPipe->InitBuffer(TopkGradPartialBuf_, BW_TOPK_PARTIAL_UB_BYTES);
}

__aicore__ inline void MegaMoeBF16Backward::GatherRoutes()
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(static_cast<int32_t>(m_) * static_cast<int32_t>(topK_), coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdx);
    GlobalTensor<QuantOutType> gradY;
    gradY.SetGlobalBuffer((__gm__ QuantOutType*)params_.aGmAddr);
    GlobalTensor<QuantOutType> ptrA0;
    ptrA0.SetGlobalBuffer((__gm__ QuantOutType*)params_.peermemInfo.ptrA0);

    // Preserve the fixed P0 window above all three P1/P2 work tiles.
    // Do not borrow their space: P1 may already fixpipe while P0-B runs.
    const uint32_t capacity = MegaMoeBF16BackwardCatlassImpl::ArchTag::UB_SIZE - BW_STAGE_BASE_BYTES;
    const uint32_t rowBytes = k_ * sizeof(bfloat16_t);
    const uint32_t batchRows = 8U * rowBytes <= capacity ? 4U :
        (4U * rowBytes <= capacity ? 2U : 1U);
    if (batchRows > 1U && batchRows * rowBytes <= 65504U) {
        const uint32_t slotElements = batchRows * k_;
        auto arena = StageUb(2U * batchRows * rowBytes);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        uint32_t batch = 0;
        for (int32_t first = 0; first < coreLen; first += batchRows, ++batch) {
            const uint32_t rows = static_cast<uint32_t>(coreLen - first) < batchRows
                ? static_cast<uint32_t>(coreLen - first) : batchRows;
            int64_t source[4];
            for (uint32_t i = 0; i < rows; ++i) {
                source[i] = static_cast<int64_t>(expandedRowIdx(coreOffset + first + i) /
                    static_cast<int32_t>(topK_)) * k_;
            }
            const auto event = batch % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
            auto slot = arena[(batch % 2U) * slotElements];
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(slot[i * k_], gradY[source[i]], k_);
            }
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            DataCopy(ptrA0[(static_cast<int64_t>(coreOffset) + first) * k_], slot, rows * k_);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        return;
    }

    const uint32_t gatherBytes = Ops::Base::CeilAlign(
        static_cast<uint32_t>(2 * k_ * sizeof(bfloat16_t)),
        static_cast<uint32_t>(ALIGN_32));
    LocalTensor<QuantOutType> dataIn0Bf16 = StageUb(gatherBytes);
    LocalTensor<QuantOutType> dataIn1Bf16 = dataIn0Bf16[k_];

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t route = coreOffset; route < coreOffset + coreLen; route++) {
        int32_t token = expandedRowIdx(route) / static_cast<int32_t>(topK_);
        // In the early-topk forward, p is applied to H before W2. Therefore
        // backward dispatches raw dY; p belongs on dH=p*dHp in P1.
        auto event = (route % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (route % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
        // dY is already BF16. Copy it directly through a ping-pong UB slot:
        // MTE3 releases the slot before its next MTE2 fill, and MTE2 finishes
        // filling the slot before MTE3 reads it.
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopy(dataInBf16, gradY[token * static_cast<int64_t>(k_)], k_);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        DataCopy(ptrA0[route * static_cast<int64_t>(k_)], dataInBf16, k_);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void MegaMoeBF16Backward::RestoreTokenPerExpert()
{
    const bool prepareMetadata = MegaMoeBF16Training::BackwardUsesGm(savedGateUpMode_, noneGmEnabled_) && splitExperts_>0;
    int32_t perRank = static_cast<int32_t>(perRankStridesInTokenPerExpert_);
    LocalTensor<int32_t> tmpBuffer = StageUb(perRank * sizeof(int32_t))
        .ReinterpretCast<int32_t>();
    // P0 uses raw UB without a TPipe. Gather has drained both ping-pong
    // events before this stage; use an explicit, paired event instead of
    // SyncFunc, which dereferences GetTPipePtr() to fetch an event ID.
    // Source-rank rows are disjoint, so distribute them across all AIVs. The
    // caller's existing SyncAll keeps the following dispatch from observing a
    // partially restored table.
    for (int32_t sourceRank = static_cast<int32_t>(aivCoreIdx_);
         sourceRank < static_cast<int32_t>(worldSize_);
         sourceRank += static_cast<int32_t>(blockAivNum_)) {
        DataCopy(tmpBuffer, savedTokenPerExpert_[sourceRank * perRank], perRank);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(tokenPerExpertWin_[sourceRank * perRank], tmpBuffer, perRank);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        if(prepareMetadata) {
            MegaMoeBF16BackwardCatlassImpl::PreparedExpertCursor cursor;
            for(uint32_t expert=0;expert<expertPerRank_;++expert) {
                uint32_t rows=0,ordinal=0;
                if(!cursor.Next(static_cast<int32_t>(groupList_.GetValue(expert)),
                    groupListType_,maxOutputSize_,rows,ordinal))continue;
                if(!MegaMoeBF16Training::UseBackwardDispatchRowReady(savedGateUpMode_,ordinal,
                    splitExperts_,rows,MegaMoeBF16BackwardCatlassImpl::L1_TILE_M,noneGmEnabled_))continue;
                MegaMoeBF16BackwardCatlassImpl::BackwardDispatchReadyView view;
                view.progressBase=reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwDispatchProgress);
                view.cumsum=reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.ptrcumsumMM);
                view.sources=worldSize_;view.producersPerSource=dispatchBlockNumPerEP_;
                view.experts=expertPerRank_;view.expert=expert;view.progressSlot=ordinal;
                view.rowLimit=static_cast<uint32_t>(maxOutputSize_);view.enabled=true;
                if(MegaMoeBF16BackwardCatlassImpl::ProducerPhaseSupported(view,blockNum_))
                    MegaMoeBF16BackwardCatlassImpl::PrepareProducerPhaseSource(view,blockNum_,rows,
                        static_cast<uint32_t>(sourceRank),nullptr);
            }
        }
    }
    if(prepareMetadata) {
        // Retire every S metadata store/atomic before the caller's original
        // AIV-only restore SyncAll. Does not reuse payload ring events 0/1.
        SetFlag<AscendC::HardEvent::S_MTE2>(7);WaitFlag<AscendC::HardEvent::S_MTE2>(7);
        SetFlag<AscendC::HardEvent::MTE2_S>(7);WaitFlag<AscendC::HardEvent::MTE2_S>(7);
    }
}

__aicore__ inline void MegaMoeBF16Backward::ResetTokenPerExpert()
{
    // Mirror of MegaMoeBF16::ResetTokenPerExpert().  The forward's count
    // exchange protocol requires the symmetric-window token table to be
    // all-zero at every kernel entry:
    //   1. moe_init_routing_v3 -> ExpertTokensCountV2 ATOMIC-ADDS the fresh
    //      routing histogram into this rank's table row, so a non-zero
    //      leftover doubles (or worse) the counts;
    //   2. CrossRankSyncCntPerExpert detects peer arrival with
    //      GmSignalWait(slot != 0) and then strips the 0x800000 tag, so a
    //      stale non-zero row is consumed as count - 0x800000 (large
    //      negative) and poisons cumsum/prevSum -> negative dispatch and
    //      combine offsets -> OOB writes that can stomp the peer window's
    //      sync/barrier region and deadlock the next kernel.
    // The forward re-establishes the invariant before it exits; the backward
    // must do the same after RestoreTokenPerExpert() dirtied the table.
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    // EP512 / four experts has a 4 MiB GM table. Reuse one bounded zero
    // tile after the P6 pipe is destroyed, rather than materializing it in UB.
    const uint64_t countElements = static_cast<uint64_t>(worldSize_) *
        static_cast<uint32_t>(perRankStridesInTokenPerExpert_);
    constexpr uint32_t chunkElements =
        MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes / sizeof(int32_t);
    const uint32_t zeroElements = countElements < chunkElements
        ? static_cast<uint32_t>(countElements) : chunkElements;
    // P6's local TPipe has already been destroyed. These immediate pairs
    // must not fetch events through its former global TPipe pointer.
    SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    LocalTensor<int32_t> tmp{AscendC::TPosition::VECIN, 0, zeroElements};
    Duplicate(tmp, 0, zeroElements);
    SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (uint64_t offset = 0; offset < countElements; offset += chunkElements) {
        const uint32_t elements = countElements - offset < chunkElements
            ? static_cast<uint32_t>(countElements - offset) : chunkElements;
        DataCopy(tokenPerExpertWin_[offset], tmp, elements);
    }
    SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

__aicore__ inline void MegaMoeBF16Backward::CopyGMToGMPerTokenV2(
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantOutType> src,
    int32_t rows, int32_t hiddenSize, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    const uint32_t ringBytes = 2 * dispatchCopyElementsPerPing_ * sizeof(QuantOutType);
    LocalTensor<QuantOutType> tmpBuffer1 = StageUb(ringBytes);
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[dispatchCopyElementsPerPing_];
    auto processCount = Ops::Base::CeilDiv(rows, static_cast<int32_t>(dispatchPingRows_));
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID eventId = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        int32_t rowNum = static_cast<int32_t>(dispatchPingRows_);
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * static_cast<int32_t>(dispatchPingRows_);
        }
        const int64_t elementOffset = static_cast<int64_t>(processIndex)
            * dispatchPingRows_ * hiddenSize;
        const int64_t elementCount = static_cast<int64_t>(rowNum) * hiddenSize;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopy(buf, src[elementOffset], elementCount);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        DataCopy(dst[elementOffset], buf, elementCount);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
}

__aicore__ inline void MegaMoeBF16Backward::DispatchTokensBackward(GM_ADDR dstBase, uint32_t expertIdx, uint32_t ordinal)
{
    MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik(AscendTimer::AIV_BW_DISPATCH, expertIdx));
    int64_t widthA = k_;
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / dispatchBlockNumPerEP_;
        uint32_t aivIdxInGroup = idx % dispatchBlockNumPerEP_;
#if defined(DEEPEP_BF16_STAGE_TIMER) && defined(DEEPEP_BF16_GRADIENT_PIPE)
        uint64_t sourceCopy=0;
#endif
        uint32_t rowStartInDst = (dstRankIdx == 0 ? 0 :
            cumsumInWorkSpace_((dstRankIdx - 1) * expertPerRank_ + expertIdx));
        if (rowStartInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = tokenPerExpertWin_(
                TokenPerExpertLayout(dstRankIdx, rankId_, expertIdx));
            if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartInDst;
            }
            // Forward saved the exclusive source offset for each local
            // expert. Read it directly, without a fixed ten-source cache or
            // an offset advanced by a capacity-clipped receive count.
            const int32_t rowSrcOffset =
                preSumBeforeRankWorkSpace_(dstRankIdx * expertPerRank_ + expertIdx);
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(dispatchBlockNumPerEP_));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
#if defined(DEEPEP_BF16_STAGE_TIMER) && defined(DEEPEP_BF16_GRADIENT_PIPE)
                sourceCopy=MegaFluxStageTimer::BeginAsync(gradient_.trace,true,
                    MegaFluxStageTimer::BWD_SOURCE_COPY_TO_RETIRE,expertIdx,-1,
                    rowDstOffsetPerCore,rowsPerCore,dstRankIdx,aivIdxInGroup);
#endif
                GlobalTensor<QuantOutType> gmRemoteA, aGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * widthA;
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
                    GetRankWinAddrWithOffset(
                        dstRankIdx, params_.peermemInfo.payloadOffset)));
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dstBase));
                if (UseDispatchRowReady(ordinal)) {
                    const auto ready = MakeDispatchRowReady(expertIdx, ordinal);
                    constexpr int32_t chunkRows = 256;
                    for (int32_t done = 0; done < rowsPerCore; done += chunkRows) {
                        const int32_t count = rowsPerCore - done < chunkRows
                            ? rowsPerCore - done : chunkRows;
                        CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA + static_cast<int64_t>(done) * widthA],
                            gmRemoteA[gmOffsetPeer + static_cast<int64_t>(done) * widthA],
                            count, widthA, pingpongId);
                        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                        AtomicAdd(ready.Slot(idx), count);
                    }
                } else {
                    CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA], gmRemoteA[gmOffsetPeer],
                        rowsPerCore, widthA, pingpongId);
                }
            }
        }
        // No TPipe exists in P0. This event type is independent of the
        // dispatch ring's outstanding MTE3_MTE2 events 0/1. Complete the
        // payload writes before PIPE_S publishes this band's ready count.
        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
#if defined(DEEPEP_BF16_STAGE_TIMER) && defined(DEEPEP_BF16_GRADIENT_PIPE)
        MegaFluxStageTimer::EndAsync(sourceCopy);
#endif
        AtomicAdd((__gm__ int32_t*)(bwFlagDispatch_ + expertIdx * INT_CACHELINE), int32_t(1));
#if defined(DEEPEP_BF16_STAGE_TIMER) && defined(DEEPEP_BF16_GRADIENT_PIPE)
        MegaFluxStageTimer::MarkAsync(gradient_.trace,true,
            MegaFluxStageTimer::BWD_SOURCE_RELEASE_REQUEST,expertIdx,-1,-1,-1,dstRankIdx,aivIdxInGroup);
#endif
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(
        AscendTimer::AIV_BW_DISPATCH, expertIdx));
}

#if MEGAFLUX_SHARED_EXPERT_BACKWARD
__aicore__ inline void MegaMoeBF16Backward::XRedispatchStamp(uint32_t phase, uint32_t edge)
{
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        if (sharedArgsV3_.timer) {
            auto out = reinterpret_cast<__gm__ int64_t*>(sharedArgsV3_.timer) +
                (phase * MegaMoeSharedExpert::kBackwardRedispatchTimerLanesV3 + aivCoreIdx_) * 2 + edge;
            WriteGmByPassDCache(out, static_cast<int64_t>(GetSystemCycle()));
        }
    }
#endif
}

__aicore__ inline void MegaMoeBF16Backward::PrepareSharedSavedSources()
{
    if constexpr (g_coreType == AscendC::AIV) {
    MEGAFLUX_STAGE_BEGIN(stageEntrySync,(g_coreType==AscendC::AIV?gradient_.trace:0),true,MegaFluxStageTimer::BW_SYNC,-1,-1,0);
    CrossRankSyncInWorldSize();
    MEGAFLUX_STAGE_END(stageEntrySync);
        DebugBwStage(BW_PROG_P0A, 0);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        GatherRoutes();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_GATHER));
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        SyncAll<true>();
        CrossRankSyncInWorldSize();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_XSYNC_GATHER));
        DebugBwStage(BW_PROG_P0A, 1);
        DebugBwStage(BW_PROG_P0B, 0);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        RestoreTokenPerExpert();
        SyncAll<true>();
        if(MegaMoeBF16Training::BackwardUsesGm(savedGateUpMode_, noneGmEnabled_) && splitExperts_>0 && aivCoreIdx_==0) {
            // word1 of expert0's existing DHp flag cacheline replaces the
            // removed chunk arrivals. All AIV metadata is complete already.
            AtomicAdd(reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDHpReady)+1,int32_t(1));
        }
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_RESTORE_TABLE));
    }
}

__aicore__ inline void MegaMoeBF16Backward::PrepareXRedispatchSources()
{
    if constexpr (g_coreType == AscendC::AIV) {
        XRedispatchStamp(0, 0);
#if MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES > 0
        if (rankId_ == 0 && aivCoreIdx_ == 0) {
            const uint64_t begin = GetSystemCycle();
            while (static_cast<uint64_t>(GetSystemCycle() - begin) <
                static_cast<uint64_t>(MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES)) {}
        }
#endif
        // Each core's Params object is private. Reuse the exact native gather
        // permutation and DMA/event protocol, first for dY and then original X.
        GatherRoutes();
        const auto oldInput = params_.aGmAddr;
        const auto oldBank = params_.peermemInfo.ptrA0;
        params_.aGmAddr = reinterpret_cast<GM_ADDR>(sharedArgsV2_.base.x);
        params_.peermemInfo.ptrA0 = params_.peermemInfo.ptrD;
        GatherRoutes();
        params_.aGmAddr = oldInput;
        params_.peermemInfo.ptrA0 = oldBank;
        // This independent gather TPipe is destroyed before later static UB
        // stages. It preserves raw BF16 bits and publishes with MTE3.
        const bool gathered = MegaMoeBF16Routing::TryGatherTopkWeights(
            params_.probsGmAddr, params_.workspaceInfo.expandedRowIdx,
            params_.peermemInfo.ptrRoutedTopkWeights,
            static_cast<uint64_t>(m_) * topK_, aivCoreIdx_, blockAivNum_, true);
        if (!gathered) {
            constexpr int32_t chunkElements = 1024;
            auto buffer = StageUb(chunkElements * sizeof(bfloat16_t));
            GlobalTensor<int32_t> indices;
            GlobalTensor<bfloat16_t> input, output;
            indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.expandedRowIdx));
            input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params_.probsGmAddr));
            output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params_.peermemInfo.ptrRoutedTopkWeights));
            int32_t count = 0, begin = 0;
            TilingByCore(static_cast<int32_t>(m_ * topK_), count, begin, 1);
            for (int32_t done = 0; done < count; done += chunkElements) {
                const int32_t take = count - done < chunkElements ? count - done : chunkElements;
                for (int32_t i = 0; i < take; ++i)
                    buffer.SetValue(i, input.GetValue(indices.GetValue(begin + done + i)));
                SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
                DataCopyPad(output[begin + done], buffer,
                    DataCopyExtParams{1U, static_cast<uint32_t>(take) * 2U, 0U, 0U, 0U});
                SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        XRedispatchStamp(0, 1);
        XRedispatchStamp(1, 0);
        // Representatives may not publish until every local source producer
        // has drained. CrossRankSync itself only has a trailing local barrier.
        SyncAll<true>();
        CrossRankSyncInWorldSize();
        RestoreTokenPerExpert();
        SyncAll<true>();
        if (splitExperts_ > 0 && aivCoreIdx_ == 0)
            AtomicAdd(reinterpret_cast<__gm__ int32_t*>(bwWs_.ptrBwFlagDHpReady) + 1, int32_t(1));
        XRedispatchStamp(1, 1);
    }
}

__aicore__ inline void MegaMoeBF16Backward::CopyRedispatchProb(
    GlobalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src, int32_t rows)
{
    constexpr int32_t chunkElements = 1024;
    auto buffer = StageUb(chunkElements * sizeof(bfloat16_t));
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    for (int32_t offset = 0; offset < rows; offset += chunkElements) {
        const int32_t take = rows - offset < chunkElements ? rows - offset : chunkElements;
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(take) * 2U, 0U, 0U, 0U};
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        DataCopyPad(buffer, src[offset], copy,
            DataCopyPadExtParams<bfloat16_t>{false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID2);
        DataCopyPad(dst[offset], buffer, copy);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
}

__aicore__ inline void MegaMoeBF16Backward::DispatchXProbBackward(
    uint32_t expert, int64_t expertRowBase)
{
    const uint64_t xOffset = params_.peermemInfo.ptrD - params_.peermemInfo.ptrBase;
    const uint64_t pOffset = params_.peermemInfo.ptrRoutedTopkWeights - params_.peermemInfo.ptrBase;
    GlobalTensor<bfloat16_t> destinationX, destinationP;
    destinationX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(savedState_.ptrA) + expertRowBase * k_);
    destinationP.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(savedState_.ptrDispatchedTopkWeights) + expertRowBase);
    int32_t pingpong = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t i = blockIdx_; i < worldSize_ * dispatchBlockNumPerEP_; i += blockNum_) {
        const uint32_t source = i / dispatchBlockNumPerEP_;
        const uint32_t producer = i % dispatchBlockNumPerEP_;
        const int64_t destinationBase = source == 0 ? 0 :
            cumsumInWorkSpace_((source - 1) * expertPerRank_ + expert);
        const int32_t count = tokenPerExpertWin_(TokenPerExpertLayout(source, rankId_, expert));
        const int64_t sourceBase = preSumBeforeRankWorkSpace_(source * expertPerRank_ + expert);
        const auto span = MegaMoeXRedispatch::SplitProducer(sourceBase, destinationBase,
            count, producer, dispatchBlockNumPerEP_);
        if (span.rows > 0) {
            GlobalTensor<bfloat16_t> remote;
            remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(GetRankWinAddrWithOffset(source, xOffset)));
            CopyGMToGMPerTokenV2(destinationX[span.destinationRow * k_],
                remote[span.sourceRow * k_], span.rows, k_, pingpong);
        }
    }
    // X and P reuse StageUb. Finish both X ring slots before binding the
    // probability staging tile; no live DMA can read overwritten UB.
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t i = blockIdx_; i < worldSize_ * dispatchBlockNumPerEP_; i += blockNum_) {
        const uint32_t source = i / dispatchBlockNumPerEP_;
        const uint32_t producer = i % dispatchBlockNumPerEP_;
        const int64_t destinationBase = source == 0 ? 0 :
            cumsumInWorkSpace_((source - 1) * expertPerRank_ + expert);
        const int32_t count = tokenPerExpertWin_(TokenPerExpertLayout(source, rankId_, expert));
        const int64_t sourceBase = preSumBeforeRankWorkSpace_(source * expertPerRank_ + expert);
        const auto span = MegaMoeXRedispatch::SplitProducer(sourceBase, destinationBase,
            count, producer, dispatchBlockNumPerEP_);
        if (span.rows > 0) {
            GlobalTensor<bfloat16_t> remote;
            remote.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(GetRankWinAddrWithOffset(source, pOffset)));
            CopyRedispatchProb(destinationP[span.destinationRow], remote[span.sourceRow], span.rows);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::ALL>();
}

__aicore__ inline void MegaMoeBF16Backward::TransportXRedispatch()
{
    if constexpr (g_coreType == AscendC::AIV) {
        cumMRows_ = 0;
        preOffset_ = 0;
        uint32_t ordinal = 0;
        if (subBlockIdx_ == 1) XRedispatchStamp(2, 0);
#if MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES > 0
        if (rankId_ == 0 && subBlockIdx_ == 1 && blockIdx_ == 0) {
            const uint64_t begin = GetSystemCycle();
            while (static_cast<uint64_t>(GetSystemCycle() - begin) <
                static_cast<uint64_t>(MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES)) {}
        }
#endif
        for (uint32_t position = 0; position < expertPerRank_; ++position) {
            const uint32_t expert = ExpertAt(position);
            if (!UpdateGroupParams(expert)) continue;
            if (subBlockIdx_ == 0) {
                DispatchTokensBackward(bwWs_.ptrDO + cumMRows_ * k_ * sizeof(bfloat16_t),
                    expert, DispatchOrdinal(expert, ordinal));
            } else {
                DispatchXProbBackward(expert, cumMRows_);
            }
            ++ordinal;
            cumMRows_ += Get<M_VALUE>(problemShape_);
        }
        if (subBlockIdx_ == 1) {
            XRedispatchStamp(2, 1);
#if MEGAFLUX_BACKWARD_RESTORE
            // XP's ring and probability buffers are already drained before
            // the panel service borrows its own UB window on this same lane.
            if (restoreReady_) {
                if (restoreMode_ == 2)
                    MegaFluxBackwardRestore::CopyExpertsPanel(replica_, restoreReady_, restoreEpoch_, gradient_.trace, noneGmEnabled_, lifetimeCache_);
                else
                    MegaFluxBackwardRestore::CopyExperts(replica_, restoreReady_, restoreEpoch_, noneGmEnabled_, lifetimeCache_);
            }
#endif
        }
        cumMRows_ = 0;
        preOffset_ = 0;
    }
}

__aicore__ inline void MegaMoeBF16Backward::RetireXRedispatchSources()
{
    if constexpr (g_coreType == AscendC::AIV) {
        XRedispatchStamp(3, 0);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
        SyncAll<true>();
        CrossRankSyncInWorldSize();
        XRedispatchStamp(3, 1);
    }
}
#endif

__aicore__ inline bool MegaMoeBF16Backward::UpdateGroupParams(uint32_t expertIdx)
{
    int32_t splitValue = 0;
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
    if(gradient_.ready) {
        int64_t prefix=0;
        if(groupListType_==0)prefix=expertIdx?groupList_.GetValue(expertIdx-1):0;
        else for(int e=0;e<expertIdx;++e)prefix+=groupList_.GetValue(e);
        cumMRows_=prefix;preOffset_=prefix;
    }
#endif
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
    if (cumMRows_ + Get<M_VALUE>(problemShape_) > maxOutputSize_) {
        Get<M_VALUE>(problemShape_) = maxOutputSize_ - cumMRows_;
    }
    if (Get<M_VALUE>(problemShape_) <= 0) {
        return false;
    }
    return true;
}

// Split one empty expert's complete gradient plane evenly over every AIV.
// Use an int32_t view so the zero fill is independent of BF16 conversion and
// every core starts at a 32-byte-aligned GM address.  The supported K/N shapes
// make every expert plane a whole number of uint32_t words.  The caller has
// already filled this AIV's fixed staging tile with zero and established its
// V -> MTE3 dependency; this routine only queues bounded UB -> GM copies.
__aicore__ inline void MegaMoeBF16Backward::ClearGradWeightPlane(
    GM_ADDR base, uint64_t planeElements, uint32_t expertIdx)
{
    constexpr uint64_t WORD_BYTES = sizeof(uint32_t);
    constexpr uint64_t WORDS_PER_32_BYTES = ALIGN_32 / WORD_BYTES;
    const uint64_t planeBytes = planeElements * sizeof(bfloat16_t);
    const uint64_t planeWords = planeBytes / WORD_BYTES;
    const uint64_t wordsPerCoreUnaligned =
        (planeWords + blockAivNum_ - 1) / blockAivNum_;
    const uint64_t wordsPerCore =
        ((wordsPerCoreUnaligned + WORDS_PER_32_BYTES - 1) /
         WORDS_PER_32_BYTES) * WORDS_PER_32_BYTES;
    const uint64_t coreOffset = static_cast<uint64_t>(aivCoreIdx_) * wordsPerCore;
    if (coreOffset >= planeWords) {
        return;
    }
    const uint64_t remainingWords = planeWords - coreOffset;
    const uint64_t coreWords =
        wordsPerCore < remainingWords ? wordsPerCore : remainingWords;
    const uint64_t expertOffset = static_cast<uint64_t>(expertIdx) * planeWords;

    GlobalTensor<int32_t> zeroGm;
    zeroGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(base) + expertOffset + coreOffset,
        coreWords);
    LocalTensor<int32_t> zeroUb =
        StageUb(BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES).ReinterpretCast<int32_t>();
    constexpr uint32_t ZERO_CHUNK_WORDS =
        BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES / sizeof(int32_t);
    for (uint64_t offset = 0; offset < coreWords; offset += ZERO_CHUNK_WORDS) {
        const uint32_t words = coreWords - offset < ZERO_CHUNK_WORDS
            ? static_cast<uint32_t>(coreWords - offset) : ZERO_CHUNK_WORDS;
        DataCopy(zeroGm[offset], zeroUb, words);
    }
}

// Non-empty experts are fully overwritten by P3/P4 GEMMs.  Empty experts do
// not launch either GEMM, so all AIVs cooperatively clear only those output
// planes while the AICs compute weight gradients for the non-empty experts.
// Reproduce UpdateGroupParams' maxOutputSize clamp: an expert beyond the
// receive-capacity boundary is also skipped by both GEMMs and must be zero.
__aicore__ inline void MegaMoeBF16Backward::ClearEmptyExpertGradWeights()
{
    int64_t previousGroupEnd = 0;
    uint64_t consumedRows = 0;
    bool zeroUbReady = false;
    const uint64_t gradWeight1PlaneElements =
        static_cast<uint64_t>(n_) * k_;
    const uint64_t gradWeight2PlaneElements =
        static_cast<uint64_t>(k_) *
        (n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF);

    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        const int64_t groupValue =
            static_cast<int32_t>(groupList_.GetValue(expertIdx));
        const int64_t routedRows = groupListType_ == 0
            ? groupValue - previousGroupEnd
            : groupValue;
        if (groupListType_ == 0) {
            previousGroupEnd = groupValue;
        }

        uint64_t effectiveRows = 0;
        if (routedRows > 0 && consumedRows < maxOutputSize_) {
            const uint64_t availableRows = maxOutputSize_ - consumedRows;
            const uint64_t positiveRows = static_cast<uint64_t>(routedRows);
            effectiveRows = positiveRows < availableRows
                ? positiveRows : availableRows;
        }
        if (effectiveRows == 0) {
#if defined(DEEPEP_BF16_EARLY_REDUCE)
            // Do not overwrite a shared empty owner's early tile SUM.
            if(gradient_.ready && !rawGradientOutput_ && expertIdx<replica_.home) {
                bool shared=false;
                for(int i=0;i<replica_.ranks*replica_.slots;++i)
                    if(MegaFluxMixed::Fresh((__gm__ int32_t*)replica_.plan+i)==replica_.rank*replica_.home+expertIdx)shared=true;
                if(shared)continue;
            }
#endif
            if (!zeroUbReady) {
                // P0 and P2 have drained their writes before this stage, so
                // the fixed staging window is free.  Produce one bounded zero
                // tile and reuse it for every empty-expert plane.  This avoids
                // InitGlobalMemory, which is only legal before any InitBuffer
                // allocation in the kernel program.
                LocalTensor<int32_t> zeroUb =
                    StageUb(BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES)
                        .ReinterpretCast<int32_t>();
                constexpr uint32_t ZERO_CHUNK_WORDS =
                    BW_GRAD_WEIGHT_ZERO_CHUNK_BYTES / sizeof(int32_t);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
                Duplicate(zeroUb, static_cast<int32_t>(0), ZERO_CHUNK_WORDS);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
                zeroUbReady = true;
            }
            ClearGradWeightPlane(
                gradWeight1GmAddr_, gradWeight1PlaneElements, expertIdx);
            ClearGradWeightPlane(
                gradWeight2GmAddr_, gradWeight2PlaneElements, expertIdx);
        } else {
            consumedRows += effectiveRows;
        }
    }
    if (zeroUbReady) {
        // All UB -> GM zero copies must finish before this AIV joins the P5
        // local/cross-rank barriers and the caller reuses the staging window.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }
}

__aicore__ inline void MegaMoeBF16Backward::UpdateGlobalBuffer(
    GMMAddrInfo &gmmAddrInfo, uint32_t phase, uint32_t expertIdx)
{
#if MEGAFLUX_BACKWARD_RESTORE
    // All waits precede tensor construction/MMAD operand prefetch. Homes
    // need no replica ready; dW GEMMs do not consume expert weights at all.
    if(restoreReady_ && restoreMode_==1 && expertIdx>=replica_.home &&
       (phase==BW_PHASE_RECOMPUTE || phase==BW_PHASE_GMM2GRAD || phase==BW_PHASE_GMM1GRAD)) {
        MegaFluxBackwardRestore::WaitExpert(restoreReady_,expertIdx-replica_.home,
            phase==BW_PHASE_GMM2GRAD?1:0,restoreEpoch_);
    }
#endif
    const int64_t base = cumMRows_;
    const uint64_t bf16 = sizeof(bfloat16_t);
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
    // A raw output does not publish a ready flag for an unwritten SHMEM partial.
    gmmAddrInfo.gradientReady=rawGradientOutput_?0:gradient_.ready;gmmAddrInfo.gradientEpoch=gradient_.epoch;
    gmmAddrInfo.gradientTrace=gradient_.trace;
    gmmAddrInfo.gradientHidden=k_;gmmAddrInfo.gradientFfn=n_/2;
#endif
#if MEGAFLUX_BACKWARD_RESTORE
    gmmAddrInfo.restoreReady=restoreReady_;
    gmmAddrInfo.restoreEpoch=restoreEpoch_;
    gmmAddrInfo.restoreMode=restoreMode_;
    gmmAddrInfo.restoreSlot=(expertIdx>=replica_.home)?
        static_cast<int>(expertIdx-replica_.home):-1;
    gmmAddrInfo.restoreW1Panels=(2*replica_.ffn+256-1)/256;
    gmmAddrInfo.restoreW2Panels=(replica_.hidden+256-1)/256;
#endif
    if (phase == BW_PHASE_RECOMPUTE) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + base * k_ * bf16;
        gmmAddrInfo.bGlobal =
#if defined(DEEPEP_BF16_REPLICA)
                (expertIdx >= replica_.home
                    ? (GM_ADDR)replica_.replica1 + (expertIdx-replica_.home)*(n_ * k_)*bf16
                    : params_.bGmAddr + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16)
#else
                params_.bGmAddr + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16
#endif
;
    } else if (phase == BW_PHASE_GMM2GRAD) {
        gmmAddrInfo.aGlobal = bwWs_.ptrDO + base * k_ * bf16;
        gmmAddrInfo.bGlobal =
#if defined(DEEPEP_BF16_REPLICA)
                (expertIdx >= replica_.home
                    ? (GM_ADDR)replica_.replica2 + (expertIdx-replica_.home)*(k_ * (n_ / 2))*bf16
                    : params_.b2GmAddr + static_cast<uint64_t>(expertIdx) * k_ * (n_ / 2) * bf16)
#else
                params_.b2GmAddr + static_cast<uint64_t>(expertIdx) * k_ * (n_ / 2) * bf16
#endif
;
    } else if (phase == BW_PHASE_GMM1GRAD) {
        gmmAddrInfo.aGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
        gmmAddrInfo.bGlobal =
#if defined(DEEPEP_BF16_REPLICA)
                (expertIdx >= replica_.home
                    ? (GM_ADDR)replica_.replica1 + (expertIdx-replica_.home)*(n_ * k_)*bf16
                    : params_.bGmAddr + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16)
#else
                params_.bGmAddr + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16
#endif
;
    } else if (phase == BW_PHASE_DW1) {
        if (params_.tilingData->transW) {
            // Canonical [N,K]: dZ^T @ Xpacked.
            gmmAddrInfo.aGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
            gmmAddrInfo.bGlobal = params_.workspaceInfo.ptrA + base * k_ * bf16;
        } else {
            // Legacy [K,N]: Xpacked^T @ dZ.
            gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + base * k_ * bf16;
            gmmAddrInfo.bGlobal = bwWs_.ptrGradZ + base * n_ * bf16;
        }
        gmmAddrInfo.cGlobal = gradWeight1GmAddr_ + static_cast<uint64_t>(expertIdx) * n_ * k_ * bf16;
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
        if(gradient_.ready && !rawGradientOutput_ && expertIdx>=replica_.home)
            gmmAddrInfo.cGlobal=(GM_ADDR)replica_.partial1+(expertIdx-replica_.home)*(n_ * k_)*bf16;
#endif
    } else if (phase == BW_PHASE_DW2) {
        if (params_.tilingData->transW) {
            // Canonical [K,F]: dO^T @ Hp.
            gmmAddrInfo.aGlobal = bwWs_.ptrDO + base * k_ * bf16;
            gmmAddrInfo.bGlobal = bwWs_.ptrHp + base * (n_ / 2) * bf16;
        } else {
            // Legacy [F,K]: Hp^T @ dO.
            gmmAddrInfo.aGlobal = bwWs_.ptrHp + base * (n_ / 2) * bf16;
            gmmAddrInfo.bGlobal = bwWs_.ptrDO + base * k_ * bf16;
        }
        gmmAddrInfo.cGlobal = gradWeight2GmAddr_ + static_cast<uint64_t>(expertIdx) * k_ * (n_ / 2) * bf16;
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
        if(gradient_.ready && !rawGradientOutput_ && expertIdx>=replica_.home)
            gmmAddrInfo.cGlobal=(GM_ADDR)replica_.partial2+(expertIdx-replica_.home)*(k_ * (n_ / 2))*bf16;
#endif
    }
}

__aicore__ inline void MegaMoeBF16Backward::CrossRankSyncInWorldSize()
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.ptrBase +
        MegaMoeBF16Training::PeerSyncCountOffset(worldSize_));
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < static_cast<int>(worldSize_); i += static_cast<int>(blockAivNum_)) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template<bool TransW>
__aicore__ inline void MegaMoeBF16Backward::GMM1RecomputePhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, ColumnMajor, RowMajor>;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true> gmm1Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    // Each phase drives its own GEMM dimensions.  P0-C computes
    // Z = Xpacked @ W1.T with GEMM {m=Me, n=N, k=K}; M is the per-expert
    // token count set by UpdateGroupParams below, N/K are fixed here.
    Get<N_VALUE>(problemShape_) = n_;
    Get<K_VALUE>(problemShape_) = k_;
    for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
        const uint32_t expertIdx=ExpertAt(expertPos);
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        const int64_t expertRowBase = cumMRows_;
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_RECOMPUTE, expertIdx);
        const int64_t planeOffset = expertRowBase *
            (n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF) * sizeof(bfloat16_t);
        GroupMatmulRecompute(
            gmm1Params, problemShape_, gmmAddrInfo,
            bwWs_.ptrRecomputeGate + planeOffset,
            bwWs_.ptrRecomputeUp + planeOffset,
            startBlockIdx, expertIdx
            MEGA_MOE_BF16_TIMER_ARG(timer_));
        cumMRows_ += Get<M_VALUE>(problemShape_);
    }
}

template<bool TransW, bool GmOutput>
__aicore__ inline void MegaMoeBF16Backward::GMM2GradPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive())startBlockIdx=workerDH_;
#endif
    vecSetSyncCom_ = 0;
    uint32_t nextTargetAiv = 0;
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive())nextTargetAiv=workerTarget_;
#endif
    const int32_t dispatchTarget = static_cast<int32_t>(worldSize_ * dispatchBlockNumPerEP_);
    // The dual target copier changes only Fixpipe subBlockId. AIC keeps one
    // BlockMmad instance and one CATLASS L1/L0 event state machine.
    BackwardGroupMatmulParam<RowMajor, WeightLayout,
        GmOutput ? 0U : DUAL_AIV_TARGET, GmOutput> gmm2Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    // P1 GEMM: dH = dO @ W2 is [Me, K] x [K, F]; N = F = n_/2, K = K.
    Get<N_VALUE>(problemShape_) = n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
    Get<K_VALUE>(problemShape_) = k_;
    uint32_t nonemptyOrdinal = 0;
    bool metadataKnown=false;
    for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
        const uint32_t expertIdx=ExpertAt(expertPos);
#if MEGAFLUX_WORKER_BACKWARD
        if(workerExpert_>=0 && expertIdx!=workerExpert_)continue;
#if MEGAFLUX_WORKER_SHARED_PREFIX
        if(workerBulk_ && expertPos<workerBeginPos_)continue;
#endif
#endif
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        const int64_t expertRowBase = cumMRows_;
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_GMM2GRAD, expertIdx);
        __gm__ int32_t* dispatchFlag = bwFlagDispatch_ + expertIdx * INT_CACHELINE;
        if constexpr (GmOutput) {
            const auto rowReady=MakeDispatchRowReady(expertIdx,DispatchOrdinal(expertIdx,nonemptyOrdinal));
            if constexpr(g_coreType==AscendC::AIC) {
                if(!metadataKnown && ProducerPhaseSupported(rowReady,blockNum_)) {
                    // Required P0 readiness only, once per AIC invocation.
                    // No dependency on any earlier source's payload.
                    while(ReadGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDHpReady)+1)!=1) {}
                    metadataKnown=true;
                }
            }
            GMM2GradSwiGLUGradGM(gmm2Params, epilogueOp_, problemShape_, gmmAddrInfo,
                bwWs_.ptrHp + expertRowBase * (n_ / 2) * sizeof(bfloat16_t),
                startBlockIdx, nextTargetAiv, expertIdx, expertRowBase,
                dispatchFlag, dispatchTarget, rowReady,
                reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDHpReady) + expertIdx * INT_CACHELINE
                MEGA_MOE_BF16_TIMER_ARG(timer_));
            if constexpr (g_coreType == AscendC::AIV) {
                // All lanes participate, including those assigned no tiles.
                // Retire dZ/Hp/partials MTE3 before publishing the expert arrival.
                PipeBarrier<PIPE_ALL>();
                AtomicAdd(reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDZReady) +
                    expertIdx * INT_CACHELINE, int32_t(1));
            }
        } else {
            GMM2GradSwiGLUGradDualAiv(gmm2Params, epilogueOp_, params_, problemShape_, gmmAddrInfo,
                startBlockIdx, vecSetSyncCom_, nextTargetAiv, expertIdx, expertRowBase,
                dispatchFlag, dispatchTarget, BW_P1_EVENT
                MEGA_MOE_BF16_TIMER_ARG(timer_));
        }
        ++nonemptyOrdinal;
        cumMRows_ += Get<M_VALUE>(problemShape_);
    }
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive()){workerDH_=startBlockIdx;workerTarget_=nextTargetAiv;}
#endif
}

template<bool TransW, bool ExpertReady>
__aicore__ inline void MegaMoeBF16Backward::GMM1GradPhase()
{
    if constexpr (ExpertReady) {
        GMM1GradReuseDOPhase<TransW>();
    } else {
        using namespace MegaMoeBF16BackwardCatlassImpl;
        using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
        GMMAddrInfo gmmAddrInfo{};
        uint32_t startBlockIdx = 0;
        vecSetSyncCom_ = 0;
        gmm2PingPongIdx_ = 0;
        BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, false> gmm1Params;
        cumMRows_ = 0;
        preOffset_ = 0;
        // P2 GEMM: dX = dZ @ W1 is [Me, N] x [N, K]; N = K, K = N.
        Get<N_VALUE>(problemShape_) = k_;
        Get<K_VALUE>(problemShape_) = n_;
        for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
        const uint32_t expertIdx=ExpertAt(expertPos);
            if (!UpdateGroupParams(expertIdx)) {
                continue;
            }
            UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_GMM1GRAD, expertIdx);
            // Backward mLoc is already expert-relative; use zero slice offset.
            // The shared Combine helper still needs nonzero offsets for MGroup slices.
            GMM1GradCombine<0>(gmm1Params, params_, problemShape_, gmmAddrInfo,
                startBlockIdx, vecSetSyncCom_, expertIdx, 0, gmm2PingPongIdx_
                MEGA_MOE_BF16_TIMER_ARG(timer_));
            cumMRows_ += Get<M_VALUE>(problemShape_);
        }
    }
}

// dO and dX have identical packed-row/K layouts. Retire dW2 for one expert
// before changing that band's lifetime; later experts occupy disjoint bands.
template<bool TransW>
__aicore__ inline void MegaMoeBF16Backward::GMM1GradReuseDOPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    using WeightLayout = std::conditional_t<TransW, RowMajor, ColumnMajor>;
    uint32_t dxStartBlockIdx = 0;
    uint32_t dw2StartBlockIdx = 0;
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive()){dxStartBlockIdx=workerDX_;dw2StartBlockIdx=workerDW2_;}
#endif
    cumMRows_ = 0;
    preOffset_ = 0;
    Get<N_VALUE>(problemShape_) = k_;
    Get<K_VALUE>(problemShape_) = n_;
    for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
        const uint32_t expertIdx=ExpertAt(expertPos);
#if MEGAFLUX_WORKER_BACKWARD
        if(workerExpert_>=0 && expertIdx!=workerExpert_)continue;
#if MEGAFLUX_WORKER_SHARED_PREFIX
        if(workerBulk_ && expertPos<workerBeginPos_)continue;
#endif
#endif
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        auto *dzReady = reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDZReady) +
            expertIdx * INT_CACHELINE;
        auto *doReleased = reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDOReleased) +
            expertIdx * INT_CACHELINE;
        auto *dxReady = reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDXReady) +
            expertIdx * INT_CACHELINE;
        const int64_t rows = Get<M_VALUE>(problemShape_);
        if constexpr (g_coreType == AscendC::AIC) {
            // This counter retires every AIV's dZ AND Hp MTE3 stores. It also
            // implies every AIC finished this expert's original dO -> dHp read.
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
            while (ReadGmByPassDCache(dzReady) < 2 * static_cast<int32_t>(blockNum_)) {}
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(
                MegaMoeBF16BackwardCatlassImpl::AIC_DZ_WAIT_EXPERT_BASE + expertIdx));
            {
                // Do not keep two CATLASS handles alive across a shared L1/L0
                // lifetime. Each scoped handle drains before the next is built.
                BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true> dw2Params;
                auto dwShape = problemShape_;
                Get<M_VALUE>(dwShape) = TransW ? k_ : n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
                Get<N_VALUE>(dwShape) = TransW ? n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF : k_;
                Get<K_VALUE>(dwShape) = rows;
                GMMAddrInfo dwAddr{};
                UpdateGlobalBuffer(dwAddr, BW_PHASE_DW2, expertIdx);
                GMM2DW(dw2Params, dwShape, dwAddr, dw2StartBlockIdx, expertIdx
                    MEGA_MOE_BF16_TIMER_ARG(timer_));
                PipeBarrier<PIPE_ALL>(); // retire this core's dO reads and dW2 writes
            }
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
            // All cores participate, even when assigned no dW2 tiles. A local
            // drain alone cannot permit another core to overwrite this band.
            AtomicAdd(doReleased, int32_t(1));
            while (ReadGmByPassDCache(doReleased) < static_cast<int32_t>(blockNum_)) {}
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(
                MegaMoeBF16BackwardCatlassImpl::AIC_DO_REUSE_WAIT_EXPERT_BASE + expertIdx));
#if MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
            if (earlyDW1Enabled_ && GradientKey(expertIdx) < 10000) {
                // dZ/Hp and every dO reader have retired above. Drain this
                // handle before constructing dX's shared L1/L0 state.
                {
                    BackwardGroupMatmulParam<ColumnMajor, RowMajor, 0, true> dw1Params;
                    GMM1DWExpert<TransW>(dw1Params, expertIdx, rows, earlyDW1Cursor_);
                    PipeBarrier<PIPE_ALL>();
                }
                // Per AIC: even a core assigned zero tiles completed its part.
                earlyDW1Done_ |= uint32_t(1) << expertIdx;
            }
#endif
        }
        {
            BackwardGroupMatmulParam<RowMajor, WeightLayout, 0, true> dxParams;
            GMMAddrInfo dxAddr{};
            UpdateGlobalBuffer(dxAddr, BW_PHASE_GMM1GRAD, expertIdx);
            GMM1GradCombineGM<WeightLayout>(dxParams, params_, problemShape_, dxAddr,
                bwWs_.ptrDX + cumMRows_ * k_ * sizeof(bfloat16_t), dxStartBlockIdx,
                expertIdx, dxReady MEGA_MOE_BF16_TIMER_ARG(timer_));
        }
        cumMRows_ += rows;
    }
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive()){workerDX_=dxStartBlockIdx;workerDW2_=dw2StartBlockIdx;}
#endif
}

#if MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
template<bool TransW, class Params>
__aicore__ inline void MegaMoeBF16Backward::GMM1DWExpert(
    Params& params, uint32_t expertIdx, int64_t rows, uint32_t& cursor)
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    auto shape = problemShape_;
    Get<M_VALUE>(shape) = TransW ? n_ : k_;
    Get<N_VALUE>(shape) = TransW ? k_ : n_;
    Get<K_VALUE>(shape) = rows;
    GMMAddrInfo addr{};
    UpdateGlobalBuffer(addr, BW_PHASE_DW1, expertIdx);
    GMM1DW(params, shape, addr, cursor, expertIdx
        MEGA_MOE_BF16_TIMER_ARG(timer_));
}
#endif

template<bool TransW>
__aicore__ inline void MegaMoeBF16Backward::GMM1DWPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
#if MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
    if (earlyDW1Enabled_) startBlockIdx = earlyDW1Cursor_;
#endif
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive())startBlockIdx=workerDW1_;
#endif
    BackwardGroupMatmulParam<ColumnMajor, RowMajor, 0, true> gmm1Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
        const uint32_t expertIdx=ExpertAt(expertPos);
#if MEGAFLUX_WORKER_BACKWARD
        if(workerExpert_>=0 && expertIdx!=workerExpert_)continue;
#if MEGAFLUX_WORKER_SHARED_PREFIX
        if(workerBulk_ && expertPos<workerBeginPos_)continue;
#endif
#endif
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        // Canonical writes [N,K] = dZ^T @ Xpacked; legacy writes
        // [K,N] = Xpacked^T @ dZ. Save Me before overwriting M so address
        // bookkeeping continues to use the expert token count.
        const int64_t me = Get<M_VALUE>(problemShape_);
#if MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
        if (earlyDW1Enabled_) {
            if (!(earlyDW1Done_ & (uint32_t(1) << expertIdx))) {
                GMM1DWExpert<TransW>(gmm1Params, expertIdx, me, startBlockIdx);
            }
            cumMRows_ += me;
            continue;
        }
#endif
        if constexpr (TransW) {
            Get<M_VALUE>(problemShape_) = n_;
            Get<N_VALUE>(problemShape_) = k_;
        } else {
            Get<M_VALUE>(problemShape_) = k_;
            Get<N_VALUE>(problemShape_) = n_;
        }
        Get<K_VALUE>(problemShape_) = me;
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_DW1, expertIdx);
        GMM1DW(gmm1Params, problemShape_, gmmAddrInfo, startBlockIdx, expertIdx
            MEGA_MOE_BF16_TIMER_ARG(timer_));
        cumMRows_ += me;
    }
#if MEGAFLUX_WORKER_BACKWARD
    if(WorkerCursorActive())workerDW1_=startBlockIdx;
#endif
}

template<bool TransW>
__aicore__ inline void MegaMoeBF16Backward::GMM2DWPhase()
{
    using namespace MegaMoeBF16BackwardCatlassImpl;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0;
    BackwardGroupMatmulParam<ColumnMajor, RowMajor, 1, true> gmm2Params;
    cumMRows_ = 0;
    preOffset_ = 0;
    for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
        const uint32_t expertIdx=ExpertAt(expertPos);
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        // Canonical writes [K,F] = dO^T @ Hp; legacy writes
        // [F,K] = Hp^T @ dO.
        const int64_t me = Get<M_VALUE>(problemShape_);
        if constexpr (TransW) {
            Get<M_VALUE>(problemShape_) = k_;
            Get<N_VALUE>(problemShape_) =
                n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
        } else {
            Get<M_VALUE>(problemShape_) =
                n_ / MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF;
            Get<N_VALUE>(problemShape_) = k_;
        }
        Get<K_VALUE>(problemShape_) = me;
        UpdateGlobalBuffer(gmmAddrInfo, BW_PHASE_DW2, expertIdx);
        GMM2DW(gmm2Params, problemShape_, gmmAddrInfo, startBlockIdx, expertIdx
            MEGA_MOE_BF16_TIMER_ARG(timer_));
        cumMRows_ += me;
    }
}

__aicore__ inline void MegaMoeBF16Backward::UnpermuteBackward()
{
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdxGather;
    expandedRowIdxGather.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<QuantOutType> ptrD;
    ptrD.SetGlobalBuffer((__gm__ QuantOutType*)params_.peermemInfo.ptrD);
    GlobalTensor<QuantOutType> gradX;
    gradX.SetGlobalBuffer((__gm__ QuantOutType*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        // The previous token's MTE3 reads dataResBf16; the next token's
        // first Vector Adds overwrites it.  Order that exact WAR edge.
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        LocalTensor<QuantOutType> dataResBf16 = DataResBuf_.Get<QuantOutType>();
        LocalTensor<QuantOutType> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<QuantOutType> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32 = DataResFp32Buf_.Get<float>();
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];

        // Precision boundary: accumulate routed BF16 dX rows in FP32 and
        // round only once when storing gradX.  This is mathematically more
        // accurate and avoids TopK-dependent BF16 cancellation.  It may not
        // reproduce torch.scatter_add_'s BF16 atomic reduction bit-for-bit,
        // because that path rounds according to its parallel reduction tree.
        for (int32_t expId = 0; expId < static_cast<int32_t>(topK_); ++expId) {
            int32_t rowIdx = expandedRowIdxGather(
                tokenIdx * static_cast<int32_t>(topK_) + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, ptrD[rowIdx * static_cast<int64_t>(k_)], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Adds(dataResFp32, dataInFp32, 0.0F, k_);
            } else {
                Add(dataResFp32, dataResFp32, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
        if (sharedEnabled_) {
            // Keep the original routed BF16 boundary, then add the separately
            // rounded shared dX. Input slot 0 is idle after the routed loop;
            // consume and restore its persistent V_MTE2 event exactly once.
            PipeBarrier<PIPE_V>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            GlobalTensor<QuantOutType> sharedDX;
            sharedDX.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(sharedArgs_.dxg));
            DataCopy(dataIn0Bf16, sharedDX[tokenIdx * static_cast<int64_t>(k_)], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            Cast(dataResFp32, dataResBf16, AscendC::RoundMode::CAST_NONE, k_);
            Cast(dataIn0Fp32, dataIn0Bf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            Add(dataResFp32, dataResFp32, dataIn0Fp32, k_);
            PipeBarrier<PIPE_V>();
            Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
            SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
#endif
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(gradX[tokenIdx * static_cast<int64_t>(k_)], dataResBf16, k_);
    }
    // Retire the last token's GM write before Unpermute's UB pipe is destroyed.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// Vector-reduce one row-major [batchRows, partialTiles] FP32 UB tile to one
// BF16 value per row. The output is packed contiguously for one MTE3 route.
__aicore__ inline void MegaMoeBF16Backward::ReduceDpPartialBatch(
    uint16_t batchRows, uint16_t partialTiles, uint16_t partialStride)
{
    constexpr uint16_t fp32PerReg =
        256U / sizeof(float);
    const uint16_t repeatsPerRow = static_cast<uint16_t>(Ops::Base::CeilDiv(
        static_cast<uint32_t>(partialTiles), static_cast<uint32_t>(fp32PerReg)));
    LocalTensor<float> partialBuf = TopkGradPartialBuf_.Get<float>();
    LocalTensor<bfloat16_t> routeBuf = TopkGradRouteBuf_.Get<bfloat16_t>();
    __ubuf__ float *partialData =
        reinterpret_cast<__ubuf__ float *>(partialBuf.GetPhyAddr());
    __ubuf__ bfloat16_t *routeData =
        reinterpret_cast<__ubuf__ bfloat16_t *>(routeBuf.GetPhyAddr());

    __VEC_SCOPE__ {
        for (uint16_t row = 0; row < batchRows; ++row) {
            AscendC::Reg::RegTensor<float> accumulated;
            AscendC::Reg::MaskReg oneLane =
                AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
            AscendC::Reg::Duplicate(accumulated, 0.0F, oneLane);
            uint16_t remaining = partialTiles;
            for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                const uint16_t lanes = remaining > fp32PerReg
                    ? fp32PerReg : remaining;
                uint32_t maskLanes = lanes;
                AscendC::Reg::MaskReg mask =
                    AscendC::Reg::UpdateMask<float>(maskLanes);
                AscendC::Reg::RegTensor<float> values;
                AscendC::Reg::RegTensor<float> rowSum;
                const uint32_t offset =
                    static_cast<uint32_t>(row) * partialStride +
                    static_cast<uint32_t>(repeat) * fp32PerReg;
                AscendC::Reg::DataCopy(values, partialData + offset);
                AscendC::Reg::ReduceSum(rowSum, values, mask);
                AscendC::Reg::Add(
                    accumulated, accumulated, rowSum, oneLane);
                remaining -= lanes;
            }
            AscendC::Reg::RegTensor<bfloat16_t> result;
            AscendC::Reg::Cast<
                bfloat16_t,
                float,
                MegaMoeBF16Impl::kBackwardCastFP32ToBF16>(
                result, accumulated, oneLane);
            AscendC::Reg::DataCopy<
                bfloat16_t,
                AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(
                routeData + row, result, oneLane);
        }
    }
}

// P6 final reduction and route. P1 has already reduced every 256-wide F tile
// to one FP32 partial. Each AIV owns a contiguous slice of a source-rank /
// expert segment, loads a row batch to UB, reduces it on PIPE_V and sends the
// packed BF16 results to the peer with one MTE3 copy.
__aicore__ inline void MegaMoeBF16Backward::ReduceAndRouteTopkGrad()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    const int32_t f = static_cast<int32_t>(n_ / SWIGLU_N_HALF);
    const int32_t partialTiles = static_cast<int32_t>(Ops::Base::CeilDiv(
        static_cast<uint64_t>(f),
        static_cast<uint64_t>(BACKWARD_DP_PARTIAL_TILE_N)));
    AscendC::GlobalTensor<float> partials;
    partials.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(bwWs_.ptrDpProducts));
    const uint64_t remoteDOffset =
        params_.peermemInfo.ptrD - params_.peermemInfo.ptrBase;
    LocalTensor<bfloat16_t> routeBuf = TopkGradRouteBuf_.Get<bfloat16_t>();
    LocalTensor<float> partialBuf = TopkGradPartialBuf_.Get<float>();
    const int32_t partialCapacity = static_cast<int32_t>(
        BW_TOPK_PARTIAL_UB_BYTES / sizeof(float));
    // Reg::DataCopy loads one full FP32 register even when the following
    // arithmetic masks the tail. Keep one register minus one element as a
    // guard so the final row's final load cannot cross the TBuf boundary.
    constexpr int32_t fp32PerReg =
        256U / sizeof(float);
    const int32_t partialStride = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<uint32_t>(partialTiles),
        static_cast<uint32_t>(ALIGN_32 / sizeof(float))));
    if (partialTiles <= 0 ||
        partialStride > partialCapacity - (fp32PerReg - 1)) {
        return;
    }
    const int32_t rowsByPartialUb =
        (partialCapacity - (fp32PerReg - 1)) / partialStride;
    const int32_t batchCapacity = rowsByPartialUb <
        static_cast<int32_t>(BW_TOPK_ROUTE_BATCH_ROWS)
        ? rowsByPartialUb
        : static_cast<int32_t>(BW_TOPK_ROUTE_BATCH_ROWS);
    int64_t previousGroupEnd = 0;
    int64_t expertBase = 0;
    for (uint32_t expert = 0; expert < expertPerRank_; ++expert) {
        const int64_t groupEnd = static_cast<int32_t>(groupList_.GetValue(expert));
        int64_t expertRows = groupListType_ == 0
            ? groupEnd - previousGroupEnd
            : groupEnd;
        previousGroupEnd = groupListType_ == 0 ? groupEnd : previousGroupEnd;
        if (expertBase + expertRows > static_cast<int32_t>(maxOutputSize_)) {
            expertRows = static_cast<int64_t>(maxOutputSize_) - expertBase;
        }
        if (expertRows <= 0) {
            continue;
        }
        int64_t rankRowBegin = 0;
        for (uint32_t dstRank = 0; dstRank < worldSize_; ++dstRank) {
            int64_t rankRows = tokenPerExpertWin_(TokenPerExpertLayout(
                dstRank, rankId_, expert));
            if (rankRowBegin + rankRows > expertRows) {
                rankRows = expertRows - rankRowBegin;
            }
            if (rankRows <= 0) {
                continue;
            }
            // Partition 32-byte ownership blocks by quotient/remainder. This
            // keeps adjacent AIV writes disjoint without disabling high-index
            // AIVs merely because ceil(rows / cores) rounded up to 16 rows.
            constexpr int32_t rowsPerBlock =
                ALIGN_32 / static_cast<int32_t>(sizeof(bfloat16_t));
            const int32_t ownershipBlocks = static_cast<int32_t>(Ops::Base::CeilDiv(
                static_cast<uint32_t>(rankRows),
                static_cast<uint32_t>(rowsPerBlock)));
            const int32_t blocksPerCore =
                ownershipBlocks / static_cast<int32_t>(blockAivNum_);
            const int32_t extraBlocks =
                ownershipBlocks % static_cast<int32_t>(blockAivNum_);
            const int32_t coreBlocks = blocksPerCore +
                (static_cast<int32_t>(aivCoreIdx_) < extraBlocks ? 1 : 0);
            const int64_t coreBlockBegin =
                static_cast<int64_t>(aivCoreIdx_) * blocksPerCore +
                (static_cast<int32_t>(aivCoreIdx_) < extraBlocks
                    ? static_cast<int32_t>(aivCoreIdx_)
                    : extraBlocks);
            const int64_t coreRowBegin = coreBlockBegin * rowsPerBlock;
            int64_t coreRows = static_cast<int64_t>(coreBlocks) * rowsPerBlock;
            if (coreRowBegin + coreRows > rankRows) {
                coreRows = rankRows - coreRowBegin;
            }
            if (coreRows > 0) {
                const int64_t dstExpertOffset = static_cast<int64_t>(
                    preSumBeforeRankWorkSpace_(
                        dstRank * expertPerRank_ + expert));
                AscendC::GlobalTensor<bfloat16_t> remoteD;
                remoteD.SetGlobalBuffer(
                    reinterpret_cast<__gm__ bfloat16_t *>(
                        GetRankWinAddrWithOffset(dstRank, remoteDOffset)));
                for (int64_t batchBegin = 0; batchBegin < coreRows;
                     batchBegin += batchCapacity) {
                    const int32_t batchRows = coreRows - batchBegin > batchCapacity
                        ? batchCapacity
                        : static_cast<int32_t>(coreRows - batchBegin);
                    const int64_t packedRow = expertBase + rankRowBegin +
                        coreRowBegin + batchBegin;
                    AscendC::DataCopyExtParams partialCopyParams{
                        static_cast<uint16_t>(batchRows),
                        static_cast<uint32_t>(partialTiles * sizeof(float)),
                        0U, 0U, 0U};
                    AscendC::DataCopyPadExtParams<float> partialPadParams{
                        false, 0U, 0U, 0U};
                    AscendC::DataCopyPad(
                        partialBuf,
                        partials[static_cast<int64_t>(packedRow) * partialTiles],
                        partialCopyParams,
                        partialPadParams);
                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    ReduceDpPartialBatch(
                        static_cast<uint16_t>(batchRows),
                        static_cast<uint16_t>(partialTiles),
                        static_cast<uint16_t>(partialStride));
                    // PIPE_V has consumed partialBuf and produced routeBuf.
                    SyncFunc<AscendC::HardEvent::V_MTE2>();
                    SyncFunc<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopyExtParams copyParams{
                        1U,
                        static_cast<uint32_t>(batchRows * sizeof(bfloat16_t)),
                        0U, 0U, 0U};
                    AscendC::DataCopyPad(
                        remoteD[dstExpertOffset + coreRowBegin + batchBegin],
                        routeBuf, copyParams);
                    // PIPE_V fills routeBuf again in the next batch.
                    SyncFunc<AscendC::HardEvent::MTE3_V>();
                }
            }
            rankRowBegin += rankRows;
        }
        expertBase += expertRows;
    }
}

__aicore__ inline void MegaMoeBF16Backward::UnpermuteTopkGrad()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen = 0;
    int32_t coreOffset = 0;
    TilingByCore(static_cast<int32_t>(m_ * topK_), coreLen, coreOffset, 1);
    __gm__ int32_t *routeToPacked = reinterpret_cast<__gm__ int32_t *>(
        params_.workspaceInfo.expandedRowIdxGather);
    __gm__ uint16_t *routedDp = reinterpret_cast<__gm__ uint16_t *>(
        params_.peermemInfo.ptrD);
    __gm__ uint16_t *gradTopk =
        reinterpret_cast<__gm__ uint16_t *>(gradTopkWeightsGmAddr_);
    for (int32_t route = coreOffset; route < coreOffset + coreLen; ++route) {
        const uint16_t dpBf16 = ReadGmByPassDCache(
            routedDp + routeToPacked[route]);
        // TilingByCore may assign adjacent BF16 outputs to different AIVs.
        // Bypass DCache so their scalar stores cannot overwrite one another
        // through cache-line writeback.
        WriteGmByPassDCache(gradTopk + route, dpBf16);
    }
}

#if MEGAFLUX_WORKER_BACKWARD
template<bool TransW>
__aicore__ inline void MegaMoeBF16Backward::WorkerBackward() {
    if constexpr(g_coreType==AIV) {
        int32_t local[32];
        for(int e=0;e<replica_.home;++e) {
            const auto end=groupList_.GetValue(e);
            local[e]=groupListType_==0?end-(e?groupList_.GetValue(e-1):0):end;
        }
        ownerService_.Init(gradient_,gradWeight1GmAddr_,gradWeight2GmAddr_,local);
    }
    int prefix=expertPerRank_;
#if MEGAFLUX_WORKER_SHARED_PREFIX
    // The stable existing order already places every shared home/replica
    // expert first. Only that prefix needs early partial production; bulk
    // unshared work retains CATLASS handles across its remaining experts.
    prefix=0;
    while(prefix<expertPerRank_ && GradientKey(ExpertAt(prefix))<10000)++prefix;
#endif
    for(int pos=0;pos<prefix;++pos) {
        workerExpert_=ExpertAt(pos);
#if MEGAFLUX_WORKER_SHARED_PREFIX
        if(!UpdateGroupParams(workerExpert_))continue;
#endif
        GMM2GradPhase<TransW,true>();
        PipeBarrier<PIPE_ALL>();
#if MEGAFLUX_WORKER_SHARED_PREFIX != 2
        if constexpr(g_coreType==AIV) {
#if MEGAFLUX_WORKER_SHARED_PREFIX
            if(subBlockIdx_==1)
#endif
                ownerService_.Service();
        }
#endif
        GMM1GradReuseDOPhase<TransW>();
        PipeBarrier<PIPE_ALL>();
#if MEGAFLUX_WORKER_SHARED_PREFIX != 2
        if constexpr(g_coreType==AIV) {
#if MEGAFLUX_WORKER_SHARED_PREFIX
            if(subBlockIdx_==1)
#endif
                ownerService_.Service();
        }
#endif
        if constexpr(g_coreType==AIC)GMM1DWPhase<TransW>();
    }
    workerExpert_=-1;
#if MEGAFLUX_WORKER_SHARED_PREFIX
    if(prefix<expertPerRank_) {
        workerBulk_=true;workerBeginPos_=prefix;
        GMM2GradPhase<TransW,true>();
        // Mode2 lends no UB/service time while later SwiGLU still needs AIV1.
        // AIV1's unshared SwiGLU work has all retired. The rest of dX/combine
        // uses AIV0, so odd owners may drain their existing unique residues.
        // No contribution is reordered inside a tile and AIV0 drains at P5.
        if constexpr(g_coreType==AIV)if(subBlockIdx_==1)ownerService_.Drain();
        GMM1GradReuseDOPhase<TransW>();
        if constexpr(g_coreType==AIC)GMM1DWPhase<TransW>();
        workerBulk_=false;
    }
#endif
    if constexpr(g_coreType==AIV) {
        PipeBarrier<PIPE_ALL>();
        AtomicAdd((__gm__ int32_t*)bwFlagAiv1Ready_,int32_t(1));
        AtomicAdd((__gm__ int32_t*)bwFlagHDone_,int32_t(1));
    }
}
#endif

__aicore__ inline void MegaMoeBF16Backward::Process()
{
    // Unsupported policy requests must retire uniformly before shared work.
    if (!backwardPolicyValid_) return;
#if MEGAFLUX_SHARED_EXPERT_BACKWARD
    if (sharedXRedispatchEnabled_ && !sharedEnabled_) {
        InitStageLayout();
        if constexpr (g_coreType == AscendC::AIV)
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_INIT));
        PrepareXRedispatchSources();
        TransportXRedispatch();
        RetireXRedispatchSources();
        // All source readers across ranks have retired before an AIC may
        // consume reconstructed P or reuse ptrD in routed backward. Shared
        // V3 obtains the equivalent fence at the end of RunBackwardS0V3.
        if constexpr (g_coreType == AscendC::AIV)
            MegaMoeSharedExpert::Publish(sharedArgsV2_.base, 0, aivCoreIdx_);
        MegaMoeSharedExpert::AcquireAll(sharedArgsV2_.base, 0, blockAivNum_);
        // On routed-only AIC, INIT includes this necessary transport wait.
        // AIV transport retains the separate bounded four-phase V3 pairs.
        if constexpr (g_coreType == AscendC::AIC)
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIC_INIT));
    }
    if (sharedEnabled_) {
        if (sharedXRedispatchEnabled_) {
            InitStageLayout();
            // ABI3 has real P0 work inside S0. Close pure native Init before
            // dynamic DO timers touch its timestamp state; V2 keeps its old
            // enclosing Init interval and exporter contract unchanged.
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(
                g_coreType == AscendC::AIC ? AIC_INIT : AIV_INIT));
            XRedispatchHooks hooks{this};
            MegaMoeSharedExpert::RunBackwardS0V3(sharedArgsV3_, params_.aGmAddr, hooks);
        } else if (sharedV2Enabled_) {
            // The saved-dispatch path only moves the original P0 source
            // preparation across S0. No X redispatch or route regeneration.
            bool overlap = false;
#if MEGAFLUX_BACKWARD_RESTORE
            overlap = sharedArgsV2_.recompute && restoreMode_ == 2;
#endif
            if (overlap) {
                InitStageLayout();
                MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(
                    g_coreType == AscendC::AIC ? AIC_INIT : AIV_INIT));
                PrepareSharedSavedSources();
            }
            MegaMoeSharedExpert::RunBackwardS0V2(sharedArgsV2_, params_.aGmAddr);
            sharedSavedSourcesPrepared_ = overlap;
        } else {
            MegaMoeSharedExpert::RunBackwardS0(sharedArgs_, params_.aGmAddr);
        }
    }
#endif
#if MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY
    // Invocation-local state. Unsupported saved modes/topologies use the
    // complete original path, including the original tail DW1 computation.
    earlyDW1Cursor_ = 0;
    earlyDW1Done_ = 0;
    earlyDW1Enabled_ = (savedGateUpMode_ == MegaMoeBF16Training::kSaveBf16 || noneGmEnabled_) &&
        gradient_.ready && restoreMode_ == 2 && expertPerRank_ <= 32 &&
        (replica_.ranks == 2 || replica_.ranks == 4 || replica_.ranks == 8);
#endif
    if (!XRedispatchEnabled()) {
    if (!sharedSavedSourcesPrepared_) {
    // P0 staging (gather slots / count-table restore / dispatch ring) is a
    // fixed static UB window above the P1/P2 tile region (StageUb), so P0
    // needs no TPipe and no ownership handover into P1.  P5/P6 allocate
    // their reduction buffers in local pipes as before.
    InitStageLayout();
    // Bridge the wakeup-to-first-stage window (rest of Init + layout setup)
    // so the head of every lane is tiled; start_cycle_ comes from Init's Tik.
    MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(
        g_coreType == AscendC::AIC ? AIC_INIT : AIV_INIT));

    // The Host-side token-table clear is ordered before this kernel on the
    // same stream. The entry device synchronization makes that local
    // ordering visible to every peer without a Host stream synchronize or
    // aclshmem_barrier_all.
    MEGAFLUX_STAGE_BEGIN(stageEntrySync,(g_coreType==AscendC::AIV?gradient_.trace:0),true,MegaFluxStageTimer::BW_SYNC,-1,-1,0);
    CrossRankSyncInWorldSize();
    MEGAFLUX_STAGE_END(stageEntrySync);

    }
    // ---- P0-A: weighted gather grad_y * topk_weights -> ptrA0 (all AIV) ----
    WriteBwProgress(BW_PROG_P0A);
    if constexpr(g_coreType == AscendC::AIV) {
        if (!sharedSavedSourcesPrepared_) {
        DebugBwStage(BW_PROG_P0A, 0);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        GatherRoutes();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_GATHER));
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        SyncAll<true>();
        CrossRankSyncInWorldSize();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_XSYNC_GATHER));
        DebugBwStage(BW_PROG_P0A, 1);
        DebugBwStage(BW_PROG_P0B, 0);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        RestoreTokenPerExpert();
        SyncAll<true>();
        if(MegaMoeBF16Training::BackwardUsesGm(savedGateUpMode_, noneGmEnabled_) && splitExperts_>0 && aivCoreIdx_==0) {
            // word1 of expert0's existing DHp flag cacheline replaces the
            // removed chunk arrivals. All AIV metadata is complete already.
            AtomicAdd(reinterpret_cast<__gm__ int32_t *>(bwWs_.ptrBwFlagDHpReady)+1,int32_t(1));
        }
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_RESTORE_TABLE));
        }
#if MEGAFLUX_BACKWARD_RESTORE
        if(restoreReady_ && subBlockIdx_==1) {
            if(restoreMode_==2)
                MegaFluxBackwardRestore::CopyExpertsPanel(replica_,restoreReady_,restoreEpoch_,gradient_.trace,noneGmEnabled_,lifetimeCache_);
            else
                MegaFluxBackwardRestore::CopyExperts(replica_,restoreReady_,restoreEpoch_,noneGmEnabled_,lifetimeCache_);
        }
#endif
        if (subBlockIdx_ == 0) {
            // AIV0: dispatch dO -> ptrDO for all experts.  Every band
            // publishes bwFlagDispatch_[expert]; the AIC's per-expert
            // DO_WAIT_e consumes it, so later experts are still in flight
            // here while P1 already computes earlier ones.
            cumMRows_ = 0;
            preOffset_ = 0;
            uint32_t nonemptyOrdinal = 0;
            for (uint32_t expertPos = 0; expertPos < expertPerRank_; ++expertPos) {
                const uint32_t expertIdx=ExpertAt(expertPos);
                if (!UpdateGroupParams(expertIdx)) {
                    continue;
                }
                const int64_t expertRowBase = cumMRows_;
                DispatchTokensBackward(
                    bwWs_.ptrDO + expertRowBase * k_ * sizeof(bfloat16_t), expertIdx,
                    DispatchOrdinal(expertIdx,nonemptyOrdinal++));
                cumMRows_ += Get<M_VALUE>(problemShape_);
            }
        }
        // No pipe teardown and no readiness publish: the staging ring is a
        // fixed static region disjoint from the P1 tile area, and AIV1 has
        // no P0-B work at all.
        DebugBwStage(BW_PROG_P0B, 1);
    }

    } // ABI3 already completed P0 under shared S0 and globally retired ptrD reads.

    // ---- P0-C: optional GMM1 fallback ----
    // BF16/MXFP8 modes consume the matching forward-persistent gate/up in P1.
    // Only save=none recomputes GMM1 into backward scratch. The direct-GM
    // fallback does not touch AIV1 UB, so it can overlap P0-A/P0-B.
    WriteBwProgress(BW_PROG_P0B);
    WriteBwProgress(BW_PROG_P0C);
    DebugBwStage(BW_PROG_P0C, 0);
    if constexpr(g_coreType == AscendC::AIV) {
        // AIVs walk the recompute phase without touching the MMAD; time the
        // bookkeeping walk (param ctor + per-expert count-table GM reads).
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
    }
    if (savedGateUpMode_ == MegaMoeBF16Training::kSaveNone) {
        if (params_.tilingData->transW) {
            GMM1RecomputePhase<true>();
        } else {
            GMM1RecomputePhase<false>();
        }
    }
    if constexpr(g_coreType == AscendC::AIV) {
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_RECOMPUTE_PASS));
    }
    DebugBwStage(BW_PROG_P0C, 1);

    // ---- P1: GMM2 Grad || dual-AIV SwiGLU Grad ----
    // No whole-stage ownership wait: P0's staging is a static region above
    // the tile area, so the fixpipe may target either paired AIV while
    // dispatch is still in flight.  P1 entry is gated per expert by
    // DO_WAIT_e (bwFlagDispatch_[e]) inside GMM2GradSwiGLUGradDualAiv, and
    // the per-lane wait is accumulated into AIV_P1_WAIT; the per-expert
    // compute lives in the AIV_SWIGLU_GRAD_e dynamic slots.  Saved gate/up
    // modes skip the GMM1 fallback above entirely, so the AIC reaches
    // DO_WAIT_0 directly and the first-expert transfer is the only P1
    // entry cost; save=none overlaps that transfer with the fallback.
    // For opt-in none/GM, every AIC completes GU recompute before P1. P1's
    // existing PIPE_ALL + all-AIC dHpReady publication also retires those GU
    // writes before either AIV reads them; no extra GU barrier is needed.
    WriteBwProgress(BW_PROG_P1);
    DebugBwStage(BW_PROG_P1, 0);
    const bool gmOutput = MegaMoeBF16Training::BackwardUsesGm(savedGateUpMode_, noneGmEnabled_);
#if defined(MEGAFLUX_OWNER_AIV1_TAIL) && MEGAFLUX_OWNER_AIV1_TAIL
    const bool aiv1TakesTail = gmOutput && savedGateUpMode_ == MegaMoeBF16Training::kSaveBf16;
#endif
#if MEGAFLUX_WORKER_BACKWARD
    cooperativeBackward_=gmOutput && gradient_.ready && !rawGradientOutput_;
    if(cooperativeBackward_) {
        if(params_.tilingData->transW)WorkerBackward<true>();else WorkerBackward<false>();
    } else {
#endif
    if (gmOutput) {
        if (params_.tilingData->transW) { GMM2GradPhase<true, true>(); }
        else { GMM2GradPhase<false, true>(); }
    } else {
        if (params_.tilingData->transW) { GMM2GradPhase<true, false>(); }
        else { GMM2GradPhase<false, false>(); }
    }
    if (!gmOutput) {
        const int32_t p1OutstandingMask = vecSetSyncCom_;
        if constexpr(g_coreType == AscendC::AIC) {
            // Pair-local tail events plus the all-AIV completion counter form one
            // detailed AIC_SPIN_DZ bar in the reworked timer view.
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
            if ((p1OutstandingMask & 1) != 0) {
                WaitForVector(BW_P1_EVENT); // AIV0 wrote its last dZ tile
            }
            if ((p1OutstandingMask & 2) != 0) {
                WaitForVector(BW_P1_EVENT + FLAG_ID_MAX); // AIV1 wrote the last dZ tile
            }
            // dW1 = dZ^T @ Xpacked reduces across the complete expert token band,
            // so pair-local WaitForVector is insufficient: every paired AIV must have
            // completed P1 before any AIC is allowed to consume dZ in P2/P3.
            // P1 contributes two blockNum bands, one arrival from every AIV0 and
            // AIV1 (the old P0 ownership band is gone with the static staging).
            const int32_t p1ReadyTarget = 2 * static_cast<int32_t>(blockNum_);
            while (ReadGmByPassDCache(bwFlagAiv1Ready_) < p1ReadyTarget) {
                const int64_t start = GetSystemCycle();
                while (GetSystemCycle() - start < 100) {}
            }
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIC_SPIN_DZ));
        }
    }
    if constexpr(g_coreType == AscendC::AIV) {
        // Each AIV publishes only after all of its dZ/Hp/dpPartials MTE3
        // writes are visible. This is the global boundary consumed above.
        PipeBarrier<PIPE_ALL>();
        AtomicAdd((__gm__ int32_t*)bwFlagAiv1Ready_, int32_t(1));
        AtomicAdd((__gm__ int32_t*)bwFlagHDone_, int32_t(1));
    }
    DebugBwStage(BW_PROG_P1, 1);

#if defined(DEEPEP_BF16_EARLY_REDUCE)
    if constexpr(g_coreType==AscendC::AIV) {
        if(gmOutput && gradient_.ready && !rawGradientOutput_ && subBlockIdx_==1) {
            // P1 arrival counts are published. GM P2 needs only AIC/AIV0.
            int32_t localRows[32];
            for(int e=0;e<replica_.home;++e) {
                const auto end=groupList_.GetValue(e);
                localRows[e]=groupListType_==0?end-(e?groupList_.GetValue(e-1):0):end;
            }
#if defined(MEGAFLUX_OWNER_AIV1_TAIL) && MEGAFLUX_OWNER_AIV1_TAIL
            if (aiv1TakesTail) {
                MegaFluxGradient::Reduce<true>(gradient_,gradWeight1GmAddr_,gradWeight2GmAddr_,localRows);
            } else
#endif
            {
                MegaFluxGradient::Reduce(gradient_,gradWeight1GmAddr_,gradWeight2GmAddr_,localRows);
            }
            gradientReducedEarly_=true;
        }
    }
#endif

    // ---- P2: GMM1 Grad || Combine (AIC + AIV0) ----
    WriteBwProgress(BW_PROG_P2);
    DebugBwStage(BW_PROG_P2, 0);
    // GM P2 writes dX into dO; legacy none uses AIV0 UB only
    // after the full P1 barrier. Each AIV enters here after its own P1 loop.
    if (gmOutput) {
#if defined(DEEPEP_BF16_EARLY_REDUCE)
        if(!gradientReducedEarly_) {
#endif
        if (params_.tilingData->transW) { GMM1GradPhase<true, true>(); }
        else { GMM1GradPhase<false, true>(); }
#if defined(DEEPEP_BF16_EARLY_REDUCE)
        }
#endif
    } else {
        if (params_.tilingData->transW) { GMM1GradPhase<true, false>(); }
        else { GMM1GradPhase<false, false>(); }
    }
    if constexpr(g_coreType == AscendC::AIC) {
        if (!gmOutput) {
            // Tail sync: ensure AIV0 consumed every dX tile before P3/P4.
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
            if (vecSetSyncCom_ == 1) {
                WaitForVector(0);
            } else if (vecSetSyncCom_ >= 2) {
                WaitForVector(gmm2PingPongIdx_);
                WaitForVector(1 - gmm2PingPongIdx_);
            }
            MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIC_P2_TAIL));
        }
        // GM dX remains live until kernel exit. P3/P4 may overlap combine;
        // the existing drained local + cross-rank P5 barrier protects ptrD.
    }
    DebugBwStage(BW_PROG_P2, 1);

    // ---- P3/P4: weight gradients (AIC) overlapped with P5 (AIV) ----
    if constexpr(g_coreType == AscendC::AIC) {
        WriteBwProgress(BW_PROG_P3);
        DebugBwStage(BW_PROG_P3, 0);
        if (params_.tilingData->transW) {
            GMM1DWPhase<true>();
        } else {
            GMM1DWPhase<false>();
        }
        DebugBwStage(BW_PROG_P3, 1);
        WriteBwProgress(BW_PROG_SPIN_HDONE);
        DebugBwStage(BW_PROG_SPIN_HDONE, 0);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        while (2 * static_cast<int32_t>(blockNum_) !=
               ReadGmByPassDCache(bwFlagHDone_)) {
            const int64_t start = GetSystemCycle();
            while (GetSystemCycle() - start < 100) {}
        }
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIC_SPIN_HP));
        DebugBwStage(BW_PROG_SPIN_HDONE, 1);
        if (!gmOutput) {
            WriteBwProgress(BW_PROG_P4);
            DebugBwStage(BW_PROG_P4, 0);
            if (params_.tilingData->transW) {
                GMM2DWPhase<true>();
            } else {
                GMM2DWPhase<false>();
            }
            DebugBwStage(BW_PROG_P4, 1);
        }
    }
#if MEGAFLUX_WORKER_BACKWARD
    } // unchanged bulk S9 path
#endif
    if constexpr(g_coreType == AscendC::AIV) {
        WriteBwProgress(BW_PROG_P5);
        DebugBwStage(BW_PROG_P5, 0);
        // P3/P4 use all AICs for non-empty expert DW.  Use all paired AIVs for
        // the complementary empty-expert zero fill so no single core clears a
        // complete weight plane and the writes overlap the AIC GEMMs.
        MEGAFLUX_STAGE_BEGIN(stageEmptyClear,gradient_.trace,true,MegaFluxStageTimer::BW_EMPTY_GRAD_CLEAR);
        ClearEmptyExpertGradWeights();
        MEGAFLUX_STAGE_END(stageEmptyClear);
#if defined(DEEPEP_BF16_GRADIENT_PIPE)
        if(gradient_.ready && !rawGradientOutput_) {
            PipeBarrier<PIPE_ALL>();
#if defined(DEEPEP_BF16_EARLY_REDUCE)
            // In GM saved mode the odd workers already reduced their own
            // tiles after P1. Shared empty owners are initialized by Reduce,
            // and ClearEmptyExpertGradWeights skips those experts. Nonempty
            // owner tiles are individually protected by ready epochs. The
            // even workers therefore need no global empty-clear rendezvous
            // here: it would unnecessarily wait for the odd consumers.
            // Keep the original rendezvous outside the early GM path. The
            // drained local + cross-rank barrier below still protects ptrD.
            if(!gmOutput)SyncAll<true>();
#else
            SyncAll<true>();
#endif
            int32_t localRows[32];
            for(int expert=0;expert<replica_.home;++expert) {
                const auto end=groupList_.GetValue(expert);
                localRows[expert]=groupListType_==0?end-(expert?groupList_.GetValue(expert-1):0):end;
            }
#if MEGAFLUX_WORKER_BACKWARD
            if(cooperativeBackward_)ownerService_.Drain();else
#endif
#if defined(DEEPEP_BF16_EARLY_REDUCE)
            if(!gradientReducedEarly_
#if defined(MEGAFLUX_OWNER_AIV1_TAIL) && MEGAFLUX_OWNER_AIV1_TAIL
                && !aiv1TakesTail
#endif
            )
#endif
            MegaFluxGradient::Reduce(gradient_,gradWeight1GmAddr_,gradWeight2GmAddr_,localRows);
        }
#endif
        // Every AIV0 may still be sending dX tiles into a peer's ptrD while
        // its paired AIV1 has already finished H recompute.  Do not publish
        // cross-rank completion until all local AIV pipelines are drained and
        // every local AIV has reached this point; otherwise a peer can start
        // top-k unpermute from a partially-written ptrD.
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        PipeBarrier<PIPE_ALL>();
        // TikPair drains before its timestamp; use a completion point rather
        // than misrepresenting a second barrier as the original drain cost.
#if defined(DEEPEP_BF16_STAGE_TIMER)
        MegaFluxStageTimer::Mark(gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,1);
#endif
        MEGAFLUX_STAGE_BEGIN(stageDxLocal,gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,2);
        SyncAll<true>();
        MEGAFLUX_STAGE_END(stageDxLocal);
        MEGAFLUX_STAGE_BEGIN(stageDxRemote,gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,3);
        CrossRankSyncInWorldSize();
        MEGAFLUX_STAGE_END(stageDxRemote);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_XSYNC_UNPERMUTE));
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        TPipe unpermutePipe;
        UnpermuteBuffInit(&unpermutePipe);
        UnpermuteBackward();
        PipeBarrier<PIPE_ALL>();
        unpermutePipe.Destroy();
        SyncAll<true>();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_UNPERMUTE));
        DebugBwStage(BW_PROG_P5, 1);

        // ptrD held routed dX until every source rank completed P5.  A local
        // barrier alone cannot make that region safe to overwrite: first drain
        // all local readers, then establish the cross-rank happens-before edge.
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        MEGAFLUX_STAGE_BEGIN(stageDpReuse,gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,4);
        CrossRankSyncInWorldSize();
        MEGAFLUX_STAGE_END(stageDpReuse);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_XSYNC_DP));
        WriteBwProgress(BW_PROG_P6);
        DebugBwStage(BW_PROG_P6, 0);
        TPipe topkGradPipe;
        TopkGradBuffInit(&topkGradPipe);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        ReduceAndRouteTopkGrad();
        PipeBarrier<PIPE_ALL>();
        topkGradPipe.Destroy();
        SyncAll<true>();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_DP_REDUCE));
        // Batched remote dp writes must be globally complete before any source
        // rank reads its ptrD route table.
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        MEGAFLUX_STAGE_BEGIN(stageDpWrites,gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,5);
        CrossRankSyncInWorldSize();
        MEGAFLUX_STAGE_END(stageDpWrites);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_XSYNC_TOPK));
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        UnpermuteTopkGrad();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_TOPK_UNPERMUTE));
        DebugBwStage(BW_PROG_P6, 1);
        // Last table reader (ReduceAndRouteTopkGrad) is done and no peer
        // writes this rank's table inside the backward kernel. Re-zero the
        // window table, then use the exit device barrier to prevent peers from
        // starting their next forward send until every rank is ready. The next forward
        // (which may directly follow this backward, e.g. repeated
        // forward+backward training iterations) starts from a clean table.
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
        ResetTokenPerExpert();
        MEGAFLUX_STAGE_BEGIN(stageExitLocal,gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,6);
        SyncAll<true>();
        MEGAFLUX_STAGE_END(stageExitLocal);
        MEGAFLUX_STAGE_BEGIN(stageExitRemote,gradient_.trace,true,MegaFluxStageTimer::BW_SYNC,-1,-1,7);
        CrossRankSyncInWorldSize();
        MEGAFLUX_STAGE_END(stageExitRemote);
        MEGA_MOE_BF16_TIMER_BLOCK(timer_.template Tok<Overwrite>(AIV_TABLE_RESET));
    }
    MEGA_MOE_BF16_TIMER_BLOCK(timerAll_.template Tok<Overwrite>(BW_KERNEL_TIMING_IDX));
}

}  // namespace MegaMoeImpl
#endif
