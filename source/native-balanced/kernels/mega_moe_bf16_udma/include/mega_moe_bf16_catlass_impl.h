/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * DeepEP-native BF16 grouped GEMM stages for MegaMoe.  This keeps the
 * existing MegaMoe expert scheduler and communication protocol; only the
 * matrix data path is replaced with BF16 x BF16 MMAD.  The A5 MMAD pipeline
 * accumulates in L0C/FP32 and Fixpipe writes BF16 tiles to UB.
 */
#ifndef DEEPEP_MEGA_MOE_BF16_UDMA_CATLASS_IMPL_H
#define DEEPEP_MEGA_MOE_BF16_UDMA_CATLASS_IMPL_H

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
// CATLASS 48f7461 pulls in unused W4A8 VF members rejected by CANN 9.2.
// Match the ordinary BF16 forward/backward include workaround.
#define CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#define CATLASS_GEMM_BLOCK_MMAD_A8W4_MX_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#undef CATLASS_GEMM_BLOCK_MMAD_A8W4_MX_HPP
#undef CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "block_epilogue_swiglu_bf16.h"
#include "copy_l0c_to_ub.hpp"
#include "mega_moe_base.h"
#include "mega_moe_bf16_timer.h"
#include "mega_moe_combine_send.h"
#include "shmem.h"

namespace MegaMoeBF16CatlassImpl {

using ArchTag = Catlass::Arch::Ascend950;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<
    ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
// Traverse three M tiles together so a complete GMM2 row-group becomes
// available to the UDMA combine sender earlier.  The legacy BF16 path keeps
// its original scheduler.
using Gmm2BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

constexpr uint32_t L1_TILE_M = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
static_assert(
    2U * MAX_SINGLE_MN * sizeof(bfloat16_t) <= ArchTag::UB_SIZE,
    "The two BF16 GMM1 output tiles must fit in A5 UB");

template <uint32_t TargetUBIdx>
struct PackedBF16TileCopyTlaToTargetUB
    : public Catlass::Gemm::Tile::PackedTileCopyTla<
          ArchTag,
          bfloat16_t,
          Catlass::layout::RowMajor,
          bfloat16_t,
          Catlass::layout::ColumnMajor,
          bfloat16_t,
          Catlass::layout::RowMajor,
          void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag,
        bfloat16_t,
        Catlass::layout::RowMajor,
        bfloat16_t,
        Catlass::layout::ColumnMajor,
        bfloat16_t,
        Catlass::layout::RowMajor,
        void>;
    using TensorL0C = typename Base::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToTargetUBTla<
        ArchTag,
        TensorL0C,
        TensorC,
        Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
        Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
        TargetUBIdx,
        false>;
};

template <uint32_t GmmIndex>
class GroupMatmulParam {
public:
    static_assert(GmmIndex < 2, "GMM1/GMM2 must target AIV0/AIV1");
    using TileCopy = PackedBF16TileCopyTlaToTargetUB<GmmIndex>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy,
        L1TileShape,
        L0TileShape,
        bfloat16_t,
        bfloat16_t,
        bfloat16_t,
        void,
        TileCopy>;
    using LayoutB = decltype(tla::MakeLayout<
        bfloat16_t, Catlass::layout::ColumnMajor>(uint32_t{}, uint32_t{}));
    using LayoutC = decltype(tla::MakeLayout<
        bfloat16_t, Catlass::layout::RowMajor>(uint32_t{}, uint32_t{}));

    __aicore__ inline GroupMatmulParam(
        uint32_t nValue,
        uint32_t kValue,
        uint32_t dispatchBlockNumPerEPValue = 0)
        : blockMmad(resource),
          blockNum(AscendC::GetBlockNum()),
          subBlockIdx(AscendC::GetSubBlockIdx()),
          blockIdx(AscendC::GetBlockIdx() / AscendC::GetTaskRation()),
          dispatchBlockNumPerEP(dispatchBlockNumPerEPValue),
          n(nValue),
          k(kValue),
          outputN(nValue / SWIGLU_N_HALF),
          layoutB(tla::MakeLayout<
              bfloat16_t, Catlass::layout::ColumnMajor>(kValue, nValue)),
          layoutC(tla::MakeLayout<
              bfloat16_t, Catlass::layout::RowMajor>(L1_TILE_M, L1_TILE_N))
    {
        l0cOutUbFirst = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
        l0cOutUbSecond = resource.ubBuf.template GetBufferByByte<bfloat16_t>(
            MAX_SINGLE_MN * sizeof(bfloat16_t));
    }

    Catlass::Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst;
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t blockIdx;
    uint32_t dispatchBlockNumPerEP;
    uint32_t n;
    uint32_t k;
    uint32_t outputN;
    LayoutB layoutB;
    LayoutC layoutC;
};

struct RowGroupCombineSenderState {
    bool active{false};
    uint32_t dstRank{0};
    uint32_t groupIdx{0};
    uint32_t nextRowGroup{0};
    uint32_t rowGroupCount{0};
    uint32_t maxRowGroups{0};
    int32_t expectedNTiles{0};
    uint64_t expertRowBase{0};
    uint64_t expertRows{0};
    uint64_t peerRowBegin{0};
    uint64_t peerRowEnd{0};
    uint64_t dstExpertOffset{0};
    uint64_t rowBytes{0};
};

constexpr uint32_t MAX_COMBINE_PEERS_PER_OWNER = 10;
// Keep the previous large-EP whole peer-expert slice path available for A/B
// or rollback, but default to row-group streaming for every EP size.  The
// current UDMA benchmark shows that multiple 64-KiB-class NBI requests can be
// accumulated before the final per-peer quiet, so large EP no longer needs to
// delay its first transfer until the complete peer-expert slice is ready.
constexpr bool ENABLE_LARGE_EP_WHOLE_PEER_SLICE = false;

struct PeerCombineSenderState {
    bool issuedRemotePut;
    bool hasCurrentExpert;
    uint32_t dstRank;
    uint32_t nextExpert;
    RowGroupCombineSenderState currentExpert;
};

struct CrossExpertRowGroupCombineSenderState {
    bool active;
    bool sendWholePeerSlice;
    bool useLegacyFallback;
    int32_t rankId;
    uint32_t registeredExpertCount;
    uint32_t ownerPeerCount;
    uint32_t nextPeerToTry;
    uint32_t maxRowGroups;
    int32_t expectedNTiles;
    uint64_t rowBytes;
    PeerCombineSenderState peers[MAX_COMBINE_PEERS_PER_OWNER];
};

__aicore__ inline void ResetRowGroupCombineSenderState(
    RowGroupCombineSenderState &state)
{
    state.active = false;
    state.dstRank = 0;
    state.groupIdx = 0;
    state.nextRowGroup = 0;
    state.rowGroupCount = 0;
    state.maxRowGroups = 0;
    state.expectedNTiles = 0;
    state.expertRowBase = 0;
    state.expertRows = 0;
    state.peerRowBegin = 0;
    state.peerRowEnd = 0;
    state.dstExpertOffset = 0;
    state.rowBytes = 0;
}

__aicore__ inline void InitCrossExpertRowGroupCombineSender(
    CrossExpertRowGroupCombineSenderState &state,
    const GroupMatmulParam<1> &gmm2Params,
    const Params &params,
    uint32_t outputN,
    int32_t rankId)
{
    if constexpr (g_coreType != AscendC::AIV) {
        return;
    }
    state.active = false;
    state.sendWholePeerSlice = false;
    state.useLegacyFallback = false;
    state.rankId = rankId;
    state.registeredExpertCount = 0;
    state.ownerPeerCount = 0;
    state.nextPeerToTry = 0;
    state.maxRowGroups = 0;
    state.expectedNTiles = 0;
    state.rowBytes = 0;
    const uint32_t worldSize = params.tilingData->epWorldSize;
    if (gmm2Params.subBlockIdx != 1 || gmm2Params.blockIdx >= worldSize) {
        return;
    }

    state.maxRowGroups = Ops::Base::CeilDiv(
        static_cast<uint32_t>(params.tilingData->maxOutputSize), L1_TILE_M);
    state.expectedNTiles = static_cast<int32_t>(
        Ops::Base::CeilDiv(outputN, L1_TILE_N));
    state.rowBytes = static_cast<uint64_t>(outputN) * sizeof(bfloat16_t);
    state.sendWholePeerSlice = ENABLE_LARGE_EP_WHOLE_PEER_SLICE &&
        worldSize > gmm2Params.blockNum;
    state.useLegacyFallback =
        worldSize > MAX_COMBINE_PEERS_PER_OWNER * gmm2Params.blockNum;
    if (state.useLegacyFallback) {
        return;
    }

    for (uint32_t dstRank = gmm2Params.blockIdx;
         dstRank < worldSize;
         dstRank += gmm2Params.blockNum) {
        if (dstRank == static_cast<uint32_t>(rankId)) {
            continue;
        }
        PeerCombineSenderState &peer = state.peers[state.ownerPeerCount];
        peer.issuedRemotePut = false;
        peer.hasCurrentExpert = false;
        peer.dstRank = dstRank;
        peer.nextExpert = 0;
        ResetRowGroupCombineSenderState(peer.currentExpert);
        ++state.ownerPeerCount;
    }
    state.active = state.ownerPeerCount != 0;
}

__aicore__ inline void RegisterCrossExpertCombineExpert(
    CrossExpertRowGroupCombineSenderState &state,
    uint32_t groupIdx)
{
    if constexpr (g_coreType != AscendC::AIV) {
        return;
    }
    if (state.active && state.registeredExpertCount < groupIdx + 1U) {
        state.registeredExpertCount = groupIdx + 1U;
    }
}

__aicore__ inline void InitRowGroupCombineSenderForExpert(
    RowGroupCombineSenderState &state,
    const CrossExpertRowGroupCombineSenderState &pipeline,
    const Params &params,
    uint32_t dstRank,
    uint32_t groupIdx,
    int32_t rankId)
{
    if constexpr (g_coreType != AscendC::AIV) {
        return;
    }
    if (!pipeline.active) {
        return;
    }
    const uint32_t worldSize = params.tilingData->epWorldSize;
    const uint32_t expertPerRank = params.tilingData->expertPerRank;
    AscendC::GlobalTensor<int32_t> groupList;
    groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.ptrcumsumMM) + (worldSize - 1U) * expertPerRank);

    uint64_t expertRowBase = 0;
    uint64_t expertRows = 0;
    if (params.tilingData->groupListType == 0U) {
        const uint64_t groupEnd = static_cast<uint64_t>(groupList.GetValue(groupIdx));
        expertRowBase = groupIdx == 0U
            ? 0U : static_cast<uint64_t>(groupList.GetValue(groupIdx - 1U));
        expertRows = groupEnd - expertRowBase;
    } else {
        for (uint32_t priorExpert = 0; priorExpert < groupIdx; ++priorExpert) {
            expertRowBase += static_cast<uint64_t>(groupList.GetValue(priorExpert));
        }
        expertRows = static_cast<uint64_t>(groupList.GetValue(groupIdx));
    }
    const uint64_t maxOutputSize = static_cast<uint64_t>(params.tilingData->maxOutputSize);
    if (expertRowBase >= maxOutputSize) {
        expertRows = 0;
    } else if (expertRowBase + expertRows > maxOutputSize) {
        expertRows = maxOutputSize - expertRowBase;
    }

    state.active = true;
    state.dstRank = dstRank;
    state.groupIdx = groupIdx;
    state.expertRowBase = expertRowBase;
    state.expertRows = expertRows;
    state.rowGroupCount = Ops::Base::CeilDiv(
        static_cast<uint32_t>(expertRows), L1_TILE_M);
    state.maxRowGroups = pipeline.maxRowGroups;
    state.expectedNTiles = pipeline.expectedNTiles;
    state.rowBytes = pipeline.rowBytes;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.peermemInfo.ptrTokenPerExpert));
    const uint32_t tokenStride = Ops::Base::CeilAlign(
        worldSize * expertPerRank, static_cast<uint32_t>(ALIGN_128));
    for (uint32_t peer = 0; peer < state.dstRank; ++peer) {
        state.peerRowBegin += static_cast<uint64_t>(tokenPerExpert(
            peer * tokenStride + rankId * expertPerRank + groupIdx));
    }
    int32_t peerRows = tokenPerExpert(
        state.dstRank * tokenStride + rankId * expertPerRank + groupIdx);
    if (state.peerRowBegin >= state.expertRows) {
        peerRows = 0;
    } else if (state.peerRowBegin + static_cast<uint64_t>(peerRows) > state.expertRows) {
        peerRows = static_cast<int32_t>(state.expertRows - state.peerRowBegin);
    }
    state.peerRowEnd = state.peerRowBegin + static_cast<uint64_t>(peerRows);

    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.ptrSumBeforeRank));
    state.dstExpertOffset = static_cast<uint64_t>(preSumBeforeRank(
        state.dstRank * expertPerRank + groupIdx));
}

__aicore__ inline bool IsCombineRowGroupReady(
    const RowGroupCombineSenderState &state,
    const Params &params,
    uint32_t rowGroupIdx)
{
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.ptrCombineReady) +
        state.groupIdx * state.maxRowGroups + rowGroupIdx;
    return AscendC::ReadGmByPassDCache(ready) >= state.expectedNTiles;
}

__aicore__ inline bool IsPeerSliceReady(
    const RowGroupCombineSenderState &state,
    const Params &params)
{
    const uint32_t firstRowGroup = static_cast<uint32_t>(
        state.peerRowBegin / L1_TILE_M);
    const uint32_t lastRowGroup = Ops::Base::CeilDiv(
        static_cast<uint32_t>(state.peerRowEnd), L1_TILE_M);
    for (uint32_t rowGroupIdx = firstRowGroup;
         rowGroupIdx < lastRowGroup;
         ++rowGroupIdx) {
        if (!IsCombineRowGroupReady(state, params, rowGroupIdx)) {
            return false;
        }
    }
    return true;
}

__aicore__ inline bool PrepareNextCombineRowGroup(
    RowGroupCombineSenderState &state,
    uint64_t &copyRowBegin,
    uint64_t &copyRowEnd)
{
    while (state.nextRowGroup < state.rowGroupCount) {
        const uint64_t groupRowBegin =
            static_cast<uint64_t>(state.nextRowGroup) * L1_TILE_M;
        const uint64_t rawGroupRowEnd = groupRowBegin + L1_TILE_M;
        const uint64_t groupRowEnd = rawGroupRowEnd < state.expertRows
            ? rawGroupRowEnd : state.expertRows;
        copyRowBegin = groupRowBegin > state.peerRowBegin
            ? groupRowBegin : state.peerRowBegin;
        copyRowEnd = groupRowEnd < state.peerRowEnd
            ? groupRowEnd : state.peerRowEnd;
        if (copyRowBegin < copyRowEnd) {
            return true;
        }
        ++state.nextRowGroup;
    }
    return false;
}

__aicore__ inline void PutCombineRows(
    RowGroupCombineSenderState &state,
    const Params &params,
    uint64_t copyRowBegin,
    uint64_t copyRowEnd)
{
    constexpr uint64_t MAX_DMA_COPY_BYTES = 0xFFFFFFE0ULL;
    const uint64_t copyBytes = (copyRowEnd - copyRowBegin) * state.rowBytes;
    __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(
        params.peermemInfo.ptrA1 +
        (state.expertRowBase + copyRowBegin) * state.rowBytes);
    __gm__ uint8_t *dst = reinterpret_cast<__gm__ uint8_t *>(
        params.peermemInfo.ptrD +
        (state.dstExpertOffset + copyRowBegin - state.peerRowBegin) * state.rowBytes);
    for (uint64_t byteOffset = 0; byteOffset < copyBytes;) {
        const uint64_t remainingBytes = copyBytes - byteOffset;
        const uint32_t currentCopyBytes = static_cast<uint32_t>(
            remainingBytes < MAX_DMA_COPY_BYTES
                ? remainingBytes : MAX_DMA_COPY_BYTES);
        aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
            dst + byteOffset, src + byteOffset,
            static_cast<__ubuf__ uint8_t *>(nullptr),
            currentCopyBytes, state.dstRank);
        byteOffset += currentCopyBytes;
    }
}

__aicore__ inline bool TrySendOneCombineTransfer(
    CrossExpertRowGroupCombineSenderState &pipeline,
    const Params &params)
{
    if constexpr (g_coreType != AscendC::AIV) {
        return false;
    }
    if (!pipeline.active) {
        return false;
    }
    PeerCombineSenderState &peer =
        pipeline.peers[pipeline.nextPeerToTry];
    ++pipeline.nextPeerToTry;
    if (pipeline.nextPeerToTry == pipeline.ownerPeerCount) {
        pipeline.nextPeerToTry = 0;
    }

    bool progressed = false;
    while (peer.nextExpert < pipeline.registeredExpertCount) {
        if (!peer.hasCurrentExpert) {
            ResetRowGroupCombineSenderState(peer.currentExpert);
            InitRowGroupCombineSenderForExpert(
                peer.currentExpert, pipeline, params, peer.dstRank,
                peer.nextExpert, pipeline.rankId);
            peer.hasCurrentExpert = true;
            progressed = true;
        }
        RowGroupCombineSenderState &state = peer.currentExpert;
        if (state.peerRowBegin >= state.peerRowEnd) {
            ++peer.nextExpert;
            peer.hasCurrentExpert = false;
            progressed = true;
            continue;
        }
        if constexpr (ENABLE_LARGE_EP_WHOLE_PEER_SLICE) {
            if (pipeline.sendWholePeerSlice) {
                if (!IsPeerSliceReady(state, params)) {
                    return progressed;
                }
                PutCombineRows(
                    state, params, state.peerRowBegin, state.peerRowEnd);
                peer.issuedRemotePut = true;
                ++peer.nextExpert;
                peer.hasCurrentExpert = false;
                return true;
            }
        }
        uint64_t copyRowBegin = 0;
        uint64_t copyRowEnd = 0;
        if (!PrepareNextCombineRowGroup(state, copyRowBegin, copyRowEnd)) {
            ++peer.nextExpert;
            peer.hasCurrentExpert = false;
            progressed = true;
            continue;
        }
        if (!IsCombineRowGroupReady(state, params, state.nextRowGroup)) {
            return progressed;
        }
        PutCombineRows(state, params, copyRowBegin, copyRowEnd);
        peer.issuedRemotePut = true;
        ++state.nextRowGroup;
        return true;
    }
    return progressed;
}

__aicore__ inline void FinalizeCrossExpertCombineSender(
    CrossExpertRowGroupCombineSenderState &pipeline,
    const Params &params)
{
    if constexpr (g_coreType != AscendC::AIV) {
        return;
    }
    if (!pipeline.active) {
        return;
    }
    uint32_t finishedPeers = 0;
    uint32_t stalledPeers = 0;
    while (finishedPeers < pipeline.ownerPeerCount) {
        if (TrySendOneCombineTransfer(pipeline, params)) {
            stalledPeers = 0;
        } else {
            ++stalledPeers;
        }
        finishedPeers = 0;
        for (uint32_t peerIdx = 0;
             peerIdx < pipeline.ownerPeerCount;
             ++peerIdx) {
            if (pipeline.peers[peerIdx].nextExpert >=
                pipeline.registeredExpertCount) {
                ++finishedPeers;
            }
        }
        if (stalledPeers >= pipeline.ownerPeerCount) {
            const int64_t start = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - start < 100) {
            }
            stalledPeers = 0;
        }
    }
    for (uint32_t peerIdx = 0;
         peerIdx < pipeline.ownerPeerCount;
         ++peerIdx) {
        PeerCombineSenderState &peer = pipeline.peers[peerIdx];
        if (peer.issuedRemotePut) {
            aclshmemx_udma_quiet(peer.dstRank);
        }
    }
}

template<uint32_t SaveMode>
__aicore__ inline void GroupMatmulSwiglu(
    GroupMatmulParam<0> &gmm1Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBF16<L1TileShape, true> &epilogueOp,
    const Params &params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo,
    uint32_t &startBlockIdx,
    int32_t &vecSetSyncCom,
    uint32_t expertIdx
    MEGA_MOE_BF16_TIMER_PARAM(timer))
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm1Params.n;
    const uint32_t k = gmm1Params.k;

    AscendC::GlobalTensor<int32_t> dispatchFlags;
    dispatchFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.bGlobal);

    if (Ops::Base::CeilDiv(m, L1_TILE_M) == 1) {
        bGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (Ops::Base::CeilDiv(n, L1_TILE_N) == 1) {
        aGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    auto layoutA = tla::MakeLayout<
        bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, gmm1Params.layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(
        gmm1Params.l0cOutUbFirst, gmm1Params.layoutC, Catlass::Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(
        gmm1Params.l0cOutUbSecond, gmm1Params.layoutC, Catlass::Arch::PositionUB{});

    BlockScheduler scheduler(
        {m, gmm1Params.outputN, k},
        Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx
            ? gmm1Params.blockIdx + gmm1Params.blockNum
            : gmm1Params.blockIdx) - startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx;
         loopIdx < coreLoops;
         loopIdx += gmm1Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1_WAIT, expertIdx));
                __gm__ int32_t *signals = reinterpret_cast<__gm__ int32_t *>(
                    params.peermemInfo.ptrDispatchSignal);
                const uint32_t expertPerRank = params.tilingData->expertPerRank;
                for (uint32_t srcRank = 0;
                     srcRank < params.tilingData->epWorldSize;
                     ++srcRank) {
                    __gm__ int32_t *slot = signals + 2 *
                        (static_cast<uint64_t>(srcRank) * expertPerRank + expertIdx);
                    while (AscendC::ReadGmByPassDCache(slot) != 1) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {}
                    }
                }
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tok<Overwrite>(
                        AscendTimer::AIC_GMM1_WAIT, expertIdx));
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1, expertIdx));
            }
            if (vecSetSyncCom) {
                WaitForVector();
            }
            auto tensorBlockA = tla::GetTile(
                tensorA,
                tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockUbFirst = tla::GetTile(
                tensorUbFirst,
                tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUbSecond = tla::GetTile(
                tensorUbSecond,
                tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            for (uint32_t weightBlock = 0;
                 weightBlock < SWIGLU_N_HALF;
                 ++weightBlock) {
                auto tensorBlockB = tla::GetTile(
                    tensorB,
                    tla::MakeCoord(
                        kLoc,
                        nLoc + weightBlock * gmm1Params.outputN),
                    tla::MakeShape(actualShape.k(), actualShape.n()));
                auto &tensorBlockUb =
                    weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond;
                gmm1Params.blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            }
            NotifyVector();
        }

        vecSetSyncCom = 1;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm1Params.subBlockIdx == 0) {
                AscendC::Shape<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                    actualShape.m(), actualShape.n(), 0, 0};
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
                    epilogueOffset{
                        mLoc * gmm1Params.outputN + nLoc,
                        0,
                        0,
                        0,
                        0,
                        0};
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                }
                WaitForCube();
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tok<Overwrite>(
                            AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_SWIGLU, expertIdx));
                }
                epilogueOp.template Run<SaveMode>(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (startLoopIdx < coreLoops) {
            MEGA_MOE_BF16_TIMER_BLOCK(
                timer.Tok<Overwrite>(
                    AscendTimer::AIC_GMM1, expertIdx));
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (gmm1Params.subBlockIdx == 0 && startLoopIdx < coreLoops) {
            MEGA_MOE_BF16_TIMER_BLOCK(
                timer.Tok<Overwrite>(
                    AscendTimer::AIV_SWIGLU, expertIdx));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

__aicore__ inline void GroupMatmul2(
    GroupMatmulParam<1> &gmm2Params,
    const Params &params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo,
    uint32_t &startBlockIdx,
    int32_t &vecSetSyncCom,
    CrossExpertRowGroupCombineSenderState &combineSender,
    uint32_t groupIdx,
    uint64_t expertRowBase,
    uint16_t &pingpongIdx,
    int32_t rankId
    MEGA_MOE_BF16_TIMER_PARAM(timer))
{
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm2Params.n;
    const uint32_t k = gmm2Params.k;

    AscendC::GlobalTensor<int32_t> swigluFlags;
    swigluFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.bGlobal);

    auto layoutA = tla::MakeLayout<
        bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, gmm2Params.layoutB, Catlass::Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(
        gmm2Params.l0cOutUbFirst, gmm2Params.layoutC, Catlass::Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(
        gmm2Params.l0cOutUbSecond, gmm2Params.layoutC, Catlass::Arch::PositionUB{});
    RegisterCrossExpertCombineExpert(combineSender, groupIdx);

    Gmm2BlockScheduler scheduler(
        {m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm2Params.blockIdx < startBlockIdx
            ? gmm2Params.blockIdx + gmm2Params.blockNum
            : gmm2Params.blockIdx) - startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx;
         loopIdx < coreLoops;
         loopIdx += gmm2Params.blockNum) {
        const Catlass::GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto outputUb = pingpongIdx == 0
            ? gmm2Params.l0cOutUbFirst
            : gmm2Params.l0cOutUbSecond;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM2_WAIT, groupIdx));
                BlockScheduler gmm1Scheduler(
                    {m, k, n}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
                const uint32_t target = gmm1Scheduler.GetCoreLoops();
                __gm__ int32_t *flag =
                    (__gm__ int32_t *)swigluFlags.GetPhyAddr();
                while (target != AscendC::ReadGmByPassDCache(flag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {}
                }
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tok<Overwrite>(
                        AscendTimer::AIC_GMM2_WAIT, groupIdx));
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM2, groupIdx));
            }
            if (vecSetSyncCom >= 2) {
                WaitForVector(pingpongIdx + FLAG_ID_MAX);
            }
            auto tensorBlockA = tla::GetTile(
                tensorA,
                tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockB = tla::GetTile(
                tensorB,
                tla::MakeCoord(kLoc, nLoc),
                tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockUb = tla::GetTile(
                tensorUb,
                tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            gmm2Params.blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            NotifyVector(pingpongIdx + FLAG_ID_MAX);
        }

        ++vecSetSyncCom;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm2Params.subBlockIdx == 1) {
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_COMBINE_WAIT, groupIdx));
                }
                WaitForCube(pingpongIdx);
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tok<Overwrite>(
                            AscendTimer::AIV_COMBINE_WAIT, groupIdx));
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_COMBINE, groupIdx));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                    mLoc,
                    nLoc,
                    n,
                    groupIdx,
                    expertRowBase,
                    rankId,
                    outputUb,
                    actualShape,
                    params);
                // Publish one completion for this N tile only after its
                // strided UB-to-GM write has reached row-major staging.  The
                // peer-owner AIV1 can send either this M row-group or a
                // coalesced peer slice after all required N tiles arrive.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
                const uint32_t maxRowGroups = Ops::Base::CeilDiv(
                    static_cast<uint32_t>(params.tilingData->maxOutputSize),
                    L1_TILE_M);
                __gm__ int32_t *combineReady = reinterpret_cast<__gm__ int32_t *>(
                    params.workspaceInfo.ptrCombineReady);
                AscendC::AtomicAdd(
                    combineReady + groupIdx * maxRowGroups + mLoc / L1_TILE_M,
                    static_cast<int32_t>(1));
                NotifyCube(pingpongIdx);
                // Release the current Ping-Pong UB before probing one peer in
                // round-robin order.  By default every EP size emits the
                // ready row-group's intersection with the peer-expert slice.
                // The retained whole-slice branch can be re-enabled by the
                // compile-time switch above.  AIC can compute the next tile
                // in the alternate UB while AIV1 enqueues UDMA.
                TrySendOneCombineTransfer(combineSender, params);
            }
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (startLoopIdx < coreLoops) {
            MEGA_MOE_BF16_TIMER_BLOCK(
                timer.Tok<Overwrite>(
                    AscendTimer::AIC_GMM2, groupIdx));
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (gmm2Params.subBlockIdx == 1 && startLoopIdx < coreLoops) {
            MEGA_MOE_BF16_TIMER_BLOCK(
                timer.Tok<Overwrite>(
                    AscendTimer::AIV_COMBINE, groupIdx));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm2Params.blockNum;
}

}  // namespace MegaMoeBF16CatlassImpl

#endif  // DEEPEP_MEGA_MOE_BF16_UDMA_CATLASS_IMPL_H
