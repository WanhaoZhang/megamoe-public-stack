/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * DeepEP-native BF16 grouped GEMM stages for MegaMoe.  This keeps the
 * existing MegaMoe expert scheduler and communication protocol; only the
 * matrix data path is replaced with BF16 x BF16 MMAD.  The A5 MMAD pipeline
 * accumulates in L0C/FP32 and Fixpipe writes BF16 tiles to UB.
 */
#ifndef DEEPEP_MEGA_MOE_BF16_CATLASS_IMPL_H
#define DEEPEP_MEGA_MOE_BF16_CATLASS_IMPL_H

#include <type_traits>

#include "kernel_operator.h"
#include "mega_moe_stage_timer.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
// CATLASS 48f7461 pulls in unused W4A8 VF members rejected by CANN 9.2.
// BF16 needs neither the W4A8 prologue nor its MMAD specialization.
// Keep this workaround local to BF16; see the stable-forward integration doc.
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
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "mega_moe_base.h"
#include "mega_moe_bf16_timer.h"
#include "mega_moe_combine_send.h"
#include "forward_prepared_phase.h"

namespace MegaMoeBF16CatlassImpl {

using ArchTag = Catlass::Arch::Ascend950;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<
    ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>;
using Gmm1L1TileShape = tla::Shape<
    tla::Int<MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows>,
    tla::Int<256>, tla::Int<256>>;
using Gmm1L0TileShape = tla::Shape<
    tla::Int<MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows>,
    tla::Int<256>, tla::Int<64>>;
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using Gmm2BlockScheduler = BlockScheduler;

constexpr uint32_t L1_TILE_M = 128;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = L1_TILE_M * L1_TILE_N;
constexpr uint32_t GMM1_TILE_M =
    MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kComputeRows;
constexpr uint32_t GMM1_TILE_ELEMENTS = GMM1_TILE_M * L1_TILE_N;
#if defined(DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE) && DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE
__aicore__ inline void ConsiderGmm1NStripeWidth(
    uint32_t width, uint32_t lanes, uint32_t mTiles, uint32_t nTiles,
    uint32_t minWaveColumns, uint32_t &bestWidth,
    uint32_t &bestAddressUnits, uint32_t &bestBRun)
{
    if (width > nTiles) {
        return;
    }
    const uint32_t activeM = mTiles < lanes ? mTiles : lanes;
    const uint32_t candidateActiveB = width < minWaveColumns ? minWaveColumns : width;
    const uint32_t activeB = candidateActiveB < nTiles ? candidateActiveB : nTiles;
    const uint32_t addressUnits = activeM * GMM1_TILE_M +
        activeB * SWIGLU_N_HALF * L1_TILE_N;
    const uint32_t bRun = Ops::Base::CeilDiv(mTiles, lanes);
    if (addressUnits < bestAddressUnits ||
        (addressUnits == bestAddressUnits && bRun > bestBRun)) {
        bestWidth = width;
        bestAddressUnits = addressUnits;
        bestBRun = bRun;
    }
}

__aicore__ inline uint32_t SelectGmm1NStripeWidth(
    uint32_t m, uint32_t outputN, uint32_t blockNum)
{
    const uint32_t mTiles = Ops::Base::CeilDiv(m, GMM1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN, L1_TILE_N);
    if (mTiles == 0 || nTiles == 0 || blockNum == 0) {
        return 0;
    }

    // The deployed platforms use 28, 32 or 36 AICs. Keep selection bounded:
    // each core evaluates only the widths that divide its AIC count.
    if (blockNum != 28 && blockNum != 32 && blockNum != 36) {
        return 0;
    }

    const uint32_t minWaveColumns = Ops::Base::CeilDiv(blockNum, mTiles);
    uint32_t bestWidth = 1;
    const uint32_t bestActiveM = mTiles < blockNum ? mTiles : blockNum;
    const uint32_t bestActiveB = minWaveColumns < nTiles ? minWaveColumns : nTiles;
    uint32_t bestAddressUnits = bestActiveM * GMM1_TILE_M +
        bestActiveB * SWIGLU_N_HALF * L1_TILE_N;
    uint32_t bestBRun = Ops::Base::CeilDiv(mTiles, blockNum);
    if (blockNum == 28) {
        ConsiderGmm1NStripeWidth(
            2, 14, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(
            4, 7, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
    } else if (blockNum == 32) {
        ConsiderGmm1NStripeWidth(
            2, 16, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(
            4, 8, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
    } else {
        ConsiderGmm1NStripeWidth(
            2, 18, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(
            3, 12, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(
            4, 9, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
        ConsiderGmm1NStripeWidth(
            6, 6, mTiles, nTiles, minWaveColumns, bestWidth, bestAddressUnits, bestBRun);
    }
    return bestWidth;
}

__aicore__ inline Catlass::GemmCoord GetGmm1AdaptiveNStripeCoord(
    uint32_t taskIdx, uint32_t m, uint32_t outputN, uint32_t stripeWidth)
{
    const uint32_t mTiles = Ops::Base::CeilDiv(m, GMM1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN, L1_TILE_N);
    const uint32_t fullStripeTasks = mTiles * stripeWidth;
    const uint32_t stripe = taskIdx / fullStripeTasks;
    const uint32_t taskInStripe = taskIdx % fullStripeTasks;
    const uint32_t stripeStart = stripe * stripeWidth;
    const uint32_t remainingN = nTiles - stripeStart;
    const uint32_t actualWidth = remainingN < stripeWidth ? remainingN : stripeWidth;
    return Catlass::GemmCoord{
        taskInStripe / actualWidth,
        stripeStart + taskInStripe % actualWidth,
        0};
}

#if defined(DEEPEP_BF16_GMM1_READY_FIRST) && DEEPEP_BF16_GMM1_READY_FIRST
__aicore__ inline uint32_t GetGmm1ReadyFirstMTile(
    uint32_t mOrdinal, uint32_t mTiles, uint32_t readyBlocksPerProducerPartition)
{
    constexpr uint32_t tilesPerReadyBlock =
        MegaMoeBF16Training::ForwardDispatchTileReadyLayout::kReadyRows / GMM1_TILE_M;
    static_assert(tilesPerReadyBlock == 2,
        "ready-first mapping requires two GMM1 M tiles per Dispatch-ready block");
    if (readyBlocksPerProducerPartition <= 1 || mTiles % tilesPerReadyBlock != 0) {
        return mOrdinal;
    }
    const uint32_t readyBlocks = mTiles / tilesPerReadyBlock;
    if (readyBlocks % readyBlocksPerProducerPartition != 0) {
        return mOrdinal;
    }
    const uint32_t producerPartitions = readyBlocks / readyBlocksPerProducerPartition;
    const uint32_t blockOrdinal = mOrdinal / tilesPerReadyBlock;
    const uint32_t tileInBlock = mOrdinal % tilesPerReadyBlock;
    const uint32_t producerRound = blockOrdinal / producerPartitions;
    const uint32_t producerPartition = blockOrdinal % producerPartitions;
    return (producerPartition * readyBlocksPerProducerPartition + producerRound) *
        tilesPerReadyBlock + tileInBlock;
}

__aicore__ inline Catlass::GemmCoord GetGmm1AdaptiveReadyFirstCoord(
    uint32_t taskIdx, uint32_t m, uint32_t outputN, uint32_t stripeWidth,
    uint32_t readyBlocksPerProducerPartition)
{
    const uint32_t mTiles = Ops::Base::CeilDiv(m, GMM1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN, L1_TILE_N);
    const uint32_t fullStripeTasks = mTiles * stripeWidth;
    const uint32_t stripe = taskIdx / fullStripeTasks;
    const uint32_t taskInStripe = taskIdx % fullStripeTasks;
    const uint32_t stripeStart = stripe * stripeWidth;
    const uint32_t remainingN = nTiles - stripeStart;
    const uint32_t actualWidth = remainingN < stripeWidth ? remainingN : stripeWidth;
    const uint32_t mOrdinal = taskInStripe / actualWidth;
    return Catlass::GemmCoord{
        GetGmm1ReadyFirstMTile(mOrdinal, mTiles, readyBlocksPerProducerPartition),
        stripeStart + taskInStripe % actualWidth,
        0};
}
#endif
#endif
static_assert(
    2U * GMM1_TILE_ELEMENTS * sizeof(bfloat16_t) <= ArchTag::UB_SIZE,
    "The two BF16 GMM1 output tiles must fit in A5 UB");

template <uint32_t TargetUBIdx, class WeightLayout>
struct PackedBF16TileCopyTlaToTargetUB
    : public Catlass::Gemm::Tile::PackedTileCopyTla<
          ArchTag,
          bfloat16_t,
          Catlass::layout::RowMajor,
          bfloat16_t,
          WeightLayout,
          bfloat16_t,
          Catlass::layout::RowMajor,
          void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag,
        bfloat16_t,
        Catlass::layout::RowMajor,
        bfloat16_t,
        WeightLayout,
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

// Ordinary forward can write raw Gate/Up to GM without an AIV handshake.
template<class WeightLayout>
struct ForwardGmTileCopy : public Catlass::Gemm::Tile::PackedTileCopyTla<
    ArchTag, bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, WeightLayout,
    bfloat16_t, Catlass::layout::RowMajor, void> {
    using Base = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag,
        bfloat16_t, Catlass::layout::RowMajor, bfloat16_t, WeightLayout,
        bfloat16_t, Catlass::layout::RowMajor, void>;
    using TensorL0C = typename Base::TensorL0C;
    template<class TensorC>
    using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToGmTla<
        ArchTag, TensorL0C, TensorC,
        Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, false>;
};

// A local dense projection with no AIV UB output, route flags, replica stripes
// or Combine acknowledgement. Shared gate/up can therefore run while both AIV
// lanes execute InitRouting; shared down can run while AIV1 still dispatches.
__aicore__ inline void SharedExpertGemmGm(
    GM_ADDR input, GM_ADDR weight, GM_ADDR output,
    uint32_t m, uint32_t n, uint32_t k)
{
    if constexpr (g_coreType == AscendC::AIC) {
        using TileCopy = ForwardGmTileCopy<Catlass::layout::ColumnMajor>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, L1TileShape, L0TileShape,
            bfloat16_t, bfloat16_t, bfloat16_t, void, TileCopy>;
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<bfloat16_t> inputGm, weightGm, outputGm;
        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input));
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        auto inputLayout = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
        // Framework Linear weights are [N,K], viewed as column-major [K,N].
        auto weightLayout = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(k, n);
        auto outputLayout = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, n);
        auto inputTensor = tla::MakeTensor(inputGm, inputLayout, Catlass::Arch::PositionGM{});
        auto weightTensor = tla::MakeTensor(weightGm, weightLayout, Catlass::Arch::PositionGM{});
        auto outputTensor = tla::MakeTensor(outputGm, outputLayout, Catlass::Arch::PositionGM{});
        BlockScheduler scheduler({m, n, k}, Catlass::MakeCoord(L1_TILE_M, L1_TILE_N));
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        for (uint32_t task = core; task < scheduler.GetCoreLoops(); task += AscendC::GetBlockNum()) {
            const auto coord = scheduler.GetBlockCoord(task);
            const auto shape = scheduler.GetActualBlockShape(coord);
            const uint32_t row = coord.m() * L1_TILE_M;
            const uint32_t column = coord.n() * L1_TILE_N;
            auto tileA = tla::GetTile(inputTensor, tla::MakeCoord(row, 0U),
                tla::MakeShape(shape.m(), shape.k()));
            auto tileB = tla::GetTile(weightTensor, tla::MakeCoord(0U, column),
                tla::MakeShape(shape.k(), shape.n()));
            auto tileC = tla::GetTile(outputTensor, tla::MakeCoord(row, column),
                tla::MakeShape(shape.m(), shape.n()));
            blockMmad(tileA, tileB, tileC, shape);
        }
        // UNIT_FLAG MMAD does not wait FIX_M in its destructor. Explicitly
        // finish FIX->GM before another stage publishes a scalar ready flag,
        // including on a core which received no output tiles.
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <uint32_t GmmIndex, bool TransW = true, bool GmOutput = false>
class GroupMatmulParam {
public:
    static_assert(GmmIndex < 2, "GMM1/GMM2 must target AIV0/AIV1");
    using WeightLayout = std::conditional_t<
        TransW, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
    using TileCopy = std::conditional_t<GmOutput,
        ForwardGmTileCopy<WeightLayout>,
        PackedBF16TileCopyTlaToTargetUB<GmmIndex, WeightLayout>>;
    using MmadL1TileShape = std::conditional_t<GmmIndex == 0, Gmm1L1TileShape, L1TileShape>;
    using MmadL0TileShape = std::conditional_t<GmmIndex == 0, Gmm1L0TileShape, L0TileShape>;
    static constexpr uint32_t kTileRows = GmmIndex == 0 ? GMM1_TILE_M : L1_TILE_M;
    using CatlassBlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy,
        MmadL1TileShape,
        MmadL0TileShape,
        bfloat16_t,
        bfloat16_t,
        bfloat16_t,
        void,
        TileCopy>;
#if defined(DEEPEP_MEGA_MOE_BF16_MGROUP_DISTINCT_MMAD_EVENTS)
    class BlockMmad : public CatlassBlockMmad {
    public:
        using Base = CatlassBlockMmad;
        static constexpr int32_t EVENT_ID_OFFSET = GmmIndex == 0 ? 0 : 4;
        static_assert(
            Base::L1A_STAGES + Base::L1B_STAGES + EVENT_ID_OFFSET <= 8,
            "MGroup L1 event IDs exceed the hardware range");
        static_assert(
            Base::L0A_STAGES + Base::L0B_STAGES + EVENT_ID_OFFSET <= 8,
            "MGroup L0 event IDs exceed the hardware range");

        CATLASS_DEVICE explicit BlockMmad(
            Catlass::Arch::Resource<ArchTag> &resource)
            : Base(resource)
        {
            if ASCEND_IS_AIC {
                if constexpr (EVENT_ID_OFFSET != 0) {
                    RemapEvents();
                }
            }
        }

        // Both persistent MMAD objects share the same physical L1/L0 space.
        // Drain the current owner's pipeline before the other object starts,
        // then re-arm its private event set for the next wave.
        CATLASS_DEVICE void HandoffSharedLocalMemory()
        {
            if ASCEND_IS_AIC {
                WaitEvents();
                SetEvents();
            }
        }

    private:
        CATLASS_DEVICE void RemapEvents()
        {
            WaitEvents();
            for (uint32_t i = 0; i < Base::L1A_STAGES; ++i) {
                this->l1AEventList[i] += EVENT_ID_OFFSET;
            }
            for (uint32_t i = 0; i < Base::L1B_STAGES; ++i) {
                this->l1BEventList[i] += EVENT_ID_OFFSET;
            }
            for (uint32_t i = 0; i < Base::L0A_STAGES; ++i) {
                this->l0AEventList[i] += EVENT_ID_OFFSET;
            }
            for (uint32_t i = 0; i < Base::L0B_STAGES; ++i) {
                this->l0BEventList[i] += EVENT_ID_OFFSET;
            }
            SetEvents();
        }

        CATLASS_DEVICE void WaitEvents()
        {
            for (uint32_t i = 0; i < Base::L1A_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                    this->l1AEventList[i]);
            }
            for (uint32_t i = 0; i < Base::L1B_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                    this->l1BEventList[i]);
            }
            for (uint32_t i = 0; i < Base::L0A_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                    this->l0AEventList[i]);
            }
            for (uint32_t i = 0; i < Base::L0B_STAGES; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                    this->l0BEventList[i]);
            }
        }

        CATLASS_DEVICE void SetEvents()
        {
            for (uint32_t i = 0; i < Base::L1A_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    this->l1AEventList[i]);
            }
            for (uint32_t i = 0; i < Base::L1B_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    this->l1BEventList[i]);
            }
            for (uint32_t i = 0; i < Base::L0A_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                    this->l0AEventList[i]);
            }
            for (uint32_t i = 0; i < Base::L0B_STAGES; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                    this->l0BEventList[i]);
            }
        }
    };
#else
    using BlockMmad = CatlassBlockMmad;
#endif
    using LayoutB = decltype(tla::MakeLayout<
        bfloat16_t, WeightLayout>(uint32_t{}, uint32_t{}));
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
              bfloat16_t, WeightLayout>(kValue, nValue)),
          layoutC(tla::MakeLayout<
              bfloat16_t, Catlass::layout::RowMajor>(kTileRows, L1_TILE_N))
    {
        l0cOutUbFirst = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
        l0cOutUbSecond = resource.ubBuf.template GetBufferByByte<bfloat16_t>(
            kTileRows * L1_TILE_N * sizeof(bfloat16_t));
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
    bool preparedPhaseEnabled = false;
#if defined(DEEPEP_BF16_REPLICA_PIPELINE)
    uint64_t pipelineTrace=0;
    uint32_t pipelineTraceTasks=0;
#if defined(DEEPEP_BF16_REPLICA_STRIPES)
    uint64_t stripeReady=0;
    int stripeEpoch=0, stripeHome=0;
#if MEGAFLUX_OPT_CACHE
    MegaFluxPipeline::ReadyCache stripeCache;
#endif
    bool firstReplicaTraced=false;
#if defined(DEEPEP_BF16_REPLICA_STREAM)
    int streamMode=0,w2FlagOffset=0;
    bool firstGmm2Traced=false;
#endif
#endif
#endif
};


template<bool DispatchTileReady = false, bool WaitForFullExpert = false,
         uint32_t SaveMode = MegaMoeBF16Training::kSaveMxFp8,
         bool TransW = true>
__aicore__ inline void GroupMatmulSwiglu(
    GroupMatmulParam<0, TransW> &gmm1Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBF16<Gmm1L1TileShape, true> &epilogueOp,
    const Params &params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo,
    uint32_t &startBlockIdx,
    int32_t &vecSetSyncCom,
    uint32_t expertIdx,
    bool useTileReady,
    uint32_t readyBlocksPerProducerPartition
    MEGA_MOE_BF16_TIMER_PARAM(timer))
{
    using ReadyLayout = MegaMoeBF16Training::ForwardDispatchTileReadyLayout;
    static_assert(ReadyLayout::kReadyRows % GMM1_TILE_M == 0,
        "Dispatch ready rows must contain whole BF16 GMM1 M tiles");
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm1Params.n;
    const uint32_t k = gmm1Params.k;

    AscendC::GlobalTensor<int32_t> dispatchFlags;
    dispatchFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.bGlobal);

    if (Ops::Base::CeilDiv(m, GMM1_TILE_M) == 1) {
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
        Catlass::MakeCoord(GMM1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx
            ? gmm1Params.blockIdx + gmm1Params.blockNum
            : gmm1Params.blockIdx) - startBlockIdx;
#if defined(DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE) && DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE
    const uint32_t nStripeWidth =
        SelectGmm1NStripeWidth(m, gmm1Params.outputN, gmm1Params.blockNum);
#if defined(DEEPEP_BF16_GMM1_AUTO_SELECT) && DEEPEP_BF16_GMM1_AUTO_SELECT
    // Ordinary M192 uses the same adaptive N stripes for every expert.
    // The isolated legacy path keeps its existing schedule selection.
#if defined(DEEPEP_BF16_M192_FORWARD)
    constexpr bool useAdaptiveSchedule = true;
#else
    const bool useAdaptiveSchedule = useTileReady;
#endif
#else
    constexpr bool useAdaptiveSchedule = true;
#endif
#endif

    MegaMoeBF16ForwardSchedule::Schedule phaseSchedule;
#if defined(DEEPEP_BF16_GMM1_AUTO_SELECT) && DEEPEP_BF16_GMM1_AUTO_SELECT
    if (gmm1Params.preparedPhaseEnabled) {
    MegaMoeBF16ForwardSchedule::View phaseView;
    phaseView.Init(gmmAddrInfo.groupFlagList2-expertIdx,
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrcumsumMM),
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1),
        params.tilingData->maxOutputSize,params.tilingData->expertPerRank,
        params.tilingData->epWorldSize,gmm1Params.blockNum,gmm1Params.dispatchBlockNumPerEP,
        params.tilingData->k);
    const bool active=g_coreType==AscendC::AIC || gmm1Params.subBlockIdx==0;
    phaseSchedule.Init(phaseView,expertIdx,0,m,gmm1Params.outputN,nStripeWidth,useTileReady&&active);
    }
#endif
    bool dispatchExpertReady = false;
    for (uint32_t loopIdx = startLoopIdx;
         loopIdx < coreLoops;
         loopIdx += gmm1Params.blockNum) {
        Catlass::GemmCoord blockCoord{0,0,0};
        if(phaseSchedule.Enabled()) {
            const auto xy=phaseSchedule.Get(loopIdx);
            blockCoord=Catlass::GemmCoord{xy.m,xy.n,0};
        } else {
#if defined(DEEPEP_BF16_GMM1_READY_FIRST) && DEEPEP_BF16_GMM1_READY_FIRST
        blockCoord =
            nStripeWidth > 0 && useAdaptiveSchedule
            ? (useTileReady
                ? GetGmm1AdaptiveReadyFirstCoord(
                    loopIdx, m, gmm1Params.outputN, nStripeWidth,
                    readyBlocksPerProducerPartition)
                : GetGmm1AdaptiveNStripeCoord(
                    loopIdx, m, gmm1Params.outputN, nStripeWidth))
            : scheduler.GetBlockCoord(loopIdx);
#elif defined(DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE) && DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE
        blockCoord = nStripeWidth > 0
            ? GetGmm1AdaptiveNStripeCoord(
                loopIdx, m, gmm1Params.outputN, nStripeWidth)
            : scheduler.GetBlockCoord(loopIdx);
#else
        blockCoord = scheduler.GetBlockCoord(loopIdx);
#endif
        }
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * GMM1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            MEGAFLUX_STAGE_BEGIN(stageInputWait,gmm1Params.pipelineTrace,false,MegaFluxStageTimer::FWD_INPUT_WAIT,expertIdx,0,mLoc,nLoc);
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1_WAIT, expertIdx));
            }
            // Streaming mode checks each task's input because static swizzling
            // can visit non-adjacent M tiles. The default expert mode waits
            // for the whole expert and removes all later input polling.
            if constexpr (DispatchTileReady && WaitForFullExpert) {
                // Expert mode publishes only partition completion. The row
                // counter layout is still reserved for the opt-in streaming
                // mode; wait on the separate expert-completion counter once.
                if (loopIdx == startLoopIdx) {
                    __gm__ int32_t *expertFlag =
                        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t target =
                        params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {}
                    }
                }
            } else if constexpr (DispatchTileReady) {
                if (!useTileReady) {
                    if (loopIdx == startLoopIdx) {
                        __gm__ int32_t *expertFlag =
                            reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                            expertIdx * INT_CACHELINE;
                        const uint32_t target =
                            params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                        while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {}
                        }
                    }
                } else if (!dispatchExpertReady) {
                    __gm__ int32_t *expertFlag =
                        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t expertTarget =
                        params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    dispatchExpertReady =
                        expertTarget == AscendC::ReadGmByPassDCache(expertFlag);
                    if (!dispatchExpertReady) {
                        const uint32_t readyBlock = mLoc / ReadyLayout::kReadyRows;
                        __gm__ int32_t *flag = gmmAddrInfo.groupFlagList2 +
                            readyBlock * INT_CACHELINE;
                        const int32_t target = static_cast<int32_t>(
                            ReadyLayout::RowsInReadyBlock(readyBlock, m));
                        while (target != AscendC::ReadGmByPassDCache(flag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {}
                        }
                    }
                }
            } else if (loopIdx == startLoopIdx) {
                const uint32_t target =
                    params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                __gm__ int32_t *flag =
                    (__gm__ int32_t *)dispatchFlags.GetPhyAddr();
                while (target != AscendC::ReadGmByPassDCache(flag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {}
                }
            }
            MEGAFLUX_STAGE_END(stageInputWait);
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tok<Overwrite>(
                        AscendTimer::AIC_GMM1_WAIT, expertIdx));
                // Keep the existing per-expert timer schema: WAIT measures
                // the first task's input wait; GMM1 includes Cube/Vector
                // handshakes and, in streaming mode only, later tile waits.
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1, expertIdx));
            }
            if (vecSetSyncCom) {
                MEGAFLUX_STAGE_BEGIN(stageVectorWait,gmm1Params.pipelineTrace,false,MegaFluxStageTimer::FWD_CUBE_WAIT_VECTOR,expertIdx,0,mLoc,nLoc);
                WaitForVector();
                MEGAFLUX_STAGE_END(stageVectorWait);
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
#if defined(DEEPEP_BF16_REPLICA_PIPELINE)
            if(gmm1Params.pipelineTrace && gmm1Params.pipelineTraceTasks<128)
                MegaFluxPipeline::Stamp(gmm1Params.pipelineTrace,gmm1Params.blockIdx,
                    68+gmm1Params.pipelineTraceTasks*2);
#endif
#if defined(DEEPEP_BF16_REPLICA_STRIPES)
            static_assert(L1_TILE_N==MegaFluxPipeline::STRIPE_N, "weight stripe must match GMM1 N tile");
            if(gmm1Params.stripeReady && expertIdx>=gmm1Params.stripeHome) {
#if MEGAFLUX_OPT_CACHE
                const int cachedStripe=(gmm1Params.streamMode&1)?(gmm1Params.outputN+L1_TILE_N-1)/L1_TILE_N-1:nLoc/L1_TILE_N;
                const int cacheKey=(expertIdx-gmm1Params.stripeHome)*((gmm1Params.outputN+L1_TILE_N-1)/L1_TILE_N)+cachedStripe+1;
                if(!gmm1Params.stripeCache.Has(gmm1Params.stripeReady,gmm1Params.stripeEpoch,0,cacheKey)) {
#endif
                MEGAFLUX_STAGE_BEGIN(stageWait,gmm1Params.pipelineTrace,false,MegaFluxStageTimer::WEIGHT_WAIT,expertIdx,0,mLoc,nLoc,-1,expertIdx-gmm1Params.stripeHome);
                const auto waitStart=MegaFluxDetail::Now(gmm1Params.pipelineTrace);
                MegaFluxPipeline::WaitStripe(gmm1Params.stripeReady,
                    expertIdx-gmm1Params.stripeHome,
#if defined(DEEPEP_BF16_REPLICA_STREAM)
                    (gmm1Params.streamMode&1)?(gmm1Params.outputN+L1_TILE_N-1)/L1_TILE_N-1:
#endif
                    nLoc/L1_TILE_N,
                    (gmm1Params.outputN+L1_TILE_N-1)/L1_TILE_N,
                    gmm1Params.blockNum,gmm1Params.stripeEpoch);
                MEGAFLUX_STAGE_END(stageWait);
                MegaFluxStageTimer::Mark(gmm1Params.pipelineTrace,false,MegaFluxStageTimer::WEIGHT_USE,expertIdx,0,mLoc,nLoc,-1,expertIdx-gmm1Params.stripeHome);
                const auto ready=MegaFluxDetail::Now(gmm1Params.pipelineTrace);
                MegaFluxDetail::Record(gmm1Params.pipelineTrace,false,MegaFluxDetail::WEIGHT_WAIT,waitStart,ready,expertIdx,0,mLoc,nLoc,-1,expertIdx-gmm1Params.stripeHome);
                MegaFluxDetail::Record(gmm1Params.pipelineTrace,false,MegaFluxDetail::WEIGHT_USE,ready,ready,expertIdx,0,mLoc,nLoc,-1,expertIdx-gmm1Params.stripeHome);
#if MEGAFLUX_OPT_CACHE
                    gmm1Params.stripeCache.Add(cacheKey);
                } else {
                    MegaFluxStageTimer::Mark(gmm1Params.pipelineTrace,false,MegaFluxStageTimer::WEIGHT_USE,expertIdx,0,mLoc,nLoc,-1,expertIdx-gmm1Params.stripeHome);
                }
#endif
            }
            const bool traceFirstReplica=gmm1Params.pipelineTrace &&
                gmm1Params.stripeReady && expertIdx>=gmm1Params.stripeHome &&
                !gmm1Params.firstReplicaTraced;
            if(traceFirstReplica)
                MegaFluxPipeline::Stamp(gmm1Params.pipelineTrace,gmm1Params.blockIdx,325);
#endif
            MEGAFLUX_STAGE_BEGIN(stageGemm,gmm1Params.pipelineTrace,false,MegaFluxStageTimer::FWD_GEMM_TILE,expertIdx,0,mLoc,nLoc);
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
            MEGAFLUX_STAGE_END(stageGemm);
#if defined(DEEPEP_BF16_REPLICA_PIPELINE)
            if(gmm1Params.pipelineTrace && gmm1Params.pipelineTraceTasks<128) {
                // Diagnostic only: drain this task's MMAD/load/Fixpipe work,
                // unlike the enclosing expert span which includes input waits.
                AscendC::PipeBarrier<PIPE_ALL>();
                MegaFluxPipeline::Stamp(gmm1Params.pipelineTrace,gmm1Params.blockIdx,
                    69+gmm1Params.pipelineTraceTasks*2);
                ++gmm1Params.pipelineTraceTasks;
                MegaFluxPipeline::Stamp(gmm1Params.pipelineTrace,gmm1Params.blockIdx,324);
            }
#endif
#if defined(DEEPEP_BF16_REPLICA_STRIPES)
            if(traceFirstReplica) {
                AscendC::PipeBarrier<PIPE_ALL>();
                MegaFluxPipeline::Stamp(gmm1Params.pipelineTrace,gmm1Params.blockIdx,326);
                gmm1Params.firstReplicaTraced=true;
            }
#endif
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
                MEGAFLUX_STAGE_BEGIN(stageCubeWait,gmm1Params.pipelineTrace,false,MegaFluxStageTimer::FWD_VECTOR_WAIT,expertIdx,0,mLoc,nLoc);
                WaitForCube();
                MEGAFLUX_STAGE_END(stageCubeWait);
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tok<Overwrite>(
                            AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_SWIGLU, expertIdx));
                }
                MEGAFLUX_STAGE_BEGIN(stageVector,gmm1Params.pipelineTrace,false,MegaFluxStageTimer::FWD_VECTOR_TILE,expertIdx,0,mLoc,nLoc);
                epilogueOp.template Run<SaveMode>(epilogueShape, epilogueOffset);
                MEGAFLUX_STAGE_END(stageVector);
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

template<bool DispatchTileReady = false, bool WaitForFullExpert = false,
         uint32_t SaveMode = MegaMoeBF16Training::kSaveMxFp8,
         bool TransW = true>
__aicore__ inline void GroupMatmulSwigluGm(
    GroupMatmulParam<0, TransW, true> &gmm1Params,
    MegaMoeBF16Impl::BlockEpilogueSwigluBF16<Gmm1L1TileShape, true> &epilogueOp,
    const Params &params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo,
    uint32_t &startBlockIdx,
    uint32_t expertIdx,
    bool useTileReady,
    uint32_t readyBlocksPerProducerPartition,
    uint64_t savedElementOffset
    MEGA_MOE_BF16_TIMER_PARAM(timer))
{
    using ReadyLayout = MegaMoeBF16Training::ForwardDispatchTileReadyLayout;
    static_assert(ReadyLayout::kReadyRows % GMM1_TILE_M == 0,
        "Dispatch ready rows must contain whole BF16 GMM1 M tiles");
    const uint32_t m = Get<M_VALUE>(problemShape);
    const uint32_t n = gmm1Params.n;
    const uint32_t k = gmm1Params.k;

    AscendC::GlobalTensor<int32_t> dispatchFlags;
    dispatchFlags.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
    AscendC::GlobalTensor<bfloat16_t> aGlobal;
    AscendC::GlobalTensor<bfloat16_t> bGlobal;
    aGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.aGlobal);
    bGlobal.SetGlobalBuffer((__gm__ bfloat16_t *)gmmAddrInfo.bGlobal);

    if (Ops::Base::CeilDiv(m, GMM1_TILE_M) == 1) {
        bGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    if (Ops::Base::CeilDiv(n, L1_TILE_N) == 1) {
        aGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    auto layoutA = tla::MakeLayout<
        bfloat16_t, Catlass::layout::RowMajor>(m, k);
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Catlass::Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, gmm1Params.layoutB, Catlass::Arch::PositionGM{});

    BlockScheduler scheduler(
        {m, gmm1Params.outputN, k},
        Catlass::MakeCoord(GMM1_TILE_M, L1_TILE_N));
    const uint32_t coreLoops = scheduler.GetCoreLoops();
    const uint32_t startLoopIdx =
        (gmm1Params.blockIdx < startBlockIdx
            ? gmm1Params.blockIdx + gmm1Params.blockNum
            : gmm1Params.blockIdx) - startBlockIdx;
#if defined(DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE) && DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE
    const uint32_t nStripeWidth =
        SelectGmm1NStripeWidth(m, gmm1Params.outputN, gmm1Params.blockNum);
#if defined(DEEPEP_BF16_GMM1_AUTO_SELECT) && DEEPEP_BF16_GMM1_AUTO_SELECT
    // Ordinary M192 uses the same adaptive N stripes for every expert.
    // The isolated legacy path keeps its existing schedule selection.
#if defined(DEEPEP_BF16_M192_FORWARD)
    constexpr bool useAdaptiveSchedule = true;
#else
    const bool useAdaptiveSchedule = useTileReady;
#endif
#else
    constexpr bool useAdaptiveSchedule = true;
#endif
#endif

    // Word 1 publishes raw Gate/Up readiness; ResetWorkSpaceFlagList clears this
    // entire cache line before Dispatch. Word 0 remains the Hp task counter.
    __gm__ int32_t *gmmReady = gmmAddrInfo.groupFlagList + 1;
    using ProgressLayout = MegaMoeBF16Training::ForwardGmTileProgressLayout;
    const ProgressLayout progressLayout(params.tilingData->n,
        params.tilingData->expertPerRank, params.tilingData->maxOutputSize,
        SaveMode, true);
    const bool useTileProgress = progressLayout.enabled &&
        MegaMoeBF16GmTileProgress::CanStream(coreLoops, gmm1Params.blockNum);
    __gm__ int32_t *producerProgress = nullptr;
    if (useTileProgress) {
        producerProgress = reinterpret_cast<__gm__ int32_t *>(
            params.workspaceInfo.ptrA2 + progressLayout.offset) +
            ProgressLayout::ProgressWordOffset(expertIdx, gmm1Params.blockIdx);
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (gmm1Params.subBlockIdx == 1) {
            startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
            return;
        }
        if (!useTileProgress) {
            MEGA_MOE_BF16_TIMER_BLOCK(timer.Tik(AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
            while (AscendC::ReadGmByPassDCache(gmmReady) != gmm1Params.blockNum) {
                const int64_t start = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - start < 100) {}
            }
            MEGA_MOE_BF16_TIMER_BLOCK(timer.Tok<Overwrite>(AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
            if (startLoopIdx < coreLoops) {
                MEGA_MOE_BF16_TIMER_BLOCK(timer.Tik(AscendTimer::AIV_SWIGLU, expertIdx));
            }
        }
    }
    AscendC::GlobalTensor<bfloat16_t> savedGate, savedUp;
    // BF16 already owns persistent raw planes. none/mxfp8 reuse Hp for
    // raw Gate; only Up is appended after the unchanged ready-counter region.
    // Legacy consumers wait for every AIC; the opt-in BF16 path waits only for
    // its paired producer's completed Gate/Up tile. Each Gate
    // tile is fully loaded before its Hp store overwrites that same tile.
    constexpr bool reuseHp = SaveMode != MegaMoeBF16Training::kSaveBf16;
    const ReadyLayout rawLayout(params.tilingData->n,
        params.tilingData->expertPerRank, params.tilingData->maxOutputSize);
    GM_ADDR rawGate = reuseHp ? params.workspaceInfo.ptrA2 : params.workspaceInfo.ptrSavedGate;
    GM_ADDR rawUp = reuseHp ? params.workspaceInfo.ptrA2 + rawLayout.totalBytes : params.workspaceInfo.ptrSavedUp;
    savedGate.SetGlobalBuffer((__gm__ bfloat16_t *)rawGate + savedElementOffset);
    savedUp.SetGlobalBuffer((__gm__ bfloat16_t *)rawUp + savedElementOffset);
    savedGate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    savedUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    auto gmLayout = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, gmm1Params.outputN);
    auto gmGate = tla::MakeTensor(savedGate, gmLayout, Catlass::Arch::PositionGM{});
    auto gmUp = tla::MakeTensor(savedUp, gmLayout, Catlass::Arch::PositionGM{});

    MegaMoeBF16ForwardSchedule::Schedule phaseSchedule;
#if defined(DEEPEP_BF16_GMM1_AUTO_SELECT) && DEEPEP_BF16_GMM1_AUTO_SELECT
    if (gmm1Params.preparedPhaseEnabled) {
    MegaMoeBF16ForwardSchedule::View phaseView;
    phaseView.Init(gmmAddrInfo.groupFlagList2-expertIdx,
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrcumsumMM),
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1),
        params.tilingData->maxOutputSize,params.tilingData->expertPerRank,
        params.tilingData->epWorldSize,gmm1Params.blockNum,gmm1Params.dispatchBlockNumPerEP,
        params.tilingData->k);
    const bool active=g_coreType==AscendC::AIC || gmm1Params.subBlockIdx==0;
    phaseSchedule.Init(phaseView,expertIdx,0,m,gmm1Params.outputN,nStripeWidth,useTileReady&&active);
    }
#endif
    bool dispatchExpertReady = false;
    for (uint32_t loopIdx = startLoopIdx;
         loopIdx < coreLoops;
         loopIdx += gmm1Params.blockNum) {
        Catlass::GemmCoord blockCoord{0,0,0};
        if(phaseSchedule.Enabled()) {
            const auto xy=phaseSchedule.Get(loopIdx);
            blockCoord=Catlass::GemmCoord{xy.m,xy.n,0};
        } else {
#if defined(DEEPEP_BF16_GMM1_READY_FIRST) && DEEPEP_BF16_GMM1_READY_FIRST
        blockCoord =
            nStripeWidth > 0 && useAdaptiveSchedule
            ? (useTileReady
                ? GetGmm1AdaptiveReadyFirstCoord(
                    loopIdx, m, gmm1Params.outputN, nStripeWidth,
                    readyBlocksPerProducerPartition)
                : GetGmm1AdaptiveNStripeCoord(
                    loopIdx, m, gmm1Params.outputN, nStripeWidth))
            : scheduler.GetBlockCoord(loopIdx);
#elif defined(DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE) && DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE
        blockCoord = nStripeWidth > 0
            ? GetGmm1AdaptiveNStripeCoord(
                loopIdx, m, gmm1Params.outputN, nStripeWidth)
            : scheduler.GetBlockCoord(loopIdx);
#else
        blockCoord = scheduler.GetBlockCoord(loopIdx);
#endif
        }
        Catlass::GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);
        const uint32_t mLoc = blockCoord.m() * GMM1_TILE_M;
        const uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        const uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr (g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1_WAIT, expertIdx));
            }
            // Streaming mode checks each task's input because static swizzling
            // can visit non-adjacent M tiles. The default expert mode waits
            // for the whole expert and removes all later input polling.
            if constexpr (DispatchTileReady && WaitForFullExpert) {
                // Expert mode publishes only partition completion. The row
                // counter layout is still reserved for the opt-in streaming
                // mode; wait on the separate expert-completion counter once.
                if (loopIdx == startLoopIdx) {
                    __gm__ int32_t *expertFlag =
                        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t target =
                        params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {}
                    }
                }
            } else if constexpr (DispatchTileReady) {
                if (!useTileReady) {
                    if (loopIdx == startLoopIdx) {
                        __gm__ int32_t *expertFlag =
                            reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                            expertIdx * INT_CACHELINE;
                        const uint32_t target =
                            params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                        while (target != AscendC::ReadGmByPassDCache(expertFlag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {}
                        }
                    }
                } else if (!dispatchExpertReady) {
                    __gm__ int32_t *expertFlag =
                        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrFlagDispatchToGmm1) +
                        expertIdx * INT_CACHELINE;
                    const uint32_t expertTarget =
                        params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                    dispatchExpertReady =
                        expertTarget == AscendC::ReadGmByPassDCache(expertFlag);
                    if (!dispatchExpertReady) {
                        const uint32_t readyBlock = mLoc / ReadyLayout::kReadyRows;
                        __gm__ int32_t *flag = gmmAddrInfo.groupFlagList2 +
                            readyBlock * INT_CACHELINE;
                        const int32_t target = static_cast<int32_t>(
                            ReadyLayout::RowsInReadyBlock(readyBlock, m));
                        while (target != AscendC::ReadGmByPassDCache(flag)) {
                            const int64_t start = AscendC::GetSystemCycle();
                            while (AscendC::GetSystemCycle() - start < 100) {}
                        }
                    }
                }
            } else if (loopIdx == startLoopIdx) {
                const uint32_t target =
                    params.tilingData->epWorldSize * gmm1Params.dispatchBlockNumPerEP;
                __gm__ int32_t *flag =
                    (__gm__ int32_t *)dispatchFlags.GetPhyAddr();
                while (target != AscendC::ReadGmByPassDCache(flag)) {
                    const int64_t start = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - start < 100) {}
                }
            }
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tok<Overwrite>(
                        AscendTimer::AIC_GMM1_WAIT, expertIdx));
                // Keep the existing per-expert timer schema: WAIT measures
                // the first task's input wait; GMM1 includes GM writes and,
                // in streaming mode only, later tile waits.
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1, expertIdx));
            }
#if defined(DEEPEP_BF16_REPLICA_STRIPES)
            // The Wuhan small-H GM epilogue is also a W1 consumer. Both gate
            // and up planes of this output stripe must be visible before the
            // first blockMmad prefetch; dispatch readiness alone is insufficient.
            static_assert(L1_TILE_N == MegaFluxPipeline::STRIPE_N,
                "weight stripe must match GM GMM1 N tile");
            if (gmm1Params.stripeReady && expertIdx >= gmm1Params.stripeHome) {
                MegaFluxPipeline::WaitStripe(gmm1Params.stripeReady,
                    expertIdx - gmm1Params.stripeHome,
#if defined(DEEPEP_BF16_REPLICA_STREAM)
                    (gmm1Params.streamMode & 1)
                        ? (gmm1Params.outputN + L1_TILE_N - 1) / L1_TILE_N - 1 :
#endif
                    nLoc / L1_TILE_N,
                    (gmm1Params.outputN + L1_TILE_N - 1) / L1_TILE_N,
                    gmm1Params.blockNum, gmm1Params.stripeEpoch);
            }
#endif
            auto tensorBlockA = tla::GetTile(
                tensorA,
                tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockGate = tla::GetTile(
                gmGate,
                tla::MakeCoord(mLoc, nLoc),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUp = tla::GetTile(
                gmUp,
                tla::MakeCoord(mLoc, nLoc),
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
                auto &tensorBlockOutput =
                    weightBlock == 0 ? tensorBlockGate : tensorBlockUp;
                gmm1Params.blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockOutput, actualShape);
            }
            if (useTileProgress) {
                // MMAD returns before its asynchronous Fixpipe stores finish.
                // Drain FIX only after both Gate/Up writes, then publish one
                // monotonic producer-local ordinal without any AIV ack. The
                // frozen prepared schedule gives producer and AIV0 identical
                // loopIdx -> GM tile mappings, including swizzled tail tiles.
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
                AscendC::WriteGmByPassDCache(producerProgress,
                    static_cast<int32_t>(MegaMoeBF16GmTileProgress::Ordinal(
                        loopIdx, startLoopIdx, gmm1Params.blockNum)));
            }
        }

        // No UB producer/consumer handshake on the GM path.
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm1Params.subBlockIdx == 0) {
                if (useTileProgress) {
                    if (loopIdx == startLoopIdx) {
                        MEGA_MOE_BF16_TIMER_BLOCK(timer.Tik(AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                    }
                    const uint32_t target = MegaMoeBF16GmTileProgress::Ordinal(
                        loopIdx, startLoopIdx, gmm1Params.blockNum);
                    while (!MegaMoeBF16GmTileProgress::Reached(
                            AscendC::ReadGmByPassDCache(producerProgress), target)) {
                        const int64_t start = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - start < 100) {}
                    }
                    if (loopIdx == startLoopIdx) {
                        MEGA_MOE_BF16_TIMER_BLOCK(timer.Tok<Overwrite>(AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                        MEGA_MOE_BF16_TIMER_BLOCK(timer.Tik(AscendTimer::AIV_SWIGLU, expertIdx));
                    }
                }
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
                epilogueOp.LoadRawTile(rawGate, rawUp, savedElementOffset,
                    actualShape.m(), actualShape.n(), Get<0>(epilogueOffset));
                // Quantized persistent state is produced from unmodified UB
                // values before ComputeSwiglu. BF16 raw planes are already saved.
                constexpr uint32_t epilogueSave = SaveMode == MegaMoeBF16Training::kSaveMxFp8
                    ? MegaMoeBF16Training::kSaveMxFp8 : MegaMoeBF16Training::kSaveNone;
                epilogueOp.template Run<epilogueSave>(epilogueShape, epilogueOffset);

            }
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        // Include every AIC, including those with no tasks for this expert.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::AtomicAdd(gmmReady, 1);
        if (startLoopIdx < coreLoops) {
            MEGA_MOE_BF16_TIMER_BLOCK(
                timer.Tok<Overwrite>(
                    AscendTimer::AIC_GMM1, expertIdx));
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (startLoopIdx < coreLoops) {
            MEGA_MOE_BF16_TIMER_BLOCK(
                timer.Tok<Overwrite>(
                    AscendTimer::AIV_SWIGLU, expertIdx));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % gmm1Params.blockNum;
}

template<bool TransW = true>
__aicore__ inline void GroupMatmul2(
    GroupMatmulParam<1, TransW> &gmm2Params,
    const Params &params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
    const GMMAddrInfo &gmmAddrInfo,
    uint32_t &startBlockIdx,
    int32_t &vecSetSyncCom,
    uint32_t groupIdx,
    uint32_t expertRowOffset,
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
            MEGAFLUX_STAGE_BEGIN(stageInputWait,gmm2Params.pipelineTrace,false,MegaFluxStageTimer::FWD_INPUT_WAIT,groupIdx,1,mLoc,nLoc);
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_BF16_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM2_WAIT, groupIdx));
                BlockScheduler gmm1Scheduler(
                    {m, k, n}, Catlass::MakeCoord(GMM1_TILE_M, L1_TILE_N));
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
            MEGAFLUX_STAGE_END(stageInputWait);
            if (vecSetSyncCom >= 2) {
                MEGAFLUX_STAGE_BEGIN(stageVectorWait,gmm2Params.pipelineTrace,false,MegaFluxStageTimer::FWD_CUBE_WAIT_VECTOR,groupIdx,1,mLoc,nLoc);
                WaitForVector(pingpongIdx + FLAG_ID_MAX);
                MEGAFLUX_STAGE_END(stageVectorWait);
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
#if defined(DEEPEP_BF16_REPLICA_STREAM)
            static_assert(L1_TILE_N==MegaFluxPipeline::STRIPE_N,"GMM2 stripe width mismatch");
            static_assert(2*L1_TILE_M*L1_TILE_N*sizeof(bfloat16_t)<=MegaFluxPipeline::COPY_UB_OFFSET,
                          "W2 copy aliases GMM2 Fixpipe UB");
            if(gmm2Params.stripeReady && groupIdx>=gmm2Params.stripeHome) {
#if MEGAFLUX_OPT_CACHE
                const auto cacheAddress=gmm2Params.stripeReady+gmm2Params.w2FlagOffset*MegaFluxPipeline::MAX_WORKERS*MegaFluxPipeline::LINE*sizeof(int32_t);
                const int cacheKey=(groupIdx-gmm2Params.stripeHome)*((gmm2Params.n+L1_TILE_N-1)/L1_TILE_N)+nLoc/L1_TILE_N+1;
                if(!gmm2Params.stripeCache.Has(cacheAddress,gmm2Params.stripeEpoch,1,cacheKey)) {
#endif
                MEGAFLUX_STAGE_BEGIN(stageWait,gmm2Params.pipelineTrace,false,MegaFluxStageTimer::WEIGHT_WAIT,groupIdx,1,mLoc,nLoc,-1,groupIdx-gmm2Params.stripeHome);
                const auto waitStart=MegaFluxDetail::Now(gmm2Params.pipelineTrace);
                MegaFluxPipeline::WaitStripe(gmm2Params.stripeReady+
                    gmm2Params.w2FlagOffset*MegaFluxPipeline::MAX_WORKERS*MegaFluxPipeline::LINE*sizeof(int32_t),
                    groupIdx-gmm2Params.stripeHome,nLoc/L1_TILE_N,
                    (gmm2Params.n+L1_TILE_N-1)/L1_TILE_N,gmm2Params.blockNum,gmm2Params.stripeEpoch);
                MEGAFLUX_STAGE_END(stageWait);
                MegaFluxStageTimer::Mark(gmm2Params.pipelineTrace,false,MegaFluxStageTimer::WEIGHT_USE,groupIdx,1,mLoc,nLoc,-1,groupIdx-gmm2Params.stripeHome);
                const auto ready=MegaFluxDetail::Now(gmm2Params.pipelineTrace);
                MegaFluxDetail::Record(gmm2Params.pipelineTrace,false,MegaFluxDetail::WEIGHT_WAIT,waitStart,ready,groupIdx,1,mLoc,nLoc,-1,groupIdx-gmm2Params.stripeHome);
                MegaFluxDetail::Record(gmm2Params.pipelineTrace,false,MegaFluxDetail::WEIGHT_USE,ready,ready,groupIdx,1,mLoc,nLoc,-1,groupIdx-gmm2Params.stripeHome);
#if MEGAFLUX_OPT_CACHE
                    gmm2Params.stripeCache.Add(cacheKey);
                } else {
                    MegaFluxStageTimer::Mark(gmm2Params.pipelineTrace,false,MegaFluxStageTimer::WEIGHT_USE,groupIdx,1,mLoc,nLoc,-1,groupIdx-gmm2Params.stripeHome);
                }
#endif
            }
            const bool traceGmm2=gmm2Params.pipelineTrace && !gmm2Params.firstGmm2Traced;
            const bool traceReplicaGmm2=gmm2Params.pipelineTrace && groupIdx>=gmm2Params.stripeHome &&
                !gmm2Params.firstReplicaTraced;
            if(traceGmm2)MegaFluxPipeline::Stamp(gmm2Params.pipelineTrace,gmm2Params.blockIdx,327);
            if(traceReplicaGmm2)MegaFluxPipeline::Stamp(gmm2Params.pipelineTrace,gmm2Params.blockIdx,329);
#endif
            MEGAFLUX_STAGE_BEGIN(stageGemm,gmm2Params.pipelineTrace,false,MegaFluxStageTimer::FWD_GEMM_TILE,groupIdx,1,mLoc,nLoc);
            gmm2Params.blockMmad(
                tensorBlockA, tensorBlockB, tensorBlockUb, actualShape);
            MEGAFLUX_STAGE_END(stageGemm);
#if defined(DEEPEP_BF16_REPLICA_STREAM)
            if(traceGmm2 || traceReplicaGmm2) {
                AscendC::PipeBarrier<PIPE_ALL>();
                if(traceGmm2) {
                    MegaFluxPipeline::Stamp(gmm2Params.pipelineTrace,gmm2Params.blockIdx,328);
                    gmm2Params.firstGmm2Traced=true;
                }
                if(traceReplicaGmm2) {
                    MegaFluxPipeline::Stamp(gmm2Params.pipelineTrace,gmm2Params.blockIdx,330);
                    gmm2Params.firstReplicaTraced=true;
                }
            }
#endif
            NotifyVector(pingpongIdx + FLAG_ID_MAX);
        }

        ++vecSetSyncCom;
        if constexpr (g_coreType == AscendC::AIV) {
            if (gmm2Params.subBlockIdx == 1) {
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_COMBINE_WAIT, groupIdx));
                }
                MEGAFLUX_STAGE_BEGIN(stageCubeWait,gmm2Params.pipelineTrace,false,MegaFluxStageTimer::FWD_VECTOR_WAIT,groupIdx,1,mLoc,nLoc);
                WaitForCube(pingpongIdx);
                MEGAFLUX_STAGE_END(stageCubeWait);
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tok<Overwrite>(
                            AscendTimer::AIV_COMBINE_WAIT, groupIdx));
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_COMBINE, groupIdx));
                }
                MEGAFLUX_STAGE_BEGIN(stageVector,gmm2Params.pipelineTrace,false,MegaFluxStageTimer::FWD_VECTOR_TILE,groupIdx,1,mLoc,nLoc);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                MegaMoeCombineImpl::CombineTokens<bfloat16_t, Catlass::GemmCoord>(
                    mLoc,
                    nLoc,
                    n,
                    groupIdx,
                    expertRowOffset,
                    rankId,
                    outputUb,
                    actualShape,
                    params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                MEGAFLUX_STAGE_END(stageVector);
                NotifyCube(pingpongIdx);
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

#endif  // DEEPEP_MEGA_MOE_BF16_CATLASS_IMPL_H
