/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef ASCEND_DEEPEP_MEGA_MOE_W4A8_MATMUL_KERNEL_H
#define ASCEND_DEEPEP_MEGA_MOE_W4A8_MATMUL_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/catlass.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/status.hpp"

// #include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../block_epilogue_swiglu_mx_quant.h"
#include "../mega_moe_combine_send.h"
#include "tile_copy.h"
#include "block_mmad_mx4_tla.h"

using namespace AscendC;
using namespace Catlass;

namespace MegaMoeFFNMxw4a8 {

    using GemmShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    using ArchTag = Arch::Ascend950;
    constexpr static bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMxW4A8<ArchTag, enableUnitFlag, 16>;
    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    constexpr uint32_t ALIGN32 = 32U;
    constexpr uint32_t SWIGLU_N_HALF = 2;
    constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
    constexpr uint32_t BUFALIGN = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32 * sizeof(bfloat16_t);
    constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM = (MAX_SINGLE_MN + ALIGN32 - 1) / ALIGN32 * ALIGN32;
    constexpr uint32_t CrossCoreMode = 4;
    constexpr uint32_t AicCrossCoreBaseFlag = 4;
    constexpr uint32_t AivCrossCoreBaseFlag = 6;

    //groupedMatmul1
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::nZ;
    using LayoutTagAScale = layout::RowMajor;
    using LayoutTagBScale = layout::ColumnMajor;
    using LayoutTagC = layout::RowMajor;
    using ElementBias = void;
    // using ElementC = float;

    struct GroupedMatmulParams {
        GemmShape matmulShape;
        uint32_t startBlockIdx{0};
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrMxScaleA;
        GM_ADDR ptrMxScaleB;
        GM_ADDR ptrCData;
        GM_ADDR ptrCScale;
        GM_ADDR ptrGmm1Output;
        GM_ADDR groupFlagList1;
        GM_ADDR groupFlagList2;
        GM_ADDR groupFlagListLocal;
        GM_ADDR gmm1ReadyFlag;
        GM_ADDR gmm2ReadyFlag;

        uint32_t groupIdx{0};
        uint16_t pingpongIdx{0};
        int32_t rankId{0};
        uint32_t expertIdx;
    };

    __aicore__ inline void WaitAivToCube(uint16_t flag){
        AscendC::CrossCoreWaitFlag<CrossCoreMode, PIPE_FIX>(AivCrossCoreBaseFlag + flag);
    //    AscendC::CrossCoreWaitFlag<CrossCoreMode, PIPE_FIX>(flag + 16);
    }

    __aicore__ inline void SetCubeToAiv(uint16_t flag){
        AscendC::CrossCoreSetFlag<CrossCoreMode, PIPE_FIX>(AicCrossCoreBaseFlag + flag);
    //    AscendC::CrossCoreSetFlag<CrossCoreMode, PIPE_FIX>(flag + 16);
    }

    __aicore__ inline void SetAivToCube1(uint16_t flag){
        AscendC::CrossCoreSetFlag<CrossCoreMode, PIPE_MTE3>(AivCrossCoreBaseFlag + flag);
    }

    __aicore__ inline void WaitCubeToAiv1(uint16_t flag){
        AscendC::CrossCoreWaitFlag<CrossCoreMode, PIPE_V>(AicCrossCoreBaseFlag + flag);
    }

    __aicore__ inline void SetAivToCube2(uint16_t flag){
        AscendC::CrossCoreSetFlag<CrossCoreMode, PIPE_MTE3>(AivCrossCoreBaseFlag + flag);
    }

    __aicore__ inline void WaitCubeToAiv2(uint16_t flag){
        AscendC::CrossCoreWaitFlag<CrossCoreMode, PIPE_MTE3>(AicCrossCoreBaseFlag + flag);
    }

   // GroupMatmulSwigluQuant
    template<
        class ElementA_,
        class ElementPrologueB_,
        class ElementB_,
        class ElementC_,
        class ElementMxScaleA_,
        class ElementMxScaleB_
    >
    class GroupMatmul1SwigluQuant {
        using ElementA =ElementA_;
        using ElementPrologueB = ElementPrologueB_;
        using ElementB = ElementB_;
        using ElementC =ElementC_;
        using ElementMxScaleA = ElementMxScaleA_;
        using ElementMxScaleB = ElementMxScaleB_;
        static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
        static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
        static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

        using TileCopy = Gemm::Tile::PackedMx4A8TileCopyTla<
                ArchTag,
                ElementA,
                LayoutTagA,
                ElementB,
                LayoutTagB,
                ElementPrologueB,
                LayoutTagB,
                ElementMxScaleA,
                LayoutTagAScale,
                ElementMxScaleB,
                LayoutTagBScale,
                ElementC,
                LayoutTagC,
                ElementBias,
                0,
                Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
                false,
                Gemm::Tile::ScaleGranularity::NO_QUANT,
                true>;

        using BlockMmad = Gemm::Block::BlockMmadTla<
                DispatchPolicy,
                L1TileShape,
                L0TileShape,
                ElementA,
                ElementB,
                ElementC,
                void,
                TileCopy>;

        using BlockEpilogue = BlockEpilogueSwigluMxQuant<ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>;

    private:
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad;
        BlockEpilogue epilogueOp;

        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t subBlockIdx = 0;
        uint32_t targetValue = 0;

        __aicore__ inline
        void waitFromDispatch(AscendC::GlobalTensor<int32_t>&  groupFlagList, int32_t targetValue){
            __gm__ int32_t *flagValueAddr = ( __gm__ int32_t *) groupFlagList.GetPhyAddr();
            while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {}
            }
        }

        __aicore__ inline void waitFromDispatchLocal(AscendC::GlobalTensor<int32_t>& groupFlagListLocal) {
            __gm__ int32_t *flagValueAddr = (__gm__ int32_t *) groupFlagListLocal.GetPhyAddr();
            while (1 != AscendC::ReadGmByPassDCache(flagValueAddr)) {  // 本地flag目标值固定为1
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {}
            }
        }

    public:
        __aicore__ inline
        GroupMatmul1SwigluQuant(){

            blockNum = AscendC::GetBlockNum();
            subBlockIdx = AscendC::GetSubBlockIdx();
            blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();

        }

        __aicore__ inline
        void start(AscendC::GlobalTensor<uint8_t>& gammaPtr, uint32_t targetValue, uint32_t n){
            this->targetValue = targetValue;
            if constexpr (g_coreType == AscendC::AIC) {
                blockMmad.start(resource);
            }else{
                if (subBlockIdx == 1){
                    blockMmad.start(resource);
                    blockMmad.loadGamma(gammaPtr);
                }else{
                    epilogueOp.Init(L1_TILE_M, n);
                }
            }
        }

        __aicore__ inline
        void end(){
            if constexpr (g_coreType == AscendC::AIC) {
                blockMmad.end();
            }else{
                if (subBlockIdx == 1){
                    blockMmad.end();
                }
            }
        }

        template <int32_t CORE_TYPE = g_coreType> __aicore__ inline
        void operator()(GroupedMatmulParams &groupedMatmulParams, AscendTimerDevice timer_);

        template <> __aicore__ inline
        void operator()<AscendC::AIC>(GroupedMatmulParams &groupedMatmulParams, AscendTimerDevice timer_){

            uint32_t m = Get<M_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t n = Get<N_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t k = Get<K_VALUE>(groupedMatmulParams.matmulShape);

            uint32_t swigluOutputN = n / SWIGLU_N_HALF;
            uint32_t scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

            auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k + scaleK);
            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagAScale, false>(m, k + scaleK);
            auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagBScale, true>(scaleK, n);
            auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, swigluOutputN);

            AscendC::GlobalTensor <int32_t> groupFlagList;
            groupFlagList.SetGlobalBuffer((__gm__ int32_t *)groupedMatmulParams.groupFlagList1);
            AscendC::GlobalTensor <int32_t> groupFlagListLocal;
            groupFlagListLocal.SetGlobalBuffer((__gm__ int32_t *)groupedMatmulParams.groupFlagListLocal);
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer((__gm__ ElementA *)groupedMatmulParams.ptrA);
            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)groupedMatmulParams.ptrMxScaleA);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)groupedMatmulParams.ptrMxScaleB);
            AscendC::GlobalTensor<ElementC> gmCList[SWIGLU_N_HALF];
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                gmCList[weightBlock].SetGlobalBuffer(
                    (__gm__ ElementC *)groupedMatmulParams.ptrGmm1Output +
                    static_cast<uint64_t>(weightBlock) * m * swigluOutputN);
            }

            if (CeilDiv(n, L1_TILE_N) == 1) {
                gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            auto tensorA = tla::MakeTensor(gmA, layoutA, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, layoutMxScaleB, Arch::PositionGM{});

            BlockScheduler matmulBlockScheduler({m, swigluOutputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = (blockIdx < groupedMatmulParams.startBlockIdx ? blockIdx + blockNum : blockIdx) - groupedMatmulParams.startBlockIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
                if (loopIdx == startLoopIdx) {
                    waitFromDispatch(groupFlagList, targetValue);
                    waitFromDispatchLocal(groupFlagListLocal);
                    TIMER_BLOCK(timer_.Tik(AscendTimer::AIC_GMM1, groupedMatmulParams.expertIdx));
                }
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(
                        tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
                );

                auto tensorBlockMxScaleA = GetTile(
                        tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()))
                );

                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto tensorBlockMxScaleB = GetTile(
                            tensorMxScaleB,
                            tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N + weightBlock * swigluOutputN),
                            tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n())
                    );
                    auto tensorC = tla::MakeTensor(gmCList[weightBlock], layoutC, Arch::PositionGM{});
                    auto tensorBlockC = GetTile(
                            tensorC,
                            tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
                    );
                    blockMmad(tensorBlockA, tensorBlockC, actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::AtomicAdd((__gm__ int32_t *)groupedMatmulParams.gmm1ReadyFlag, int32_t(1));
            TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIC_GMM1, groupedMatmulParams.expertIdx));
            groupedMatmulParams.startBlockIdx = (groupedMatmulParams.startBlockIdx + coreLoops) % blockNum;
        }

        template <> __aicore__ inline
        void operator()<AscendC::AIV>( GroupedMatmulParams &groupedMatmulParams, AscendTimerDevice timer_){

            uint32_t m = Get<M_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t n = Get<N_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t k = Get<K_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t swigluOutputN = n / SWIGLU_N_HALF;
            uint32_t scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

            auto prologueLayoutB = tla::MakeLayout<ElementPrologueB, LayoutTagB>(k, n);

            AscendC::GlobalTensor<ElementPrologueB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementPrologueB*)groupedMatmulParams.ptrB);

            if (CeilDiv(m, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            // tensorBlayout 修改
            auto tensorB = tla::MakeTensor(gmB, prologueLayoutB, Arch::PositionGM{});

            BlockScheduler matmulBlockScheduler({m, swigluOutputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = (blockIdx < groupedMatmulParams.startBlockIdx ? blockIdx + blockNum : blockIdx) - groupedMatmulParams.startBlockIdx;

            if (subBlockIdx == 0) {
                __gm__ int32_t *readyFlag = (__gm__ int32_t *)groupedMatmulParams.gmm1ReadyFlag;
                while (AscendC::ReadGmByPassDCache(readyFlag) != blockNum) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                if (subBlockIdx == 0) {
                    Std::tuple <int64_t, int64_t, int64_t, int64_t> epilogueShape{actualBlockShape.m(), actualBlockShape.n(), 0, 0};
                    int64_t dataOffset = blockCoord.m() * L1_TILE_M * swigluOutputN + blockCoord.n() * L1_TILE_N;
                    int64_t scaleOffset = blockCoord.m() * L1_TILE_M * CeilDiv(swigluOutputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                            CeilDiv(blockCoord.n() * L1_TILE_N, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
                    if (loopIdx == startLoopIdx) {
                        TIMER_BLOCK(timer_.Tik(AscendTimer::AIV_SWIGLU, groupedMatmulParams.expertIdx));
                    }
                    AscendC::SetCtrlSpr<60, 60>(0);
                    uint64_t gmm1TileOffset =
                        (static_cast<uint64_t>(blockCoord.m()) * L1_TILE_M * swigluOutputN +
                         static_cast<uint64_t>(blockCoord.n()) * L1_TILE_N) * sizeof(ElementC);
                    epilogueOp(
                        epilogueShape,
                        groupedMatmulParams.ptrGmm1Output + gmm1TileOffset,
                        groupedMatmulParams.ptrGmm1Output +
                            static_cast<uint64_t>(m) * swigluOutputN * sizeof(ElementC) + gmm1TileOffset,
                        swigluOutputN,
                        groupedMatmulParams.ptrCData + dataOffset,
                        groupedMatmulParams.ptrCScale + scaleOffset,
                        groupedMatmulParams.groupFlagList2);
                } else {
                    for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                        auto tensorBlockB = GetTile(
                                tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N + weightBlock * swigluOutputN),
                                tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
                        );
                        if (loopIdx == startLoopIdx && weightBlock == 0) {
                            TIMER_BLOCK(timer_.Tik(AscendTimer::AIV_GMM1_PROLOGUE, groupedMatmulParams.expertIdx));
                        }
                        blockMmad.Prologue(tensorBlockB, actualBlockShape);
                    }
                }
            }
            if (subBlockIdx == 0) {
                TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_SWIGLU, groupedMatmulParams.expertIdx));
            } else {
                TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_GMM1_PROLOGUE, groupedMatmulParams.expertIdx));
            }
            groupedMatmulParams.startBlockIdx = (groupedMatmulParams.startBlockIdx + coreLoops) % blockNum;
        }
    };

template<
        class ElementA_,
        class ElementPrologueB_,
        class ElementB_,
        class ElementC_,
        class ElementMxScaleA_,
        class ElementMxScaleB_
>
class GroupMatmul2Combine {
    using ElementA = ElementA_;
    using ElementPrologueB = ElementPrologueB_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementMxScaleA = ElementMxScaleA_;
    using ElementMxScaleB = ElementMxScaleB_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    using TileCopy = Gemm::Tile::PackedMx4A8TileCopyTla<
            ArchTag,
            ElementA,
            LayoutTagA,
            ElementB,
            LayoutTagB,
            ElementPrologueB,
            LayoutTagB,
            ElementMxScaleA,
            LayoutTagAScale,
            ElementMxScaleB,
            LayoutTagBScale,
            ElementC,
            LayoutTagC,
            ElementBias,
            0,
            Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
            false,
            Gemm::Tile::ScaleGranularity::NO_QUANT,
            true>;

    using BlockMmad = Gemm::Block::BlockMmadTla<
            DispatchPolicy,
            L1TileShape,
            L0TileShape,
            ElementA,
            ElementB,
            ElementC,
            void,
            TileCopy>;

    private:
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad;

        uint32_t blockNum = 0;
        uint32_t blockIdx = 0;
        uint32_t subBlockIdx = 0;

        __aicore__ inline
        void waitFromGMM1(AscendC::GlobalTensor <int32_t>& groupFlagList, uint32_t targetValue) {
            __gm__ int32_t * flagValueAddr = (__gm__ int32_t *) groupFlagList.GetPhyAddr();
            while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {}
            }
        }

    public:
        __aicore__ inline
        GroupMatmul2Combine() {

            blockNum = AscendC::GetBlockNum();
            subBlockIdx = GetSubBlockIdx();
            blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();

        }

        __aicore__ inline
        void start(AscendC::GlobalTensor<uint8_t>& gammaPtr){
            if constexpr (g_coreType == AscendC::AIC) {
                blockMmad.start(resource);
            }else{
                if (subBlockIdx == 1){
                    blockMmad.start(resource);
                    blockMmad.loadGamma(gammaPtr);
                }

            }
        }

        __aicore__ inline
        void end(){
            if constexpr (g_coreType == AscendC::AIC) {
                blockMmad.end();
            }else{
                if (subBlockIdx == 1){
                    blockMmad.end();
                }
            }
        }

        template<int32_t CORE_TYPE = g_coreType>
        __aicore__ inline
        void operator()(const Params &params,
                        GroupedMatmulParams &groupedMatmulParams,
                        AscendTimerDevice timer_);

        template<>
        __aicore__ inline
        void operator()<AscendC::AIC>(
                const Params &params,
                GroupedMatmulParams &groupedMatmulParams, AscendTimerDevice timer_) {

            uint32_t m = Get<M_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t n = Get<K_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t k = Get<N_VALUE>(groupedMatmulParams.matmulShape) / SWIGLU_N_HALF;
            uint32_t scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

            auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
            auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagAScale, false>(m, scaleK);
            auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagBScale, true>(scaleK, n);
            auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);

            AscendC::GlobalTensor <int32_t> groupFlagList;
            groupFlagList.SetGlobalBuffer((__gm__ int32_t *)groupedMatmulParams.groupFlagList2);
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer((__gm__ ElementA *)groupedMatmulParams.ptrA);
            AscendC::GlobalTensor<ElementMxScaleA> gmMxScaleA;
            gmMxScaleA.SetGlobalBuffer((__gm__ ElementMxScaleA *)groupedMatmulParams.ptrMxScaleA);
            AscendC::GlobalTensor<ElementMxScaleB> gmMxScaleB;
            gmMxScaleB.SetGlobalBuffer((__gm__ ElementMxScaleB *)groupedMatmulParams.ptrMxScaleB);
            AscendC::GlobalTensor<ElementC> gmC;
            gmC.SetGlobalBuffer((__gm__ ElementC *)groupedMatmulParams.ptrCData);

            if (CeilDiv(n, L1_TILE_N) == 1) {
                gmA.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            auto tensorA = tla::MakeTensor(gmA, layoutA, Arch::PositionGM{});
            auto tensorMxScaleA = tla::MakeTensor(gmMxScaleA, layoutMxScaleA, Arch::PositionGM{});
            auto tensorMxScaleB = tla::MakeTensor(gmMxScaleB, layoutMxScaleB, Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(gmC, layoutC, Arch::PositionGM{});

            BlockScheduler matmulBlockScheduler({m, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = (blockIdx < groupedMatmulParams.startBlockIdx ? blockIdx + blockNum : blockIdx) - groupedMatmulParams.startBlockIdx;
            BlockScheduler target({m, k, n}, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t targetValue = target.GetCoreLoops();

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
                if (loopIdx == startLoopIdx) {
                    waitFromGMM1(groupFlagList, targetValue);
                    TIMER_BLOCK(timer_.Tik(AscendTimer::AIC_GMM2, groupedMatmulParams.expertIdx));
                }
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                auto tensorBlockA = GetTile(
                        tensorA, tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())
                );
                auto tensorBlockMxScaleB = GetTile(
                        tensorMxScaleB,
                        tla::MakeCoord(blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n())
                );
                auto tensorBlockMxScaleA = GetTile(
                        tensorMxScaleA,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K / MX_SCALE_GROUP_NUM),
                        tla::MakeShape(actualBlockShape.m(), CeilDiv<MX_SCALE_GROUP_NUM>(actualBlockShape.k()))
                );
                auto tensorBlockC = GetTile(
                        tensorC,
                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.n() * L1_TILE_N),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n())
                );

                // GMM2 writes the complete expert output to GM. Combine is
                // launched after every AIC has finished this expert.
                blockMmad(
                        tensorBlockA, tensorBlockC, actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::AtomicAdd(
                (__gm__ int32_t *)groupedMatmulParams.gmm2ReadyFlag,
                int32_t(1));
            TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIC_GMM2, groupedMatmulParams.expertIdx));
            groupedMatmulParams.startBlockIdx = (groupedMatmulParams.startBlockIdx + coreLoops) % blockNum;
        }

        template<>
        __aicore__ inline
        void operator()<AscendC::AIV>(const Params &params,
                                    GroupedMatmulParams &groupedMatmulParams,
                                    AscendTimerDevice timer_) {

            uint32_t m = Get<M_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t n = Get<K_VALUE>(groupedMatmulParams.matmulShape);
            uint32_t k = Get<N_VALUE>(groupedMatmulParams.matmulShape) / SWIGLU_N_HALF;
            uint32_t scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

            auto prologueLayoutB = tla::MakeLayout<ElementPrologueB, LayoutTagB>(k, n);

            AscendC::GlobalTensor <ElementPrologueB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementPrologueB *)groupedMatmulParams.ptrB);

            if ((m, L1_TILE_M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            auto tensorB = tla::MakeTensor(gmB, prologueLayoutB, Arch::PositionGM{});

            BlockScheduler matmulBlockScheduler({m, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
            uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = (blockIdx < groupedMatmulParams.startBlockIdx ? blockIdx + blockNum : blockIdx) -
                                    groupedMatmulParams.startBlockIdx;

            if (subBlockIdx == 1) {
                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
                    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                    auto tensorBlockB = GetTile(
                            tensorB, tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                            tla::MakeShape(actualBlockShape.k(), actualBlockShape.n())
                    );
                    if (loopIdx == startLoopIdx) {
                        TIMER_BLOCK(timer_.Tik(AscendTimer::AIV_GMM2_PROLOGUE, groupedMatmulParams.expertIdx));
                    }
                    blockMmad.Prologue(tensorBlockB, actualBlockShape);
                }
                TIMER_BLOCK(timer_.template Tok<Overwrite>(
                    AscendTimer::AIV_GMM2_PROLOGUE, groupedMatmulParams.expertIdx));
            }

            if (subBlockIdx == 0) {
                __gm__ int32_t *readyFlag = (__gm__ int32_t *)groupedMatmulParams.gmm2ReadyFlag;
                while (AscendC::ReadGmByPassDCache(readyFlag) != blockNum) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }

                AscendC::GlobalTensor<ElementC> currentExpertOutputGm;
                currentExpertOutputGm.SetGlobalBuffer((__gm__ ElementC *)groupedMatmulParams.ptrCData);
                TIMER_BLOCK(timer_.Tik(AscendTimer::AIV_COMBINE, groupedMatmulParams.expertIdx));
                MegaMoeCombineImpl::CombineTokensParallel<ElementC>(
                    groupedMatmulParams.groupIdx, groupedMatmulParams.rankId,
                    currentExpertOutputGm, static_cast<int32_t>(m), params);
                TIMER_BLOCK(timer_.template Tok<Overwrite>(
                    AscendTimer::AIV_COMBINE, groupedMatmulParams.expertIdx));
            }

            groupedMatmulParams.startBlockIdx = (groupedMatmulParams.startBlockIdx + coreLoops) % blockNum;
        }
};

}  // namespace MegaMoeFFNMxw4a8
#endif
