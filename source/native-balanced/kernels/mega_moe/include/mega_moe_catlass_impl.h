/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef MEGA_MOE_CATLASS_IMPL_H
#define MEGA_MOE_CATLASS_IMPL_H


#include "kernel_operator.h"

#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"

#include "catlass/gemm/block/block_mmad.hpp"

#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_base.h"
#include "mega_moe_timer.h"
#include "tile_copy.hpp"
#include "mega_moe_combine_send.h"

using namespace AscendC;
using namespace Catlass;

namespace MegaMoeCatlassImpl {

using ArchTag = Arch::Ascend950;
constexpr static bool enableUnitFlag = true;
using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<128>>;
using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
constexpr uint32_t ALIGN32 = 32U;
constexpr uint32_t L1_TILE_M = 256;
constexpr uint32_t L1_TILE_N = 256;
constexpr uint32_t L1_TILE_K = 256;
constexpr uint32_t SWIGLU_N_HALF = 2;
constexpr uint32_t MAX_SINGLE_MN = 256 * 256;
constexpr uint32_t BUFALIGN = (MAX_SINGLE_MN + 31U) / ALIGN32 *ALIGN32 * sizeof(bfloat16_t);
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM = (MAX_SINGLE_MN + 31U) / ALIGN32 * ALIGN32;

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB, uint32_t GmmIndex>
class GroupMatmulParam {
public:
    using LayoutMxScaleA = decltype(tla::MakeMxScaleLayout<
        ElementMxScaleA, layout::RowMajor, false>(uint32_t{}, uint32_t{}));
    using LayoutB = decltype(tla::MakeLayout<ElementB, layout::ColumnMajor>(uint32_t{}, uint32_t{}));
    using LayoutMxScaleB = decltype(tla::MakeMxScaleLayout<
        ElementMxScaleB, layout::ColumnMajor, true>(uint32_t{}, uint32_t{}));
    using LayoutC = decltype(tla::MakeLayout<ElementC, layout::RowMajor>(uint32_t{}, uint32_t{}));
    using TileCopy = Gemm::Tile::PackedMxTileCopyTlaToTargetUB<
        ArchTag, ElementA, layout::RowMajor, ElementB, layout::ColumnMajor,
        ElementMxScaleA, LayoutMxScaleA, ElementMxScaleB, LayoutMxScaleB,
        ElementC, layout::RowMajor, void, GmmIndex>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    __aicore__ inline GroupMatmulParam(
        uint32_t nValue, uint32_t kValue, uint32_t dispatchBlockNumPerEPValue = 0)
        : blockMmad(resource),
          blockNum(GetBlockNum()),
          subBlockIdx(GetSubBlockIdx()),
          blockIdx(GetBlockIdx() / GetTaskRation()),
          dispatchBlockNumPerEP(dispatchBlockNumPerEPValue),
          n(nValue),
          k(kValue),
          outputN(nValue / SWIGLU_N_HALF),
          scaleK(CeilDiv(kValue, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
          layoutB(tla::MakeLayout<ElementB, layout::ColumnMajor>(kValue, nValue)),
          layoutMxScaleB(tla::MakeMxScaleLayout<ElementMxScaleB, layout::ColumnMajor, true>(scaleK, nValue)),
          layoutC(tla::MakeLayout<ElementC, layout::RowMajor>(L1_TILE_M, L1_TILE_N))
    {
        int64_t ubOffset = 0;
        l0cOutUbFirst = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
        ubOffset += MAX_SINGLE_MN_ALIGN32_NUM * sizeof(ElementC);
        l0cOutUbSecond = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
    }

    Arch::Resource<ArchTag> resource;
    BlockMmad blockMmad;
    LocalTensor<ElementC> l0cOutUbFirst;
    LocalTensor<ElementC> l0cOutUbSecond;
    uint32_t blockNum;
    uint32_t subBlockIdx;
    uint32_t blockIdx;
    uint32_t dispatchBlockNumPerEP;
    uint32_t n;
    uint32_t k;
    uint32_t outputN;
    uint32_t scaleK;
    LayoutB layoutB;
    LayoutMxScaleB layoutMxScaleB;
    LayoutC layoutC;
};

// GroupMatmulSwigluQuant - GMM1 + SwiGLU + Quant
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmulSwigluQuant(
    GroupMatmulParam<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, 0>& gmm1Params,
    BlockEpilogueSwigluMxQuant<L1TileShape, ElementA, ElementC, ElementMxScaleA, ElementMxScaleB, true>& epilogueOp,
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom,
    uint32_t expertIdx MEGA_MOE_TIMER_PARAM(timer))
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = gmm1Params.n;
    uint32_t k = gmm1Params.k;
    uint32_t outputN = gmm1Params.outputN;
    uint32_t blockNum = gmm1Params.blockNum;
    uint32_t subBlockIdx = gmm1Params.subBlockIdx;
    uint32_t blockIdx = gmm1Params.blockIdx;
    uint32_t dispatchBlockNumPerEP = gmm1Params.dispatchBlockNumPerEP;

    GlobalTensor<int32_t> groupFlagListGm2;
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList2);

    auto scaleK = gmm1Params.scaleK;
    auto LayoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto LayoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(m, scaleK);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;
    GlobalTensor<ElementMxScaleA> aScaleGlobalTensor;
    GlobalTensor<ElementMxScaleB> bScaleGlobalTensor;


    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal));
    bScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal));

    // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
    // data reads will bypass L2 Cache.
    if (CeilDiv(m, L1_TILE_M) == 1) {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }
    if (CeilDiv(n, L1_TILE_N) == 1) {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }

    auto tensorA = tla::MakeTensor(aGlobalTensor, LayoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobalTensor, gmm1Params.layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobalTensor, LayoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobalTensor, gmm1Params.layoutMxScaleB, Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(gmm1Params.l0cOutUbFirst, gmm1Params.layoutC, Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(gmm1Params.l0cOutUbSecond, gmm1Params.layoutC, Arch::PositionUB{});

    BlockScheduler scheduler({m, outputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
    uint32_t coreLoops = scheduler.GetCoreLoops();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);

        uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        if constexpr(g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1_WAIT, expertIdx));
                uint32_t targetValue = params.tilingData->epWorldSize * dispatchBlockNumPerEP;
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
                while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }
                MEGA_MOE_TIMER_BLOCK(
                    timer.template Tok<Overwrite>(
                        AscendTimer::AIC_GMM1_WAIT, expertIdx));
                MEGA_MOE_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM1, expertIdx));
            }
            if (vecSetSyncCom) {
                WaitForVector();
            }
            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockMxScaleA = GetTile(tensorMxScaleA,
                tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
            auto tensorBlockUbFirst = GetTile(tensorUbFirst, tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockUbSecond = GetTile(tensorUbSecond, tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));

            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = GetTile(tensorB,
                    tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                    tla::MakeShape(actualShape.k(), actualShape.n()));
                auto tensorBlockMxScaleB = GetTile(tensorMxScaleB,
                    tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                    tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
                gmm1Params.blockMmad(tensorBlockA, tensorBlockB,
                                     weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
                                     actualShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            }
            NotifyVector();
        }

        vecSetSyncCom = 1;

        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx == 0) {
                Std::tuple <int64_t, int64_t, int64_t, int64_t> epilogueShape{actualShape.m(), actualShape.n(), 0, 0};
                Std::tuple <int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                        mLoc * outputN + nLoc,
                        mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                        CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                        0, 0, 0, 0
                };
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                }
                WaitForCube();
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_TIMER_BLOCK(
                        timer.template Tok<Overwrite>(
                            AscendTimer::AIV_SWIGLU_WAIT, expertIdx));
                    MEGA_MOE_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_SWIGLU, expertIdx));
                }
                AscendC::SetCtrlSpr<60, 60>(0);
                epilogueOp(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (startLoopIdx < coreLoops) {
            MEGA_MOE_TIMER_BLOCK(
                timer.template Tok<Overwrite>(
                    AscendTimer::AIC_GMM1, expertIdx));
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx == 0 && startLoopIdx < coreLoops) {
            MEGA_MOE_TIMER_BLOCK(
                timer.template Tok<Overwrite>(
                    AscendTimer::AIV_SWIGLU, expertIdx));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}

// GroupMatmul2 - GMM2矩阵乘法 + Combine
template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementMxScaleA, typename ElementMxScaleB>
__aicore__ inline void GroupMatmul2(
    GroupMatmulParam<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, 1>& gmm2Params,
    const Params& params,
    const AscendC::Shape<int64_t, int64_t, int64_t, int64_t>& problemShape,
    const GMMAddrInfo& gmmAddrInfo,
    uint32_t& startBlockIdx,
    int32_t& vecSetSyncCom2,
    uint32_t groupIdx, uint16_t& pingpongIdx, int32_t rankId
    MEGA_MOE_TIMER_PARAM(timer))
{
    constexpr uint32_t GMM2_BLOCK_SIZE = L1_TILE_M * L1_TILE_N;

    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = gmm2Params.n;
    uint32_t k = gmm2Params.k;
    uint32_t blockNum = gmm2Params.blockNum;
    uint32_t subBlockIdx = gmm2Params.subBlockIdx;
    uint32_t blockIdx = gmm2Params.blockIdx;

    GlobalTensor<int32_t> groupFlagListGm;
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)gmmAddrInfo.groupFlagList);

    auto scaleK = gmm2Params.scaleK;
    auto LayoutA = tla::MakeLayout<ElementA, layout::RowMajor>(m, k);
    auto LayoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, layout::RowMajor, false>(m, scaleK);

    GlobalTensor<ElementA> aGlobalTensor;
    GlobalTensor<ElementB> bGlobalTensor;
    GlobalTensor<ElementMxScaleA> aScaleGlobalTensor;
    GlobalTensor<ElementMxScaleB> bScaleGlobalTensor;

    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(gmmAddrInfo.aGlobal));
    bGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(gmmAddrInfo.bGlobal));
    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleA*>(gmmAddrInfo.aScaleGlobal));
    bScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMxScaleB*>(gmmAddrInfo.bScaleGlobal));

    // Matrix A or Matrix B does not have duplicate data reads. Setting L2 Cache to Disable,
    // data reads will bypass L2 Cache.
    if (CeilDiv(m, L1_TILE_M) == 1) {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        bGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }
    if (CeilDiv(n, L1_TILE_N) == 1) {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    } else {
        aGlobalTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    }

    auto tensorA = tla::MakeTensor(aGlobalTensor, LayoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobalTensor, gmm2Params.layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobalTensor, LayoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobalTensor, gmm2Params.layoutMxScaleB, Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(gmm2Params.l0cOutUbFirst, gmm2Params.layoutC, Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(gmm2Params.l0cOutUbSecond, gmm2Params.layoutC, Arch::PositionUB{});

    BlockScheduler scheduler({m, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
    uint32_t coreLoops = scheduler.GetCoreLoops();
    uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        GemmCoord blockCoord = scheduler.GetBlockCoord(loopIdx);
        GemmCoord actualShape = scheduler.GetActualBlockShape(blockCoord);

        uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        uint32_t kLoc = blockCoord.k() * L1_TILE_K;

        auto tensorUb = pingpongIdx == 0 ? tensorUbFirst : tensorUbSecond;
        auto l0cOutUbGMM2 = pingpongIdx == 0 ? gmm2Params.l0cOutUbFirst : gmm2Params.l0cOutUbSecond;

        if constexpr(g_coreType == AscendC::AIC) {
            if (loopIdx == startLoopIdx) {
                MEGA_MOE_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM2_WAIT, groupIdx));
                BlockScheduler gmmBlockScheduler({m, k, n}, MakeCoord(L1_TILE_M, L1_TILE_N));
                uint32_t targetLoops = gmmBlockScheduler.GetCoreLoops();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                while(targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {}
                }
                MEGA_MOE_TIMER_BLOCK(
                    timer.template Tok<Overwrite>(
                        AscendTimer::AIC_GMM2_WAIT, groupIdx));
                MEGA_MOE_TIMER_BLOCK(
                    timer.Tik(AscendTimer::AIC_GMM2, groupIdx));
            }
            if (vecSetSyncCom2 >= 2) {
                WaitForVector(pingpongIdx + FLAG_ID_MAX);
            }

            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc),
                tla::MakeShape(actualShape.m(), actualShape.k()));
            auto tensorBlockMxScaleA = GetTile(tensorMxScaleA,
                tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                tla::MakeShape(actualShape.m(), CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM)));
            auto tensorBlockUb = GetTile(tensorUb, tla::MakeCoord(0, 0),
                tla::MakeShape(actualShape.m(), actualShape.n()));
            auto tensorBlockB = GetTile(tensorB,
                tla::MakeCoord(kLoc, nLoc),
                tla::MakeShape(actualShape.k(), actualShape.n()));
            auto tensorBlockMxScaleB = GetTile(tensorMxScaleB,
                tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                tla::MakeShape(CeilDiv(actualShape.k(), MXFP_SCALE_GROUP_NUM), actualShape.n()));
            gmm2Params.blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb,
                                 actualShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            NotifyVector(pingpongIdx + FLAG_ID_MAX);
        }

        vecSetSyncCom2++;

        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx == 1) {
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_COMBINE_WAIT, groupIdx));
                }
                WaitForCube(pingpongIdx);
                if (loopIdx == startLoopIdx) {
                    MEGA_MOE_TIMER_BLOCK(
                        timer.template Tok<Overwrite>(
                            AscendTimer::AIV_COMBINE_WAIT, groupIdx));
                    MEGA_MOE_TIMER_BLOCK(
                        timer.Tik(AscendTimer::AIV_COMBINE, groupIdx));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                MegaMoeCombineImpl::CombineTokens<ElementC, GemmCoord>(mLoc, nLoc, n, groupIdx, rankId, l0cOutUbGMM2,
                                                                       actualShape, params);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                NotifyCube(pingpongIdx);
            }
        }
        pingpongIdx = 1 - pingpongIdx;
    }
    if constexpr (g_coreType == AscendC::AIC) {
        if (startLoopIdx < coreLoops) {
            MEGA_MOE_TIMER_BLOCK(
                timer.template Tok<Overwrite>(
                    AscendTimer::AIC_GMM2, groupIdx));
        }
    }
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx == 1 && startLoopIdx < coreLoops) {
            MEGA_MOE_TIMER_BLOCK(
                timer.template Tok<Overwrite>(
                    AscendTimer::AIV_COMBINE, groupIdx));
        }
    }
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}

}  // namespace MegaMoeCatlassImpl

#endif
