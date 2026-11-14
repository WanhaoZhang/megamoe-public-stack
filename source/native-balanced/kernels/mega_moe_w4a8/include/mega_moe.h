/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_w4a8.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"

namespace AscendC {

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

}

#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "tla/layout.hpp"
#include "mega_moe_grouped_matmul/mega_moe_matmul_kernel.h"
#include "../../common/timer/AscendTimerV2_device.hpp"
#include "shmem.h"

using namespace AscendC;

namespace MegaMoeW4A8Impl {
using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= 3) && (QuantMode != 4)),
        fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= 3), fp8_e8m0_t, float>::type;
    using WeightType = float4_e2m1x2_t;
    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeW4A8TilingData* tilingData, GM_ADDR timerAddr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit(TPipe *tPipe);
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void ComputeAfterSum();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void DispatchTokens(PeermemInfo &peermemInfo);
    __aicore__ inline bool UpdateGMMParams(MegaMoeFFNMxw4a8::GroupedMatmulParams& groupedMatmulParams, uint32_t expertIdx, uint32_t gmmIdx);
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline uint32_t getTokensNumInCurrentExpert(uint32_t expertIdx);
    __aicore__ inline void UnpermuteV2();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerExpert(GlobalTensor<QuantOutType> dst, GlobalTensor<QuantOutType> src, uint32_t dataLen, uint32_t remoteRankId, int32_t idx);

    __aicore__ inline void RemoteAtomicAdd(GlobalTensor<int32_t> dst, int32_t value, uint32_t remoteRankId, int32_t idx){
        if(remoteRankId == rankId_){
            // 本卡搬运走 SDMA（无法原子加），所以单独用一个 local flag 槽
            // (idx*2+1) 置1，供 GMM1 的 waitFromDispatchLocal 轮询(目标值固定1)。
            // 远端计数的 groupFlagList1 槽 (idx*2) 由各远端卡的 udma_atomic_add 累加，
            // 本卡不往该槽加。即使 rowsCopyCnt<=0（本卡对本expert无token）也必须置
            // local flag，否则 GMM1 会一直等该 flag==1 而卡死。
            auto ub_buf = SdmaWorkSpaceBuf_.template Get<uint8_t>();
            __ubuf__ uint8_t* ub_ptr = reinterpret_cast<__ubuf__ uint8_t*>(ub_buf.GetPhyAddr());
            __gm__ uint8_t* onesBase = reinterpret_cast<__gm__ uint8_t*>(params_.peermemInfo.ptrAllOnes);
            __gm__ int32_t* localFlagSlot = const_cast<__gm__ int32_t*>(gmFlagRemote[(idx * 2 + 1) * INT_CACHELINE].GetPhyAddr());
            aclshmemx_sdma_get_nbi(reinterpret_cast<__gm__ uint8_t*>(localFlagSlot),
                                   onesBase, ub_ptr, SDMA_UB_SIZE, SIZE_INT_32, rankId_, EVENT_ID0);
        }else{
            // The remote flag is the consumer-visible completion point for
            // the preceding SO payload put on the same peer/QP.
            aclshmemx_udma_atomic_add<
                int32_t, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
                (__gm__ int32_t*)dst.GetPhyAddr(), value, remoteRankId);
        }
    }

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,
        int32_t& coreOffset, int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2; // 取到vec核总数
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
        return dim0 * syncBufAlign + dim1 * expertPerRank_ + dim2;
    }

    __aicore__ inline void GmSignalWait(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

    TBuf<> CrossRankSyncBuf_;
    TBuf<> BeforeSumBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> AfterSumBuf_;
    TBuf<> SdmaWorkSpaceBuf_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> ptrBeforeRow_;
    GlobalTensor<int32_t> ptrAfterRow_;
    GlobalTensor<int32_t> ptrAfterRowLast_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm1_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<uint8_t> weightGamma_;
    LocalTensor<float> topKWeightsTensor_;
    GlobalTensor <int32_t> gmFlagRemote;
    AscendTimerDevice timer_;
    AscendTimerDevice timerAll_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertNums_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;

    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();

    uint32_t hiddenAndScaleSize = 0;
    uint32_t scaleK1 = 0;
    uint32_t syncBufAlign = 0;
    uint32_t syncBufAlignBytes = 0;

};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
    MegaMoeW4A8TilingData* tilingData, GM_ADDR timerAddr)
{

    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    expertNums_ = worldSize_ * expertPerRank_;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    Get<N_VALUE>(problemShape_) = tilingData->n;
    Get<K_VALUE>(problemShape_) = k_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    scaleK1 = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    hiddenAndScaleSize = k_ + scaleK1;
    syncBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(expertNums_), static_cast<uint32_t>(ALIGN_128));
    syncBufAlignBytes = syncBufAlign * SIZE_INT_32;

    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.bScaleGmAddr =  weightScales1;
    params_.b2ScaleGmAddr = weightScales2;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;

    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrBeforeColumn);
    cumsumInWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrAfterColumn);
    ptrBeforeRow_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrBeforeRow);
    ptrAfterRow_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrAfterRow);
    ptrAfterRowLast_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrAfterRowLast);
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    weightGamma_.SetGlobalBuffer((__gm__ uint8_t*)params_.workspaceInfo.ptrWeightGamma);
    gmFlagRemote.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrCommFlag));

    tokenPerExpertCurRankAddr_ = params_.peermemInfo.ptrTokenPerExpert +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);

    TIMER_BLOCK(timer_.Init(timerAddr));
    TIMER_BLOCK(timerAll_.Init(timerAddr));
    TIMER_BLOCK(timerAll_.Tik());
    TIMER_BLOCK(timer_.Tik());
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIV_DISPATCH, 1));
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIV_SWIGLU, expertPerRank_));
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIV_COMBINE, expertPerRank_));
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIC_GMM1, expertPerRank_));
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIC_GMM2, expertPerRank_));
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIV_GMM1_PROLOGUE, expertPerRank_));
    TIMER_BLOCK(timer_.setDynamicActualIter(
        AscendTimer::AIV_GMM2_PROLOGUE, expertPerRank_));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit(TPipe *tPipe)
{
    // 对CrossRankSync、Dispatch、GetCumSum过程涉及到的LocalTensor初始化管理
    tPipe->InitBuffer(CrossRankSyncBuf_, syncBufAlignBytes * 3);
    tPipe->InitBuffer(AfterSumBuf_, syncBufAlignBytes * worldSize_);
    tPipe->InitBuffer(CopyTempBuf_, BUFFER_ALIGN);
    tPipe->InitBuffer(SdmaWorkSpaceBuf_, SDMA_UB_SIZE);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetWorkSpaceFlagList()
{
    // workSpace groupFlagList清零
    if constexpr(g_coreType == AIV) {
        if (aivCoreIdx_ == 0) {
            groupFlagListGm1_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrCommFlag);
            int32_t flagNum = expertPerRank_ * 2 * INT_CACHELINE;
            LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
            Duplicate(initTensor, 0, flagNum);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(groupFlagListGm1_, initTensor, flagNum);
            // 初始化SDMA本地flag置1源区（全1），确保CopyGMToGMPerExpert本地分支能用
            GlobalTensor<int32_t> allOnesGm;
            allOnesGm.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrAllOnes);
            int32_t onesNum = INT_CACHELINE;
            Duplicate(initTensor, 1, onesNum);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(allOnesGm, initTensor, onesNum);
        }
        if (aivCoreIdx_ == 1) {
            groupFlagListGm2_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
            int32_t flagNum = expertPerRank_ * INT_CACHELINE;
            LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
            Duplicate(initTensor, 0, flagNum);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(groupFlagListGm2_, initTensor, flagNum);
            SyncFunc<AscendC::HardEvent::MTE3_V>();

            GlobalTensor<int32_t> gmm1ReadyFlag, gmm2ReadyFlag;
            gmm1ReadyFlag.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrGmm1ReadyFlag);
            gmm2ReadyFlag.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrGmm2ReadyFlag);
            Duplicate(initTensor, 0, flagNum);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(gmm1ReadyFlag, initTensor, flagNum);
            DataCopy(gmm2ReadyFlag, initTensor, flagNum);
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>:: CopyGMToGMPerExpert(GlobalTensor<QuantOutType> dst, GlobalTensor<QuantOutType> src, uint32_t dataLen, uint32_t remoteRankId, int32_t idx){
        uint64_t dataSize = static_cast<uint64_t>(dataLen) * sizeof(QuantOutType);
        //本地搬运
        if (remoteRankId == rankId_){

            auto ub_buf = SdmaWorkSpaceBuf_.template Get<uint8_t>();
            __ubuf__ uint8_t* ub_ptr = reinterpret_cast<__ubuf__ uint8_t*>(ub_buf.GetPhyAddr());
            aclshmemx_sdma_get_nbi((__gm__ uint8_t* )dst.GetPhyAddr(), (__gm__ uint8_t* )src.GetPhyAddr(),
                                   ub_ptr, SDMA_UB_SIZE, static_cast<uint32_t>(dataSize), rankId_, EVENT_ID0);
        }else{
            // SHMEM defaults to NO order. Use SO so the following SO atomic
            // completion flag cannot become visible before this payload.
            aclshmemx_udma_put_nbi<
                uint8_t, PIPE_S, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
                (__gm__ uint8_t* )dst.GetPhyAddr(),
                (__gm__ uint8_t* )src.GetPhyAddr(),
                (__ubuf__ uint8_t* )nullptr,
                static_cast<uint32_t>(dataSize), remoteRankId);
        }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ComputeAfterSum()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    int32_t expertNumsAligned = (expertNums_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> afterSumBuf = AfterSumBuf_.Get<int32_t>();

    //ptrAfterRow_
    if (aivCoreIdx_ == 0) {
        Duplicate(afterSumBuf, 0, expertNumsAligned);
        DataCopyPad(afterSumBuf[expertNumsAligned], tokenPerExpertWin_,
            {static_cast<uint16_t>(worldSize_),
            static_cast<uint16_t>(expertNums_ * sizeof(int32_t)),
            static_cast<uint16_t>(syncBufAlignBytes - expertNums_ * sizeof(int32_t)), 0},{});
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        for (uint32_t i = 2; i <= worldSize_; ++i) {
            Add(afterSumBuf[i * expertNumsAligned], afterSumBuf[i * expertNumsAligned], afterSumBuf[(i - 1) * expertNumsAligned], expertNums_);
            PipeBarrier<PIPE_V>();
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(ptrAfterRow_, afterSumBuf[expertNumsAligned * rankId_],
                {1, static_cast<uint16_t>(expertNums_ * sizeof(int32_t)), 0, 0});
        DataCopyPad(ptrAfterRowLast_, afterSumBuf[expertNumsAligned * (worldSize_)],
                {1, static_cast<uint16_t>(expertNums_ * sizeof(int32_t)), 0, 0});
    }

    //ptrAftercloumn_
    if (aivCoreIdx_ == 1) {
        DataCopyPad(afterSumBuf, tokenPerExpertWin_[expertPerRank_ * rankId_],
                {static_cast<uint16_t>(worldSize_),
                static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
                static_cast<uint16_t>(syncBufAlignBytes - expertPerRank_ * sizeof(int32_t)), 0},
                {});

        SyncFunc<AscendC::HardEvent::MTE2_V>();

        for (uint32_t i = 1; i < worldSize_; ++i) {
            Add(afterSumBuf[i * expertPerRankAligned], afterSumBuf[i * expertPerRankAligned], afterSumBuf[(i - 1) * expertPerRankAligned], expertPerRank_);
            PipeBarrier<PIPE_V>();
        }

        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(cumsumInWorkSpace_, afterSumBuf,
                {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer = CrossRankSyncBuf_.Get<int32_t>();
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(expertTokenNumsOut_, tmpCopyBuffer, copyNum);
}


template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncCntPerExpert()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t numPerCore = syncBufAlign;
    LocalTensor<int32_t> tmpBuffer = CrossRankSyncBuf_.Get<int32_t>();
    LocalTensor<int32_t> beforeSumRowBuf = tmpBuffer[syncBufAlign];
    LocalTensor<int32_t> beforeSumColumnBuf = beforeSumRowBuf[syncBufAlign];

    // 给对端发送
    uint64_t offsetInDstWinAddr = syncBufAlignBytes * rankId_ + PEERMEM_DATA_OFFSET;
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockNum_) {
        if (dstRankIdx == rankId_) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr_));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr);
        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(tmpBuffer, srcAddress, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(dstAddress, tmpBuffer, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    // 本端接收
    int32_t intPer512 = ALIGN_512 / sizeof(int32_t);
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockNum_) {
        if (dstRankIdx != rankId_) {
            for (int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* syncCheck = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrTokenPerExpert)
                                                                             + TokenPerExpertLayout(dstRankIdx, 0, checkIdx);
                GmSignalWait(syncCheck, 0);
            }
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], tmpBuffer, numPerCore);
        } else {
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_S>();

            int perSum = 0;
            for (int32_t i = 0; i <  expertNums_; i++) {
                beforeSumRowBuf(i) = perSum;
                perSum += tmpBuffer(i);
            }
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopy(ptrBeforeRow_, beforeSumRowBuf, numPerCore);
        }

        PipeBarrier<PIPE_ALL>();
        int32_t perSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rankId_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                beforeSumColumnBuf(j) = perSum;
                j++;
            }
            perSum += tmpBuffer(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(preSumBeforeRankWorkSpace_[dstRankIdx * expertPerRank_], beforeSumColumnBuf,
                    DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }

    // PipeBarrier<PIPE_MTE3>();
    SyncAll<true>();
    ComputeAfterSum();
    SyncAll<true>();
}

//todo worldsize 超出aicore数量时需修改
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchTokens(PeermemInfo &peermemInfo)
{
    if constexpr(g_coreType == AIC) return;
    if( subBlockIdx_ > 0 ) return;

    uint32_t widthA = k_; // H 长度
    uint32_t widthAScale = scaleK1;
    for (int dstRankIdx = blockIdx_; dstRankIdx < worldSize_; dstRankIdx += blockNum_) {
        // statements
        //单个核负责向单张卡的远端写
        prevSum_ = ptrBeforeRow_(dstRankIdx * expertPerRank_);
        uint32_t preOffset = 0;

        for (uint32_t idx = 0; idx < expertPerRank_; idx ++) {
            uint32_t rowStartInDst = ptrAfterRow_(dstRankIdx * expertPerRank_ + idx) + preOffset;
            preOffset += ptrAfterRowLast_(dstRankIdx * expertPerRank_ + idx);
            if (rowStartInDst < maxOutputSize_) {
                int32_t rowsCopyCnt = tokenPerExpertWin_(TokenPerExpertLayout(rankId_, dstRankIdx, idx));

                if (rowsCopyCnt <= 0) {
                    RemoteAtomicAdd(gmFlagRemote[idx * 2 * INT_CACHELINE], int32_t(1), dstRankIdx, idx);
                    continue;
                }
                if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                    rowsCopyCnt = maxOutputSize_ - rowStartInDst;
                }
                int32_t rowStartInSrc = prevSum_;
                prevSum_ += rowsCopyCnt;
                int32_t rowSrcOffset = rowStartInSrc * (widthA + widthAScale);
                int32_t rowDstOffset = rowStartInDst * (widthA + widthAScale);
                int64_t dataLen = rowsCopyCnt * (widthA + widthAScale);
                //远端写data
                GlobalTensor <QuantOutType> gmLocalSrc, gmRemoteDst;
                gmLocalSrc.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(peermemInfo.ptrA0));
                gmRemoteDst.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(peermemInfo.ptrA1));
                CopyGMToGMPerExpert(gmRemoteDst[rowDstOffset], gmLocalSrc[rowSrcOffset], dataLen, dstRankIdx, idx);
                //远端写flag
                RemoteAtomicAdd(gmFlagRemote[idx * 2 * INT_CACHELINE], int32_t(1), dstRankIdx, idx);
            }
        }
    }
}

//当前专家的token数量
template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::getTokensNumInCurrentExpert(uint32_t expertIdx){

    uint32_t currentM = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    if (Get<IDX_M_OFFSET>(baseOffset_) + currentM > maxOutputSize_) {
        currentM = maxOutputSize_ - Get<IDX_M_OFFSET>(baseOffset_);
    }
    Get<IDX_M_OFFSET>(baseOffset_) += currentM;

    return currentM;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGMMParams(MegaMoeFFNMxw4a8::GroupedMatmulParams& groupedMatmulParams, uint32_t expertIdx, uint32_t gmmIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    Get<M_VALUE>(problemShape_) = getTokensNumInCurrentExpert(expertIdx);
    if (Get<M_VALUE>(problemShape_) <= 0) {
        return false;
    }

    uint64_t m = Get<M_VALUE>(problemShape_);
    uint64_t n = Get<N_VALUE>(problemShape_);
    uint64_t k = Get<K_VALUE>(problemShape_);
    groupedMatmulParams.matmulShape = problemShape_;
    groupedMatmulParams.expertIdx = expertIdx;

    if(gmmIdx == 0){
        groupedMatmulParams.ptrA = params_.peermemInfo.ptrA1 + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        groupedMatmulParams.ptrMxScaleA = params_.peermemInfo.ptrA1Scale + Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        groupedMatmulParams.ptrB = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) / 2;
        groupedMatmulParams.ptrMxScaleB = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        groupedMatmulParams.groupFlagList1 = params_.peermemInfo.ptrCommFlag + Get<IDX_FLAG_OFFSET>(baseOffset_) * 2 * INT_CACHELINE * SIZE_INT_32;
        groupedMatmulParams.groupFlagListLocal = params_.peermemInfo.ptrCommFlag + (Get<IDX_FLAG_OFFSET>(baseOffset_) * 2 + 1) * INT_CACHELINE * SIZE_INT_32;
        // groupedMatmulParams.groupFlagList1 = params_.peermemInfo.ptrCommFlag + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * SIZE_INT_32;
        groupedMatmulParams.groupFlagList2 = params_.workspaceInfo.ptrFlagSwiGluToGmm2 + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * SIZE_INT_32;
        groupedMatmulParams.gmm1ReadyFlag = params_.workspaceInfo.ptrGmm1ReadyFlag +
                                             Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * SIZE_INT_32;

        groupedMatmulParams.ptrCData = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        groupedMatmulParams.ptrCScale = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        groupedMatmulParams.ptrGmm1Output = params_.workspaceInfo.ptrGmm1Output +
                                            Get<IDX_C_OFFSET>(baseOffset_) * SWIGLU_N_HALF * sizeof(bfloat16_t);

        Get<IDX_A_OFFSET>(baseOffset_) += m * hiddenAndScaleSize;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * hiddenAndScaleSize;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK1;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) += m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    }else{

        groupedMatmulParams.ptrA = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        groupedMatmulParams.ptrMxScaleA = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        groupedMatmulParams.ptrB = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) / 2;
        groupedMatmulParams.ptrMxScaleB = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        groupedMatmulParams.groupFlagList2 = params_.workspaceInfo.ptrFlagSwiGluToGmm2 + Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * SIZE_INT_32;
        groupedMatmulParams.gmm2ReadyFlag = params_.workspaceInfo.ptrGmm2ReadyFlag +
                                             Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE * SIZE_INT_32;
        groupedMatmulParams.ptrCData = params_.workspaceInfo.ptrGmm2Output +
                                       Get<IDX_Y2_OFFSET>(baseOffset_) * sizeof(bfloat16_t);

        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) += m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
                k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;

        groupedMatmulParams.rankId = rankId_;
        groupedMatmulParams.groupIdx = expertIdx;
        groupedMatmulParams.startBlockIdx = 0;
    }

    Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
            static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp{AscendC::TPosition::VECIN, 0, 0};
    Duplicate(tmp, 0, num);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteV2() {
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    AscendC::GlobalTensor<int32_t> expertIdxTensor;
    expertIdxTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    AscendC::GlobalTensor<int32_t> expandedRowIdxGather;
    expandedRowIdxGather.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    AscendC::GlobalTensor<bfloat16_t> probs;
    probs.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
    AscendC::GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.ptrD);
    AscendC::GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::LocalTensor<bfloat16_t> dataRes{AscendC::TPosition::VECIN, 0, 0};
        AscendC::LocalTensor<bfloat16_t> dataIn0 = dataRes[k_];
        AscendC::LocalTensor<bfloat16_t> dataIn1 = dataRes[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            int32_t expIdx = expertIdxTensor(tokenIdx * topK_ + expId);
            int32_t epIdx = expIdx / expertPerRank_;
            int32_t rowIdx = expandedRowIdxGather(tokenIdx * topK_ + expId);
            bfloat16_t expScale = probs(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataIn = (expId % 2 == 0) ? dataIn0 : dataIn1;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopy(dataIn, expandedX[rowIdx * k_], k_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
            if (expId == 0) {
                AscendC::Muls(dataRes, dataIn, expScale, k_);
            } else {
                AscendC::Muls(dataIn, dataIn, expScale, k_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dataRes, dataRes, dataIn, k_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(output[tokenIdx * k_], dataRes, k_);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

//全卡同步
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.ptrBase + 48 * 1024);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer_.template Tok<Overwrite>(
            AscendTimer::AIV_INIT_PARAM));
    }

    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    // 1. moe_init_routing_v3 + 量化

    moe_init_routing_v3<bfloat16_t, QuantOutType>(
        params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
        params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx,
        tokenPerExpertCurRankAddr_, nullptr,
        params_.workspaceInfo.expandedRowIdxGather,
        params_.workspaceInfo.ptrA2, &params_.tilingData->moeInitRoutingTilingData,
        params_.tilingData->moeInitRoutingTilingData.tilingKey, timer_);

    SyncAll<true>();
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer_.template Tok<Overwrite>(
            AscendTimer::AIV_INITROUTING_GATHEROUT_QUANT));
    }
    // 2. buffer申请 & workspace Init
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);
    // The registered DeepEP operator can be invoked repeatedly. Reset both
    // dispatch->GMM1 and SwiGLU->GMM2 completion counters for every launch;
    // their consumers use strict-equality waits.
    ResetWorkSpaceFlagList();
    SyncAll<true>();
    // 3. cnt值全ep域内的发送以及本端接收整理
    TIMER_BLOCK(timer_.Tik());
    CrossRankSyncCntPerExpert();
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_ALLGATHER));
        if (subBlockIdx_ == 0) {
            TIMER_BLOCK(timer_.Tik(AscendTimer::AIV_DISPATCH, 0));
        }
    }

    DispatchTokens(params_.peermemInfo);
    // if constexpr (g_coreType == AscendC::AIV) {
    //     if (subBlockIdx_ == 1) {
    //         TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_DISPATCH, 0));
    //     }
    // }

    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 0) {
            TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_DISPATCH, 0));
        }
    }
    firstPipe.Destroy();

    SyncAll<false>();





    //5. GMM1
    MegaMoeFFNMxw4a8::GroupedMatmulParams groupedMatmulParams;
    using GroupedMatmul1 = MegaMoeFFNMxw4a8::GroupMatmul1SwigluQuant<QuantOutType, WeightType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>;
    GroupedMatmul1 groupedMatmul1;
    //expertPerRank_
    groupedMatmul1.start(weightGamma_, worldSize_ - 1, Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF);
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGMMParams(groupedMatmulParams, expertIdx, 0)) {
            continue;
        }
        groupedMatmul1(groupedMatmulParams, timer_);
    }
    groupedMatmul1.end();

    // 5. GMM2
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    using GroupedMatmul2 = MegaMoeFFNMxw4a8::GroupMatmul2Combine<QuantOutType, WeightType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>;
    GroupedMatmul2 groupedMatmul2;
    groupedMatmul2.start(weightGamma_);
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGMMParams(groupedMatmulParams, expertIdx, 1)) {
            continue;
        }
        groupedMatmul2(params_, groupedMatmulParams, timer_);
    }
    groupedMatmul2.end();
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer_.Tik());
    }
    SyncAll<true>();
    if constexpr (g_coreType == AscendC::AIV) {
        TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_WAIT_COMB));
    }

    // 6. unpermute & 清理win区上放置token的位置
    if constexpr(g_coreType == AIV) {
        TIMER_BLOCK(timer_.Tik());
        ResetTokenPerExpert();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        TIMER_BLOCK(timer_.template Tok<Overwrite>(
            AscendTimer::AIV_UNPERMUTE_WAIT));
        TIMER_BLOCK(timer_.Tik());
        UnpermuteV2();
        PipeBarrier<PIPE_ALL>();
        TIMER_BLOCK(timer_.template Tok<Overwrite>(AscendTimer::AIV_UNPERMUTE));
    }

    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    TIMER_BLOCK(timerAll_.template Tok<Overwrite>(
        AscendTimer::KERNEL_TIMING_IDX));
}

}   // namespace MegaMoeW4A8Impl
#endif
