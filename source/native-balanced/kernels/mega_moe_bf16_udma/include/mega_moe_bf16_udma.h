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
 * \file mega_moe_bf16_udma.h
 * \brief
 */

#ifndef MEGA_MOE_BF16_UDMA_H
#define MEGA_MOE_BF16_UDMA_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "mega_moe_combine_send.h"
#include "tla/layout.hpp"
#include "shmem.h"
#include "../../mega_moe_bf16/include/block_epilogue_swiglu_bf16.h"
#include "mega_moe_bf16_catlass_impl.h"
#include "../../mega_moe_bf16/include/mega_moe_bf16_timer.h"
#include "../../mega_moe_bf16/include/mega_moe_forward_save.h"
#include "../../mega_moe_bf16/include/mega_moe_training_workspace.h"
#include "../../mega_moe_bf16/include/mega_moe_topk_weight_gather.h"
#include "../../mega_moe_bf16/include/mega_moe_unpermute.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;
constexpr uint32_t TOPK_WEIGHT_COPY_BUFFER_BYTES = 4U * 1024U;
constexpr uint32_t TOPK_WEIGHT_COPY_ELEMENTS =
    TOPK_WEIGHT_COPY_BUFFER_BYTES / sizeof(bfloat16_t);

class MegaMoeBF16UDMA {
public:
    using QuantOutType = bfloat16_t;
    using QuantScaleOutType = float;
    __aicore__ inline MegaMoeBF16UDMA() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
        GM_ADDR trainingSavedStateGM, GM_ADDR forwardTransientWorkspaceGM,
        MegaMoeTilingData *tilingData, GM_ADDR timerAddr = nullptr);
    __aicore__ inline void Process();
    template<uint32_t SaveMode> __aicore__ inline void ProcessWithSaveMode();

private:
    __aicore__ inline void FirstBuffInit(TPipe *tPipe);
    __aicore__ inline void SecondBuffInit(TPipe *tPipe);
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void SaveForwardState();
    __aicore__ inline void GatherRoutedTopkWeights();
    __aicore__ inline void DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
    __aicore__ inline void UnpermuteV2();
    __aicore__ inline void EndSync();
    __aicore__ inline void EndGMM2Sync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
                int32_t rows, int32_t hiddenSize, int32_t& pingpongId);
    __aicore__ inline void DispatchAllTokensUDMA();
    __aicore__ inline void DispatchTopkWeightsMTE(
        uint32_t expertIdx, uint64_t &expertRowBase);
    __aicore__ inline void SendCombineExpertUDMA(
        uint32_t expertIdx, uint64_t expertRowBase);
    __aicore__ inline void PublishControlSignal(uint32_t dstRank, uint32_t expertIdx);

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
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
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
    TBuf<> PrevSumBuf_;
    TBuf<> TopkWeightCopyBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> CumsumBuf_;
    TBuf<> ResetFlagBuf_;
    TBuf<> DataResBuf_;
    TBuf<> DataResFp32Buf_;
    TBuf<> ResetTokenBuf_;
    TBuf<> topKWeightsBuf_;
    TBuf<> SdmaWorkSpaceBuf_;
    TBuf<> SaveStateBuf_;
    // One AIV1 owns at most ten peers (worldSize_ <= 10 * blockNum_).  Keep
    // the flattened send-layout and receive-layout cursors across experts so
    // DispatchAllTokensUDMA does not repeatedly rescan all prior experts.
    uint64_t dispatchSrcRowCursor_[10] = {0};
    uint64_t dispatchDstExpertBase_[10] = {0};

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GM_ADDR tokenPerExpertCurRankAddr_;
    GM_ADDR trainingSavedStateGM_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    LocalTensor<float> topKWeightsTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t gmm1VecSetSyncCom_ = 0;
    int32_t gmm2VecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    int32_t targetFlag_;
    int32_t prevSum_[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // worldSize_ <= 10 * blockNum_

    uint64_t preOffset_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint32_t dispatchCopyElementsPerPing_ = 0;
    static constexpr uint32_t SDMA_UB_SIZE = 64U;

    using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
    using BlockEpilogue = MegaMoeBF16Impl::BlockEpilogueSwigluBF16<
            MegaMoeBF16CatlassImpl::L1TileShape, true>;
    BlockEpilogue epilogueOp_;//GMMswiglu调用
#if defined(ENABLE_TIMER)
    AscendTimerDevice timer_;
    AscendTimerDevice timerAll_;
#endif
};

__aicore__ inline void MegaMoeBF16UDMA::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR trainingSavedStateGM,
    GM_ADDR forwardTransientWorkspaceGM,
    MegaMoeTilingData *tilingData,
    GM_ADDR timerAddr)
{
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
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
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    // DeepEP passes each local expert's weights in one contiguous tensor.
    // Unlike the original aclTensorList ABI, these are already raw GM addresses.
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.bScaleGmAddr = weightScales1;
    params_.b2ScaleGmAddr = weightScales2;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    trainingSavedStateGM_ = trainingSavedStateGM;
    params_.workspaceInfo = WorkspaceInfo(
        trainingSavedStateGM, forwardTransientWorkspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
        params_.workspaceInfo.ptrFlagSwiGluToGmm2,
        params_.workspaceInfo.ptrDispatchedTopkWeights,
        nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256,
        tilingData->swigluLimit,
        params_.workspaceInfo.ptrSavedGate, params_.workspaceInfo.ptrSavedUp,
        params_.workspaceInfo.ptrSavedGateScale, params_.workspaceInfo.ptrSavedUpScale,
        params_.workspaceInfo.ptrSavedGateHigh, params_.workspaceInfo.ptrSavedUpLow,
        params_.workspaceInfo.ptrSavedUpHigh});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.ptrTokenPerExpert +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
#if defined(ENABLE_TIMER)
    timer_.Init(timerAddr);
    timerAll_.Init(timerAddr);
    timerAll_.Tik();
    timer_.Tik();
    timer_.setDynamicActualIter(AscendTimer::AIV_DISPATCH, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIV_SWIGLU_WAIT, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIV_SWIGLU, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIV_COMBINE_WAIT, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIV_COMBINE, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_GMM1_WAIT, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_GMM1, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_GMM2_WAIT, expertPerRank_);
    timer_.setDynamicActualIter(AscendTimer::AIC_GMM2, expertPerRank_);
#endif
}

__aicore__ inline void MegaMoeBF16UDMA::FirstBuffInit(TPipe *tPipe)
{
    // 对CrossRankSync、Dispatch、GetCumSum过程涉及到的LocalTensor初始化管理
    uint32_t syncBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t prevSumBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128))
        * sizeof(int32_t) * expertPerRank_;
    tPipe->InitBuffer(CrossRankSyncBuf_, syncBufAlign);
    tPipe->InitBuffer(PrevSumBuf_, prevSumBufAlign);
    tPipe->InitBuffer(
        TopkWeightCopyBuf_, TOPK_WEIGHT_COPY_BUFFER_BYTES);
    uint32_t getCumsumBufAlign = (AlignUp(worldSize_ * expertPerRank_, ALIGN_128)) * worldSize_ * sizeof(int32_t);
    tPipe->InitBuffer(CumsumBuf_, getCumsumBufAlign);
    tPipe->InitBuffer(SdmaWorkSpaceBuf_, SDMA_UB_SIZE);
    uint32_t maxResetBufAlign = Ops::Base::CeilDiv(
        static_cast<int32_t>(expertPerRank_ * INT_CACHELINE * sizeof(int32_t) * 2), static_cast<int32_t>(ALIGN_32));
    tPipe->InitBuffer(ResetFlagBuf_, maxResetBufAlign);
}

__aicore__ inline void MegaMoeBF16UDMA::SecondBuffInit(TPipe *tPipe)
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(DataResBuf_, dataResBufAlign);
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    tPipe->InitBuffer(ResetTokenBuf_, num);
    tPipe->InitBuffer(DataResFp32Buf_, dataResBufAlign * 2);
    tPipe->InitBuffer(topKWeightsBuf_, Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32)));
    topKWeightsTensor_ = topKWeightsBuf_.Get<float>();
    TBuf<> tempBuf;
    tPipe->InitBuffer(tempBuf, Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32)));
    LocalTensor<bfloat16_t> tempLocal = tempBuf.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
    topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
}

__aicore__ inline void MegaMoeBF16UDMA::ResetWorkSpaceFlagList()
{
    // workSpace groupFlagList清零
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    if constexpr(g_coreType == AIV) {
        if (aivCoreIdx_ == 0) {
            int32_t flagNum = expertPerRank_ * INT_CACHELINE * 2;
            //LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
            LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
            Duplicate(initTensor, 0, flagNum);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(groupFlagListGm_, initTensor, flagNum);
            __gm__ int32_t *dispatchSignals = reinterpret_cast<__gm__ int32_t *>(
                params_.peermemInfo.ptrDispatchSignal);
            for (uint32_t i = 0; i < worldSize_ * expertPerRank_ * 2; ++i) {
                AscendC::WriteGmByPassDCache(dispatchSignals + i, static_cast<int32_t>(0));
            }
            AscendC::WriteGmByPassDCache(
                reinterpret_cast<__gm__ int32_t *>(params_.peermemInfo.ptrAllOnes64),
                static_cast<int32_t>(1));
            AscendC::WriteGmByPassDCache(
                reinterpret_cast<__gm__ int32_t *>(params_.peermemInfo.ptrAllOnes64) + 1,
                static_cast<int32_t>(0));
        }
        // combineReady is much larger than the legacy expert flags.  Clear
        // contiguous cache-line ranges with all AIVs instead of issuing one
        // scalar bypass-cache store per element from AIV0.  The SyncAll after
        // this function makes every range visible before GMM2 can publish or
        // consume row-group readiness counts.
        const uint32_t readyCount = expertPerRank_ * Ops::Base::CeilDiv(
            static_cast<uint32_t>(maxOutputSize_), static_cast<uint32_t>(128));
        const uint32_t readyCacheLines = Ops::Base::CeilDiv(
            readyCount, static_cast<uint32_t>(INT_CACHELINE));
        const uint32_t baseCacheLinesPerAiv = readyCacheLines / blockAivNum_;
        const uint32_t extraCacheLines = readyCacheLines % blockAivNum_;
        const uint32_t extraBeforeThisAiv = aivCoreIdx_ < extraCacheLines ?
            aivCoreIdx_ : extraCacheLines;
        const uint32_t firstCacheLine =
            aivCoreIdx_ * baseCacheLinesPerAiv + extraBeforeThisAiv;
        const uint32_t thisAivCacheLines = baseCacheLinesPerAiv +
            static_cast<uint32_t>(aivCoreIdx_ < extraCacheLines);
        const uint32_t lastCacheLine = firstCacheLine + thisAivCacheLines;
        if (firstCacheLine < lastCacheLine) {
            constexpr uint32_t RESET_CHUNK_ELEMENTS = 512;
            LocalTensor<int32_t> initTensor{TPosition::VECIN, 0, 0};
            Duplicate(initTensor, 0, RESET_CHUNK_ELEMENTS);
            SyncFunc<AscendC::HardEvent::V_MTE3>();

            GlobalTensor<int32_t> combineReady;
            combineReady.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                params_.workspaceInfo.ptrCombineReady));
            uint32_t elementOffset = firstCacheLine * INT_CACHELINE;
            uint32_t remainingElements =
                (lastCacheLine - firstCacheLine) * INT_CACHELINE;
            while (remainingElements > 0) {
                const uint32_t currentElements = remainingElements < RESET_CHUNK_ELEMENTS ?
                    remainingElements : RESET_CHUNK_ELEMENTS;
                DataCopy(combineReady[elementOffset], initTensor, currentElements);
                elementOffset += currentElements;
                remainingElements -= currentElements;
            }
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
    }
}

__aicore__ inline void MegaMoeBF16UDMA::GetCumsumForMMAIV()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpBuffer1 = CumsumBuf_.Get<int32_t>();
    DataCopyPad(tmpBuffer1, tokenPerExpertWin_[rankId_ * expertPerRank_],
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
         static_cast<uint16_t>((AlignUp(worldSize_ * expertPerRank_, ALIGN_128)
             - expertPerRank_) * sizeof(int32_t)), 0},
        {});
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    for (uint32_t i = 1; i < worldSize_; ++i) {
        Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
            tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(cumsumInWorkSpace_, tmpBuffer1,
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>((expertPerRank_) * sizeof(int32_t)), 0, 0});
}

__aicore__ inline void MegaMoeBF16UDMA::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer = CrossRankSyncBuf_.Get<int32_t>();
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(
        expertTokenNumsOut_,
        tmpCopyBuffer,
        DataCopyParams{
            1,
            static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
            0,
            0});
}

__aicore__ inline void MegaMoeBF16UDMA::SaveForwardState()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1) {
        return;
    }

    uint64_t actualRows = 0;
    if (groupListType_ == 0) {
        actualRows = static_cast<uint64_t>(
            groupList_.GetValue(expertPerRank_ - 1));
    } else {
        for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
            actualRows += static_cast<uint64_t>(groupList_.GetValue(expertIdx));
        }
    }
    if (actualRows > maxOutputSize_) {
        actualRows = maxOutputSize_;
    }
    const MegaMoeBF16Training::TrainingSavedStateLayout savedLayout(
        m_, k_, topK_, worldSize_, expertPerRank_, maxOutputSize_, params_.tilingData->n, MegaMoeBF16Training::SaveGateUpAsMxFp8(params_.tilingData->swigluLimit, params_.tilingData->savedGateUpMode), params_.tilingData->savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(params_.tilingData->swigluLimit, params_.tilingData->savedGateUpMode));
    auto copyBuffer = SaveStateBuf_.Get<uint8_t>();
    // UDMA receives packed X in symmetric ptrA1. Snapshot its valid prefix
    // before Combine reuses that storage; routing metadata is already saved.
    MegaMoeForwardSave::CopyContiguous(
        trainingSavedStateGM_ + savedLayout.ptrAOffset,
        params_.peermemInfo.ptrA1,
        actualRows * k_ * sizeof(bfloat16_t),
        blockIdx_, blockNum_, copyBuffer);

    const uint64_t perRankStride = Ops::Base::CeilAlign(
        static_cast<uint64_t>(worldSize_) * expertPerRank_,
        static_cast<uint64_t>(MegaMoeMeta::kTokenCountStrideAlignment));
    const uint64_t countsBytes = worldSize_ * perRankStride * sizeof(int32_t);
    // The token-count table is mutable SHMEM and still needs a snapshot.
    MegaMoeForwardSave::CopyContiguous(
        trainingSavedStateGM_ + savedLayout.savedTokenCountsOffset,
        params_.peermemInfo.ptrTokenPerExpert,
        countsBytes,
        blockIdx_, blockNum_, copyBuffer);
}

__aicore__ inline void MegaMoeBF16UDMA::GatherRoutedTopkWeights()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    const int64_t totalRoutes = static_cast<int64_t>(m_) * topK_;
    const int64_t routesPerCore = Ops::Base::CeilDiv(
        totalRoutes, static_cast<int64_t>(blockAivNum_));
    const int64_t indexBase = static_cast<int64_t>(aivCoreIdx_) * routesPerCore;
    const int64_t elements =
        indexBase >= totalRoutes
            ? 0
            : indexBase + routesPerCore > totalRoutes
            ? totalRoutes - indexBase
            : routesPerCore;
    if (elements == 0) {
        return;
    }

    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.expandedRowIdx));
    GlobalTensor<bfloat16_t> topkWeights;
    topkWeights.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(params_.probsGmAddr));
    GlobalTensor<bfloat16_t> routedTopkWeights;
    routedTopkWeights.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(
            params_.peermemInfo.ptrRoutedTopkWeights));
    LocalTensor<bfloat16_t> buffer =
        TopkWeightCopyBuf_.Get<bfloat16_t>();

    const int64_t processCount = Ops::Base::CeilDiv(
        elements, static_cast<int64_t>(TOPK_WEIGHT_COPY_ELEMENTS));
    for (int64_t processIndex = 0;
         processIndex < processCount;
         ++processIndex) {
        const int64_t chunkOffset =
            processIndex * TOPK_WEIGHT_COPY_ELEMENTS;
        const int64_t chunkElements =
            chunkOffset + TOPK_WEIGHT_COPY_ELEMENTS > elements
                ? elements - chunkOffset
                : TOPK_WEIGHT_COPY_ELEMENTS;
        for (int64_t index = 0; index < chunkElements; ++index) {
            const int64_t sortedRow = indexBase + chunkOffset + index;
            const int32_t originalRoute =
                expandedRowIdx.GetValue(sortedRow);
            buffer.SetValue(
                index, topkWeights.GetValue(originalRoute));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams copyParams{
            1U,
            static_cast<uint32_t>(
                chunkElements * sizeof(bfloat16_t)),
            0U,
            0U,
            0U};
        DataCopyPad(
            routedTopkWeights[indexBase + chunkOffset],
            buffer,
            copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

__aicore__ inline void MegaMoeBF16UDMA::EndSync()
{
    if (gmm1VecSetSyncCom_ == 0) {
        return;
    }
    if constexpr(g_coreType == AIC) {
        WaitForVector();
    }
}

__aicore__ inline void MegaMoeBF16UDMA::EndGMM2Sync()
{
    if constexpr(g_coreType == AIV) {
        return;
    }
    if (gmm2VecSetSyncCom_ <= 0) {
        return;
    } else if (gmm2VecSetSyncCom_ == 1) {
        WaitForVector(FLAG_ID_MAX);
    } else {
        WaitForVector(gmm2PingPongIdx_ + FLAG_ID_MAX);
        WaitForVector(1 - gmm2PingPongIdx_ + FLAG_ID_MAX);
    }
}

__aicore__ inline void MegaMoeBF16UDMA::CrossRankSyncCntPerExpert()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t numPerCore = Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_),
            static_cast<uint32_t>(ALIGN_128));
    LocalTensor<int32_t> tmpBuffer = CrossRankSyncBuf_.Get<int32_t>();
    LocalTensor<int32_t> prevSumBuf = PrevSumBuf_.Get<int32_t>();
    // 给对端发送
    uint64_t offsetInDstWinAddr = perRankStridesInTokenPerExpert_ * sizeof(int32_t) * rankId_ + PEERMEM_DATA_OFFSET;
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
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
    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
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
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rankId_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(preSumBeforeRankWorkSpace_[dstRankIdx * expertPerRank_], prevSumBuf,
                    DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
    SyncAll<true>();
    if (aivCoreIdx_ == 0) {
        GetCumsumForMMAIV();
    }
}

__aicore__ inline void MegaMoeBF16UDMA::CopyGMToGMPerTokenV2(
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
    int32_t rows, int32_t hiddenSize, int32_t& pingpongId)
{
    (void)dstScale;
    constexpr int32_t BufferNum = 2;
    LocalTensor<QuantOutType> tmpBuffer1 = CopyTempBuf_.Get<QuantOutType>();
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[
        dispatchCopyElementsPerPing_];
    auto processCount = Ops::Base::CeilDiv(rows, dispatchRows_);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID eventId = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf =
            pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        int32_t rowNum = dispatchRows_;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows_;
        }
        const int64_t elementOffset =
            static_cast<int64_t>(processIndex) * dispatchRows_ * hiddenSize;
        const int64_t elementCount =
            static_cast<int64_t>(rowNum) * hiddenSize;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopy(buf, src[elementOffset], elementCount);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        DataCopy(dst[elementOffset], buf, elementCount);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
}

__aicore__ inline void MegaMoeBF16UDMA::DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
    int64_t widthA = Get<K_VALUE>(problemShape_); // H 长度
    int64_t widthAScale = 0;
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / dispatchBlockNumPerEP_;
        uint32_t aivIdxInGroup = idx % dispatchBlockNumPerEP_;
        uint32_t rowStartInDst = (dstRankIdx == 0 ? 0 :
            cumsumInWorkSpace_((dstRankIdx - 1) * expertPerRank_ + expertIdx));
        uint32_t prevSumOffset = idx / blockNum_;
        if (prevSum_[prevSumOffset] == 0) {
            prevSum_[prevSumOffset] = preSumBeforeRankWorkSpace_(idx / dispatchBlockNumPerEP_ * expertPerRank_);
        }
        if (rowStartInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = tokenPerExpertWin_(TokenPerExpertLayout(dstRankIdx, rankId_, expertIdx));
            if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartInDst;
            }
            int32_t rowSrcOffset = prevSum_[prevSumOffset];
            prevSum_[prevSumOffset] += rowsCopyCnt;
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(dispatchBlockNumPerEP_));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor <QuantOutType> gmRemoteA, aGlobalTensor;
                GlobalTensor <QuantScaleOutType> aScaleGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetAScale = rowDstOffsetPerCore * widthAScale;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * (widthA + widthAScale);
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(GetRankWinAddrWithOffset(dstRankIdx,
                    PEERMEM_DATA_OFFSET + perRankStridesInTokenPerExpert_ * worldSize_ * sizeof(uint32_t))));
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(gmmAddrInfo.aGlobal));
                aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
                    gmmAddrInfo.aScaleGlobal));
                CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA], aScaleGlobalTensor[gmOffsetAScale],
                    gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

__aicore__ inline void MegaMoeBF16UDMA::PublishControlSignal(
    uint32_t dstRank, uint32_t expertIdx)
{
    const uint64_t signalOffset = params_.peermemInfo.ptrDispatchSignal -
        params_.peermemInfo.ptrBase +
        (static_cast<uint64_t>(rankId_) * expertPerRank_ + expertIdx) * sizeof(uint64_t);
    __gm__ int32_t *dstSignal = reinterpret_cast<__gm__ int32_t *>(
        GetRankWinAddrWithOffset(dstRank, signalOffset));
    AscendC::WriteGmByPassDCache(dstSignal, static_cast<int32_t>(1));
    AscendC::WriteGmByPassDCache(dstSignal + 1, static_cast<int32_t>(0));
}

__aicore__ inline void MegaMoeBF16UDMA::DispatchTopkWeightsMTE(
    uint32_t expertIdx, uint64_t &expertRowBase)
{
    // Receiver-pull, like ordinary BF16: source routed weights are already
    // published before the count exchange. Store directly into persistent
    // [expert][source rank][row] order, matching the UDMA activation receive.
    const uint64_t routedOffset = params_.peermemInfo.ptrRoutedTopkWeights -
        params_.peermemInfo.ptrBase;
    auto buffer = TopkWeightCopyBuf_.Get<bfloat16_t>();
    uint64_t rowsBeforeSource = 0;
    for (uint32_t srcRank = 0; srcRank < worldSize_; ++srcRank) {
        const uint64_t rawRows = tokenPerExpertWin_(
            TokenPerExpertLayout(srcRank, rankId_, expertIdx));
        const uint64_t dstRow = expertRowBase + rowsBeforeSource;
        rowsBeforeSource += rawRows;
        if ((static_cast<uint64_t>(expertIdx) * worldSize_ + srcRank) %
                blockNum_ != blockIdx_ || dstRow >= maxOutputSize_) {
            continue;
        }
        const uint64_t remaining = maxOutputSize_ - dstRow;
        const uint64_t rows = rawRows < remaining ? rawRows : remaining;
        const uint64_t srcRow = preSumBeforeRankWorkSpace_(
            srcRank * expertPerRank_ + expertIdx);
        GlobalTensor<bfloat16_t> src, dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            GetRankWinAddrWithOffset(srcRank, routedOffset)) + srcRow);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            params_.workspaceInfo.ptrDispatchedTopkWeights) + dstRow);
        for (uint64_t offset = 0; offset < rows;) {
            const uint64_t left = rows - offset;
            const uint32_t count = static_cast<uint32_t>(
                left < TOPK_WEIGHT_COPY_ELEMENTS ? left : TOPK_WEIGHT_COPY_ELEMENTS);
            const DataCopyExtParams copyParams{
                1U, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
            const DataCopyPadExtParams<bfloat16_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(buffer, src[offset], copyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(dst[offset], buffer, copyParams);
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            offset += count;
        }
    }
    expertRowBase += rowsBeforeSource;
}

__aicore__ inline void MegaMoeBF16UDMA::DispatchAllTokensUDMA()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1) {
        return;
    }

    // ptrA0 is flattened as [dstRank][expert][row].  Initialize the source
    // cursor once for every peer owned by this AIV1; subsequent experts only
    // advance it by their raw row count.  ptrA1 is flattened as
    // [expert][srcRank][row], so its expert base advances by the total rows
    // received from all source ranks.
    uint32_t ownerPeerIdx = 0;
    for (uint32_t dstRank = blockIdx_; dstRank < worldSize_;
         dstRank += blockNum_, ++ownerPeerIdx) {
        uint64_t srcRowBase = 0;
        for (uint32_t priorDst = 0; priorDst < dstRank; ++priorDst) {
            for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
                srcRowBase += tokenPerExpertWin_(
                    TokenPerExpertLayout(rankId_, priorDst, expertIdx));
            }
        }
        dispatchSrcRowCursor_[ownerPeerIdx] = srcRowBase;
        dispatchDstExpertBase_[ownerPeerIdx] = 0;
    }

    uint64_t packedWeightExpertBase = 0;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        MEGA_MOE_BF16_TIMER_BLOCK(
            timer_.Tik(AscendTimer::AIV_DISPATCH, expertIdx));
        ownerPeerIdx = 0;
        for (uint32_t dstRank = blockIdx_; dstRank < worldSize_;
             dstRank += blockNum_, ++ownerPeerIdx) {
            // Keep the complete Dispatch transport path on AIV1: local SDMA
            // and remote UDMA use the same sub-block ownership model as the
            // standalone mixed Dispatch protocol test.
            const int32_t rows = tokenPerExpertWin_(
                TokenPerExpertLayout(rankId_, dstRank, expertIdx));
            const uint64_t srcRowOffset = dispatchSrcRowCursor_[ownerPeerIdx];
            dispatchSrcRowCursor_[ownerPeerIdx] += static_cast<uint64_t>(rows);

            uint64_t rowsBeforeRank = 0;
            uint64_t totalExpertRows = 0;
            for (uint32_t srcRank = 0; srcRank < worldSize_; ++srcRank) {
                const uint64_t srcRows = static_cast<uint64_t>(tokenPerExpertWin_(
                    TokenPerExpertLayout(srcRank, dstRank, expertIdx)));
                if (srcRank < rankId_) {
                    rowsBeforeRank += srcRows;
                }
                totalExpertRows += srcRows;
            }
            const uint64_t dstRowOffset =
                dispatchDstExpertBase_[ownerPeerIdx] + rowsBeforeRank;
            dispatchDstExpertBase_[ownerPeerIdx] += totalExpertRows;
            const uint64_t srcElementOffset = srcRowOffset * k_;
            const uint64_t dstElementOffset = dstRowOffset * k_;
            __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(
                params_.peermemInfo.ptrA0 + srcElementOffset * sizeof(bfloat16_t));
            __gm__ uint8_t *dst = reinterpret_cast<__gm__ uint8_t *>(
                params_.peermemInfo.ptrA1 + dstElementOffset * sizeof(bfloat16_t));
            __gm__ int32_t *signal = reinterpret_cast<__gm__ int32_t *>(
                params_.peermemInfo.ptrDispatchSignal) + 2 *
                (static_cast<uint64_t>(rankId_) * expertPerRank_ + expertIdx);

            int32_t copyRows = rows;
            if (dstRowOffset >= maxOutputSize_) {
                copyRows = 0;
            } else if (dstRowOffset + copyRows > maxOutputSize_) {
                copyRows = static_cast<int32_t>(maxOutputSize_ - dstRowOffset);
            }
            if (copyRows <= 0) {
                if (dstRank == rankId_) {
                    PublishControlSignal(dstRank, expertIdx);
                } else {
                    aclshmemx_udma_atomic_add(signal, static_cast<int32_t>(1), dstRank);
                }
                continue;
            }
            const uint32_t bytes = static_cast<uint32_t>(
                static_cast<uint64_t>(copyRows) * k_ * sizeof(bfloat16_t));
            if (dstRank == rankId_) {
                auto sdmaBuffer = SdmaWorkSpaceBuf_.template Get<uint8_t>();
                __ubuf__ uint8_t *sdmaScratch = reinterpret_cast<__ubuf__ uint8_t *>(
                    sdmaBuffer.GetPhyAddr());
                aclshmemx_sdma_get_nbi(dst, src, sdmaScratch, SDMA_UB_SIZE,
                                       bytes, rankId_, EVENT_ID0);
                __gm__ uint8_t *localSignal = reinterpret_cast<__gm__ uint8_t *>(
                    params_.peermemInfo.ptrDispatchSignal +
                    (static_cast<uint64_t>(rankId_) * expertPerRank_ + expertIdx) * sizeof(uint64_t));
                aclshmemx_sdma_get_nbi(localSignal,
                    reinterpret_cast<__gm__ uint8_t *>(params_.peermemInfo.ptrAllOnes64),
                    sdmaScratch, SDMA_UB_SIZE, sizeof(uint64_t), rankId_, EVENT_ID0);
            } else {
                // SO orders the activation before its completion atomic.
                // Packed weights use receiver-side MTE and the local barrier
                // before compute; they no longer share this transport signal.
                aclshmemx_udma_put_nbi<
                    uint8_t, PIPE_S, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
                    dst, src, static_cast<__ubuf__ uint8_t *>(nullptr),
                    bytes, dstRank);
                aclshmemx_udma_atomic_add<
                    int32_t, ACLSHMEMX_UDMA_OP_CONFIG_SO>(
                    signal, static_cast<int32_t>(1), dstRank);
            }
        }
        DispatchTopkWeightsMTE(expertIdx, packedWeightExpertBase);
        MEGA_MOE_BF16_TIMER_BLOCK(
            timer_.template Tok<Overwrite>(
                AscendTimer::AIV_DISPATCH, expertIdx));
    }
}

__aicore__ inline void MegaMoeBF16UDMA::SendCombineExpertUDMA(
    uint32_t expertIdx, uint64_t expertRowBase)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Match the legacy BF16 ownership model: AIV0 executes SwiGLU, while
    // AIV1 owns both Dispatch and Combine.  The peer mapping is identical to
    // DispatchAllTokensUDMA; at very large EP one AIV1 may own multiple peers.
    if (subBlockIdx_ != 1) {
        return;
    }

    GlobalTensor<int32_t> groupList;
    groupList.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params_.workspaceInfo.ptrcumsumMM) + (worldSize_ - 1) * expertPerRank_);
    __gm__ int32_t *combineReady = reinterpret_cast<__gm__ int32_t *>(
        params_.workspaceInfo.ptrCombineReady);
    const uint32_t rowGroupSize = MegaMoeBF16CatlassImpl::L1_TILE_M;
    const uint32_t maxRowGroups = Ops::Base::CeilDiv(
        static_cast<uint32_t>(maxOutputSize_), rowGroupSize);
    const int32_t expectedNTiles = static_cast<int32_t>(Ops::Base::CeilDiv(
        k_, MegaMoeBF16CatlassImpl::L1_TILE_N));
    const uint64_t rowBytes = static_cast<uint64_t>(k_) * sizeof(bfloat16_t);
    constexpr uint64_t MAX_DMA_COPY_BYTES = 0xFFFFFFE0ULL;

    const uint64_t groupValue = static_cast<uint64_t>(groupList.GetValue(expertIdx));
    const uint64_t previousGroupValue = expertIdx == 0
        ? 0 : static_cast<uint64_t>(groupList.GetValue(expertIdx - 1));
    const uint64_t rawExpertRows = groupListType_ == 0
        ? groupValue - previousGroupValue
        : groupValue;
    const uint64_t remainingRows = expertRowBase >= maxOutputSize_
        ? 0 : maxOutputSize_ - expertRowBase;
    const uint64_t expertRows = rawExpertRows < remainingRows
        ? rawExpertRows : remainingRows;
    const uint32_t rowGroupCount = Ops::Base::CeilDiv(
        static_cast<uint32_t>(expertRows), rowGroupSize);

    for (uint32_t dstRank = blockIdx_; dstRank < worldSize_; dstRank += blockNum_) {
        // The local slice is written directly from the live GMM2 output UB by
        // MegaMoeCombineImpl::CombineTokens, matching the legacy BF16 path.
        // This sender therefore owns remote UDMA only and needs no extra UB
        // scratch/TPipe on AIV1.
        if (dstRank == rankId_) {
            continue;
        }
        bool issuedRemotePut = false;
        uint64_t peerRowInExpert = 0;
        for (uint32_t peer = 0; peer < dstRank; ++peer) {
            peerRowInExpert += tokenPerExpertWin_(
                TokenPerExpertLayout(peer, rankId_, expertIdx));
        }
        const int32_t rawRows = tokenPerExpertWin_(
            TokenPerExpertLayout(dstRank, rankId_, expertIdx));
        int32_t rows = rawRows;
        if (peerRowInExpert >= expertRows) {
            rows = 0;
        } else if (peerRowInExpert + rows > expertRows) {
            rows = static_cast<int32_t>(expertRows - peerRowInExpert);
        }
        const uint64_t dstExpertOffset = preSumBeforeRankWorkSpace_(
            dstRank * expertPerRank_ + expertIdx);
        const uint64_t peerRowEnd = peerRowInExpert + static_cast<uint64_t>(rows);
        for (uint32_t rowGroupIdx = 0; rowGroupIdx < rowGroupCount; ++rowGroupIdx) {
            const uint64_t groupRowBegin = static_cast<uint64_t>(rowGroupIdx) * rowGroupSize;
            const uint64_t rawGroupRowEnd = groupRowBegin + rowGroupSize;
            const uint64_t groupRowEnd = rawGroupRowEnd < expertRows
                ? rawGroupRowEnd : expertRows;
            const uint64_t copyRowBegin = groupRowBegin > peerRowInExpert
                ? groupRowBegin : peerRowInExpert;
            const uint64_t copyRowEnd = groupRowEnd < peerRowEnd
                ? groupRowEnd : peerRowEnd;
            if (copyRowBegin >= copyRowEnd) {
                continue;
            }

            // This owner has already completed all of its GMM2 loops for the
            // current expert before entering here, so waiting for the other
            // AIV1 cores cannot block output that this owner still needs to
            // produce itself.
            __gm__ int32_t *ready = combineReady +
                expertIdx * maxRowGroups + rowGroupIdx;
            while (AscendC::ReadGmByPassDCache(ready) < expectedNTiles) {
                const int64_t start = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - start < 100) {
                }
            }

            const uint64_t copyRows = copyRowEnd - copyRowBegin;
            const uint64_t copyBytes = copyRows * rowBytes;
            __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(
                params_.peermemInfo.ptrA1 +
                (expertRowBase + copyRowBegin) * rowBytes);
            __gm__ uint8_t *dst = reinterpret_cast<__gm__ uint8_t *>(
                params_.peermemInfo.ptrD +
                (dstExpertOffset + copyRowBegin - peerRowInExpert) * rowBytes);
            for (uint64_t byteOffset = 0; byteOffset < copyBytes;) {
                const uint64_t remainingBytes = copyBytes - byteOffset;
                const uint32_t currentCopyBytes = static_cast<uint32_t>(
                    remainingBytes < MAX_DMA_COPY_BYTES
                        ? remainingBytes : MAX_DMA_COPY_BYTES);
                aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
                    dst + byteOffset, src + byteOffset,
                    static_cast<__ubuf__ uint8_t *>(nullptr),
                    currentCopyBytes, dstRank);
                byteOffset += currentCopyBytes;
            }
            issuedRemotePut = true;
        }
        if (issuedRemotePut) {
            // Drain once per expert.  Besides bounding SQ occupancy, this also
            // guarantees staging for this expert is no longer in use before
            // the owner advances to the next GMM2 expert.
            aclshmemx_udma_quiet(dstRank);
        }
    }
}

__aicore__ inline bool MegaMoeBF16UDMA::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
    if (Get<IDX_M_OFFSET>(baseOffset_) + Get<M_VALUE>(problemShape_) > maxOutputSize_) {
        Get<M_VALUE>(problemShape_) = maxOutputSize_ - Get<IDX_M_OFFSET>(baseOffset_);
    }

    // split_m，when m=0, skip
    if (Get<M_VALUE>(problemShape_) <= 0) {
        return false;
    }

    return true;
}

__aicore__ inline void MegaMoeBF16UDMA::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.peermemInfo.ptrA1 +
            Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale +
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_), Get<IDX_M_OFFSET>(baseOffset_), 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);//GMMSwiglu调用
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagDispatchToGmm1 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
}

__aicore__ inline void MegaMoeBF16UDMA::ResetTokenPerExpert()
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


__aicore__ inline void MegaMoeBF16UDMA::UnpermuteV2() {
    if constexpr(g_coreType == AscendC::AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (MegaMoeBF16Routing::TryUnpermute(
            params_.workspaceInfo.expandedRowIdxGather, params_.peermemInfo.ptrD,
            params_.y2GmAddr, k_, topK_, coreOffset, coreLen,
            Catlass::Arch::Ascend950::UB_SIZE)) {
        return;
    }
    AscendC::GlobalTensor<int32_t> expandedRowIdxGather;
    expandedRowIdxGather.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
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
            int32_t rowIdx = expandedRowIdxGather(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataIn = (expId % 2 == 0) ? dataIn0 : dataIn1;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopy(dataIn, expandedX[rowIdx * k_], k_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
            if (expId == 0) {
                AscendC::Adds(dataRes, dataIn, 0.0F, k_);
            } else {
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

__aicore__ inline void MegaMoeBF16UDMA::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.ptrD);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        LocalTensor<bfloat16_t> dataResBf16 = DataResBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32 = DataResFp32Buf_.Get<float>();
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK_ + expId);
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[rowIdx * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            // bf16 -> fp32
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(dataResFp32, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32, dataResFp32, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(output[tokenIdx * k_], dataResBf16, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

__aicore__ inline void MegaMoeBF16UDMA::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.ptrBase + 48 * 1024);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

__aicore__ inline void MegaMoeBF16UDMA::Process()
{
    using namespace MegaMoeBF16Training;
    switch (ResolveSavedGateUpMode(params_.tilingData->savedGateUpMode, params_.tilingData->swigluLimit)) {
        case kSaveNone: ProcessWithSaveMode<kSaveNone>(); break;
        case kSaveBf16: ProcessWithSaveMode<kSaveBf16>(); break;
        default: ProcessWithSaveMode<kSaveMxFp8>(); break;
    }
}

template<uint32_t SaveMode>
__aicore__ inline void MegaMoeBF16UDMA::ProcessWithSaveMode()
{
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.template Tok<Overwrite>(AscendTimer::AIV_INIT_PARAM);
        timer_.Tik();
    }
#endif
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    // 1. moe_init_routing_v3 + 量化
    moe_init_routing_v3<bfloat16_t, QuantOutType>(params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
            params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx, tokenPerExpertCurRankAddr_, nullptr,
            params_.workspaceInfo.expandedRowIdxGather,
            params_.peermemInfo.ptrA1, &params_.tilingData->moeInitRoutingTilingData,
            params_.tilingData->moeInitRoutingTilingData.tilingKey);

    SyncAll<true>();
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.template Tok<Overwrite>(AscendTimer::AIV_INITROUTING_SORT);
        timer_.Tik();
    }
#endif

    // Complete the gather-only UB phase before allocating transport buffers.
    // Keep it in the same Timer interval as the existing scalar fallback.
    bool routedWeightsReady = false;
    if constexpr (g_coreType == AscendC::AIV) {
        routedWeightsReady = MegaMoeBF16Routing::TryGatherTopkWeights(
            params_.probsGmAddr, params_.workspaceInfo.expandedRowIdx,
            params_.peermemInfo.ptrRoutedTopkWeights,
            static_cast<uint64_t>(m_) * topK_, aivCoreIdx_, blockAivNum_);
    }

    // 2. buffer申请 & workspace Init
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);
    if (!routedWeightsReady) {
        GatherRoutedTopkWeights();
    }
    ResetWorkSpaceFlagList();
    SyncAll<true>();
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.template Tok<Overwrite>(
            AscendTimer::AIV_INITROUTING_EXPERT_TOKEN_COUNT);
        timer_.Tik();
    }
#endif

    // 3. cnt值全ep域内的发送以及本端接收整理
    CrossRankSyncCntPerExpert();
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.template Tok<Overwrite>(AscendTimer::AIV_ALLGATHER);
    }
#endif
    // Make token-count/cumsum metadata visible to every AIC before data
    // dispatch starts. Each active GMM1 AIC waits for the current expert's
    // per-source transport signals directly.
    if constexpr (g_coreType == AscendC::AIV) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<false>();
    if constexpr (g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 0) {
            ExpertTokenNumCopyOut();
        }
    }
    DispatchAllTokensUDMA();
    // Complete every AIV1's MTE weight pulls before the local all-core barrier
    // releases GMM1/SwiGLU. Remote activation readiness still uses SO/SDMA flags.
    if constexpr (g_coreType == AscendC::AIV) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    // Match W4A8: all peer owners finish enqueueing their ordered data+flag
    // operations before compute starts, but neither UDMA nor SDMA data is
    // drained here.  GMM1 may consume an early expert as soon as its remote
    // atomic slots and local SDMA flag arrive while later experts remain in
    // flight.
    SyncAll<false>();
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
        }
    }
    if (subBlockIdx_ == 0) {
        PipeBarrier<PIPE_ALL>();
        firstPipe.Destroy();
    }

    // 4. for循环按照本卡专家id进行Dispatch，内部gmm1SwigluQuant按照基本块进行流水
    GMMAddrInfo gmmAddrInfo{};
    {
        uint32_t gmm1N = Get<N_VALUE>(problemShape_);
        uint32_t gmm1K = Get<K_VALUE>(problemShape_);
        epilogueOp_.UpdateNextProblem({0, gmm1N / SWIGLU_N_HALF, gmm1K, 0});
            MegaMoeBF16CatlassImpl::GroupMatmulParam<0> gmm1Params(
                gmm1N, gmm1K, params_.tilingData->blockNumPerEP);
            for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
                if (!UpdateGroupParams(expertIdx)) {
                    continue;
                }
                UpdateGlobalBuffer(gmmAddrInfo, 0);
                if constexpr(g_coreType == AscendC::AIV) {
                }
                MegaMoeBF16CatlassImpl::GroupMatmulSwiglu<SaveMode>(
                    gmm1Params, epilogueOp_, params_, problemShape_,
                    gmmAddrInfo, startBlockIdx_, gmm1VecSetSyncCom_, expertIdx
                    MEGA_MOE_BF16_TIMER_ARG(timer_));
            }
    }

    // ptrA1 is reused as Combine staging.  Persist its valid GMM1-input
    // prefix, plus backward routing metadata, before GMM2 is allowed to start.
    SyncAll<false>();
    if constexpr(g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1) {
            TPipe savePipe;
            savePipe.InitBuffer(
                SaveStateBuf_, MegaMoeForwardSave::kCopyBufferBytes);
            SaveForwardState();
            PipeBarrier<PIPE_ALL>();
            savePipe.Destroy();
        }
    }
    SyncAll<false>();

    if constexpr(g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1) {
            // firstPipe(dispatch的CopyTempBuf_)与GMM2 fixpipe目标UB(AIV1)物理重叠。
            // 先通知本核cube启动GMM2，再利用AIV1空闲时间回收其负责peer的
            // Dispatch UDMA完成项，避免最终Combine quiet承担前序Dispatch等待。
            NotifyCube(GMM2_UB_READY_FLAG);
            for (uint32_t dstRank = blockIdx_; dstRank < worldSize_;
                 dstRank += blockNum_) {
                if (dstRank != rankId_) {
                    aclshmemx_udma_quiet(dstRank);
                }
            }
        }
    }
    // cube仅等待本核AIV1(dispatch)释放UB后再进入GMM2 fixpipe；不等待AIV0(swiglu)，
    // 从而保留“gmm2与swiglu尾块并行”，swiglu逐专家由groupFlagList继续把关。
    if constexpr(g_coreType == AscendC::AIC) {
        WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);
    }

    // 5. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    {
        uint32_t gmm2N = Get<K_VALUE>(problemShape_);
        uint32_t gmm2K = Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF;
            MegaMoeBF16CatlassImpl::GroupMatmulParam<1> gmm2Params(
                gmm2N, gmm2K);
            MegaMoeBF16CatlassImpl::CrossExpertRowGroupCombineSenderState combineSender;
            MegaMoeBF16CatlassImpl::InitCrossExpertRowGroupCombineSender(
                combineSender, gmm2Params, params_, gmm2N, rankId_);
            bool runGmm2 = true;
            if constexpr (g_coreType == AscendC::AIV) {
                runGmm2 = subBlockIdx_ == 1;
            }
            if (runGmm2) {
                for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
                    if (!UpdateGroupParams(expertIdx)) {
                        continue;
                    }
                    UpdateGlobalBuffer(gmmAddrInfo, 1);
                    const uint64_t expertRowBase = Get<IDX_M_OFFSET>(baseOffset_);
                    MegaMoeBF16CatlassImpl::GroupMatmul2(
                        gmm2Params, params_, problemShape_, gmmAddrInfo,
                        startBlockIdx_, gmm2VecSetSyncCom_,
                        combineSender,
                        expertIdx,
                        expertRowBase,
                        gmm2PingPongIdx_, rankId_
                        MEGA_MOE_BF16_TIMER_ARG(timer_));
                    if constexpr (g_coreType == AscendC::AIV) {
                        // Extremely large EP values that exceed the bounded
                        // per-owner sender state retain the legacy fallback.
                        // Normal large-EP configurations are handled by the
                        // multi-peer sender inside GroupMatmul2.
                        if (combineSender.useLegacyFallback) {
                            SendCombineExpertUDMA(expertIdx, expertRowBase);
                        }
                    }
                }
            }
            // Preserve outstanding work across expert boundaries.  The peer
            // owner blocks only after all GMM2 experts have been issued, then
            // finalizes unsent slices and drains every owned peer before the
            // global Combine barrier.
            if constexpr (g_coreType == AscendC::AIV) {
                if (subBlockIdx_ == 1) {
                    MEGA_MOE_BF16_TIMER_BLOCK(timer_.Tik());
                }
            }
            MegaMoeBF16CatlassImpl::FinalizeCrossExpertCombineSender(
                combineSender, params_);
            if constexpr (g_coreType == AscendC::AIV) {
                if (subBlockIdx_ == 1) {
                    MEGA_MOE_BF16_TIMER_BLOCK(
                        timer_.template Tok<Overwrite>(
                            AscendTimer::AIV_INITROUTING_GATHEROUT_QUANT));
                }
            }
    }
    EndSync();
    EndGMM2Sync();
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.Tik();
    }
#endif
    SyncAll<true>();
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.template Tok<Overwrite>(AscendTimer::AIV_WAIT_COMB);
    }
#endif

    // 6. unpermute & 清理win区上放置token的位置
    if constexpr(g_coreType == AIV) {
        //TPipe secondPipe;
        //SecondBuffInit(&secondPipe);
        ResetTokenPerExpert();
        SyncAll<true>();
#if defined(ENABLE_TIMER)
        timer_.Tik();
#endif
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
#if defined(ENABLE_TIMER)
        timer_.template Tok<Overwrite>(AscendTimer::AIV_UNPERMUTE_WAIT);
        timer_.Tik();
#endif
        UnpermuteV2();
        PipeBarrier<PIPE_ALL>();
#if defined(ENABLE_TIMER)
        timer_.template Tok<Overwrite>(AscendTimer::AIV_UNPERMUTE);
#endif
        //secondPipe.Destroy();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
#if defined(ENABLE_TIMER)
    timerAll_.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
}

}   // namespace MegaMoeImpl
#endif
