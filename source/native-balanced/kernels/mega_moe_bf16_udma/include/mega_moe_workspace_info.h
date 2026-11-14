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
 * \file mega_moe_workspace_info.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_INFO_H
#define MEGA_MOE_WORKSPACE_INFO_H

#include "kernel_operator.h"
#include "../../mega_moe_bf16/include/mega_moe_training_workspace.h"

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t FLAG_ID_MAX = 16;
// GMM1(dispatch/swiglu)阶段结束、GMM2开始前，AIV1->AIC的一次性同步：
// 用一个独立的flag槽位(避开combine使用的pingpong 0/1)，值2 => AIV1本地槽8，
// cube侧WaitForVector(2+FLAG_ID_MAX)路由到AIV1。
constexpr uint16_t GMM2_UB_READY_FLAG = 2;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t PEERMEM_DATA_OFFSET = 1024 * 60;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
constexpr int32_t MAX_AICORE_NUM = 36;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_BF_16 = 2U;
constexpr int64_t SIZE_INT_32 = 4U;
} // namespace

struct WorkspaceInfo {
    GM_ADDR gmCtx;
    GM_ADDR ptrAScale;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;
    GM_ADDR ptrcumsumMM;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;
    GM_ADDR ptrDispatchedTopkWeights;
    GM_ADDR ptrSavedGate{nullptr};
    GM_ADDR ptrSavedUp{nullptr};
    GM_ADDR ptrSavedGateScale{nullptr};
    GM_ADDR ptrSavedUpScale{nullptr};
    GM_ADDR ptrSavedGateHigh{nullptr};
    GM_ADDR ptrSavedUpLow{nullptr};
    GM_ADDR ptrSavedUpHigh{nullptr};
    GM_ADDR ptrFlagSwiGluToGmm2;
    GM_ADDR ptrFlagDispatchToGmm1;
    GM_ADDR ptrCombineReady;
    int64_t workspaceSize;
    __aicore__ inline WorkspaceInfo() = default;
    __aicore__ inline WorkspaceInfo(
        GM_ADDR savedStateBase,
        GM_ADDR transientWorkspaceBase,
        const MegaMoeTilingData *tilingData)
    {
        const MegaMoeBF16Training::TrainingSavedStateLayout savedLayout(
            tilingData->m, tilingData->k, tilingData->topK,
            tilingData->epWorldSize, tilingData->expertPerRank,
            tilingData->maxOutputSize, tilingData->n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tilingData->swigluLimit, tilingData->savedGateUpMode), tilingData->savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tilingData->swigluLimit, tilingData->savedGateUpMode));

        const MegaMoeBF16Training::UDMAForwardTransientWorkspaceLayout
            transientLayout(tilingData->n, tilingData->expertPerRank,
                            tilingData->maxOutputSize);

        // Persistent routing metadata has no duplicate transient allocation.
        ptrcumsumMM = savedStateBase + savedLayout.cumsumOffset;
        expandedRowIdx = savedStateBase + savedLayout.expandedRowIdxOffset;
        expandedRowIdxGather =
            savedStateBase + savedLayout.expandedRowIdxGatherOffset;
        ptrSumBeforeRank = savedStateBase + savedLayout.sumBeforeRankOffset;
        ptrDispatchedTopkWeights =
            savedStateBase + savedLayout.dispatchedTopkWeightsOffset;
        ptrSavedGate = savedLayout.gateUpSaved ? savedStateBase + savedLayout.gateOffset : nullptr;
        ptrSavedUp = savedLayout.gateUpSaved ? savedStateBase + savedLayout.upOffset : nullptr;
        ptrSavedGateScale = savedLayout.gateUpMxFp8 ? savedStateBase + savedLayout.gateScaleOffset : nullptr;
        ptrSavedUpScale = savedLayout.gateUpMxFp8 ? savedStateBase + savedLayout.upScaleOffset : nullptr;
        ptrSavedGateHigh = savedLayout.clipMaskKind ? savedStateBase + savedLayout.gateHighOffset : nullptr;
        ptrSavedUpLow = savedLayout.clipMaskKind ? savedStateBase + savedLayout.upLowOffset : nullptr;
        ptrSavedUpHigh = savedLayout.clipMaskKind ? savedStateBase + savedLayout.upHighOffset : nullptr;
        ptrA2 = transientWorkspaceBase + transientLayout.ptrA2Offset;
        ptrFlagSwiGluToGmm2 =
            transientWorkspaceBase + transientLayout.swigluToGmm2FlagOffset;
        ptrFlagDispatchToGmm1 =
            transientWorkspaceBase + transientLayout.dispatchToGmm1FlagOffset;
        ptrCombineReady =
            transientWorkspaceBase + transientLayout.combineReadyOffset;
        workspaceSize = static_cast<int64_t>(transientLayout.totalBytes);
        // BF16 does not consume scale buffers; preserve non-null aliases.
        ptrAScale = ptrA2;
        ptrA2Scale = ptrcumsumMM;
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H
