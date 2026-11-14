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
 * \file mega_moe_combine_send.h
 * \brief Token-grained combine send from complete GMM2 workspace output.
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H
#include "kernel_operator.h"
#include "mega_moe_base.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {
template <typename ElementC>
__aicore__ inline void CombineTokensParallel(
    uint32_t groupIdx, uint32_t rankId, GlobalTensor<ElementC> currentExpertOutputGm,
    int32_t currentExpertTokenCount, const Params& params)
{
    int32_t hiddenSize = static_cast<int32_t>(params.tilingData->k);
    uint32_t rowBytes = static_cast<uint32_t>(hiddenSize) * sizeof(ElementC);
    LocalTensor<ElementC> tokenRowUb{
        AscendC::TPosition::VECIN, 0, static_cast<uint32_t>(hiddenSize)};

    // W4A8 keeps the two AIV roles separate: AIV1 runs weight Prologue,
    // while only AIV0 participates in token-grained combine.
    int32_t aivCoreCount = static_cast<int32_t>(GetBlockNum());
    int32_t aivCoreId = static_cast<int32_t>(GetBlockIdx() / GetTaskRation());
    int32_t rowsPerCore = Ops::Base::CeilDiv(
        static_cast<uint32_t>(currentExpertTokenCount), static_cast<uint32_t>(aivCoreCount));
    int32_t coreRowStart = aivCoreId * rowsPerCore;
    int32_t coreRowEnd = coreRowStart + rowsPerCore;
    if (coreRowStart >= currentExpertTokenCount) {
        return;
    }
    if (coreRowEnd > currentExpertTokenCount) {
        coreRowEnd = currentExpertTokenCount;
    }
    int32_t alignedExpertCount = Ops::Base::CeilAlign(
        static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
        static_cast<uint32_t>(ALIGN_128));
    int32_t sourceSegmentStart = 0;
    GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrBeforeColumn);

    DataCopyExtParams rowCopyParams{1U, rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<ElementC> rowPadParams{false, 0U, 0U, 0U};
    uint64_t remoteOutputOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
    GlobalTensor<ElementC> remoteOutputGm;
    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->epWorldSize; dstEpIdx++) {
        int32_t lenRankInExpert = tokenPerExpertWin(
            dstEpIdx * alignedExpertCount + rankId * params.tilingData->expertPerRank + groupIdx);
        int32_t dstExpertOffset = preSumBeforeRankWorkSpace(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
        int32_t sourceSegmentEnd = sourceSegmentStart + lenRankInExpert;

        if (sourceSegmentStart >= coreRowEnd) {
            break;
        }
        if (sourceSegmentEnd <= coreRowStart) {
            sourceSegmentStart = sourceSegmentEnd;
            continue;
        }

        int32_t copyRowStart = sourceSegmentStart > coreRowStart ? sourceSegmentStart : coreRowStart;
        int32_t copyRowEnd = sourceSegmentEnd < coreRowEnd ? sourceSegmentEnd : coreRowEnd;
        remoteOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(
            GetRankWinAddrWithOffset(dstEpIdx, remoteOutputOffset)));

        for (int32_t expertRow = copyRowStart; expertRow < copyRowEnd; ++expertRow) {
            int32_t dstRow = dstExpertOffset + expertRow - sourceSegmentStart;
            DataCopyPad(tokenRowUb,
                currentExpertOutputGm[static_cast<int64_t>(expertRow) * hiddenSize],
                rowCopyParams, rowPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            DataCopyPad(remoteOutputGm[static_cast<int64_t>(dstRow) * hiddenSize],
                tokenRowUb, rowCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        sourceSegmentStart = sourceSegmentEnd;
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
