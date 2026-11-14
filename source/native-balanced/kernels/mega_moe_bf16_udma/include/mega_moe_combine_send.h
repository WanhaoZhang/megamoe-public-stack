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
 * \brief
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H
#include "catlass/gemm_coord.hpp"
#include "kernel_operator.h"
#include "mega_moe_base.h"
using namespace AscendC;

namespace MegaMoeCombineImpl {
template <typename ElementMMadOut2, typename GemmCoord>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, uint32_t groupIdx,
    uint64_t expertRowBase, uint32_t rankId,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, GemmCoord& actualBlockShape, const Params& params)
{
    // UDMA only accepts contiguous GM messages.  Preserve remote rows in the
    // local expert-row-major staging layout; keep local rows on the legacy
    // direct UB-to-ptrD path.  Avoiding local rows in staging removes the
    // duplicate GM write without changing the sender's source offsets.
    AscendC::GlobalTensor<ElementMMadOut2> gmStaging;
    gmStaging.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2 *>(
        params.peermemInfo.ptrA1));
    AscendC::GlobalTensor<int32_t> tokenPerExpertWin, preSumBeforeRankWorkSpace;
    tokenPerExpertWin.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    preSumBeforeRankWorkSpace.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);
    const uint32_t tokenStride = Ops::Base::CeilAlign(
        static_cast<uint32_t>(params.tilingData->epWorldSize * params.tilingData->expertPerRank),
        static_cast<uint32_t>(ALIGN_128));
    const int32_t stTile = static_cast<int32_t>(mLoc);
    const int32_t edTile = stTile + static_cast<int32_t>(actualBlockShape.m());
    int32_t stRankInExpert = 0;
    for (uint32_t dstRank = 0; dstRank < params.tilingData->epWorldSize; ++dstRank) {
        const int32_t rankRows = tokenPerExpertWin(
            dstRank * tokenStride + rankId * params.tilingData->expertPerRank + groupIdx);
        const int32_t edRankInExpert = stRankInExpert + rankRows;
        if (stRankInExpert >= edTile) {
            break;
        }
        if (edRankInExpert <= stTile) {
            stRankInExpert = edRankInExpert;
            continue;
        }

        const int32_t stData = tla::max(stRankInExpert, stTile);
        const int32_t edData = tla::min(edRankInExpert, edTile);
        const int32_t lenData = edData - stData;
        const uint64_t srcOffsetInTile = static_cast<uint64_t>(stData - stTile);
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = static_cast<uint16_t>(lenData);
        ub2GmParams.blockLen = actualBlockShape.n() * sizeof(ElementMMadOut2);
        ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(ElementMMadOut2);

        if (dstRank == rankId) {
            // Preserve the legacy local Combine path while the GMM2 tile is
            // still live in UB.
            AscendC::GlobalTensor<ElementMMadOut2> gmLocalD;
            gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
                params.peermemInfo.ptrD));
            const uint64_t dstExpertOffset = preSumBeforeRankWorkSpace(
                rankId * params.tilingData->expertPerRank + groupIdx);
            const uint64_t dstOffsetInExpert = static_cast<uint64_t>(
                stData - stRankInExpert);
            const uint64_t gmLocalOffset =
                (dstExpertOffset + dstOffsetInExpert) * n + nLoc;
            AscendC::DataCopyPad(
                gmLocalD[gmLocalOffset],
                l0cOutUbGMM2[srcOffsetInTile * actualBlockShape.n()],
                ub2GmParams);
        } else {
            const uint64_t gmStagingOffset =
                (expertRowBase + static_cast<uint64_t>(stData)) * n + nLoc;
            AscendC::DataCopyPad(
                gmStaging[gmStagingOffset],
                l0cOutUbGMM2[srcOffsetInTile * actualBlockShape.n()],
                ub2GmParams);
        }
        stRankInExpert = edRankInExpert;
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_BF16_UDMA_COMBINE_SEND_H
