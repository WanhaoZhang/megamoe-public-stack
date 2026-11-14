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

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__ [aicore]
#else
// #define GM_ADDR uint8_t*
#define HOST_DEVICE
#endif

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
constexpr uint32_t SDMA_UB_SIZE = 64U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_BF_16 = 2U;
constexpr int64_t SIZE_INT_32 = 4U;
} // namespace

struct WorkspaceInfo {
    GM_ADDR ptrWeightGamma;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;
    GM_ADDR ptrBeforeRow;
    GM_ADDR ptrBeforeColumn;
    GM_ADDR ptrAfterRow;
    GM_ADDR ptrAfterRowLast;
    GM_ADDR ptrAfterColumn;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrGmm1Output;
    GM_ADDR ptrGmm2Output;
    GM_ADDR ptrFlagSwiGluToGmm2;
    GM_ADDR ptrGmm1ReadyFlag;
    GM_ADDR ptrGmm2ReadyFlag;
    GM_ADDR ptrBase;
    int64_t workspaceSize{0};

    __aicore__ inline WorkspaceInfo() = default;
    __aicore__ inline WorkspaceInfo(GM_ADDR base, const MegaMoeW4A8TilingData *tilingData)
    {
        int64_t offset = 0;
        ptrBase = base;
        ptrWeightGamma = AllocateAddr(offset, 512);
        ptrA2 = AllocateAddr(offset,
                             Ops::Base::CeilAlign((int64_t)(SIZE_INT_8 * tilingData->maxOutputSize *
                                                          tilingData->n / SWIGLU_N_HALF), ALIGN_512));
        ptrA2Scale = AllocateAddr(offset,
                                  Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->n /
                                                               SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM), ALIGN_512));
        // before comm
        ptrBeforeRow = AllocateAddr(offset,
                                   Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
                                                                tilingData->expertPerRank, ALIGN_512));
        ptrBeforeColumn = AllocateAddr(offset,
                                       Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
                                                                tilingData->expertPerRank, ALIGN_512));
        // after comm
        ptrAfterRow = AllocateAddr(offset,
                                       Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
                                                            tilingData->expertPerRank, ALIGN_512));
        ptrAfterRowLast = AllocateAddr(offset,
                                       Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
                                                            tilingData->expertPerRank, ALIGN_512));
        ptrAfterColumn = AllocateAddr(offset,
                                        Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
                                                             tilingData->expertPerRank, ALIGN_512));
        // rounting 排序后的映射
        expandedRowIdx = AllocateAddr(offset,
                                      SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->m), ALIGN_256) *
                                              tilingData->topK);
        //expandedRowIdx 的反映射
        expandedRowIdxGather = AllocateAddr(offset,
                                            SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->m), ALIGN_256) *
                                            tilingData->topK);

        // GMM1 writes complete BF16 Gate/Up rows here. SwiGLU reads them from
        // GM so Cube execution no longer waits for AIV epilogue consumption.
        ptrGmm1Output = AllocateAddr(offset,
                                     Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_BF_16) *
                                                          tilingData->maxOutputSize * tilingData->n,
                                                          ALIGN_512));

        // GMM2 first writes complete BF16 expert rows here. Combine then
        // redistributes those rows across all AIV cores at token granularity.
        ptrGmm2Output = AllocateAddr(offset,
                                     Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_BF_16) *
                                                          tilingData->maxOutputSize * tilingData->k,
                                                          ALIGN_512));

        ptrFlagSwiGluToGmm2 = AllocateAddr(offset,
                                           Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->expertPerRank) * INT_CACHELINE * SIZE_INT_32, ALIGN_512));
        ptrGmm1ReadyFlag = AllocateAddr(offset,
                                        Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->expertPerRank) *
                                                             INT_CACHELINE * SIZE_INT_32,
                                                             ALIGN_512));
        ptrGmm2ReadyFlag = AllocateAddr(offset,
                                        Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->expertPerRank) *
                                                             INT_CACHELINE * SIZE_INT_32,
                                                             ALIGN_512));
    }

private:
    __aicore__ inline GM_ADDR
    AllocateAddr(int64_t& offset,
    int64_t bytes
    ) const
    {
        GM_ADDR addr = ptrBase + offset;
        offset += bytes;
        return addr;
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H
