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
 * \file mega_moe_base.h
 * \brief
 */

#ifndef MEGA_MOE_BASE_H
#define MEGA_MOE_BASE_H

#include "lib/std/tuple.h"
#include "mega_moe_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t epRankSize;
    uint64_t winSize;
    uint64_t epHcclBuffer[1024];
};

struct PeermemInfo {
    GM_ADDR ptrBase;
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrCommFlag;
    GM_ADDR ptrAllOnes; // SDMA 本地flag置1的源区（全1），在ptrCommFlag尾端
    GM_ADDR ptrA0;      // initrouting结果，包括x+scale
    GM_ADDR ptrA1;     //dispatch 后的 x+scale tilingData->maxOutputSize
    GM_ADDR ptrA1Scale;
    GM_ADDR ptrD;

    __aicore__ inline PeermemInfo() = default;

    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeW4A8TilingData *tilingData) {
        int64_t offset = PEERMEM_DATA_OFFSET;
        ptrBase = base;
        ptrTokenPerExpert = AllocateAddr(offset, tilingData->epWorldSize *
            Ops::Base::CeilAlign(tilingData->epWorldSize * tilingData->expertPerRank * SIZE_INT_32, ALIGN_512));
        ptrCommFlag = AllocateAddr(offset,
            Ops::Base::CeilAlign(tilingData->expertPerRank * 2 * INT_CACHELINE * SIZE_INT_32, ALIGN_512));
        ptrAllOnes = AllocateAddr(offset,
            Ops::Base::CeilAlign(INT_CACHELINE * SIZE_INT_32, ALIGN_512));
        ptrA0 = AllocateAddr(offset,
            Ops::Base::CeilAlign(tilingData->m * tilingData->topK * (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * SIZE_INT_8, ALIGN_512));
        ptrA1 = AllocateAddr(offset,
            Ops::Base::CeilAlign(tilingData->maxOutputSize * (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * SIZE_INT_8, ALIGN_512));
        ptrA1Scale = ptrA1 + tilingData->k * SIZE_INT_8;
        ptrD = AllocateAddr(offset,
            Ops::Base::CeilAlign(SIZE_BF_16 * tilingData->m * tilingData->topK * tilingData->k, ALIGN_512));
    }

private:
    __aicore__ inline GM_ADDR
    AllocateAddr(int64_t& offset, int64_t bytes) const
    {
        GM_ADDR addr = ptrBase + offset;
        offset += bytes;
        return addr;
    }
};

struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeW4A8TilingData *tilingData;
};

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

// __aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
// {
//     __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
//     uint64_t tensorPtrOffset = *dataAddr;
//     __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
//     return reinterpret_cast<GM_ADDR>(*(retPtr + index));
// }

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

#endif
