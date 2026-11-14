/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "mega_moe.h"
#include "mega_moe_tiling.h"
#include "mega_moe_quant_constants.h"
#include "ascend_deepep/kernels/launch.h"

using namespace AscendC;
using namespace MegaMoeImpl;

template<typename DTYPE_X, typename DTYPE_Y, typename DTYPE_TOPK_WEIGHTS,
         uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType>
#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void mega_moe(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR timerAddr,
    MegaMoeTilingData tilingData)
{
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

    if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DispatchQuantOutType> op;
        op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData,
                timerAddr);
        op.Process();
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void mega_moe_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* context,
    uint8_t* x,
    uint8_t* topk_ids,
    uint8_t* topk_weights,
    uint8_t* weight1,
    uint8_t* weight2,
    uint8_t* weight_scales1,
    uint8_t* weight_scales2,
    uint8_t* x_active_mask,
    uint8_t* scales,
    uint8_t* y,
    uint8_t* expert_token_nums,
    uint8_t* workspace,
    uint8_t* timer,
    const uint8_t* tiling_host,
    uint64_t ffts_addr)
{
    (void)ffts_addr;
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    mega_moe<bfloat16_t, bfloat16_t, bfloat16_t,
             DISPATCH_QUANT_MODE_MXFP, DISPATCH_QUANT_OUT_TYPE_E4M3FN>
        <<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2,
            weight_scales1, weight_scales2, x_active_mask, scales, y,
            expert_token_nums, workspace, timer, tiling);
}

extern "C" ASCEND_DEEPEP_EXPORT int mega_moe_timer_enabled()
{
#if defined(ENABLE_TIMER)
    return 1;
#else
    return 0;
#endif
}
