/**
 * MegaMoe W4A8 device entry for Ascend DeepEP.
 */

#include "kernel_operator.h"
#include "mega_moe.h"
#include "mega_moe_tiling.h"
#include "mega_moe_quant_constants.h"
#include "ascend_deepep/kernels/launch.h"

using namespace AscendC;
using namespace MegaMoeW4A8Impl;

template<typename DTYPE_X, typename DTYPE_Y, typename DTYPE_TOPK_WEIGHTS,
         uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType>
#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void mega_moe_w4a8(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weightScales1,
    GM_ADDR weightScales2, GM_ADDR xActiveMask, GM_ADDR scales,
    GM_ADDR timerAddr, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
    MegaMoeW4A8TilingData tilingData)
{
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS,
            DispatchQuantOutType> op;
        op.Init(context, x, topkIds, topkWeights, weight1, weight2,
            xActiveMask, weightScales1, weightScales2, scales, yOut,
            expertTokenNumsOut, workspaceGM, &tilingData, timerAddr);
        op.Process();
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void mega_moe_w4a8_kernel_do(
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
    uint8_t* timer,
    uint8_t* y,
    uint8_t* expert_token_nums,
    uint8_t* workspace,
    const uint8_t* tiling_host,
    uint64_t ffts_addr)
{
    (void)ffts_addr;
    const auto& tiling =
        *reinterpret_cast<const MegaMoeW4A8TilingData*>(tiling_host);
    mega_moe_w4a8<bfloat16_t, bfloat16_t, bfloat16_t,
        DISPATCH_QUANT_MODE_MXFP, DISPATCH_QUANT_OUT_TYPE_E4M3FN>
        <<<block_dim, nullptr, stream>>>(
            context, x, topk_ids, topk_weights, weight1, weight2,
            weight_scales1, weight_scales2, x_active_mask, scales, timer, y,
            expert_token_nums, workspace, tiling);
}
