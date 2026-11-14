/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
#include "kernel_operator.h"
#include "mega_moe_quant_constants.h"
#include "mega_moe_tiling.h"
#include "mega_moe_bf16_mgroup.h"
#include "ascend_deepep/kernels/launch.h"

using namespace AscendC;
using namespace MegaMoeImpl;

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void mega_moe_bf16_mgroup(
    GM_ADDR context,
    GM_ADDR x,
    GM_ADDR topkIds,
    GM_ADDR topkWeights,
    GM_ADDR weight1,
    GM_ADDR weight2,
    GM_ADDR timerAddr,
    GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut,
    GM_ADDR trainingSavedStateGM,
    GM_ADDR forwardTransientWorkspaceGM,
    MegaMoeTilingData tilingData)
{
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    MegaMoeBF16MGroup op;
    op.Init(
        context,
        x,
        topkIds,
        topkWeights,
        weight1,
        weight2,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        yOut,
        expertTokenNumsOut,
        nullptr,
        nullptr,
        nullptr,
        trainingSavedStateGM,
        &tilingData,
        timerAddr,
        forwardTransientWorkspaceGM,
        nullptr,
        false);
    op.Process();
}

extern "C" ASCEND_DEEPEP_EXPORT void mega_moe_bf16_mgroup_kernel_do(
    uint32_t block_dim,
    void *stream,
    uint8_t *context,
    uint8_t *x,
    uint8_t *topk_ids,
    uint8_t *topk_weights,
    uint8_t *weight1,
    uint8_t *weight2,
    uint8_t *timer,
    uint8_t *y,
    uint8_t *expert_token_nums,
    uint8_t *training_saved_state,
    uint8_t *forward_transient_workspace,
    const uint8_t *tiling_host,
    uint64_t ffts_addr)
{
    (void)ffts_addr;
    const auto &tiling =
        *reinterpret_cast<const MegaMoeTilingData *>(tiling_host);
    mega_moe_bf16_mgroup<<<block_dim, nullptr, stream>>>(
        context,
        x,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        timer,
        y,
        expert_token_nums,
        training_saved_state,
        forward_transient_workspace,
        tiling);
}

extern "C" ASCEND_DEEPEP_EXPORT int mega_moe_bf16_mgroup_timer_enabled()
{
#if defined(ENABLE_TIMER)
    return 1;
#else
    return 0;
#endif
}
