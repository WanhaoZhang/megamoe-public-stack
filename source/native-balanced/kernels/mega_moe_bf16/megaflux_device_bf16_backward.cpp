#define DEEPEP_BF16_REPLICA 1
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * DeepEP MegaMoe BF16 training backward kernel entry.
 *
 * ABI:
 *   mega_moe_bf16_backward_kernel_do(
 *       block_dim, stream,
 *       context,           Mc2MoeContext (peermem window addresses + rank)
 *       grad_y,            [M, K] BF16
 *       topk_weights,      [M, T] BF16
 *       weight1,           canonical [E,N,K] or legacy [E,K,N] BF16
 *       weight2,           canonical [E,K,F] or legacy [E,F,K] BF16
 *       grad_x,            [M, K] BF16 (host pre-zeroed)
 *       grad_topk_weights, [M, T] BF16 (host pre-zeroed)
 *       grad_weight1,      same shape/layout as weight1 (host pre-zeroed)
 *       grad_weight2,      same shape/layout as weight2 (host pre-zeroed)
 *       training_saved_state, UInt8 compact forward-persistent state
 *       dispatch_output,      packed X followed by packed TopK weights
 *       backward_workspace, UInt8 backward-only scratch
 *       tiling_host, ffts_addr)
 */
#include "kernel_operator.h"
#include "mega_moe_quant_constants.h"
#include "mega_moe_tiling.h"
#include "mega_moe_bf16_backward.h"
#include "ascend_deepep/kernels/launch.h"

#include "mega_moe_replica_mixed.h"
#include "mega_moe_device_descriptor.h"
using namespace AscendC;
using namespace MegaMoeImpl;

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void megaflux_device_bf16_backward(
    GM_ADDR context,
    GM_ADDR gradY,
    GM_ADDR topkWeights,
    GM_ADDR weight1,
    GM_ADDR weight2,
    GM_ADDR gradX,
    GM_ADDR gradTopkWeights,
    GM_ADDR gradWeight1,
    GM_ADDR gradWeight2,
    GM_ADDR trainingSavedState,
    GM_ADDR dispatchOutput,
    GM_ADDR backwardWorkspace,
    GM_ADDR timerAddr,
    MegaMoeTilingData tilingData, MegaMoeTilingData homeTiling, MegaMoeDeviceReplicaDescriptor descriptor, uint32_t splitExperts)
{
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    // Planner is earlier in this stream. All ranks observe the same collective
    // status and skip every compute/copy/reduction collective on failure.
    auto header=(__gm__ int32_t*)descriptor.header;
    if(descriptor.abi!=2 || MegaFluxMixed::Fresh(header+3)!=0 ||
       MegaFluxMixed::Fresh(header+2)!=descriptor.epoch ||
       MegaFluxMixed::Fresh(header+1)>descriptor.capacity ||
       MegaFluxMixed::Fresh(header+7)!=descriptor.capacity)return;
    const bool active=MegaFluxMixed::Fresh(header)>0;
    if(!active)tilingData=homeTiling;
    MegaMoeBF16Backward op;
    op.Init(
        context,
        gradY,
        topkWeights,
        weight1,
        weight2,
        gradX,
        gradTopkWeights,
        gradWeight1,
        gradWeight2,
        trainingSavedState,
        dispatchOutput,
        backwardWorkspace,
        &tilingData, splitExperts,
        timerAddr);
    op.EnableReplica(descriptor.replica);
    op.Process();
    if(active)MegaFluxMixed::Reduce(descriptor.replica,op,gradWeight1,gradWeight2);
}

extern "C" ASCEND_DEEPEP_EXPORT void megaflux_device_bf16_backward_ready_k_kernel_do(
    uint32_t block_dim,
    void *stream,
    uint8_t *context,
    uint8_t *grad_y,
    uint8_t *topk_weights,
    uint8_t *weight1,
    uint8_t *weight2,
    uint8_t *grad_x,
    uint8_t *grad_topk_weights,
    uint8_t *grad_weight1,
    uint8_t *grad_weight2,
    uint8_t *training_saved_state,
    uint8_t *dispatch_output,
    uint8_t *backward_workspace,
    uint8_t *timer,
    const uint8_t *tiling_host,
    const uint8_t *home_tiling_host,
    uint64_t ffts_addr, MegaMoeDeviceReplicaDescriptor descriptor, uint32_t splitExperts)
{
    (void)ffts_addr;
    const auto &tiling =
        *reinterpret_cast<const MegaMoeTilingData *>(tiling_host);
    megaflux_device_bf16_backward<<<block_dim, nullptr, stream>>>(
        context,
        grad_y,
        topk_weights,
        weight1,
        weight2,
        grad_x,
        grad_topk_weights,
        grad_weight1,
        grad_weight2,
        training_saved_state,
        dispatch_output,
        backward_workspace,
        timer,
        tiling,*reinterpret_cast<const MegaMoeTilingData *>(home_tiling_host),descriptor,splitExperts);
}

extern "C" ASCEND_DEEPEP_EXPORT int megaflux_device_bf16_backward_timer_enabled()
{
#if defined(ENABLE_TIMER)
    return 1;
#else
    return 0;
#endif
}
