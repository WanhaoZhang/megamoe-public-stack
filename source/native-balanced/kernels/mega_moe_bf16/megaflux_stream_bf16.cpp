#define DEEPEP_BF16_REPLICA_STREAM 1
#define DEEPEP_BF16_REPLICA_STRIPES 1
#define DEEPEP_BF16_REPLICA_PIPELINE 1
#define DEEPEP_BF16_REPLICA 1
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Standalone DeepEP MegaMoe BF16 entry.  The implementation uses DeepEP's
 * routing/dispatch/combine pipeline and only references CATLASS primitives
 * for BF16 MMAD; it does not call the CATCCOS DispatchFFNCombine operator.
 */
#define DEEPEP_BF16_M192_FORWARD 1
#include "kernel_operator.h"
#include "mega_moe_quant_constants.h"
#include "mega_moe_tiling.h"
#include "mega_moe_bf16.h"
#include "ascend_deepep/kernels/launch.h"

#include "mega_moe_replica_mixed.h"
#include "mega_moe_stream_descriptor.h"
using namespace AscendC;
using namespace MegaMoeImpl;

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void megaflux_stream_bf16(
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
    GM_ADDR dispatchOutputGM,
    GM_ADDR forwardTransientWorkspaceGM,
    MegaMoeTilingData tilingData, MegaMoeTilingData homeTiling, MegaMoeStreamDescriptor pipeline)
{
    const auto& descriptor=pipeline.device;
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
    if(!active)topkIds=(GM_ADDR)descriptor.logical;
    MegaMoeBF16 op;
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
        dispatchOutputGM);
    op.EnableDispatchTileReady(forwardTransientWorkspaceGM);
    op.EnableReplica(descriptor.replica);
    if(active)op.EnableReplicaPipeline(pipeline.ready,pipeline.trace,descriptor.epoch);
    if(active)op.EnableStream(pipeline.stripes,pipeline.mode);
    op.Process();
}

extern "C" ASCEND_DEEPEP_EXPORT void megaflux_stream_bf16_kernel_do(
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
    uint8_t *dispatch_output,
    uint8_t *forward_transient_workspace,
    const uint8_t *tiling_host,
    const uint8_t *home_tiling_host,
    uint64_t ffts_addr, MegaMoeStreamDescriptor pipeline)
{
    (void)ffts_addr;
    const auto &tiling =
        *reinterpret_cast<const MegaMoeTilingData *>(tiling_host);
    megaflux_stream_bf16<<<block_dim, nullptr, stream>>>(
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
        dispatch_output,
        forward_transient_workspace,
        tiling,*reinterpret_cast<const MegaMoeTilingData *>(home_tiling_host),pipeline);
}

extern "C" ASCEND_DEEPEP_EXPORT int megaflux_stream_bf16_timer_enabled()
{
#if defined(ENABLE_TIMER)
    return 1;
#else
    return 0;
#endif
}
