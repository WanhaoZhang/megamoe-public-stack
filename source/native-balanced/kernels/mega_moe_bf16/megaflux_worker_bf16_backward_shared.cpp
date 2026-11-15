/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#define DEEPEP_BF16_WORKER_PIPELINE 1
#if defined(ENABLE_TIMER)
#define DEEPEP_BF16_STAGE_TIMER 1
#endif
#define DEEPEP_BF16_EARLY_REDUCE 1
#define DEEPEP_BF16_GRADIENT_PIPE 1
#define DEEPEP_BF16_REPLICA 1

#include "kernel_operator.h"
#include "mega_moe_quant_constants.h"
#include "mega_moe_tiling.h"
#include "mega_moe_bf16_backward.h"
#include "ascend_deepep/kernels/launch.h"
#include "mega_moe_replica_mixed.h"
#include "mega_moe_full_descriptor.h"
#include "mega_moe_shared_expert_types.h"

using namespace AscendC;
using namespace MegaMoeImpl;


#if !MEGAFLUX_SHARED_EXPERT_BACKWARD
#error "Dedicated shared backward entry requires MEGAFLUX_SHARED_EXPERT_BACKWARD=1"
#endif

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void megaflux_worker_bf16_shared_backward_kernel(
    GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR gradX,
    GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
    GM_ADDR trainingSavedState, GM_ADDR dispatchOutput,
    GM_ADDR backwardWorkspace, GM_ADDR timerAddr,
    MegaMoeTilingData tilingData, MegaMoeTilingData homeTiling,
    MegaMoeFullDescriptor pipeline, uint64_t splitExperts,
    MegaMoeSharedExpert::SharedExpertBackwardArgs shared)
{
    const auto& descriptor=pipeline.forward.device;
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    // Preserve the existing collective planner failure/epoch boundary. Shared
    // work must not begin on a rank whose routed launch was rejected.
    auto header=(__gm__ int32_t*)descriptor.header;
    if(descriptor.abi!=2 || MegaFluxMixed::Fresh(header+3)!=0 ||
       MegaFluxMixed::Fresh(header+2)!=descriptor.epoch ||
       MegaFluxMixed::Fresh(header+1)>descriptor.capacity ||
       MegaFluxMixed::Fresh(header+7)!=descriptor.capacity)return;
    const bool active=MegaFluxMixed::Fresh(header)>0;
    if(!active)tilingData=homeTiling;
    MegaMoeBF16Backward op;
    op.Init(context,gradY,topkWeights,weight1,weight2,
        gradX,gradTopkWeights,gradWeight1,gradWeight2,
        trainingSavedState,dispatchOutput,backwardWorkspace,
        &tilingData,splitExperts,timerAddr,&shared);
    op.EnableReplica(descriptor.replica);
    if(active)op.EnableGradientPipe(pipeline.grad_ready,pipeline.grad_trace,descriptor.epoch);
#if MEGAFLUX_BACKWARD_RESTORE
    if(active)op.EnableBackwardRestore(pipeline.forward.stripes,descriptor.epoch,
        MEGAFLUX_BACKWARD_RESTORE);
#endif
    op.Process();
}

extern "C" ASCEND_DEEPEP_EXPORT void megaflux_worker_bf16_backward_shared(
    uint32_t block_dim, void* stream,
    uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* grad_x,
    uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output,
    uint8_t* backward_workspace, uint8_t* timer,
    const uint8_t* tiling_host, const uint8_t* home_tiling_host,
    uint64_t ffts_addr, MegaMoeFullDescriptor pipeline, uint32_t splitExperts,
    MegaMoeSharedExpert::SharedExpertBackwardArgs shared)
{
    (void)ffts_addr;
    const auto& tiling=*reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    const auto& home=*reinterpret_cast<const MegaMoeTilingData*>(home_tiling_host);
    megaflux_worker_bf16_shared_backward_kernel<<<block_dim,nullptr,stream>>>(
        context,grad_y,topk_weights,weight1,weight2,grad_x,grad_topk_weights,
        grad_weight1,grad_weight2,training_saved_state,dispatch_output,
        backward_workspace,timer,tiling,home,pipeline,splitExperts,shared);
}

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void megaflux_worker_bf16_shared_backward_v2_kernel(
    GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR gradX,
    GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
    GM_ADDR trainingSavedState, GM_ADDR dispatchOutput,
    GM_ADDR backwardWorkspace, GM_ADDR timerAddr,
    MegaMoeTilingData tilingData, MegaMoeTilingData homeTiling,
    MegaMoeFullDescriptor pipeline, uint64_t splitExperts,
    MegaMoeSharedExpert::SharedExpertBackwardArgsV2 shared)
{
    const auto& descriptor=pipeline.forward.device;
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    // Preserve the existing collective planner failure/epoch boundary. Shared
    // work must not begin on a rank whose routed launch was rejected.
    auto header=(__gm__ int32_t*)descriptor.header;
    if(descriptor.abi!=2 || MegaFluxMixed::Fresh(header+3)!=0 ||
       MegaFluxMixed::Fresh(header+2)!=descriptor.epoch ||
       MegaFluxMixed::Fresh(header+1)>descriptor.capacity ||
       MegaFluxMixed::Fresh(header+7)!=descriptor.capacity)return;
    if(shared.version!=MegaMoeSharedExpert::kBackwardVersionV2 || shared.recompute>1 ||
       shared.base.version!=MegaMoeSharedExpert::kBackwardVersion ||
       shared.base.bytes!=sizeof(MegaMoeSharedExpert::SharedExpertBackwardArgs) ||
       shared.base.flags!=0 || !shared.raw_gate || !shared.raw_up)return;
    const bool active=MegaFluxMixed::Fresh(header)>0;
    if(!active)tilingData=homeTiling;
    MegaMoeBF16Backward op;
    op.Init(context,gradY,topkWeights,weight1,weight2,
        gradX,gradTopkWeights,gradWeight1,gradWeight2,
        trainingSavedState,dispatchOutput,backwardWorkspace,
        &tilingData,splitExperts,timerAddr,&shared.base);
    op.EnableReplica(descriptor.replica);
    if(active)op.EnableGradientPipe(pipeline.grad_ready,pipeline.grad_trace,descriptor.epoch);
#if MEGAFLUX_BACKWARD_RESTORE
    if(active)op.EnableBackwardRestore(pipeline.forward.stripes,descriptor.epoch,
        MEGAFLUX_BACKWARD_RESTORE);
#endif
    op.EnableSharedExpertV2(shared);
    op.Process();
}

extern "C" ASCEND_DEEPEP_EXPORT void megaflux_worker_bf16_backward_shared_v2(
    uint32_t block_dim, void* stream,
    uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* grad_x,
    uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output,
    uint8_t* backward_workspace, uint8_t* timer,
    const uint8_t* tiling_host, const uint8_t* home_tiling_host,
    uint64_t ffts_addr, MegaMoeFullDescriptor pipeline, uint32_t splitExperts,
    MegaMoeSharedExpert::SharedExpertBackwardArgsV2 shared)
{
    (void)ffts_addr;
    const auto& tiling=*reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    const auto& home=*reinterpret_cast<const MegaMoeTilingData*>(home_tiling_host);
    megaflux_worker_bf16_shared_backward_v2_kernel<<<block_dim,nullptr,stream>>>(
        context,grad_y,topk_weights,weight1,weight2,grad_x,grad_topk_weights,
        grad_weight1,grad_weight2,training_saved_state,dispatch_output,
        backward_workspace,timer,tiling,home,pipeline,splitExperts,shared);
}

extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_backward_v3_version() { return 3; }
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_routed_redispatch_version() { return 2; }
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_max_ep() { return 64; }
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_redispatch_timer_phases() {
    return MegaMoeSharedExpert::kBackwardRedispatchTimerPhasesV3;
}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_redispatch_timer_lanes() {
    return MegaMoeSharedExpert::kBackwardRedispatchTimerLanesV3;
}
extern "C" ASCEND_DEEPEP_EXPORT uint64_t megaflux_x_redispatch_delay_source_cycles() {
    return static_cast<uint64_t>(MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES);
}
extern "C" ASCEND_DEEPEP_EXPORT uint64_t megaflux_x_redispatch_delay_pull_cycles() {
    return static_cast<uint64_t>(MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES);
}

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void megaflux_worker_bf16_shared_backward_v3_kernel(
    GM_ADDR context, GM_ADDR gradY, GM_ADDR topkWeights,
    GM_ADDR weight1, GM_ADDR weight2, GM_ADDR gradX,
    GM_ADDR gradTopkWeights, GM_ADDR gradWeight1, GM_ADDR gradWeight2,
    GM_ADDR trainingSavedState, GM_ADDR dispatchOutput,
    GM_ADDR backwardWorkspace, GM_ADDR timerAddr,
    MegaMoeTilingData tilingData, MegaMoeTilingData homeTiling,
    MegaMoeFullDescriptor pipeline, uint64_t splitExperts,
    MegaMoeSharedExpert::SharedExpertBackwardArgsV3 shared)
{
    const auto& descriptor = pipeline.forward.device;
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    auto header = reinterpret_cast<__gm__ int32_t*>(descriptor.header);
    if (descriptor.abi != 2 || MegaFluxMixed::Fresh(header + 3) != 0 ||
        MegaFluxMixed::Fresh(header + 2) != descriptor.epoch ||
        MegaFluxMixed::Fresh(header + 1) > descriptor.capacity ||
        MegaFluxMixed::Fresh(header + 7) != descriptor.capacity) {
        return;
    }
    const auto& v2 = shared.shared;
    const bool routedOnly = shared.flags == MegaMoeSharedExpert::kBackwardRoutedOnlyRedispatchFlag;
    if (shared.version != MegaMoeSharedExpert::kBackwardVersionV3 ||
        (!routedOnly && shared.flags != MegaMoeSharedExpert::kBackwardRedispatchFlag) ||
        v2.version != MegaMoeSharedExpert::kBackwardVersionV2 || v2.recompute > 1 ||
        v2.base.version != MegaMoeSharedExpert::kBackwardVersion ||
        v2.base.bytes != sizeof(MegaMoeSharedExpert::SharedExpertBackwardArgs) ||
        v2.base.flags != 0 || (!routedOnly && (!v2.raw_gate || !v2.raw_up)) ||
        (routedOnly && (!v2.base.control || v2.base.epoch == 0)) || !v2.base.x) {
        return;
    }
    const bool active = MegaFluxMixed::Fresh(header) > 0;
    if (!active) tilingData = homeTiling;
    if (!MegaMoeBF16Training::SharedSavedModeBit(
            MegaMoeBF16Training::ResolveSavedGateUpMode(
                tilingData.savedGateUpMode, tilingData.swigluLimit)) ||
        v2.base.m != tilingData.m || v2.base.hidden != tilingData.k) {
        return;
    }
    MegaMoeBF16Backward op;
    op.Init(context, gradY, topkWeights, weight1, weight2, gradX,
        gradTopkWeights, gradWeight1, gradWeight2, trainingSavedState,
        dispatchOutput, backwardWorkspace, &tilingData, splitExperts,
        timerAddr, routedOnly ? nullptr : &v2.base);
    op.EnableReplica(descriptor.replica);
    if (active) op.EnableGradientPipe(pipeline.grad_ready, pipeline.grad_trace, descriptor.epoch);
    if(routedOnly)op.EnableLifetime(pipeline.forward.trace);
#if MEGAFLUX_BACKWARD_RESTORE
    if (active) op.EnableBackwardRestore(pipeline.forward.stripes, descriptor.epoch,
        MEGAFLUX_BACKWARD_RESTORE);
#endif
    if (routedOnly) op.EnableRoutedRedispatch(shared);
    else op.EnableSharedExpertV3(shared);
    op.Process();
}

extern "C" ASCEND_DEEPEP_EXPORT void megaflux_worker_bf16_backward_shared_v3(
    uint32_t block_dim, void* stream,
    uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* grad_x,
    uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output,
    uint8_t* backward_workspace, uint8_t* timer,
    const uint8_t* tiling_host, const uint8_t* home_tiling_host,
    uint64_t ffts_addr, MegaMoeFullDescriptor pipeline, uint32_t splitExperts,
    MegaMoeSharedExpert::SharedExpertBackwardArgsV3 shared)
{
    (void)ffts_addr;
    const auto& tiling = *reinterpret_cast<const MegaMoeTilingData*>(tiling_host);
    const auto& home = *reinterpret_cast<const MegaMoeTilingData*>(home_tiling_host);
    megaflux_worker_bf16_shared_backward_v3_kernel<<<block_dim, nullptr, stream>>>(
        context, grad_y, topk_weights, weight1, weight2, grad_x,
        grad_topk_weights, grad_weight1, grad_weight2, training_saved_state,
        dispatch_output, backward_workspace, timer, tiling, home, pipeline,
        splitExperts, shared);
}
