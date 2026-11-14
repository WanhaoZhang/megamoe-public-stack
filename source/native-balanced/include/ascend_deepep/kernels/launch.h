#pragma once

#include <cstdint>

struct MegaMoeFullDescriptor;
struct MegaMoeSharedExpertDescriptor;
namespace MegaMoeSharedExpert { struct SharedExpertBackwardArgsV2; struct SharedExpertBackwardArgsV3; }

#if defined(__GNUC__)
#define ASCEND_DEEPEP_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_DEEPEP_EXPORT
#endif

extern "C" {

// V2 is separate from the packed-GU ABI1 worker entry.
ASCEND_DEEPEP_EXPORT void megaflux_worker_bf16_backward_shared_v2(
    uint32_t block_dim, void* stream,
    uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* grad_x,
    uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output,
    uint8_t* backward_workspace, uint8_t* timer,
    const uint8_t* tiling_host, const uint8_t* home_tiling_host,
    uint64_t ffts_addr, MegaMoeFullDescriptor pipeline, uint32_t split_experts,
    MegaMoeSharedExpert::SharedExpertBackwardArgsV2 shared);

// Original-X redispatch is an independent opt-in ABI; V2 remains unchanged.
ASCEND_DEEPEP_EXPORT void megaflux_worker_bf16_backward_shared_v3(
    uint32_t block_dim, void* stream,
    uint8_t* context, uint8_t* grad_y, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* grad_x,
    uint8_t* grad_topk_weights, uint8_t* grad_weight1, uint8_t* grad_weight2,
    uint8_t* training_saved_state, uint8_t* dispatch_output,
    uint8_t* backward_workspace, uint8_t* timer,
    const uint8_t* tiling_host, const uint8_t* home_tiling_host,
    uint64_t ffts_addr, MegaMoeFullDescriptor pipeline, uint32_t split_experts,
    MegaMoeSharedExpert::SharedExpertBackwardArgsV3 shared);

ASCEND_DEEPEP_EXPORT void dispatch_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_idx,
    uint8_t* topk_weights,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_idx,
    uint8_t* recv_topk_weights,
    uint8_t* handle,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void combine_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* expert_output,
    uint8_t* handle,
    uint8_t* topk_weights,
    uint8_t* symmetric_buffer,
    uint8_t* out,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void allgather_matmul_kernel_do(
    void* stream,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* symmetric_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void mega_moe_kernel_do(
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
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void mega_moe_w4a8_kernel_do(
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
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void mega_moe_bf16_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* context,
    uint8_t* x,
    uint8_t* topk_ids,
    uint8_t* topk_weights,
    uint8_t* weight1,
    uint8_t* weight2,
    uint8_t* timer,
    uint8_t* y,
    uint8_t* expert_token_nums,
    uint8_t* training_saved_state,
    uint8_t* dispatch_output,
    uint8_t* forward_transient_workspace,
    const uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void mega_moe_bf16_backward_ready_k_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* context,
    uint8_t* grad_y,
    uint8_t* topk_weights,
    uint8_t* weight1,
    uint8_t* weight2,
    uint8_t* grad_x,
    uint8_t* grad_topk_weights,
    uint8_t* grad_weight1,
    uint8_t* grad_weight2,
    uint8_t* training_saved_state,
    uint8_t* dispatch_output,
    uint8_t* backward_workspace,
    uint8_t* timer,
    const uint8_t* tiling_host,
    uint64_t ffts_addr, uint32_t split_experts);

// Independent forward-only shared ABI. The existing S9 entry is unchanged.
ASCEND_DEEPEP_EXPORT void megaflux_early_bf16_shared_kernel_do(
    uint32_t block_dim, void* stream,
    uint8_t* context, uint8_t* x, uint8_t* topk_ids, uint8_t* topk_weights,
    uint8_t* weight1, uint8_t* weight2, uint8_t* timer, uint8_t* y,
    uint8_t* expert_token_nums, uint8_t* training_saved_state,
    uint8_t* dispatch_output, uint8_t* forward_transient_workspace,
    const uint8_t* tiling_host, const uint8_t* home_tiling_host,
    uint64_t ffts_addr, MegaMoeFullDescriptor pipeline,
    MegaMoeSharedExpertDescriptor shared);

ASCEND_DEEPEP_EXPORT void mega_moe_bf16_udma_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* context,
    uint8_t* x,
    uint8_t* topk_ids,
    uint8_t* topk_weights,
    uint8_t* weight1,
    uint8_t* weight2,
    uint8_t* timer,
    uint8_t* y,
    uint8_t* expert_token_nums,
    uint8_t* training_saved_state,
    uint8_t* forward_transient_workspace,
    const uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT void mega_moe_bf16_mgroup_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* context,
    uint8_t* x,
    uint8_t* topk_ids,
    uint8_t* topk_weights,
    uint8_t* weight1,
    uint8_t* weight2,
    uint8_t* timer,
    uint8_t* y,
    uint8_t* expert_token_nums,
    uint8_t* training_saved_state,
    uint8_t* forward_transient_workspace,
    const uint8_t* tiling_host,
    uint64_t ffts_addr);

ASCEND_DEEPEP_EXPORT int mega_moe_bf16_timer_enabled();
ASCEND_DEEPEP_EXPORT int mega_moe_bf16_udma_timer_enabled();
ASCEND_DEEPEP_EXPORT int mega_moe_bf16_mgroup_timer_enabled();
ASCEND_DEEPEP_EXPORT int mega_moe_timer_enabled();

}  // extern "C"
