#include <vector>
#pragma once

#include <cstddef>
#include <cstdint>

#include <torch/extension.h>

#include "ascend_deepep/kernels/types.h"

struct MegaMoeTilingData;
struct MegaMoeW4A8TilingData;
namespace MegaMoeSharedExpert { struct SharedExpertBackwardArgs; struct SharedExpertBackwardArgsV2; struct SharedExpertBackwardArgsV3; }

void LaunchDispatchKernel(
    torch::Tensor x,
    torch::Tensor topk_idx,
    const c10::optional<torch::Tensor>& topk_weights,
    torch::Tensor recv_x,
    torch::Tensor recv_topk_idx,
    const c10::optional<torch::Tensor>& recv_topk_weights,
    torch::Tensor handle,
    const ascend_deepep::DispatchTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr);

void LaunchCombineKernel(
    torch::Tensor expert_output,
    torch::Tensor handle,
    const c10::optional<torch::Tensor>& topk_weights,
    torch::Tensor out,
    const ascend_deepep::CombineTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr);

void LaunchAllGatherMatmulKernel(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    const ascend_deepep::AllGatherMatmulTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr);

void LaunchMegaMoeKernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor weight_scales1,
    torch::Tensor weight_scales2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr);

void LaunchMegaMoeW4A8Kernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor weight_scales1,
    torch::Tensor weight_scales2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor workspace,
    const MegaMoeW4A8TilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr);

void LaunchMegaMoeBF16Kernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor dispatch_output,
    torch::Tensor forward_transient_workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr,
    const std::vector<int64_t>& replica = {});

bool MegaMoeBF16SwigluDecoupled(uint64_t hidden, uint32_t saved_mode);
bool MegaMoeBF16WorkerSwigluDecoupled(uint64_t hidden, uint32_t saved_mode);
bool MegaMoeBF16WorkerGmTileReady();
bool MegaMoeBF16WorkerBackwardNoneGmSupported();
int MegaMoeBF16WorkerSharedForwardTimerSchema();
void CheckMegaMoeBF16RoutedRedispatchCapability(int world);
void CheckMegaMoeBF16SharedWorldSize(int world);
bool MegaMoeBF16WorkerSharedTilePipeline();
void CheckMegaMoeBF16SharedSavedMode(uint32_t saved_mode, bool backward);

void InitializeMegaMoeBF16Context(
    torch::Tensor context,
    void* symmetric_buffer,
    int rank,
    int world_size);

void LaunchMegaMoeBF16MGroupKernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor forward_transient_workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr);

void LaunchMegaMoeBF16BackwardKernel(
    torch::Tensor context,
    torch::Tensor grad_y,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor grad_x,
    torch::Tensor grad_topk_weights,
    torch::Tensor grad_weight1,
    torch::Tensor grad_weight2,
    torch::Tensor training_saved_state,
    torch::Tensor dispatch_output,
    torch::Tensor backward_workspace,
    torch::Tensor timer,
    const MegaMoeTilingData& tiling,
    uint32_t splitExperts,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr,
    const std::vector<int64_t>& replica = {});

void CheckMegaMoeBF16SharedBackwardCapability();
void LaunchMegaMoeBF16BackwardSharedKernel(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
    torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
    torch::Tensor grad_weight2, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, torch::Tensor backward_workspace,
    const MegaMoeTilingData& tiling, uint32_t split_experts, uint32_t block_dim,
    void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
    int rank, int world_size, uint64_t ffts_addr,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgs& shared);

void CheckMegaMoeBF16SharedBackwardCapabilityV2(bool timer);
void LaunchMegaMoeBF16BackwardSharedKernelV2(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
    torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
    torch::Tensor grad_weight2, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, torch::Tensor backward_workspace, torch::Tensor timer,
    const MegaMoeTilingData& tiling, uint32_t split_experts, uint32_t block_dim,
    void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
    int rank, int world_size, uint64_t ffts_addr,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgsV2& shared);

void CheckMegaMoeBF16SharedBackwardCapabilityV3(bool timer);
void LaunchMegaMoeBF16BackwardSharedKernelV3(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
    torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
    torch::Tensor grad_weight2, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, torch::Tensor backward_workspace, torch::Tensor timer,
    const MegaMoeTilingData& tiling, uint32_t split_experts, uint32_t block_dim,
    void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
    int rank, int world_size, uint64_t ffts_addr,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgsV3& shared);

void LaunchMegaMoeBF16UDMAKernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor forward_transient_workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr);
