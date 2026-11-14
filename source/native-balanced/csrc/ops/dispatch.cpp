#include "csrc/ops/dispatch.h"

#include <algorithm>
#include <cstdint>

#include "ascend_deepep/kernels/types.h"
#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"

namespace {

uint32_t SelectBlockDim(int64_t work_items) {
    if (work_items <= 0) {
        return 1;
    }
    return static_cast<uint32_t>(std::min<int64_t>(48, work_items));
}

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, torch::Tensor> Dispatch(
    torch::Tensor x,
    torch::Tensor topk_idx,
    c10::optional<torch::Tensor> topk_weights,
    int64_t num_experts,
    int64_t num_max_tokens_per_rank,
    int64_t expert_alignment,
    int rank,
    int world_size,
    ShmemInstance& shmem,
    void* symmetric_buffer) {
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(topk_idx, "topk_idx");
    TORCH_CHECK(x.dim() == 2, "x must have shape [num_tokens, hidden]");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must have shape [num_tokens, num_topk]");
    TORCH_CHECK(x.size(0) == topk_idx.size(0), "x and topk_idx must have the same num_tokens");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "the baseline kernel currently supports float16 x");
    TORCH_CHECK(topk_idx.scalar_type() == torch::kInt32, "topk_idx must be int32");
    TORCH_CHECK(num_experts > 0, "num_experts must be greater than zero");
    TORCH_CHECK(num_experts % world_size == 0, "num_experts must be divisible by world_size");
    TORCH_CHECK(expert_alignment > 0, "expert_alignment must be greater than zero");

    const int64_t num_tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t num_topk = topk_idx.size(1);
    const int64_t route_count = num_tokens * num_topk;
    TORCH_CHECK(
        num_max_tokens_per_rank >= route_count,
        "num_max_tokens_per_rank must be at least num_tokens * num_topk for the baseline layout");

    if (topk_weights.has_value()) {
        CheckNpuContiguous(*topk_weights, "topk_weights");
        TORCH_CHECK(topk_weights->sizes() == topk_idx.sizes(), "topk_weights shape must match topk_idx");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
    }

    auto recv_x = torch::zeros({num_max_tokens_per_rank, hidden}, x.options());
    auto recv_topk_idx = torch::full(
        {num_max_tokens_per_rank, num_topk},
        -1,
        topk_idx.options());

    c10::optional<torch::Tensor> recv_topk_weights = c10::nullopt;
    if (topk_weights.has_value()) {
        recv_topk_weights = torch::zeros(
            {num_max_tokens_per_rank, num_topk},
            topk_weights->options());
    }

    auto handle = torch::full(
        {num_tokens, num_topk, 4},
        -1,
        topk_idx.options().dtype(torch::kInt32));

    ascend_deepep::DispatchTilingData tiling{};
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.num_tokens = static_cast<uint32_t>(num_tokens);
    tiling.hidden = static_cast<uint32_t>(hidden);
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.num_experts = static_cast<uint32_t>(num_experts);
    tiling.num_max_tokens_per_rank = static_cast<uint32_t>(num_max_tokens_per_rank);
    tiling.expert_alignment = static_cast<uint32_t>(expert_alignment);
    tiling.block_dim = SelectBlockDim(route_count);
    tiling.has_topk_weights = topk_weights.has_value() ? 1U : 0U;
    tiling.reserved = 0;

    LaunchDispatchKernel(
        x,
        topk_idx,
        topk_weights,
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        handle,
        tiling,
        symmetric_buffer,
        shmem.FftsAddr());

    return {recv_x, recv_topk_idx, recv_topk_weights, handle};
}

}  // namespace ascend_deepep::ops
