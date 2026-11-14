#include "csrc/ops/combine.h"

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

std::tuple<torch::Tensor> Combine(
    torch::Tensor expert_output,
    torch::Tensor handle,
    c10::optional<torch::Tensor> topk_weights,
    int rank,
    int world_size,
    ShmemInstance& shmem,
    void* symmetric_buffer) {
    CheckNpuContiguous(expert_output, "expert_output");
    CheckNpuContiguous(handle, "handle");
    TORCH_CHECK(expert_output.dim() == 2, "expert_output must have shape [num_recv_tokens, hidden]");
    TORCH_CHECK(handle.dim() == 3 && handle.size(2) == 4, "handle must have shape [num_tokens, num_topk, 4]");
    TORCH_CHECK(expert_output.scalar_type() == torch::kFloat16, "the baseline kernel supports float16 expert_output");
    TORCH_CHECK(handle.scalar_type() == torch::kInt32, "handle must be int32");

    const int64_t num_tokens = handle.size(0);
    const int64_t num_topk = handle.size(1);
    const int64_t hidden = expert_output.size(1);

    if (topk_weights.has_value()) {
        CheckNpuContiguous(*topk_weights, "topk_weights");
        TORCH_CHECK(
            topk_weights->dim() == 2 &&
                topk_weights->size(0) == num_tokens &&
                topk_weights->size(1) == num_topk,
            "topk_weights must have shape [num_tokens, num_topk]");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
    }

    auto out = torch::zeros({num_tokens, hidden}, expert_output.options());

    ascend_deepep::CombineTilingData tiling{};
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.num_tokens = static_cast<uint32_t>(num_tokens);
    tiling.hidden = static_cast<uint32_t>(hidden);
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.block_dim = SelectBlockDim(num_tokens);
    tiling.has_topk_weights = topk_weights.has_value() ? 1U : 0U;
    tiling.reserved = 0;

    LaunchCombineKernel(
        expert_output,
        handle,
        topk_weights,
        out,
        tiling,
        symmetric_buffer,
        shmem.FftsAddr());

    return {out};
}

}  // namespace ascend_deepep::ops
