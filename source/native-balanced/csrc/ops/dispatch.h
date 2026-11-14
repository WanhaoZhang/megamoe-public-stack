#pragma once

#include <cstdint>
#include <tuple>

#include <torch/extension.h>

namespace deepep { class ShmemInstance; }

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
    deepep::ShmemInstance& shmem,
    void* symmetric_buffer = nullptr);

}  // namespace ascend_deepep::ops
