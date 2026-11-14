#pragma once

#include <tuple>

#include <torch/extension.h>

namespace deepep { class ShmemInstance; }

namespace ascend_deepep::ops {

std::tuple<torch::Tensor> Combine(
    torch::Tensor expert_output,
    torch::Tensor handle,
    c10::optional<torch::Tensor> topk_weights,
    int rank,
    int world_size,
    deepep::ShmemInstance& shmem,
    void* symmetric_buffer = nullptr);

}  // namespace ascend_deepep::ops
