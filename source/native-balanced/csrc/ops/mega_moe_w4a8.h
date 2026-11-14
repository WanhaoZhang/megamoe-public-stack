#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include <torch/extension.h>

namespace deepep { class ShmemInstance; }

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, torch::Tensor> MegaMoeW4A8(
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor weight_scales1,
    torch::Tensor weight_scales2,
    int64_t max_recv_tokens,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    deepep::ShmemInstance& shmem,
    void* symmetric_buffer,
    torch::Tensor timer);

}  // namespace ascend_deepep::ops
