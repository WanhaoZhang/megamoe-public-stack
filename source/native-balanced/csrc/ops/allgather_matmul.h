#pragma once

#include <cstddef>

#include <torch/extension.h>

namespace deepep { class ShmemInstance; }

namespace ascend_deepep::ops {

torch::Tensor AllGatherMatmul(
    torch::Tensor a,
    torch::Tensor b,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    deepep::ShmemInstance& shmem,
    void* symmetric_buffer = nullptr);

torch::Tensor AllGatherMatmulOut(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor out,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    deepep::ShmemInstance& shmem,
    void* symmetric_buffer = nullptr);

}  // namespace ascend_deepep::ops
