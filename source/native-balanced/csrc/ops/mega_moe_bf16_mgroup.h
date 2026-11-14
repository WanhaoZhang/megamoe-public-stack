#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <string>

#include <torch/extension.h>

#include "csrc/ops/mega_moe_bf16_result.h"

namespace deepep { class ShmemInstance; }

namespace ascend_deepep::ops {

MegaMoeBF16ForwardResult MegaMoeBF16MGroup(
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    int64_t max_recv_tokens,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    deepep::ShmemInstance &shmem,
    void *symmetric_buffer,
    torch::Tensor timer,
    const std::string &save_gmm1_output = "bf16");

}  // namespace ascend_deepep::ops
