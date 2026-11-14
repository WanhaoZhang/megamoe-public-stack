#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <string>

#include <torch/extension.h>

namespace deepep { class ShmemInstance; }
struct MegaMoeTilingData;

namespace ascend_deepep::ops::detail {

struct MegaMoeHostTiling {
    uint64_t workspace_bytes;
    uint64_t peer_bytes;
};

// Spell out both trailing arguments so old bool-only calls cannot silently
// become swiglu_limit=1.0 after a merge.
MegaMoeHostTiling MakeMegaMoeHostTiling(
    MegaMoeTilingData& tiling,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t topk,
    int64_t world_size,
    int64_t expert_per_rank,
    int64_t max_recv_tokens,
    int rank,
    bool bf16,
    double swiglu_limit,
    bool bf16_direct_dispatch,
    const std::string &save_gmm1_output = "auto");

MegaMoeHostTiling GetBf16EarlyTopkWeightBufferBytes(
    const MegaMoeTilingData& tiling);

}  // namespace ascend_deepep::ops::detail

namespace ascend_deepep::ops {

std::tuple<torch::Tensor, torch::Tensor> MegaMoe(
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
