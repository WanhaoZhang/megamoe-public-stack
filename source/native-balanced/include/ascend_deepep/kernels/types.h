#pragma once

#include <cstdint>

namespace ascend_deepep {

struct alignas(8) DispatchTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t expert_alignment;
    uint32_t block_dim;
    uint32_t has_topk_weights;
    uint32_t reserved;
};

struct alignas(8) CombineTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t hidden;
    uint32_t num_topk;
    uint32_t block_dim;
    uint32_t has_topk_weights;
    uint32_t reserved;
};

struct alignas(8) AllGatherMatmulTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t m0;
    uint32_t n0;
    uint32_t k0;
    uint32_t comm_tile_m;
    uint32_t comm_interval;
    uint32_t comm_npu_split;
    uint32_t comm_data_split;
    uint32_t comm_block_m;
    uint32_t block_dim;
    uint32_t reserved;
};

}  // namespace ascend_deepep
