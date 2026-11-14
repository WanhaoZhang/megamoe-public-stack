#include <vector>
#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <string>

#include <torch/extension.h>

#include "csrc/ops/mega_moe_bf16_result.h"

namespace deepep { class ShmemInstance; }
struct MegaMoeTilingData;

namespace ascend_deepep::ops {

MegaMoeBF16ForwardResult MegaMoeBF16(
    torch::Tensor context,
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

// Training forward ABI returns
// (y, expert_token_nums, training_saved_state, dispatch_output,
//  forward_transient_workspace). The framework chooses whether to retain the
// independent dispatch output for backward; transient scratch is never saved.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
MegaMoeBF16Forward(
    torch::Tensor context,
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
    double swiglu_limit = 0.0,
    const std::string &save_gmm1_output = "bf16",
    const std::vector<int64_t>& replica = {});

// Forward-only shared expert ABI. The first five values match Forward;
// raw Gate/Up, activation, fresh readiness storage and optional shared timer
// follow. This entry does not authorize the routed-only backward for its Y.
using MegaMoeBF16SharedForwardResult = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

MegaMoeBF16SharedForwardResult MegaMoeBF16SharedForward(
    torch::Tensor context,
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
    double swiglu_limit,
    const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica,
    torch::Tensor shared_weight_gate,
    torch::Tensor shared_weight_up,
    torch::Tensor shared_weight_down);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor> MegaMoeBF16Backward(
    torch::Tensor context,
    torch::Tensor grad_y,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor dispatch_output,
    int64_t max_recv_tokens,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    deepep::ShmemInstance &shmem,
    void *symmetric_buffer,
    torch::Tensor timer,
    double swiglu_limit = 0.0,
    const std::string &save_gmm1_output = "bf16",
    const std::vector<int64_t>& replica = {});

// Explicit shared-backward ABI; the original routed-only method is unchanged.
// raw_gu is the actual unclipped BF16 forward [M,2*Fs], gate columns first.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
MegaMoeBF16BackwardShared(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state, torch::Tensor dispatch_output,
    int64_t max_recv_tokens, int rank, int world_size,
    std::size_t symmetric_buffer_bytes, deepep::ShmemInstance &shmem,
    void *symmetric_buffer, double swiglu_limit,
    const std::string &save_gmm1_output, const std::vector<int64_t>& replica,
    torch::Tensor x, torch::Tensor shared_wg, torch::Tensor shared_wu,
    torch::Tensor shared_wd, torch::Tensor raw_gu, torch::Tensor activation,
    double shared_limit);

// V2 uses independent planes, optional shared-only recomputation and raw Timer.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
MegaMoeBF16BackwardSharedV2(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state, torch::Tensor dispatch_output,
    int64_t max_recv_tokens, int rank, int world_size,
    std::size_t symmetric_buffer_bytes, deepep::ShmemInstance &shmem,
    void *symmetric_buffer, double swiglu_limit,
    const std::string &save_gmm1_output, const std::vector<int64_t>& replica,
    torch::Tensor x, torch::Tensor shared_wg, torch::Tensor shared_wu,
    torch::Tensor shared_wd, torch::Tensor raw_gate, torch::Tensor raw_up,
    torch::Tensor activation, double shared_limit, bool recompute,
    torch::Tensor timer = torch::Tensor(), torch::Tensor shared_timer = torch::Tensor());

// V3 reconstructs packed X/prob inside the shared backward kernel. The caller
// passes an empty NPU UInt8 dispatch tensor. The eighth result owns the newly
// allocated packed buffer through stream completion and is diagnostic-readable.
using MegaMoeBF16SharedBackwardV3Result = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
MegaMoeBF16SharedBackwardV3Result MegaMoeBF16BackwardSharedV3(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state, torch::Tensor dispatch_output,
    int64_t max_recv_tokens, int rank, int world_size,
    std::size_t symmetric_buffer_bytes, deepep::ShmemInstance &shmem,
    void *symmetric_buffer, double swiglu_limit,
    const std::string &save_gmm1_output, const std::vector<int64_t>& replica,
    torch::Tensor x, torch::Tensor shared_wg, torch::Tensor shared_wu,
    torch::Tensor shared_wd, torch::Tensor raw_gate, torch::Tensor raw_up,
    torch::Tensor activation, double shared_limit, bool recompute,
    torch::Tensor timer = torch::Tensor(), torch::Tensor shared_timer = torch::Tensor(),
    torch::Tensor redispatch_timer = torch::Tensor());

// Routed-only original-X redispatch: no shared weights or shared mathematics.
// Returns dx, dp, dw1, dw2 and the freshly reconstructed packed X/prob buffer.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MegaMoeBF16BackwardRedispatch(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state, int64_t max_recv_tokens,
    int rank, int world_size, std::size_t symmetric_buffer_bytes,
    deepep::ShmemInstance &shmem, void *symmetric_buffer, double swiglu_limit,
    const std::string &save_gmm1_output, const std::vector<int64_t>& replica,
    torch::Tensor original_x, torch::Tensor timer = torch::Tensor(),
    torch::Tensor redispatch_timer = torch::Tensor());

}  // namespace ascend_deepep::ops
