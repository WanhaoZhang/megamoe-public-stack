#include "csrc/ops/mega_moe_w4a8.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <string>
#include <type_traits>

#include <tiling/platform/platform_ascendc.h>

#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"
#include "kernels/mega_moe_w4a8/include/mega_moe_tiling.h"
#include "kernels/mega_moe_w4a8/include/mega_moe_tiling_host.h"

namespace {

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckByteTensor(const torch::Tensor& tensor, const char* name) {
    CheckNpuContiguous(tensor, name);
    TORCH_CHECK(tensor.element_size() == 1, name, " must use byte storage");
}

MegaMoeW4A8TilingData MakeTiling(
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t topk,
    int64_t world_size,
    int64_t expert_per_rank,
    int64_t max_recv_tokens,
    int rank) {
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aic_num = platform->GetCoreNumAic();
    const uint32_t aiv_num = platform->GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    TORCH_CHECK(
        aic_num > 0 && aiv_num > 0 && ub_size > 0,
        "failed to query AscendC platform resources for MegaMoe W4A8");

    MegaMoeW4A8TilingData tiling{};
    // Preserve the standalone W4A8 tiling value. Unlike W8A8, this kernel's
    // dispatch loop emits one completion per source rank and its GMM1 wait
    // target is world_size, so this field does not multiply the wait target.
    constexpr uint32_t blocks_per_ep = 2;
    const bool ok = optiling_w4a8::SetInitTilingData<
        uint16_t, DISPATCH_QUANT_OUT_TYPE_E4M3FN>(
        tiling,
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(k),
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(topk),
        static_cast<uint32_t>(world_size),
        static_cast<uint32_t>(expert_per_rank),
        blocks_per_ep,
        static_cast<uint32_t>(max_recv_tokens),
        static_cast<uint32_t>(rank),
        16,
        0,
        aiv_num,
        ub_size);
    TORCH_CHECK(ok, "failed to generate MegaMoe W4A8 tiling data");
    return tiling;
}

}  // namespace

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
    ShmemInstance& shmem,
    void* symmetric_buffer,
    torch::Tensor timer) {
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(topk_ids, "topk_ids");
    CheckNpuContiguous(topk_weights, "topk_weights");
    CheckByteTensor(weight1, "weight1");
    CheckByteTensor(weight2, "weight2");
    CheckByteTensor(weight_scales1, "weight_scales1");
    CheckByteTensor(weight_scales2, "weight_scales2");

    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(
        topk_weights.scalar_type() == torch::kBFloat16,
        "topk_weights must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must have shape [m, topk]");
    TORCH_CHECK(
        topk_weights.sizes() == topk_ids.sizes(),
        "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topk_ids.size(0), "x and topk_ids token counts differ");
    TORCH_CHECK(
        weight_scales1.dim() == 4 && weight_scales2.dim() == 4,
        "W4A8 weight scales must be four-dimensional physical-layout tensors");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topk_ids.size(1);
    const int64_t expert_per_rank = weight_scales1.size(0);
    const int64_t n = weight_scales1.size(1);
    TORCH_CHECK(
        m > 0 && k > 0 && n > 0 && topk > 0 && expert_per_rank > 0,
        "all MegaMoe W4A8 dimensions must be positive");
    TORCH_CHECK(world_size >= 2, "MegaMoe W4A8 requires world_size >= 2");
    TORCH_CHECK(world_size <= 1024, "MegaMoe W4A8 supports at most 1024 EP ranks");
    TORCH_CHECK(expert_per_rank <= 64, "MegaMoe W4A8 supports at most 64 local experts");
    TORCH_CHECK(
        expert_per_rank * world_size <= 1024,
        "MegaMoe W4A8 total expert count must not exceed 1024");
    TORCH_CHECK(max_recv_tokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(
        k % 64 == 0 && n % 128 == 0,
        "MegaMoe W4A8 requires k % 64 == 0 and n % 128 == 0");

    TORCH_CHECK(
        weight_scales1.size(0) == expert_per_rank &&
            weight_scales1.size(1) == n &&
            weight_scales1.size(2) == k / 64 &&
            weight_scales1.size(3) == 2,
        "weight_scales1 must have shape [experts, n, k / 64, 2]");
    TORCH_CHECK(
        weight_scales2.size(0) == expert_per_rank &&
            weight_scales2.size(1) == k &&
            weight_scales2.size(2) == n / 128 &&
            weight_scales2.size(3) == 2,
        "weight_scales2 must have shape [experts, k, n / 128, 2]");
    TORCH_CHECK(
        weight1.numel() == expert_per_rank * n * k / 2,
        "weight1 must contain packed+nZ FP4 storage of experts*n*k/2 bytes");
    TORCH_CHECK(
        weight2.numel() == expert_per_rank * k * n / 4,
        "weight2 must contain packed+nZ FP4 storage of experts*k*n/4 bytes");

    const auto tiling = MakeTiling(
        m, k, n, topk, world_size, expert_per_rank, max_recv_tokens, rank);
    const auto workspace_info = optiling_w4a8::CalcWorkspaceSize(tiling);
    const auto peer_info = optiling_w4a8::CalcPeermemSize(tiling);
    TORCH_CHECK(
        peer_info.totalSize <= symmetric_buffer_bytes,
        "MegaMoe W4A8 local_mem_size is too small: need ",
        peer_info.totalSize,
        " bytes, got ",
        symmetric_buffer_bytes);

    auto y = torch::zeros({m, k}, x.options());
    auto expert_token_nums = torch::zeros(
        {expert_per_rank}, topk_ids.options().dtype(torch::kInt32));
    auto workspace = torch::zeros(
        {static_cast<int64_t>(workspace_info.totalSize)},
        x.options().dtype(torch::kUInt8));
    auto context = torch::empty({8208}, x.options().dtype(torch::kUInt8));

    LaunchMegaMoeW4A8Kernel(
        context,
        x,
        topk_ids,
        topk_weights,
        weight1,
        weight2,
        weight_scales1,
        weight_scales2,
        timer,
        y,
        expert_token_nums,
        workspace,
        tiling,
        static_cast<uint32_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
        symmetric_buffer,
        symmetric_buffer_bytes,
        rank,
        world_size,
        shmem.FftsAddr());
    return {y, expert_token_nums};
}

}  // namespace ascend_deepep::ops
