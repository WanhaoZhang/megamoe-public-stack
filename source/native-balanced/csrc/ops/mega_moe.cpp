#include "csrc/ops/mega_moe.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include <tiling/platform/platform_ascendc.h>

#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"
#include "kernels/mega_moe/include/mega_moe_tiling.h"
#include "kernels/mega_moe/include/mega_moe_tiling_host.h"
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"

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
    TORCH_CHECK(
        tensor.element_size() == 1,
        name,
        " must use a one-byte MXFP8 storage dtype");
}

bool ResolveCompactTokenCountExchange(
    bool bf16DirectDispatch,
    int64_t worldSize,
    int64_t expertPerRank)
{
    if (!bf16DirectDispatch) {
        return false;
    }
    const char *rawMode = std::getenv(
        "MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE");
    const std::string mode = rawMode == nullptr || rawMode[0] == '\0'
        ? "auto" : rawMode;
    if (mode == "auto") {
        return MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(
            static_cast<uint64_t>(worldSize),
            static_cast<uint64_t>(expertPerRank));
    }
    if (mode == "dense") {
        return false;
    }
    TORCH_CHECK(
        mode == "compact",
        "MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE must be auto, dense or compact; got '",
        mode, "'");
    TORCH_CHECK(
        MegaMoeBF16Training::CanUseCompactTokenCountExchange(
            static_cast<uint64_t>(worldSize),
            static_cast<uint64_t>(expertPerRank)),
        "forced compact BF16 token-count exchange requires 2..1024 EP ranks, "
        "1..8 local experts");
    return true;
}

}  // namespace

namespace ascend_deepep::ops::detail {

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
    const std::string &save_gmm1_output) {
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aic_num = platform->GetCoreNumAic();
    const uint32_t aiv_num = platform->GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    TORCH_CHECK(
        aic_num > 0 && aiv_num > 0 && ub_size > 0,
        "failed to query AscendC platform resources");
    // Ordinary BF16 forward and backward read saved/direct source offsets.
    // UDMA and legacy kernels still require their bookkeeping guard.
    TORCH_CHECK(!bf16_direct_dispatch || bf16,
        "direct Dispatch offsets require a BF16 kernel");
    TORCH_CHECK(
        bf16_direct_dispatch || world_size <= static_cast<int64_t>(aic_num) * 10,
        "world_size exceeds the kernel dispatch bookkeeping capacity for ",
        aic_num,
        " AIC cores");

    if (bf16_direct_dispatch) {
        // Backward consumes state from the matching ordinary forward, so
        // keep its forward-capacity checks too. Backward also checks its own
        // phase-by-phase UB footprint in MegaMoeBF16Backward().
        const MegaMoeBF16Training::ForwardDispatchUbLayout dispatchUb(
            world_size, expert_per_rank, k, 2);
        TORCH_CHECK(dispatchUb.totalBytes <= ub_size,
            "MegaMoe BF16 Dispatch requires ", dispatchUb.totalBytes,
            " UB bytes, but the device has ", ub_size);
        TORCH_CHECK(MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes <= ub_size,
            "MegaMoe BF16 count reset chunk exceeds device UB capacity");
    }

    const float normalizedSwigluLimit = swiglu_limit > 0.0
        ? static_cast<float>(swiglu_limit) : kMegaMoeSwigluLimitDisabled;
    using namespace MegaMoeBF16Training;
    uint32_t saveMode = kSaveAuto;
    if (save_gmm1_output == "none") saveMode = kSaveNone;
    else if (save_gmm1_output == "bf16") saveMode = kSaveBf16;
    else if (save_gmm1_output == "mxfp8") saveMode = kSaveMxFp8;
    else TORCH_CHECK(save_gmm1_output == "auto",
        "save_gmm1_output must be auto, none, bf16 or mxfp8");
    TORCH_CHECK(bf16 || saveMode == kSaveAuto, "save_gmm1_output is a BF16-only option");
    saveMode = ResolveSavedGateUpMode(saveMode, normalizedSwigluLimit);
    if (bf16 && saveMode == kSaveMxFp8) {
        const uint64_t quantUbBytes = SaveGateUpClipMasks(normalizedSwigluLimit, saveMode)
            ? SavedGateUpQuantUbLayout::totalBytesWithClipMasks : SavedGateUpQuantUbLayout::totalBytes;
        const uint64_t ubBudget = std::min<uint64_t>(ub_size, 248ULL * 1024ULL);
        TORCH_CHECK(quantUbBytes <= ubBudget,
            "MegaMoe BF16 saved MXFP8 epilogue requires ",
            quantUbBytes, " UB bytes, have ", ubBudget);
        TORCH_CHECK((n / 2) % 32 == 0, "saved MXFP8 requires F aligned to 32");
    }

    const uint32_t blocks_per_ep =
        std::max<uint32_t>(1, aic_num / world_size);
    bool ok = false;
    if (bf16) {
        ok = optiling::SetInitTilingData<
            uint16_t,
            DISPATCH_QUANT_MODE_NO_QUANT>(
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
            2,
            1,
            aiv_num,
            ub_size);
    } else {
        ok = optiling::SetInitTilingData<
            uint16_t,
            DISPATCH_QUANT_OUT_TYPE_E4M3FN>(
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
            2,
            1,
            aiv_num,
            ub_size);
    }
    TORCH_CHECK(
        ok,
        bf16 ? "failed to generate MegaMoe BF16 tiling data"
             : "failed to generate MegaMoe tiling data");
    tiling.swigluLimit = normalizedSwigluLimit;
    tiling.savedGateUpMode = bf16 ? saveMode : kSaveAuto;
    tiling.compactTokenCountExchange = ResolveCompactTokenCountExchange(
        bf16_direct_dispatch, world_size, expert_per_rank) ? 1U : 0U;
    const auto workspace_info = optiling::CalcWorkspaceSize(tiling);
    const auto peer_info = optiling::CalcPeermemSize(tiling);
    uint64_t peerBytes = static_cast<uint64_t>(peer_info.totalSize);
    if (bf16_direct_dispatch) {
        // CalcPeermemSize retains the 60 KiB legacy header used by the
        // non-BF16 and experimental variants. Ordinary BF16 uses its 72 KiB
        // synchronization header and, at large EP, an aligned compact-count
        // inbox before the unchanged dense token-count table.
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
            static_cast<uint64_t>(world_size),
            static_cast<uint64_t>(expert_per_rank),
            tiling.compactTokenCountExchange != 0);
        peerBytes += countLayout.tokenCountsOffset -
            MegaMoeBF16Training::kPeerBaseDataOffset;
    }
    return {
        static_cast<uint64_t>(workspace_info.totalSize),
        peerBytes};
}

MegaMoeHostTiling GetBf16EarlyTopkWeightBufferBytes(
    const MegaMoeTilingData& tiling) {
    return {
        static_cast<uint64_t>(
            optiling::CalcBf16EarlyTopkWeightWorkspaceSize(tiling)),
        static_cast<uint64_t>(
            optiling::CalcBf16EarlyTopkWeightPeermemSize(tiling))};
}

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
    TORCH_CHECK(
        topk_ids.scalar_type() == torch::kInt32,
        "topk_ids must be int32");
    TORCH_CHECK(
        topk_weights.scalar_type() == torch::kBFloat16,
        "topk_weights must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(
        topk_ids.dim() == 2,
        "topk_ids must have shape [m, topk]");
    TORCH_CHECK(
        topk_weights.sizes() == topk_ids.sizes(),
        "topk_weights shape must match topk_ids");
    TORCH_CHECK(
        x.size(0) == topk_ids.size(0),
        "x and topk_ids must have the same token count");
    TORCH_CHECK(
        weight1.dim() == 3,
        "weight1 must have shape [local_experts, n, k]");
    TORCH_CHECK(
        weight2.dim() == 3,
        "weight2 must have shape [local_experts, k, n / 2]");
    TORCH_CHECK(
        weight_scales1.dim() == 4,
        "weight_scales1 must have shape "
        "[local_experts, n, ceil(k / 64), 2]");
    TORCH_CHECK(
        weight_scales2.dim() == 4,
        "weight_scales2 must have shape "
        "[local_experts, k, ceil((n / 2) / 64), 2]");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topk_ids.size(1);
    const int64_t expert_per_rank = weight1.size(0);
    const int64_t n = weight1.size(1);
    TORCH_CHECK(
        m > 0 && k > 0 && topk > 0 && expert_per_rank > 0 && n > 0,
        "all MegaMoe dimensions must be positive");
    TORCH_CHECK(world_size >= 2, "MegaMoe requires world_size >= 2");
    TORCH_CHECK(world_size <= 768, "MegaMoe supports at most 768 EP ranks");
    TORCH_CHECK(m <= 16384, "MegaMoe supports at most 16384 tokens per rank");
    TORCH_CHECK(
        expert_per_rank <= 16,
        "MegaMoe supports at most 16 local experts");
    TORCH_CHECK(
        expert_per_rank * world_size <= 1024,
        "the total number of experts must not exceed 1024");
    TORCH_CHECK(max_recv_tokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(
        k % 64 == 0 && n % 128 == 0,
        "k must be divisible by 64 and weight1.shape[1] by 128");
    TORCH_CHECK(
        weight1.size(2) == k,
        "weight1.shape[2] must equal x.shape[1]");
    TORCH_CHECK(
        weight2.size(0) == expert_per_rank &&
            weight2.size(1) == k &&
            weight2.size(2) * 2 == n,
        "weight2 must have shape "
        "[local_experts, k, weight1.shape[1] / 2]");
    TORCH_CHECK(
        weight_scales1.size(0) == expert_per_rank &&
            weight_scales1.size(1) == n &&
            weight_scales1.size(2) == (k + 63) / 64 &&
            weight_scales1.size(3) == 2,
        "weight_scales1 has an invalid MXFP scale layout");
    TORCH_CHECK(
        weight_scales2.size(0) == expert_per_rank &&
            weight_scales2.size(1) == k &&
            weight_scales2.size(2) == (n / 2 + 63) / 64 &&
            weight_scales2.size(3) == 2,
        "weight_scales2 has an invalid MXFP scale layout");

    MegaMoeTilingData tiling{};
    const auto host_tiling = detail::MakeMegaMoeHostTiling(
        tiling,
        m,
        k,
        n,
        topk,
        world_size,
        expert_per_rank,
        max_recv_tokens,
        rank,
        false,
        0.0,
        false);
    TORCH_CHECK(
        host_tiling.peer_bytes <= symmetric_buffer_bytes,
        "MegaMoe local_mem_size is too small: need ",
        host_tiling.peer_bytes,
        " bytes, got ",
        symmetric_buffer_bytes);

    auto y = torch::zeros({m, k}, x.options());
    auto expert_token_nums = torch::zeros(
        {expert_per_rank},
        topk_ids.options().dtype(torch::kInt32));
    auto workspace = torch::zeros(
        {static_cast<int64_t>(host_tiling.workspace_bytes)},
        x.options().dtype(torch::kUInt8));
    auto context = torch::empty(
        {static_cast<int64_t>(
            sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025)},
        x.options().dtype(torch::kUInt8));

    LaunchMegaMoeKernel(
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
            platform_ascendc::PlatformAscendCManager::GetInstance()
                ->GetCoreNumAic()),
        symmetric_buffer,
        symmetric_buffer_bytes,
        rank,
        world_size,
        shmem.FftsAddr());
    return {y, expert_token_nums};
}

}  // namespace ascend_deepep::ops
