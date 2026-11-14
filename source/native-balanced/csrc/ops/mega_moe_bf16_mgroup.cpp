#include "csrc/ops/mega_moe_bf16_mgroup.h"
#include "csrc/ops/mega_moe.h"

#include <cstdint>

#include <tiling/platform/platform_ascendc.h>

#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"
#include "kernels/mega_moe/include/mega_moe_tiling.h"
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"

namespace {

void CheckNpuContiguous(const torch::Tensor &tensor, const char *name)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

namespace ascend_deepep::ops {

MegaMoeBF16ForwardResult MegaMoeBF16MGroup(
    torch::Tensor x,
    torch::Tensor topkIds,
    torch::Tensor topkWeights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    int64_t maxRecvTokens,
    int rank,
    int worldSize,
    std::size_t symmetricBufferBytes,
    ShmemInstance &shmem,
    void *symmetricBuffer,
    torch::Tensor timer, const std::string &save_gmm1_output)
{
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(topkIds, "topk_ids");
    CheckNpuContiguous(topkWeights, "topk_weights");
    CheckNpuContiguous(weight1, "weight1");
    CheckNpuContiguous(weight2, "weight2");

    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(topkIds.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16,
                "topk_weights must be bfloat16");
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16,
                "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16,
                "weight2 must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(topkIds.dim() == 2, "topk_ids must have shape [m, topk]");
    TORCH_CHECK(topkWeights.sizes() == topkIds.sizes(),
                "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topkIds.size(0),
                "x and topk_ids must have the same token count");
    TORCH_CHECK(weight1.dim() == 3,
                "weight1 must have shape [local_experts, n, k]");
    TORCH_CHECK(weight2.dim() == 3,
                "weight2 must have shape [local_experts, k, n / 2]");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const int64_t n = weight1.size(1);
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 MGroup dimensions must be positive");
    TORCH_CHECK(worldSize >= 2,
                "MegaMoe BF16 MGroup requires world_size >= 2");
    TORCH_CHECK(worldSize <= 768,
                "MegaMoe BF16 MGroup supports at most 768 EP ranks");
    TORCH_CHECK(m <= 16384,
                "MegaMoe BF16 MGroup supports at most 16384 tokens per rank");
    TORCH_CHECK(expertPerRank <= 16,
                "MegaMoe BF16 MGroup supports at most 16 local experts");
    TORCH_CHECK(expertPerRank * worldSize <= 1024,
                "MegaMoe BF16 MGroup total expert count must not exceed 1024");
    TORCH_CHECK(maxRecvTokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0,
                "MegaMoe BF16 MGroup requires k % 256 == 0 and n % 512 == 0");
    TORCH_CHECK(weight1.size(2) == k,
                "weight1 must have shape [local_experts, n, k]");
    TORCH_CHECK(weight2.size(0) == expertPerRank &&
                    weight2.size(1) == k && weight2.size(2) * 2 == n,
                "weight2 must have shape [local_experts, k, n / 2]");

    MegaMoeTilingData tiling{};
    const auto hostTiling = detail::MakeMegaMoeHostTiling(
        tiling, m, k, n, topk, worldSize, expertPerRank,
        maxRecvTokens, rank, true, 0.0, false, save_gmm1_output);
    const auto earlyWeightBuffers =
        detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes =
        hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes,
                "MegaMoe BF16 MGroup local_mem_size is too small: need ",
                peerBytes, " bytes, got ", symmetricBufferBytes,
                ". BF16 dispatch payloads use two bytes per element.");

    auto y = torch::zeros({m, k}, x.options());
    auto expertTokenNums = torch::zeros(
        {expertPerRank}, topkIds.options().dtype(torch::kInt32));
    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode), tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode));
    const MegaMoeBF16Training::ForwardTransientWorkspaceLayout
        transientWorkspaceLayout(n, expertPerRank, maxRecvTokens);
    const uint64_t perRankTokenCountStride =
        MegaMoeBF16Training::AlignUp(
            worldSize * expertPerRank,
            MegaMoeBF16Training::kAlignment128);
    const uint64_t legacyTrainingTailBytes =
        MegaMoeBF16Training::AlignUp(
            worldSize * perRankTokenCountStride *
                MegaMoeBF16Training::kInt32Bytes,
            MegaMoeBF16Training::kAlignment512) +
        MegaMoeBF16Training::AlignUp(
            MegaMoeBF16Training::kBackwardProgressElements *
                MegaMoeBF16Training::kInt32Bytes,
            MegaMoeBF16Training::kAlignment128);
    TORCH_CHECK(
        savedStateLayout.legacyBytes + transientWorkspaceLayout.totalBytes ==
            hostTiling.workspace_bytes + earlyWeightBuffers.workspace_bytes +
                legacyTrainingTailBytes,
        "MegaMoe BF16 MGroup split workspace layout does not match the "
        "legacy forward workspace capacity");
    // Preserve the original initialization contract, without zero-filling
    // the large appended gate/up planes. Every valid tile overwrites them.
    auto trainingSavedState = torch::empty(
        {static_cast<int64_t>(savedStateLayout.totalBytes)},
        x.options().dtype(torch::kUInt8));
    trainingSavedState.narrow(
        0, 0, static_cast<int64_t>(savedStateLayout.legacyBytes)).zero_();
    constexpr int64_t kMGroupRows = 256;
    constexpr int64_t kReadyCachelineBytes = 16 * sizeof(int32_t);
    constexpr int64_t kReadyAlignment = 128;
    const int64_t maxGroupsPerExpert =
        (static_cast<int64_t>(tiling.maxOutputSize) + kMGroupRows - 1) /
            kMGroupRows;
    const int64_t readySlots = expertPerRank * maxGroupsPerExpert;
    const int64_t readyBytesUnaligned = readySlots * kReadyCachelineBytes;
    const int64_t readyBytes =
        (readyBytesUnaligned + kReadyAlignment - 1) /
        kReadyAlignment * kReadyAlignment;
    const int64_t transientWorkspaceBytes =
        static_cast<int64_t>(transientWorkspaceLayout.totalBytes) + readyBytes;
    auto forwardTransientWorkspace = torch::zeros(
        {transientWorkspaceBytes},
        x.options().dtype(torch::kUInt8));
    auto context = torch::empty(
        {static_cast<int64_t>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025)},
        x.options().dtype(torch::kUInt8));

    LaunchMegaMoeBF16MGroupKernel(
        context, x, topkIds, topkWeights, weight1, weight2, timer,
        y, expertTokenNums, trainingSavedState, forwardTransientWorkspace,
        tiling,
        static_cast<uint32_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()
                ->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize,
        shmem.FftsAddr());
    return {y, expertTokenNums, trainingSavedState};
}

}  // namespace ascend_deepep::ops
