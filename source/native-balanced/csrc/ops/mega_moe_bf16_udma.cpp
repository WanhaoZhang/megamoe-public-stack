#include "csrc/ops/mega_moe_bf16_udma.h"
#include "csrc/ops/mega_moe.h"

#include <cstdint>
#include <limits>
#include <tiling/platform/platform_ascendc.h>

#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"
#include "kernels/mega_moe/include/mega_moe_tiling.h"
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#include "kernels/mega_moe_bf16_udma/include/mega_moe_bf16_udma_tiling_host.h"

namespace {

void CheckNpuContiguous(const torch::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

namespace ascend_deepep::ops {

MegaMoeBF16ForwardResult MegaMoeBF16UDMA(
    torch::Tensor x, torch::Tensor topkIds, torch::Tensor topkWeights,
    torch::Tensor weight1, torch::Tensor weight2, int64_t maxRecvTokens,
    int rank, int worldSize, std::size_t symmetricBufferBytes,
    ShmemInstance &shmem, void *symmetricBuffer, torch::Tensor timer, const std::string &save_gmm1_output)
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
    TORCH_CHECK(x.dim() == 2 && topkIds.dim() == 2,
                "x and topk_ids must be rank-2 tensors");
    TORCH_CHECK(topkWeights.sizes() == topkIds.sizes(),
                "topk_weights shape must match topk_ids");
    TORCH_CHECK(x.size(0) == topkIds.size(0),
                "x and topk_ids must have the same token count");
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3,
                "weight tensors must be rank-3 tensors");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const int64_t n = weight1.size(1);
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 UDMA dimensions must be positive");
    TORCH_CHECK(m <= 16384,
                "MegaMoe BF16 UDMA supports at most 16384 tokens per rank");
    TORCH_CHECK(worldSize >= 2 && worldSize <= 768,
                "MegaMoe BF16 UDMA requires 2 <= world_size <= 768");
    TORCH_CHECK(expertPerRank <= 16 && expertPerRank * worldSize <= 1024,
                "MegaMoe BF16 UDMA expert count exceeds the supported range");
    TORCH_CHECK(maxRecvTokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0,
                "MegaMoe BF16 UDMA requires k % 256 == 0 and n % 512 == 0");
    TORCH_CHECK(weight1.size(2) == k && weight2.size(0) == expertPerRank &&
                    weight2.size(1) == k && weight2.size(2) * 2 == n,
                "weight shapes do not match the MegaMoe BF16 contract");
    TORCH_CHECK(static_cast<uint64_t>(maxRecvTokens) * k * sizeof(uint16_t) <=
                    std::numeric_limits<uint32_t>::max(),
                "one BF16 Dispatch segment exceeds the UDMA 32-bit length limit");
    TORCH_CHECK(static_cast<uint64_t>(k) * sizeof(uint16_t) <=
                    std::numeric_limits<uint32_t>::max(),
                "one BF16 Combine row exceeds the UDMA 32-bit length limit");

    MegaMoeTilingData tiling{};
    const auto baseHostTiling = detail::MakeMegaMoeHostTiling(
        tiling, m, k, n, topk, worldSize, expertPerRank,
        maxRecvTokens, rank, true, 0.0, false, save_gmm1_output);
    const auto udmaExtra = MegaMoeBF16UDMAHost::CalcExtraBufferSize(
        m, topk, worldSize, expertPerRank, maxRecvTokens, k);
    const MegaMoeBF16Training::UDMAForwardTransientWorkspaceLayout
        transientWorkspaceLayout(n, expertPerRank, maxRecvTokens);
    const uint64_t peerBytes =
        baseHostTiling.peer_bytes + udmaExtra.peerBytes;
    const uint64_t workspaceBytes = transientWorkspaceLayout.totalBytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes,
                "MegaMoe BF16 UDMA local_mem_size is too small: need ",
                peerBytes, " bytes, got ", symmetricBufferBytes);

    auto y = torch::zeros({m, k}, x.options());
    auto expertTokenNums = torch::zeros(
        {expertPerRank}, topkIds.options().dtype(torch::kInt32));
    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode), tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode));
    // Preserve the original initialization contract, without zero-filling
    // the large appended gate/up planes. Every valid tile overwrites them.
    auto trainingSavedState = torch::empty(
        {static_cast<int64_t>(savedStateLayout.totalBytes)},
        x.options().dtype(torch::kUInt8));
    trainingSavedState.narrow(
        0, 0, static_cast<int64_t>(savedStateLayout.legacyBytes)).zero_();
    auto forwardTransientWorkspace = torch::zeros(
        {static_cast<int64_t>(workspaceBytes)}, x.options().dtype(torch::kUInt8));
    auto context = torch::empty(
        {static_cast<int64_t>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025)},
        x.options().dtype(torch::kUInt8));

    LaunchMegaMoeBF16UDMAKernel(
        context, x, topkIds, topkWeights, weight1, weight2, timer, y,
        expertTokenNums, trainingSavedState, forwardTransientWorkspace, tiling,
        static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()
                                  ->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize, shmem.FftsAddr());
    return {y, expertTokenNums, trainingSavedState};
}

}  // namespace ascend_deepep::ops
