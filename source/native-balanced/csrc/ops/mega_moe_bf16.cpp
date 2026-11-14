#include "csrc/ops/mega_moe_bf16.h"
#include "csrc/ops/mega_moe.h"

#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>
#include <utility>

#include <tiling/platform/platform_ascendc.h>

#include "csrc/common/check.h"
#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"
#include "kernels/mega_moe/include/mega_moe_tiling.h"
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#include "kernels/mega_moe_bf16/include/mega_moe_saved_meta.h"
#include "kernels/mega_moe_bf16/include/mega_moe_shared_expert_types.h"
#include "kernels/mega_moe_bf16/include/mega_moe_shared_expert_descriptor.h"
#include "kernels/common/timer/AscendTimerV2.hpp"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr std::size_t kBackwardTopkPartialUbBytes = 128ULL * 1024ULL;
constexpr std::size_t kA5Fp32PerVectorRegister = 64ULL;

void CheckNpuContiguous(const torch::Tensor &tensor, const char *name)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        name,
        " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor AllocateForwardTransientWorkspace(
    const MegaMoeBF16Training::ForwardDispatchTileReadyLayout &dispatchReadyLayout,
    const torch::Tensor &like, const MegaMoeTilingData &tiling, bool worker = false)
{
    // The transient payload is overwritten before use. The forward kernel
    // resets every control flag at the start of each invocation, before its
    // existing pre-Dispatch SyncAll publishes those resets to consumers.
    TORCH_CHECK(
        dispatchReadyLayout.totalBytes <=
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "MegaMoe BF16 forward transient workspace exceeds int64 capacity");

    const auto savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(
        tiling.savedGateUpMode, tiling.swigluLimit);
    const bool decoupled = worker ? MegaMoeBF16WorkerSwigluDecoupled(tiling.k, savedMode)
                                 : MegaMoeBF16SwigluDecoupled(tiling.k, savedMode);
    const MegaMoeBF16Training::ForwardGmTileProgressLayout progressLayout(
        tiling.n, tiling.expertPerRank, tiling.maxOutputSize, savedMode, decoupled,
        worker && MegaMoeBF16WorkerGmTileReady());
    const uint64_t extraBytes = progressLayout.totalBytes - dispatchReadyLayout.totalBytes;
    TORCH_CHECK(
        extraBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
            dispatchReadyLayout.totalBytes,
        "MegaMoe BF16 decoupled transient workspace exceeds int64 capacity");

    return torch::empty(
        {static_cast<int64_t>(progressLayout.totalBytes)},
        like.options().dtype(torch::kUInt8));
}

struct Bf16WeightLayoutInfo {
    int64_t n;
    bool transW;
};

Bf16WeightLayoutInfo ResolveBf16WeightLayout(
    const torch::Tensor &weight1,
    const torch::Tensor &weight2,
    int64_t k)
{
    const int64_t expertPerRank = weight1.size(0);
    const bool canonical =
        weight1.size(2) == k &&
        weight2.size(0) == expertPerRank &&
        weight2.size(1) == k &&
        weight1.size(1) % 2 == 0 &&
        weight2.size(2) == weight1.size(1) / 2;
    if (canonical) {
        return {weight1.size(1), true};
    }

    const bool legacy =
        weight1.size(1) == k &&
        weight2.size(0) == expertPerRank &&
        weight2.size(2) == k &&
        weight1.size(2) % 2 == 0 &&
        weight2.size(1) == weight1.size(2) / 2;
    TORCH_CHECK(
        legacy,
        "MegaMoe BF16 weights must use either canonical layout "
        "weight1 [E, N, K], weight2 [E, K, N / 2], or legacy Megatron "
        "layout weight1 [E, K, N], weight2 [E, N / 2, K]");
    return {weight1.size(2), false};
}

torch::Tensor AllocateTrainingSavedState(
    const MegaMoeBF16Training::TrainingSavedStateLayout &savedStateLayout,
    const torch::Tensor &like)
{
    // The compact ordinary-BF16 layout begins at cumsumOffset=0. Embedded
    // layouts used by MGroup/UDMA still skip their ptrA payload here. Keep only
    // the metadata/control portion zero-initialized; gate/up planes are written
    // by every valid forward tile.
    const uint64_t initOffset = savedStateLayout.cumsumOffset;
    const uint64_t initBytes = savedStateLayout.legacyBytes - initOffset;
    TORCH_CHECK(
        savedStateLayout.ptrAOffset <= initOffset &&
            initOffset <= savedStateLayout.legacyBytes &&
            savedStateLayout.legacyBytes <= savedStateLayout.totalBytes &&
            savedStateLayout.totalBytes <=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "invalid MegaMoe BF16 training saved-state initialization range");

    auto savedState = torch::empty(
        {static_cast<int64_t>(savedStateLayout.totalBytes)},
        like.options().dtype(torch::kUInt8));
    savedState.narrow(
        0,
        static_cast<int64_t>(initOffset),
        static_cast<int64_t>(initBytes)).zero_();
    return savedState;
}

std::size_t MegaMoeWindowControlRegionBytes(
    int64_t worldSize,
    int64_t expertPerRank,
    bool compactTokenCountExchange)
{
    // Clear the optional compact inbox and the unchanged dense token table.
    // Forward/backward payload starts at payloadOffset and is never cleared.
    const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(
        static_cast<uint64_t>(worldSize),
        static_cast<uint64_t>(expertPerRank), compactTokenCountExchange);
    return static_cast<std::size_t>(MegaMoeBF16Training::AlignUp(
        countLayout.payloadOffset,
        MegaMoeBF16Training::kAlignment512));
}

void ClearMegaMoeWindowControlRegion(
    void *symmetricBuffer,
    std::size_t symmetricBufferBytes,
    int64_t worldSize,
    int64_t expertPerRank,
    bool compactTokenCountExchange)
{
    const std::size_t controlBytes =
        MegaMoeWindowControlRegionBytes(
            worldSize, expertPerRank, compactTokenCountExchange);
    const std::size_t clearOffset = static_cast<std::size_t>(
        MegaMoeBF16Training::PeerDataOffset(worldSize));
    TORCH_CHECK(
        controlBytes <= symmetricBufferBytes,
        "MegaMoe BF16 SHMEM control region exceeds the symmetric buffer: need ",
        controlBytes,
        " bytes, got ",
        symmetricBufferBytes);
    TORCH_CHECK(clearOffset <= controlBytes,
        "invalid MegaMoe BF16 SHMEM control clear range");
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    ASCEND_DEEPEP_CHECK_ACL(aclrtMemsetAsync(
        static_cast<uint8_t *>(symmetricBuffer) + clearOffset,
        symmetricBufferBytes - clearOffset,
        0,
        controlBytes - clearOffset,
        stream));
}

constexpr std::size_t Bf16Align(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

struct MegaMoeBF16BackwardWorkspaceLayout {
    std::size_t controlOffset;
    std::size_t totalBytes;

    explicit MegaMoeBF16BackwardWorkspaceLayout(const MegaMoeTilingData &tiling,
        uint32_t splitExperts, bool noneGm = false)
    {
        const std::size_t e = tiling.expertPerRank;
        const std::size_t r = tiling.maxOutputSize;
        const std::size_t k = tiling.k;
        const std::size_t n = tiling.n;
        const std::size_t f = n / 2;
        const std::size_t dpPartialTiles =
            (f + BACKWARD_DP_PARTIAL_TILE_N - 1) /
            BACKWARD_DP_PARTIAL_TILE_N;
        std::size_t offset = 0;
        offset += Bf16Align(2 * r * k, 512);
        const uint32_t savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(
            tiling.savedGateUpMode, tiling.swigluLimit);
        if (savedMode == MegaMoeBF16Training::kSaveNone) {
            offset += Bf16Align(2 * r * f, 512);
            offset += Bf16Align(2 * r * f, 512);
        }
        offset += Bf16Align(2 * r * n, 512);
        offset += Bf16Align(2 * r * f, 512);
        offset += Bf16Align(4 * r * dpPartialTiles, 512);
        // dHp reuses Hp and saved-mode dX reuses dO; no extra payload.
        controlOffset = offset;
        offset += Bf16Align(4 * e * 16, 128);
        offset += Bf16Align(4, 32);
        offset += Bf16Align(4, 32);
        offset += Bf16Align(4 * e * 16, 128);
        offset += Bf16Align(4 * e * 16, 128); // dZ ready, one arrival per AIV
        offset += Bf16Align(4 * e * 16, 128); // dX ready, one arrival per AIC
        if (MegaMoeBF16Training::BackwardUsesGm(savedMode, noneGm)) {
            offset += Bf16Align(4 * e * 16, 128); // all AICs retired dO reads
        }
        totalBytes = MegaMoeBF16Training::BackwardDispatchProgressLayout::TotalBytes(
            offset, splitExperts, tiling.epWorldSize, tiling.blockNumPerEP, savedMode, noneGm);
    }
};

torch::Tensor AllocateBackwardWorkspace(
    const MegaMoeBF16BackwardWorkspaceLayout &layout,
    const torch::Tensor &like)
{
    // dO, recomputed gate/up, dZ, Hp and dP partials are payload regions.
    // Their valid packed rows are fully overwritten before any consumer runs.
    // Atomic dispatch/ready/done counters and the per-producer progress suffix require zero.
    TORCH_CHECK(
        layout.controlOffset <= layout.totalBytes &&
            layout.totalBytes <=
                static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
        "invalid MegaMoe BF16 backward workspace initialization range");
    auto workspace = torch::empty(
        {static_cast<int64_t>(layout.totalBytes)},
        like.options().dtype(torch::kUInt8));
    workspace.narrow(
        0,
        static_cast<int64_t>(layout.controlOffset),
        static_cast<int64_t>(layout.totalBytes - layout.controlOffset)).zero_();
    return workspace;
}

}  // namespace

namespace ascend_deepep::ops {

MegaMoeBF16ForwardResult MegaMoeBF16(
    torch::Tensor context,
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
    TORCH_CHECK(
        topkWeights.scalar_type() == torch::kBFloat16,
        "topk_weights must be bfloat16");
    TORCH_CHECK(
        weight1.scalar_type() == torch::kBFloat16,
        "weight1 must be bfloat16");
    TORCH_CHECK(
        weight2.scalar_type() == torch::kBFloat16,
        "weight2 must be bfloat16");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(topkIds.dim() == 2, "topk_ids must have shape [m, topk]");
    TORCH_CHECK(
        topkWeights.sizes() == topkIds.sizes(),
        "topk_weights shape must match topk_ids");
    TORCH_CHECK(
        x.size(0) == topkIds.size(0),
        "x and topk_ids must have the same token count");
    TORCH_CHECK(
        weight1.dim() == 3,
        "weight1 must be a rank-3 canonical or legacy tensor");
    TORCH_CHECK(
        weight2.dim() == 3,
        "weight2 must be a rank-3 canonical or legacy tensor");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0);
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(
        m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
        "all MegaMoe BF16 dimensions must be positive");
    TORCH_CHECK(worldSize >= 2, "MegaMoe BF16 requires world_size >= 2");
    TORCH_CHECK(
        worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
        "MegaMoe BF16 supports at most 1024 EP ranks");
    TORCH_CHECK(m <= 16384, "MegaMoe BF16 supports at most 16384 tokens per rank");
    TORCH_CHECK(
        expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts),
        "MegaMoe BF16 supports at most 32 local experts");
    TORCH_CHECK(
        expertPerRank * worldSize <=
            static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
        "MegaMoe BF16 forward total expert count must not exceed 2048");
    TORCH_CHECK(topk <= expertPerRank * worldSize,
        "MegaMoe BF16 topk must not exceed the total expert count");
    TORCH_CHECK(maxRecvTokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(
        k % 256 == 0 && n % 512 == 0,
        "MegaMoe BF16 requires k % 256 == 0 and n % 512 == 0");
    MegaMoeTilingData tiling{};
    const auto hostTiling = detail::MakeMegaMoeHostTiling(
        tiling,
        m,
        k,
        n,
        topk,
        worldSize,
        expertPerRank,
        maxRecvTokens,
        rank,
        true,
        0.0,  // Raw compute has no truncation parameter.
        true, save_gmm1_output);
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    const auto earlyWeightBuffers =
        detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode), tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode));
    const MegaMoeBF16Training::ForwardTransientWorkspaceLayout
        transientWorkspaceLayout(n, expertPerRank, maxRecvTokens);
    const MegaMoeBF16Training::ForwardDispatchTileReadyLayout
        dispatchTileReadyLayout(n, expertPerRank, maxRecvTokens);
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
        "MegaMoe BF16 split workspace layout does not match the legacy "
        "forward workspace capacity");
    const uint64_t peerBytes =
        hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(
        peerBytes <= symmetricBufferBytes,
        "MegaMoe BF16 local_mem_size is too small: need ",
        peerBytes,
        " bytes, got ",
        symmetricBufferBytes,
        ". BF16 dispatch payloads use two bytes per element.");

    // Both unpermute implementations overwrite every output token row once.
    auto y = torch::empty({m, k}, x.options());
    // The device no-replica branch selects home-only tiling and writes only
    // home entries. Keep the full physical-slot ABI, with unused replica
    // counts zero so callers can sum initialized counts for packed X/P.
    auto expertTokenNums = torch::zeros(
        {expertPerRank}, topkIds.options().dtype(torch::kInt32));
    auto trainingSavedState = AllocateTrainingSavedState(savedStateLayout, x);
    auto forwardTransientWorkspace = AllocateForwardTransientWorkspace(
        dispatchTileReadyLayout, x, tiling);
    ClearMegaMoeWindowControlRegion(
        symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
        tiling.compactTokenCountExchange != 0);

    LaunchMegaMoeBF16Kernel(
        context,
        x,
        topkIds,
        topkWeights,
        weight1,
        weight2,
        timer,
        y,
        expertTokenNums,
        trainingSavedState,
        torch::Tensor(),
        forwardTransientWorkspace,
        tiling,
        static_cast<uint32_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()
                ->GetCoreNumAic()),
        symmetricBuffer,
        symmetricBufferBytes,
        rank,
        worldSize,
        shmem.FftsAddr());
    return {
        y,
        expertTokenNums,
        trainingSavedState};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
MegaMoeBF16Forward(
    torch::Tensor context,
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
    torch::Tensor timer,
    double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica)
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
                "weight1 and weight2 must be rank-3 tensors");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t topk = topkIds.size(1);
    const int64_t expertPerRank = weight1.size(0) + (replica.empty() ? 0 : replica.at(9));
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 dimensions must be positive");
    TORCH_CHECK(
        worldSize >= 2 &&
            worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
        "MegaMoe BF16 requires 2..1024 EP ranks");
    TORCH_CHECK(
        m <= 16384 &&
            expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts) &&
            expertPerRank * worldSize <=
                static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
        "MegaMoe BF16 forward shape exceeds supported routing limits "
        "(32 local, 2048 total experts)");
    TORCH_CHECK(topk <= expertPerRank * worldSize,
                "MegaMoe BF16 topk must not exceed the total expert count");
    TORCH_CHECK(maxRecvTokens > 0, "max_recv_tokens must be positive");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0,
                "MegaMoe BF16 requires k % 256 == 0 and n % 512 == 0");
    MegaMoeTilingData tiling{};
    const auto hostTiling = detail::MakeMegaMoeHostTiling(
        tiling, m, k, n, topk, worldSize, expertPerRank,
        maxRecvTokens, rank, true, swigluLimit, true, save_gmm1_output);
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    if (replica.size() == 40) {
        CheckMegaMoeBF16SharedSavedMode(MegaMoeBF16Training::ResolveSavedGateUpMode(
            tiling.savedGateUpMode, tiling.swigluLimit), false);
    }
    const auto earlyWeightBuffers =
        detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes =
        hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes,
                "MegaMoe BF16 local_mem_size is too small: need ",
                peerBytes, " bytes, got ", symmetricBufferBytes);

    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode), tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false);
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(
        k, maxRecvTokens);
    const MegaMoeBF16Training::ForwardTransientWorkspaceLayout
        transientWorkspaceLayout(n, expertPerRank, maxRecvTokens);
    const MegaMoeBF16Training::ForwardDispatchTileReadyLayout
        dispatchTileReadyLayout(n, expertPerRank, maxRecvTokens);
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
        savedStateLayout.legacyBytes + dispatchOutputLayout.totalBytes +
                transientWorkspaceLayout.totalBytes ==
            hostTiling.workspace_bytes + earlyWeightBuffers.workspace_bytes +
                legacyTrainingTailBytes,
        "MegaMoe BF16 split workspace layout does not match the legacy "
        "forward workspace capacity");

    // Both unpermute implementations overwrite every output token row once.
    auto y = torch::empty({m, k}, x.options());
    // The device no-replica branch selects home-only tiling and writes only
    // home entries. Keep the full physical-slot ABI, with unused replica
    // counts zero so callers can sum initialized counts for packed X/P.
    auto expertTokenNums = torch::zeros(
        {expertPerRank}, topkIds.options().dtype(torch::kInt32));
    auto trainingSavedState = AllocateTrainingSavedState(savedStateLayout, x);
    auto dispatchOutput = torch::empty(
        {static_cast<int64_t>(dispatchOutputLayout.totalBytes)},
        x.options().dtype(torch::kUInt8));
    auto forwardTransientWorkspace = AllocateForwardTransientWorkspace(
        dispatchTileReadyLayout, x, tiling,
        replica.size() == 40 || (replica.size() == 26 && (replica[25] == 1 || replica[25] == 3)));
    ClearMegaMoeWindowControlRegion(
        symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
        tiling.compactTokenCountExchange != 0);

    LaunchMegaMoeBF16Kernel(
        context, x, topkIds, topkWeights, weight1, weight2, timer,
        y, expertTokenNums, trainingSavedState, dispatchOutput,
        forwardTransientWorkspace,
        tiling,
        static_cast<uint32_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()
                ->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize,
        shmem.FftsAddr(), replica);
    return {y, expertTokenNums, trainingSavedState, dispatchOutput,
            forwardTransientWorkspace};
}

MegaMoeBF16SharedForwardResult MegaMoeBF16SharedForward(
    torch::Tensor context, torch::Tensor x, torch::Tensor topkIds,
    torch::Tensor topkWeights, torch::Tensor weight1, torch::Tensor weight2,
    int64_t maxRecvTokens, int rank, int worldSize,
    std::size_t symmetricBufferBytes, ShmemInstance &shmem,
    void *symmetricBuffer, torch::Tensor timer, double swigluLimit,
    const std::string &saveGmm1Output, const std::vector<int64_t>& replica,
    torch::Tensor sharedWeightGate, torch::Tensor sharedWeightUp,
    torch::Tensor sharedWeightDown)
{
    CheckNpuContiguous(x, "x");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16,
                "shared expert requires BF16 x [M,H]");
    CheckMegaMoeBF16SharedWorldSize(worldSize);
    TORCH_CHECK(replica.size() == 26 && replica[25] == 1,
                "shared expert forward requires the unchanged S9 selector-1 descriptor");
    TORCH_CHECK(std::isfinite(swigluLimit) && swigluLimit >= 0.0,
                "shared expert limit must be finite and nonnegative");
    const int64_t m = x.size(0), hidden = x.size(1);
    const std::array<std::pair<torch::Tensor, const char *>, 3> sharedWeights{{
        {sharedWeightGate, "shared_weight_gate"},
        {sharedWeightUp, "shared_weight_up"},
        {sharedWeightDown, "shared_weight_down"}}};
    for (const auto &entry : sharedWeights) {
        CheckNpuContiguous(entry.first, entry.second);
        TORCH_CHECK(entry.first.device() == x.device() &&
                        entry.first.scalar_type() == torch::kBFloat16 &&
                        entry.first.dim() == 2,
                    entry.second, " must be a rank-2 BF16 tensor on x's device");
    }
    const int64_t ffn = sharedWeightGate.size(0);
    TORCH_CHECK(m > 0 && hidden > 0 && hidden % 256 == 0 &&
                    ffn > 0 && ffn % 256 == 0 &&
                    sharedWeightGate.size(1) == hidden &&
                    sharedWeightUp.sizes() == sharedWeightGate.sizes() &&
                    sharedWeightDown.size(0) == hidden &&
                    sharedWeightDown.size(1) == ffn,
                "shared weights require [Fs,H], [Fs,H], [H,Fs] with H/Fs multiples of 256");
    auto rawGate = torch::empty({m, ffn}, x.options());
    auto rawUp = torch::empty({m, ffn}, x.options());
    auto activation = torch::empty({m, ffn}, x.options());
    // A new call never observes readiness left by a previous pool user.
    const char* tileSetting = std::getenv("MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE");
    TORCH_CHECK(!tileSetting || std::strcmp(tileSetting, "0") == 0 ||
                    std::strcmp(tileSetting, "1") == 0,
                "MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE must be 0/1");
    const bool tiled = tileSetting && tileSetting[0] == '1';
    TORCH_CHECK(!tiled || MegaMoeBF16WorkerSharedTilePipeline(),
                "shared tile pipeline requested but selected worker is too old");
    const int timerSchema = MegaMoeBF16WorkerSharedForwardTimerSchema();
    TORCH_CHECK(timerSchema == 1 || timerSchema == 2 || timerSchema == 3, "unknown shared forward Timer schema");
    const uint64_t readyElements = tiled
        ? MegaMoeSharedExpertLayout::ReadyElements(m, ffn)
        : MegaMoeSharedExpertLayout::kReadyElements;
    const uint64_t timerElements = timerSchema == 3 ? 1024 : tiled
        ? MegaMoeSharedExpertLayout::TimerElements(m, hidden, ffn)
        : (timerSchema == 2 ? MegaMoeSharedExpertLayout::kTileTimerOffset
                            : MegaMoeSharedExpertLayout::kTimerElements);
    TORCH_CHECK(readyElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                    timerElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "shared tile control storage exceeds int64 capacity");
    auto ready = tiled
        ? torch::zeros({static_cast<int64_t>(readyElements)}, x.options().dtype(torch::kInt32))
        : torch::zeros({3, 128, 32}, x.options().dtype(torch::kInt32));
    auto sharedTimer = timer.defined()
        ? (timerSchema == 1 && !tiled
            ? torch::zeros({4, 128, 2}, x.options().dtype(torch::kInt64))
            : torch::zeros({static_cast<int64_t>(timerElements)}, x.options().dtype(torch::kInt64)))
        : torch::empty({0}, x.options().dtype(torch::kInt64));
    MegaMoeSharedExpertDescriptor shared{};
    shared.abi = 1;
    shared.weight_gate = reinterpret_cast<uint64_t>(sharedWeightGate.data_ptr());
    shared.weight_up = reinterpret_cast<uint64_t>(sharedWeightUp.data_ptr());
    shared.weight_down = reinterpret_cast<uint64_t>(sharedWeightDown.data_ptr());
    shared.raw_gate = reinterpret_cast<uint64_t>(rawGate.data_ptr());
    shared.raw_up = reinterpret_cast<uint64_t>(rawUp.data_ptr());
    shared.activation = reinterpret_cast<uint64_t>(activation.data_ptr());
    shared.ready = reinterpret_cast<uint64_t>(ready.data_ptr());
    shared.tokens = static_cast<uint64_t>(m);
    shared.hidden = static_cast<uint64_t>(hidden);
    shared.ffn = static_cast<uint64_t>(ffn);
    shared.timer = sharedTimer.numel() ? reinterpret_cast<uint64_t>(sharedTimer.data_ptr()) : 0;
    shared.flags = (tiled ? MegaMoeSharedExpertLayout::kFlagTilePipeline : 0) |
        (timer.defined() && timerSchema == 2 ? MegaMoeSharedExpertLayout::kFlagDetailTimer : 0);
    static_assert(sizeof(shared) == 14 * sizeof(int64_t), "shared descriptor ABI changed");
    std::vector<int64_t> extended(replica);
    const auto offset = extended.size();
    extended.resize(offset + 14);
    std::memcpy(extended.data() + offset, &shared, sizeof(shared));
    auto [y, counts, state, dispatch, workspace] = MegaMoeBF16Forward(
        context, x, topkIds, topkWeights, weight1, weight2, maxRecvTokens,
        rank, worldSize, symmetricBufferBytes, shmem, symmetricBuffer,
        timer, swigluLimit, saveGmm1Output, extended);
    return {y, counts, state, dispatch, workspace,
            rawGate, rawUp, activation, ready, sharedTimer};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
MegaMoeBF16BackwardImpl(
    torch::Tensor context,
    torch::Tensor gradY,
    torch::Tensor topkWeights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor expertTokenNums,
    torch::Tensor trainingSavedState,
    torch::Tensor dispatchOutput,
    int64_t maxRecvTokens,
    int rank,
    int worldSize,
    std::size_t symmetricBufferBytes,
    ShmemInstance &shmem,
    void *symmetricBuffer,
    torch::Tensor timer,
    double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgs* shared,
    const MegaMoeSharedExpert::SharedExpertBackwardArgsV2* sharedV2 = nullptr,
    const MegaMoeSharedExpert::SharedExpertBackwardArgsV3* sharedV3 = nullptr)
{
    const int64_t m = gradY.size(0);
    const int64_t k = gradY.size(1);
    TORCH_CHECK(weight1.dim() == 3 && weight2.dim() == 3,
                "weight1 and weight2 must be rank-3 tensors");
    const int64_t expertPerRank = weight1.size(0) + (replica.empty() ? 0 : replica.at(9));
    const auto weightLayout = ResolveBf16WeightLayout(weight1, weight2, k);
    const int64_t n = weightLayout.n;
    TORCH_CHECK(topkWeights.dim() == 2 && topkWeights.size(0) == m,
                "topk_weights must have shape [m, topk]");
    const int64_t topk = topkWeights.size(1);
    TORCH_CHECK(expertTokenNums.numel() == expertPerRank,
                "expert_token_nums must have one entry per local expert");
    // Only persistent saved-state is consumed by backward.
    TORCH_CHECK(trainingSavedState.scalar_type() == torch::kUInt8 &&
                    trainingSavedState.device().type() ==
                        c10::DeviceType::PrivateUse1,
                "training_saved_state must be an NPU UInt8 tensor");
    TORCH_CHECK(m > 0 && k > 0 && topk > 0 && expertPerRank > 0 && n > 0,
                "all MegaMoe BF16 backward dimensions must be positive");
    TORCH_CHECK(
        worldSize >= 2 &&
            worldSize <= static_cast<int64_t>(MegaMoeBF16Training::kMaxEpRanks),
        "MegaMoe BF16 backward requires 2..1024 EP ranks");
    TORCH_CHECK(
        m <= 16384 &&
            expertPerRank <= static_cast<int64_t>(MegaMoeBF16Training::kMaxLocalExperts) &&
            expertPerRank * worldSize <=
                static_cast<int64_t>(MegaMoeBF16Training::kMaxTotalExperts),
        "MegaMoe BF16 backward shape exceeds supported routing limits "
        "(32 local, 2048 total experts)");
    TORCH_CHECK(topk <= expertPerRank * worldSize,
                "MegaMoe BF16 backward topk must not exceed total expert count");
    TORCH_CHECK(maxRecvTokens > 0,
                "MegaMoe BF16 backward max_recv_tokens must be positive");
    TORCH_CHECK(
        maxRecvTokens <= std::numeric_limits<int32_t>::max(),
        "MegaMoe BF16 backward max_recv_tokens exceeds int32 routing capacity");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0,
                "MegaMoe BF16 backward requires k % 256 == 0 and n % 512 == 0");
    TORCH_CHECK(gradY.scalar_type() == torch::kBFloat16,
                "grad_y must be bfloat16");
    TORCH_CHECK(topkWeights.scalar_type() == torch::kBFloat16,
                "topk_weights must be bfloat16");
    TORCH_CHECK(weight1.scalar_type() == torch::kBFloat16,
                "weight1 must be bfloat16");
    TORCH_CHECK(weight2.scalar_type() == torch::kBFloat16,
                "weight2 must be bfloat16");
    TORCH_CHECK(expertTokenNums.scalar_type() == torch::kInt32,
                "expert_token_nums must be int32");
    CheckNpuContiguous(gradY, "grad_y");
    CheckNpuContiguous(topkWeights, "topk_weights");
    CheckNpuContiguous(weight1, "weight1");
    CheckNpuContiguous(weight2, "weight2");
    CheckNpuContiguous(trainingSavedState, "training_saved_state");
    CheckNpuContiguous(dispatchOutput, "dispatch_output");
    const MegaMoeBF16Training::DispatchOutputLayout dispatchOutputLayout(
        k, maxRecvTokens);
    TORCH_CHECK(
        dispatchOutput.scalar_type() == torch::kUInt8 &&
            static_cast<uint64_t>(dispatchOutput.numel()) >=
                dispatchOutputLayout.totalBytes,
        "dispatch_output must be an NPU UInt8 tensor with at least ",
        dispatchOutputLayout.totalBytes, " bytes");

    MegaMoeTilingData tiling{};
    const auto hostTiling = detail::MakeMegaMoeHostTiling(
        tiling, m, k, n, topk, worldSize, expertPerRank,
        maxRecvTokens, rank, true, swigluLimit, true, save_gmm1_output);
    tiling.transW = weightLayout.transW;
    tiling.transW2 = weightLayout.transW;
    const auto earlyWeightBuffers =
        detail::GetBf16EarlyTopkWeightBufferBytes(tiling);
    const uint64_t peerBytes =
        hostTiling.peer_bytes + earlyWeightBuffers.peer_bytes;
    TORCH_CHECK(peerBytes <= symmetricBufferBytes,
                "MegaMoe BF16 backward local_mem_size is too small: need ",
                peerBytes, " bytes, got ", symmetricBufferBytes);
    if (sharedV3 != nullptr) {
        // Same offsets as device PeermemInfo: counts, dY bank, aligned P,
        // then ptrD. V3 borrows ptrD for X only until the EP read-retire fence.
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout counts(
            static_cast<uint64_t>(worldSize), static_cast<uint64_t>(expertPerRank),
            tiling.compactTokenCountExchange != 0);
        const uint64_t sourceRows=static_cast<uint64_t>(m)*static_cast<uint64_t>(topk);
        const uint64_t sourceBankBytes=sourceRows*static_cast<uint64_t>(k)*2ULL;
        const uint64_t sourceProbabilityBytes=MegaMoeBF16Training::AlignUp(
            sourceRows*2ULL,MegaMoeBF16Training::kAlignment512);
        const uint64_t sourceXEnd=counts.payloadOffset+sourceBankBytes+
            sourceProbabilityBytes+sourceBankBytes;
        TORCH_CHECK(sourceXEnd<=peerBytes && sourceXEnd<=symmetricBufferBytes,
            "V3 sorted X/prob source banks exceed the existing symmetric window");
    }

    const std::size_t ubTileBytes = 3ULL * 128ULL * 256ULL * 2ULL;
    // MXFP8 backward decodes 16 rows at a time after the three persistent
    // BF16 work tiles: one FP8 data chunk, compact/padded E8M0 scales,
    // expanded BF16/FP32 scales, a temporary FP32 decoded chunk and three
    // optional clipping-mask chunks.
    constexpr std::size_t kMxDecodeRows = 16ULL;
    constexpr std::size_t kMxTileColumns = 256ULL;
    const std::size_t ubMxFp8TileBytes = ubTileBytes +
        kMxDecodeRows * kMxTileColumns +
        kMxDecodeRows * 32ULL +
        kMxDecodeRows * (kMxTileColumns / 32ULL) *
            (2ULL * sizeof(uint16_t) + 4ULL * sizeof(float)) +
        kMxDecodeRows * kMxTileColumns * sizeof(float) +
        3ULL * kMxDecodeRows * 32ULL;
    const std::size_t ubRestoreBytes =
        Bf16Align(static_cast<std::size_t>(worldSize) * expertPerRank, 128ULL) * 4ULL;
    const std::size_t dispatchElements = Bf16Align(
        static_cast<std::size_t>(tiling.dispatchRows) * static_cast<std::size_t>(k),
        32ULL);
    const std::size_t ubP0ScratchBytes =
        4ULL * dispatchElements +
        Bf16Align(2ULL * static_cast<std::size_t>(k) * sizeof(uint16_t), 32ULL) +
        ubRestoreBytes;
    const std::size_t ubUnpermuteBytes =
        3ULL * Bf16Align(6ULL * static_cast<std::size_t>(k), 32ULL);
    const std::size_t dpPartialTiles =
        (static_cast<std::size_t>(n / 2) +
            BACKWARD_DP_PARTIAL_TILE_N - 1) /
        BACKWARD_DP_PARTIAL_TILE_N;
    const std::size_t alignedDpPartialTiles =
        Bf16Align(dpPartialTiles * sizeof(float), 32ULL) / sizeof(float);
    TORCH_CHECK(
        (alignedDpPartialTiles + kA5Fp32PerVectorRegister - 1) * sizeof(float) <=
            kBackwardTopkPartialUbBytes,
        "MegaMoe BF16 backward n=", n,
        " exceeds the P6 partial-reduction UB capacity");
    const std::size_t ubTopkBytes = kBackwardTopkPartialUbBytes +
        Bf16Align(1024ULL * sizeof(uint16_t), 32ULL);
    const std::size_t ubResetBytes =
        MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes;
    // These phases own UB at different times. Include the post-P6 count
    // reset and the P6 route buffer, not just the P0 and SwiGLU allocations.
    const uint32_t savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(
        tiling.savedGateUpMode, tiling.swigluLimit);
    if (sharedV2 || sharedV3) CheckMegaMoeBF16SharedSavedMode(savedMode, true);
    const char *noneGmText = std::getenv("MEGAFLUX_BACKWARD_NONE_GM");
    TORCH_CHECK(noneGmText == nullptr || std::string(noneGmText) == "0" ||
                    std::string(noneGmText) == "1",
                "MEGAFLUX_BACKWARD_NONE_GM must be 0 or 1");
    const bool noneGm = noneGmText != nullptr && std::string(noneGmText) == "1";
    if (noneGm) {
        TORCH_CHECK(shared == nullptr,
                    "legacy shared ABI1 does not support none GM; use shared V2/V3");
        TORCH_CHECK(savedMode == MegaMoeBF16Training::kSaveNone &&
                        replica.size() == 26 && (replica[25] == 1 || replica[25] == 3),
                    "none GM backward requires save=none and the S9 early/external-home worker");
        TORCH_CHECK(MegaMoeBF16WorkerBackwardNoneGmSupported(),
                    "none GM backward requires a matched worker rebuilt with MEGAFLUX_BACKWARD_NONE_GM=1");
    }
    const std::size_t p1UbBytes =
        savedMode == MegaMoeBF16Training::kSaveMxFp8
            ? ubMxFp8TileBytes : ubTileBytes;
    const std::size_t ubFootprint = std::max({
        p1UbBytes, ubP0ScratchBytes, ubUnpermuteBytes, ubTopkBytes, ubResetBytes});
    uint64_t deviceUbBytes = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(
        platform_ascendc::CoreMemType::UB, deviceUbBytes);
    const uint64_t ubBudget = std::min<uint64_t>(deviceUbBytes, 248ULL * 1024ULL);
    TORCH_CHECK(ubFootprint <= ubBudget,
                "MegaMoe BF16 backward k=", k,
                " exceeds the UB budget (kernel needs ", ubFootprint,
                " bytes, have ", ubBudget, ")");

    const MegaMoeBF16Training::TrainingSavedStateLayout savedStateLayout(
        m, k, topk, worldSize, expertPerRank, maxRecvTokens, n, MegaMoeBF16Training::SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode), tiling.savedGateUpMode != MegaMoeBF16Training::kSaveNone, MegaMoeBF16Training::SaveGateUpClipMasks(tiling.swigluLimit, tiling.savedGateUpMode), false);
    TORCH_CHECK(
        static_cast<std::size_t>(trainingSavedState.numel()) *
                trainingSavedState.element_size() >=
            savedStateLayout.totalBytes,
        "MegaMoe BF16 training_saved_state is too small: need ",
        savedStateLayout.totalBytes, " bytes");
    uint32_t splitExperts = MegaMoeBF16Training::DefaultBackwardDispatchSplitExperts(
        tiling.savedGateUpMode, static_cast<uint32_t>(expertPerRank), noneGm);
    // Explicit k is a diagnostic override; legacy none still forces k=0 below.
    const char *splitText = std::getenv("MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS");
    if (splitText != nullptr) {
        if (std::string(splitText) == "all") {
            splitExperts = static_cast<uint32_t>(expertPerRank);
        } else {
            char *end = nullptr;
            const long value = std::strtol(splitText, &end, 10);
            TORCH_CHECK(end != splitText && *end == '\0' && value >= 0 &&
                            value <= expertPerRank,
                        "MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS must be 0..localE or all");
            splitExperts = static_cast<uint32_t>(value);
        }
    }
    if (!MegaMoeBF16Training::BackwardUsesGm(tiling.savedGateUpMode, noneGm)) splitExperts = 0;
    const MegaMoeBF16BackwardWorkspaceLayout backwardLayout(tiling, splitExperts, noneGm);
    auto backwardWorkspace = AllocateBackwardWorkspace(backwardLayout, gradY);
    // Device replica entries select home/expanded tiling after reading their
    // header. The host allocation uses expanded capacity: at EP8/home16/slot1
    // its progress offset points inside the home saved gate plane. Progress
    // is diagnostic-only, never a readiness signal. The matching forward
    // initializes both possible metadata prefixes, and every worker writes its
    // current phase at backward entry. Do not clear a device-selected payload
    // using a host-assumed offset. Ordinary (non-replica) behavior is unchanged.
    if (replica.empty()) {
        trainingSavedState.narrow(
            0, static_cast<int64_t>(savedStateLayout.backwardProgressOffset),
            static_cast<int64_t>(MegaMoeBF16Training::AlignUp(
                MegaMoeBF16Training::kBackwardProgressElements *
                    MegaMoeBF16Training::kInt32Bytes,
                MegaMoeBF16Training::kAlignment128))).zero_();
    }

    // P2 writes every gradX token row and P6 writes every TopK route exactly
    // once.  The backward kernel likewise overwrites every non-empty expert's
    // weight-gradient plane and clears only the experts whose effective row
    // count is zero, so none of these outputs needs a full pre-clear here.
    auto gradX = torch::empty({m, k}, gradY.options());
    auto gradTopkWeights = torch::empty_like(topkWeights);
    auto shape1=weight1.sizes().vec(), shape2=weight2.sizes().vec();
    shape1[0]=expertPerRank; shape2[0]=expertPerRank;
    auto gradWeight1 = torch::empty(shape1,weight1.options());
    auto gradWeight2 = torch::empty(shape2,weight2.options());
    ClearMegaMoeWindowControlRegion(
        symmetricBuffer, symmetricBufferBytes, worldSize, expertPerRank,
        tiling.compactTokenCountExchange != 0);

    if (sharedV3 != nullptr) {
        LaunchMegaMoeBF16BackwardSharedKernelV3(
            context, gradY, topkWeights, weight1, weight2,
            gradX, gradTopkWeights, gradWeight1, gradWeight2, trainingSavedState,
            dispatchOutput, backwardWorkspace, timer, tiling,
            MegaMoeBF16Training::EncodeBackwardDispatchPolicy(splitExperts, noneGm),
            static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
            symmetricBuffer, symmetricBufferBytes, rank, worldSize,
            shmem.FftsAddr(), replica, *sharedV3);
    } else if (sharedV2 != nullptr) {
        LaunchMegaMoeBF16BackwardSharedKernelV2(
            context, gradY, topkWeights, weight1, weight2,
            gradX, gradTopkWeights, gradWeight1, gradWeight2, trainingSavedState,
            dispatchOutput, backwardWorkspace, timer, tiling,
            MegaMoeBF16Training::EncodeBackwardDispatchPolicy(splitExperts, noneGm),
            static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
            symmetricBuffer, symmetricBufferBytes, rank, worldSize,
            shmem.FftsAddr(), replica, *sharedV2);
    } else if (shared != nullptr) {
        LaunchMegaMoeBF16BackwardSharedKernel(
            context, gradY, topkWeights, weight1, weight2,
            gradX, gradTopkWeights, gradWeight1, gradWeight2, trainingSavedState,
            dispatchOutput, backwardWorkspace, tiling, splitExperts,
            static_cast<uint32_t>(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic()),
            symmetricBuffer, symmetricBufferBytes, rank, worldSize,
            shmem.FftsAddr(), replica, *shared);
    } else {
    LaunchMegaMoeBF16BackwardKernel(
        context, gradY, topkWeights, weight1, weight2,
        gradX, gradTopkWeights, gradWeight1, gradWeight2, trainingSavedState,
        dispatchOutput,
        backwardWorkspace, timer, tiling,
        MegaMoeBF16Training::EncodeBackwardDispatchPolicy(splitExperts, noneGm),
        static_cast<uint32_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()
                ->GetCoreNumAic()),
        symmetricBuffer, symmetricBufferBytes, rank, worldSize,
        shmem.FftsAddr(), replica);
    }
    return {gradX, gradTopkWeights, gradWeight1, gradWeight2,
            backwardWorkspace};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
MegaMoeBF16Backward(
    torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
    torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
    int64_t maxRecvTokens, int rank, int worldSize,
    std::size_t symmetricBufferBytes, ShmemInstance &shmem, void *symmetricBuffer,
    torch::Tensor timer, double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica) {
    return MegaMoeBF16BackwardImpl(context, gradY, topkWeights, weight1, weight2,
        expertTokenNums, trainingSavedState, dispatchOutput, maxRecvTokens,
        rank, worldSize, symmetricBufferBytes, shmem, symmetricBuffer, timer,
        swigluLimit, save_gmm1_output, replica, nullptr);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
MegaMoeBF16BackwardShared(
    torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
    torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
    int64_t maxRecvTokens, int rank, int worldSize,
    std::size_t symmetricBufferBytes, ShmemInstance &shmem, void *symmetricBuffer,
    double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica, torch::Tensor x,
    torch::Tensor wg, torch::Tensor wu, torch::Tensor wd,
    torch::Tensor gu, torch::Tensor activation, double sharedLimit) {
    TORCH_CHECK(gradY.dim()==2, "shared backward grad_y must be [M,H]");
    const auto m=gradY.size(0), h=gradY.size(1);
    TORCH_CHECK(m>0 && m<=16384 && h>0 && h<=16384 && h%256==0,
                "shared backward requires 0<M,H<=16384 and H divisible by 256");
    TORCH_CHECK(worldSize==2 || worldSize==4 || worldSize==8,
                "shared backward first ABI supports EP2/4/8");
    TORCH_CHECK(replica.size()==26 && (replica[25]==1 || replica[25]==3),
                "shared backward requires the S9 early replica descriptor ABI");
    TORCH_CHECK(save_gmm1_output=="bf16", "shared backward requires routed BF16 saved state");
    TORCH_CHECK(std::isfinite(sharedLimit) && sharedLimit>0 &&
                std::isfinite(static_cast<float>(sharedLimit)) &&
                static_cast<float>(sharedLimit)>0,
                "shared_limit must be positive and finite in FP32");
    TORCH_CHECK(wg.dim()==2 && wu.dim()==2 && wd.dim()==2,
                "shared weights must be rank-2 Wg/Wu[Fs,H], Wd[H,Fs]");
    const auto f=wg.size(0);
    TORCH_CHECK(f>0 && f%256==0 && f<=16384,
                "shared backward Fs must be positive, divisible by 256, and <=16384");
    TORCH_CHECK(wg.size(1)==h && wu.size(0)==f && wu.size(1)==h &&
                wd.size(0)==h && wd.size(1)==f, "shared weight geometry mismatch");
    TORCH_CHECK(x.dim()==2 && x.size(0)==m && x.size(1)==h,
                "shared X must be original local [M,H]");
    TORCH_CHECK(gu.dim()==2 && gu.size(0)==m && gu.size(1)==2*f,
                "raw_gu must be unclipped row-interleaved [M,2Fs], gate first");
    TORCH_CHECK(activation.dim()==2 && activation.size(0)==m && activation.size(1)==f,
                "shared activation must be actual forward BF16 [M,Fs]");
    for (const auto& value : {gradY,x,wg,wu,wd,gu,activation,
                             topkWeights,weight1,weight2})
        TORCH_CHECK(value.scalar_type()==torch::kBFloat16,
                    "all numerical inputs of shared backward ABI1 must be BF16");
    for (const auto& item : std::vector<std::pair<const char*,torch::Tensor>>{
            {"grad_y",gradY},{"X",x},{"Wg",wg},{"Wu",wu},{"Wd",wd},
            {"raw_gu",gu},{"activation",activation},
            {"topk_weights",topkWeights},{"weight1",weight1},{"weight2",weight2},
            {"expert_token_nums",expertTokenNums},{"training_saved_state",trainingSavedState},
            {"dispatch_output",dispatchOutput}}) {
        CheckNpuContiguous(item.second,item.first);
        TORCH_CHECK(item.second.device()==gradY.device(), item.first," must share grad_y device");
    }
    // Reject a mismatched/old SO before clearing any routed window control.
    CheckMegaMoeBF16SharedBackwardCapability();
    TORCH_CHECK(ResolveBf16WeightLayout(weight1,weight2,h).transW,
                "shared backward requires canonical routed weights");
    const auto aic=platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(aic>0 && aic<=32,"shared backward control supports at most 32 AIC");
    auto dwg=torch::empty_like(wg),dwu=torch::empty_like(wu),dwd=torch::empty_like(wd);
    const int64_t mf=m*f,mh=m*h;
    auto scratch=torch::empty({3*mf+2*mh},gradY.options());
    // Fresh per-call storage and epoch1: no mutable global generation or pool alias.
    auto control=torch::zeros({MegaMoeSharedExpert::kControlWords},gradY.options().dtype(torch::kInt32));
    auto addr=[](const torch::Tensor& value) {return reinterpret_cast<uint64_t>(value.data_ptr());};
    MegaMoeSharedExpert::SharedExpertBackwardArgs shared{};
    shared.x=addr(x);shared.gu=addr(gu);shared.a=addr(activation);
    shared.wg=addr(wg);shared.wu=addr(wu);shared.wd=addr(wd);
    shared.dwg=addr(dwg);shared.dwu=addr(dwu);shared.dwd=addr(dwd);
    shared.da=addr(scratch);shared.dg=shared.da+mf*2;
    shared.du=shared.dg+mf*2;shared.dxg=shared.du+mf*2;
    shared.dxu=shared.dxg+mh*2;shared.control=addr(control);
    shared.m=static_cast<uint32_t>(m);shared.hidden=static_cast<uint32_t>(h);
    shared.ffn=static_cast<uint32_t>(f);shared.epoch=1;
    shared.limit=static_cast<float>(sharedLimit);shared.version=MegaMoeSharedExpert::kBackwardVersion;
    shared.bytes=sizeof(shared);shared.flags=0;
    auto routed=MegaMoeBF16BackwardImpl(context,gradY,topkWeights,weight1,weight2,
        expertTokenNums,trainingSavedState,dispatchOutput,maxRecvTokens,rank,
        worldSize,symmetricBufferBytes,shmem,symmetricBuffer,torch::Tensor(),
        swigluLimit,save_gmm1_output,replica,&shared);
    // Allocated and consumed on the current stream, like the original routed
    // private workspace. No tensor/pointer survives into another invocation.
    return {std::get<0>(routed),std::get<1>(routed),std::get<2>(routed),
            std::get<3>(routed),dwg,dwu,dwd};
}

static MegaMoeBF16SharedBackwardV3Result MegaMoeBF16BackwardSharedPlanesImpl(
    torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
    torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
    int64_t maxRecvTokens, int rank, int worldSize,
    std::size_t symmetricBufferBytes, ShmemInstance &shmem, void *symmetricBuffer,
    double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica, torch::Tensor x,
    torch::Tensor wg, torch::Tensor wu, torch::Tensor wd,
    torch::Tensor rawGate, torch::Tensor rawUp, torch::Tensor activation, double sharedLimit,
    bool recompute, torch::Tensor timer, torch::Tensor sharedTimer,
    bool redispatchX, torch::Tensor redispatchTimer) {
    TORCH_CHECK(gradY.dim()==2, "shared backward grad_y must be [M,H]");
    const auto m=gradY.size(0), h=gradY.size(1);
    TORCH_CHECK(m>0 && m<=16384 && h>0 && h<=16384 && h%256==0,
                "shared backward requires 0<M,H<=16384 and H divisible by 256");
    CheckMegaMoeBF16SharedWorldSize(worldSize);
    TORCH_CHECK(replica.size()==26 && (replica[25]==1 || replica[25]==3),
                "shared backward requires the S9 early replica descriptor ABI");
    TORCH_CHECK(std::isfinite(sharedLimit) && sharedLimit>=0 &&
                std::isfinite(static_cast<float>(sharedLimit)) &&
                (sharedLimit==0 || static_cast<float>(sharedLimit)>0),
                "shared_limit must be nonnegative and finite in FP32");
    TORCH_CHECK(wg.dim()==2 && wu.dim()==2 && wd.dim()==2,
                "shared weights must be rank-2 Wg/Wu[Fs,H], Wd[H,Fs]");
    const auto f=wg.size(0);
    TORCH_CHECK(f>0 && f%256==0 && f<=16384,
                "shared backward Fs must be positive, divisible by 256, and <=16384");
    TORCH_CHECK(wg.size(1)==h && wu.size(0)==f && wu.size(1)==h &&
                wd.size(0)==h && wd.size(1)==f, "shared weight geometry mismatch");
    TORCH_CHECK(x.dim()==2 && x.size(0)==m && x.size(1)==h,
                "shared X must be original local [M,H]");
    if (recompute) {
        TORCH_CHECK(rawGate.numel()==0 && rawUp.numel()==0 && activation.numel()==0,
            "shared V2 recompute requires three empty BF16 state tensors");
    } else {
        TORCH_CHECK(rawGate.dim()==2 && rawGate.size(0)==m && rawGate.size(1)==f &&
                    rawUp.sizes()==rawGate.sizes() && activation.sizes()==rawGate.sizes(),
            "shared V2 keep requires independent raw G/U and activation [M,Fs]");
    }
    for (const auto& value : {gradY,x,wg,wu,wd,rawGate,rawUp,activation,
                             topkWeights,weight1,weight2})
        TORCH_CHECK(value.scalar_type()==torch::kBFloat16,
                    "all numerical inputs of shared backward V2 must be BF16");
    for (const auto& item : std::vector<std::pair<const char*,torch::Tensor>>{
            {"grad_y",gradY},{"X",x},{"Wg",wg},{"Wu",wu},{"Wd",wd},
            {"raw_gate",rawGate},{"raw_up",rawUp},{"activation",activation},
            {"topk_weights",topkWeights},{"weight1",weight1},{"weight2",weight2},
            {"expert_token_nums",expertTokenNums},{"training_saved_state",trainingSavedState},
            {"dispatch_output",dispatchOutput}}) {
        CheckNpuContiguous(item.second,item.first);
        TORCH_CHECK(item.second.device()==gradY.device(), item.first," must share grad_y device");
    }
    // Reject a mismatched/old SO before clearing any routed window control.
    if (redispatchX) {
        CheckMegaMoeBF16SharedBackwardCapabilityV3(timer.defined());
        TORCH_CHECK(dispatchOutput.scalar_type()==torch::kUInt8 && dispatchOutput.numel()==0,
            "shared V3 original-X redispatch requires an empty NPU UInt8 dispatch_output");
        TORCH_CHECK(maxRecvTokens>0 && maxRecvTokens<=std::numeric_limits<int32_t>::max(),
            "shared V3 receive capacity must fit positive int32");
        TORCH_CHECK(timer.defined()==redispatchTimer.defined(),
            "V3 native/redispatch Timer pair mismatch");
    } else {
        CheckMegaMoeBF16SharedBackwardCapabilityV2(timer.defined());
        TORCH_CHECK(!redispatchTimer.defined(), "V2 must not receive an X redispatch Timer");
    }
    TORCH_CHECK(timer.defined()==sharedTimer.defined(), "V2 native/shared Timer pair mismatch");
    if (timer.defined()) {
        for (const auto& item : std::vector<std::pair<const char*,torch::Tensor>>{
                {"native_timer",timer},{"shared_timer",sharedTimer}}) {
            CheckNpuContiguous(item.second,item.first);
            TORCH_CHECK(item.second.device()==gradY.device() && item.second.scalar_type()==torch::kInt64,
                item.first," must be contiguous NPU int64 on grad_y device");
        }
        TORCH_CHECK(timer.numel()>=AscendTimer::TOTAL_BUFFER_SIZE &&
                    sharedTimer.dim()==3 && sharedTimer.size(0)==10 &&
                    sharedTimer.size(1)==128 && sharedTimer.size(2)==2,
            "V2 Timer buffers have an incompatible shape");
        if (redispatchX) {
            CheckNpuContiguous(redispatchTimer,"redispatch_timer");
            TORCH_CHECK(redispatchTimer.device()==gradY.device() &&
                        redispatchTimer.scalar_type()==torch::kInt64 &&
                        redispatchTimer.dim()==3 &&
                        redispatchTimer.size(0)==MegaMoeSharedExpert::kBackwardRedispatchTimerPhases &&
                        redispatchTimer.size(1)==MegaMoeSharedExpert::kBackwardRedispatchTimerLanes &&
                        redispatchTimer.size(2)==2,
                "V3 X redispatch Timer buffer has an incompatible shape");
        }
    }
    if (recompute) {
        rawGate=torch::empty({m,f},gradY.options());
        rawUp=torch::empty({m,f},gradY.options());
        activation=torch::empty({m,f},gradY.options());
    }
    TORCH_CHECK(ResolveBf16WeightLayout(weight1,weight2,h).transW,
                "shared backward requires canonical routed weights");
    const auto aic=platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(aic>0 && aic<=32,"shared backward control supports at most 32 AIC");
    auto dwg=torch::empty_like(wg),dwu=torch::empty_like(wu),dwd=torch::empty_like(wd);
    const int64_t mf=m*f,mh=m*h;
    auto scratch=torch::empty({3*mf+2*mh},gradY.options());
    // Fresh per-call storage and epoch1: no mutable global generation or pool alias.
    auto control=torch::zeros({MegaMoeSharedExpert::kControlWords},gradY.options().dtype(torch::kInt32));
    auto addr=[](const torch::Tensor& value) {return reinterpret_cast<uint64_t>(value.data_ptr());};
    MegaMoeSharedExpert::SharedExpertBackwardArgsV2 extended{};
    auto& shared=extended.base;
    extended.raw_gate=addr(rawGate);extended.raw_up=addr(rawUp);
    extended.version=MegaMoeSharedExpert::kBackwardVersionV2;
    extended.recompute=recompute ? 1 : 0;
    extended.timer=timer.defined() ? addr(sharedTimer) : 0;
    shared.x=addr(x);shared.gu=0;shared.a=addr(activation);
    shared.wg=addr(wg);shared.wu=addr(wu);shared.wd=addr(wd);
    shared.dwg=addr(dwg);shared.dwu=addr(dwu);shared.dwd=addr(dwd);
    shared.da=addr(scratch);shared.dg=shared.da+mf*2;
    shared.du=shared.dg+mf*2;shared.dxg=shared.du+mf*2;
    shared.dxu=shared.dxg+mh*2;shared.control=addr(control);
    shared.m=static_cast<uint32_t>(m);shared.hidden=static_cast<uint32_t>(h);
    shared.ffn=static_cast<uint32_t>(f);shared.epoch=1;
    shared.limit=sharedLimit==0 ? kMegaMoeSwigluLimitDisabled : static_cast<float>(sharedLimit);shared.version=MegaMoeSharedExpert::kBackwardVersion;
    shared.bytes=sizeof(shared);shared.flags=0;
    MegaMoeSharedExpert::SharedExpertBackwardArgsV3 originalX{};
    if (redispatchX) {
        const MegaMoeBF16Training::DispatchOutputLayout layout(h,maxRecvTokens);
        TORCH_CHECK(layout.totalBytes<=static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    "V3 packed X/prob allocation exceeds int64");
        dispatchOutput=torch::empty({static_cast<int64_t>(layout.totalBytes)},
                                    gradY.options().dtype(torch::kUInt8));
        originalX.shared=extended;
        originalX.version=MegaMoeSharedExpert::kBackwardVersionV3;
        originalX.flags=1;
        originalX.timer=timer.defined() ? addr(redispatchTimer) : 0;
    }
    auto routed=MegaMoeBF16BackwardImpl(context,gradY,topkWeights,weight1,weight2,
        expertTokenNums,trainingSavedState,dispatchOutput,maxRecvTokens,rank,
        worldSize,symmetricBufferBytes,shmem,symmetricBuffer,timer,
        swigluLimit,save_gmm1_output,replica,nullptr,
        redispatchX ? nullptr : &extended, redispatchX ? &originalX : nullptr);
    // Allocated and consumed on the current stream, like the original routed
    // private workspace. No tensor/pointer survives into another invocation.
    return {std::get<0>(routed),std::get<1>(routed),std::get<2>(routed),
            std::get<3>(routed),dwg,dwu,dwd,
            redispatchX ? dispatchOutput : torch::Tensor()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
MegaMoeBF16BackwardSharedV2(
    torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
    torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
    int64_t maxRecvTokens, int rank, int worldSize,
    std::size_t symmetricBufferBytes, ShmemInstance &shmem, void *symmetricBuffer,
    double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica, torch::Tensor x,
    torch::Tensor wg, torch::Tensor wu, torch::Tensor wd,
    torch::Tensor rawGate, torch::Tensor rawUp, torch::Tensor activation, double sharedLimit,
    bool recompute, torch::Tensor timer, torch::Tensor sharedTimer) {
    auto values=MegaMoeBF16BackwardSharedPlanesImpl(context,gradY,topkWeights,
        weight1,weight2,expertTokenNums,trainingSavedState,dispatchOutput,
        maxRecvTokens,rank,worldSize,symmetricBufferBytes,shmem,symmetricBuffer,
        swigluLimit,save_gmm1_output,replica,x,wg,wu,wd,rawGate,rawUp,activation,
        sharedLimit,recompute,timer,sharedTimer,false,torch::Tensor());
    return {std::get<0>(values),std::get<1>(values),std::get<2>(values),
            std::get<3>(values),std::get<4>(values),std::get<5>(values),std::get<6>(values)};
}

MegaMoeBF16SharedBackwardV3Result MegaMoeBF16BackwardSharedV3(
    torch::Tensor context, torch::Tensor gradY, torch::Tensor topkWeights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expertTokenNums,
    torch::Tensor trainingSavedState, torch::Tensor dispatchOutput,
    int64_t maxRecvTokens, int rank, int worldSize,
    std::size_t symmetricBufferBytes, ShmemInstance &shmem, void *symmetricBuffer,
    double swigluLimit, const std::string &save_gmm1_output,
    const std::vector<int64_t>& replica, torch::Tensor x,
    torch::Tensor wg, torch::Tensor wu, torch::Tensor wd,
    torch::Tensor rawGate, torch::Tensor rawUp, torch::Tensor activation, double sharedLimit,
    bool recompute, torch::Tensor timer, torch::Tensor sharedTimer,
    torch::Tensor redispatchTimer) {
    return MegaMoeBF16BackwardSharedPlanesImpl(context,gradY,topkWeights,
        weight1,weight2,expertTokenNums,trainingSavedState,dispatchOutput,
        maxRecvTokens,rank,worldSize,symmetricBufferBytes,shmem,symmetricBuffer,
        swigluLimit,save_gmm1_output,replica,x,wg,wu,wd,rawGate,rawUp,activation,
        sharedLimit,recompute,timer,sharedTimer,true,redispatchTimer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MegaMoeBF16BackwardRedispatch(
    torch::Tensor context, torch::Tensor gradY, torch::Tensor prob,
    torch::Tensor w1, torch::Tensor w2, torch::Tensor counts,
    torch::Tensor state, int64_t capacity, int rank, int world,
    std::size_t windowBytes, ShmemInstance &shmem, void *window,
    double limit, const std::string &saveMode, const std::vector<int64_t>& replica,
    torch::Tensor x, torch::Tensor timer, torch::Tensor redispatchTimer) {
    CheckNpuContiguous(x, "original_x");
    TORCH_CHECK(x.sizes()==gradY.sizes() && x.device()==gradY.device() &&
                x.scalar_type()==torch::kBFloat16,
                "original_x must match contiguous BF16 grad_y [M,H]");
    TORCH_CHECK(replica.size()==26 && replica[25]==1,
                "routed redispatch requires the standalone S9 descriptor");
    TORCH_CHECK(capacity>0 && capacity<=std::numeric_limits<int32_t>::max(),
                "routed redispatch capacity must fit positive int32");
    CheckMegaMoeBF16RoutedRedispatchCapability(world);
    const auto aic=platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(aic>0 && aic<=32,"routed redispatch retirement control requires at most 32 AIC");
    TORCH_CHECK(timer.defined()==redispatchTimer.defined(),
                "routed redispatch Timer pair mismatch");
    if (redispatchTimer.defined()) {
        CheckNpuContiguous(redispatchTimer, "redispatch_timer");
        TORCH_CHECK(redispatchTimer.device()==x.device() &&
                    redispatchTimer.scalar_type()==torch::kInt64 &&
                    redispatchTimer.dim()==3 && redispatchTimer.size(0)==4 &&
                    redispatchTimer.size(1)==128 && redispatchTimer.size(2)==2,
                    "routed redispatch Timer expects int64 [4,128,2]");
    }
    const MegaMoeBF16Training::DispatchOutputLayout layout(x.size(1),capacity);
    TORCH_CHECK(layout.totalBytes<=static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "routed redispatch allocation overflow");
    auto packed=torch::empty({static_cast<int64_t>(layout.totalBytes)},x.options().dtype(torch::kUInt8));
    // One fresh cacheline per local vector participant. This is a device
    // read-lease fence, not a shared activation plane or a Host synchronize.
    auto retired=torch::zeros({MegaMoeSharedExpert::kMaxParticipants *
        MegaMoeSharedExpert::kControlLineWords},x.options().dtype(torch::kInt32));
    MegaMoeSharedExpert::SharedExpertBackwardArgsV3 transport{};
    transport.version=MegaMoeSharedExpert::kBackwardVersionV3;
    transport.flags=MegaMoeSharedExpert::kBackwardRoutedOnlyRedispatchFlag; // Explicit routed-only capability; no fake shared tensors.
    transport.timer=redispatchTimer.defined() ? reinterpret_cast<uint64_t>(redispatchTimer.data_ptr()) : 0;
    transport.shared.version=MegaMoeSharedExpert::kBackwardVersionV2;
    auto& base=transport.shared.base;
    base.x=reinterpret_cast<uint64_t>(x.data_ptr());
    base.control=reinterpret_cast<uint64_t>(retired.data_ptr());
    base.epoch=1; // Fresh zeroed control allocation per invocation.
    base.m=static_cast<uint32_t>(x.size(0));
    base.hidden=static_cast<uint32_t>(x.size(1));
    base.version=MegaMoeSharedExpert::kBackwardVersion;
    base.bytes=sizeof(base);
    auto result=MegaMoeBF16BackwardImpl(context,gradY,prob,w1,w2,counts,state,packed,
        capacity,rank,world,windowBytes,shmem,window,timer,limit,saveMode,replica,
        nullptr,nullptr,&transport);
    return {std::get<0>(result),std::get<1>(result),std::get<2>(result),std::get<3>(result),packed};
}

}  // namespace ascend_deepep::ops
