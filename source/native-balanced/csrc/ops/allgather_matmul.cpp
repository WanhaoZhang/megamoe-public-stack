#include "csrc/ops/allgather_matmul.h"

#include <cstdint>

#include "ascend_deepep/kernels/types.h"
#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"

namespace {

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

constexpr std::size_t kCatccosAgmmSymmetricBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr int64_t kAgmmM0 = 128;
constexpr int64_t kAgmmN0 = 256;
constexpr int64_t kAgmmK0 = 256;
constexpr int64_t kAgmmCommTileM = 64;
constexpr int64_t kAgmmCommInterval = 3;
constexpr int64_t kAgmmCommNpuSplit = 1;
constexpr int64_t kAgmmCommDataSplit = 20;
constexpr int64_t kAgmmCommBlockM = 64;
constexpr int64_t kAgmmMaxBlockCount = 2;
constexpr int64_t kAgmmInputDtypeBytes = 2;

struct AllGatherMatmulMeta {
    int64_t m;
    int64_t n;
    ascend_deepep::AllGatherMatmulTilingData tiling;
};

AllGatherMatmulMeta MakeAllGatherMatmulMeta(
    const torch::Tensor& a,
    const torch::Tensor& b,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes) {
    CheckNpuContiguous(a, "a");
    CheckNpuContiguous(b, "b");
    TORCH_CHECK(a.dim() == 2, "a must have shape [m, k]");
    TORCH_CHECK(b.dim() == 2, "b must have shape [k, n]");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(0), "a.shape[1] must equal b.shape[0]");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must be float16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must be float16");

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "m, n and k must be positive");
    TORCH_CHECK(
        n < 65536 && n % 256 == 0,
        "the minimal catccos allgather_matmul path only supports no-padding B; require n < 65536 and n % 256 == 0");
    TORCH_CHECK(
        symmetric_buffer_bytes >= kCatccosAgmmSymmetricBytes,
        "catccos allgather_matmul requires AllGatherMatmul local_mem_size >= ",
        kCatccosAgmmSymmetricBytes);

    const int64_t k_loops = (k + kAgmmK0 - 1) / kAgmmK0;
    const int64_t max_peer_mem_per_rank =
        static_cast<int64_t>(kCatccosAgmmSymmetricBytes) /
        kAgmmInputDtypeBytes /
        world_size /
        kAgmmMaxBlockCount;
    TORCH_CHECK(
        kAgmmCommInterval * kAgmmM0 * kAgmmK0 * k_loops < max_peer_mem_per_rank,
        "catccos allgather_matmul shape exceeds the default symmetric workspace");

    ascend_deepep::AllGatherMatmulTilingData tiling{};
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.m0 = static_cast<uint32_t>(kAgmmM0);
    tiling.n0 = static_cast<uint32_t>(kAgmmN0);
    tiling.k0 = static_cast<uint32_t>(kAgmmK0);
    tiling.comm_tile_m = static_cast<uint32_t>(kAgmmCommTileM);
    tiling.comm_interval = static_cast<uint32_t>(kAgmmCommInterval);
    tiling.comm_npu_split = static_cast<uint32_t>(kAgmmCommNpuSplit);
    tiling.comm_data_split = static_cast<uint32_t>(kAgmmCommDataSplit);
    tiling.comm_block_m = static_cast<uint32_t>(kAgmmCommBlockM);
    tiling.block_dim = 20;
    tiling.reserved = 0;

    return {m, n, tiling};
}

void CheckAllGatherMatmulOut(
    const torch::Tensor& out,
    const torch::Tensor& a,
    const AllGatherMatmulMeta& meta,
    int world_size) {
    CheckNpuContiguous(out, "out");
    TORCH_CHECK(out.device() == a.device(), "out must be on the same device as a");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be float16");
    TORCH_CHECK(out.dim() == 2, "out must have shape [m * world_size, n]");
    TORCH_CHECK(
        out.size(0) == meta.m * world_size && out.size(1) == meta.n,
        "out must have shape [",
        meta.m * world_size,
        ", ",
        meta.n,
        "]");
}

}  // namespace

namespace ascend_deepep::ops {

torch::Tensor AllGatherMatmul(
    torch::Tensor a,
    torch::Tensor b,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    ShmemInstance& shmem,
    void* symmetric_buffer) {
    const auto meta = MakeAllGatherMatmulMeta(a, b, rank, world_size, symmetric_buffer_bytes);
    auto out = torch::empty({meta.m * world_size, meta.n}, a.options());
    LaunchAllGatherMatmulKernel(a, b, out, meta.tiling, symmetric_buffer, shmem.FftsAddr());
    return out;
}

torch::Tensor AllGatherMatmulOut(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor out,
    int rank,
    int world_size,
    std::size_t symmetric_buffer_bytes,
    ShmemInstance& shmem,
    void* symmetric_buffer) {
    const auto meta = MakeAllGatherMatmulMeta(a, b, rank, world_size, symmetric_buffer_bytes);
    CheckAllGatherMatmulOut(out, a, meta, world_size);
    LaunchAllGatherMatmulKernel(a, b, out, meta.tiling, symmetric_buffer, shmem.FftsAddr());
    return out;
}

}  // namespace ascend_deepep::ops
