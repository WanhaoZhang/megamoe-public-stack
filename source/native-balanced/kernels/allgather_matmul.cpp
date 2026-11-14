#include <climits>

#include "acl/acl.h"

#include "ascend_deepep/kernels/launch.h"
#include "ascend_deepep/kernels/types.h"

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
#include "ascend950_allgather_matmul_device.h"
#else
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catccos/catccos.hpp"
#include "catccos/comm/block/comm_block.hpp"
#include "catccos/comm/block/comm_block_swizzle.hpp"
#include "catccos/comm/comm_dispatch_policy.hpp"
#include "catccos/comm/tile/tile_remote_copy.hpp"
#include "catccos/detail/remote_copy_type.hpp"
#include "catccos/dgemm/block/block_swizzle_allgather.hpp"
#include "catccos/dgemm/device/device_dgemm.hpp"
#include "catccos/dgemm/kernel/allgather_matmul_with_local_mm_opt.hpp"
#include "padding.h"
#endif
#include "catccos/dgemm/device/kernel_adapter.hpp"

using namespace AscendC;
using namespace Catccos;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;

template <
    class ElementA_, class LayoutA_,
    class ElementB_, class LayoutB_,
    class ElementC_, class LayoutC_>
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
using AllGatherMatmulNoPaddingConfig =
    Ascend950AllGatherMatmulConfig_M0_128<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_>;
#else
struct AllGatherMatmulNoPaddingConfig {
    using ArchTag = Catlass::Arch::AtlasA2;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA_, LayoutA_>;
    using BType = Catlass::Gemm::GemmType<ElementB_, LayoutB_>;
    using CType = Catlass::Gemm::GemmType<ElementC_, LayoutC_>;

    using BlockMmad =
        Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using BlockCommScheduler = Catccos::Comm::Block::BlockCommSwizzle<true, void, 0>;

    using TileRemoteCopy = Comm::Tile::TileRemoteCopy<
        ArchTag,
        true,
        AType,
        AType,
        void,
        Catccos::detail::CopyDirect::Put,
        Catccos::detail::CopyTransport::Mte>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, 2, true>;
    using BlockComm = Comm::Block::CommBlock<CommDispatchPolicy, AType, AType, void, TileRemoteCopy, TileScheduler>;

    using Kernel = DGemm::Kernel::AllGatherMatmul<
        void,
        BlockMmad,
        BlockComm,
        BlockMmadScheduler,
        BlockCommScheduler,
        2>;
    using Device = Catccos::DGemm::Device::DeviceDGemm<Kernel>;
};
#endif

using Config = AllGatherMatmulNoPaddingConfig<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>;
using DeviceOp = Config::Device;

template <class Device>
void RunDeviceOpAsync(Device& device_op, aclrtStream stream, uint32_t block_dim, uint64_t ffts_addr) {
    using Kernel = typename Device::Kernel;
    if (ffts_addr == 0) {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<block_dim, nullptr, stream>>>(device_op.params(), nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<block_dim, nullptr, stream>>>(
            device_op.params(),
            ffts_addr,
            nullptr);
    }
}

extern "C" ASCEND_DEEPEP_EXPORT void allgather_matmul_kernel_do(
    void* stream,
    uint8_t* a,
    uint8_t* b,
    uint8_t* c,
    uint8_t* symmetric_buffer,
    uint8_t* tiling_host,
    uint64_t ffts_addr) {
    auto* td = reinterpret_cast<ascend_deepep::AllGatherMatmulTilingData*>(tiling_host);

    Catlass::GemmCoord problem_shape{td->m, td->n, td->k};
    Catlass::MatrixCoord comm_core_split{td->comm_data_split, td->comm_npu_split};
    Catlass::MatrixCoord comm_block_shape{td->comm_block_m, UINT_MAX / 2};
    Catlass::MatrixCoord comm_tile_shape{td->comm_tile_m / 2, td->n0};

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
    DeviceOp::Arguments args{
        problem_shape,
        td->rank,
        td->world_size,
        td->comm_interval,
        a,
        b,
        c,
        symmetric_buffer,
        comm_core_split,
        comm_block_shape,
        comm_tile_shape};
#else
    DeviceOp::Arguments args{
        problem_shape,
        td->rank,
        td->world_size,
        td->comm_interval,
        a,
        b,
        c,
        nullptr,
        symmetric_buffer,
        comm_core_split,
        comm_block_shape,
        comm_tile_shape};
#endif

    DeviceOp device_op;
    device_op.Initialize(args);
    // DeviceDGemm::Run synchronizes the stream, which is illegal during NPU graph capture.
    RunDeviceOpAsync(device_op, static_cast<aclrtStream>(stream), td->block_dim, ffts_addr);
}
