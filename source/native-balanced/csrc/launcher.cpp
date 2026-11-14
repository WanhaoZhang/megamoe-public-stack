#include "kernels/mega_moe_bf16/include/mega_moe_replica_descriptor.h"
#include "kernels/mega_moe_bf16/include/mega_moe_device_descriptor.h"
#include "kernels/mega_moe_bf16/include/mega_moe_pipeline_descriptor.h"
#include "kernels/mega_moe_bf16/include/mega_moe_stripe_descriptor.h"
#include "kernels/mega_moe_bf16/include/mega_moe_stream_descriptor.h"
#include "kernels/mega_moe_bf16/include/mega_moe_full_descriptor.h"
#include "kernels/mega_moe_bf16/include/mega_moe_shared_expert_types.h"
#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#include "kernels/mega_moe_bf16/include/mega_moe_shared_expert_descriptor.h"
#include <cstring>
#include "launcher.h"
#include "csrc/ops/mega_moe.h"

#include <acl/acl.h>
#include <array>
#include <dlfcn.h>

#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "ascend_deepep/kernels/launch.h"
#include "kernels/mega_moe/include/mega_moe_tiling.h"
#include "kernels/common/timer/AscendTimerV2.hpp"

namespace {

MegaMoeTilingData MakeDeviceHomeTiling(const MegaMoeTilingData& tiling,
                                     const MegaMoeDeviceReplicaDescriptor& descriptor) {
    TORCH_CHECK(descriptor.header && descriptor.logical && descriptor.abi==2 &&
                descriptor.epoch>0 && descriptor.capacity>0,"invalid device descriptor");
    MegaMoeTilingData home{};
    ascend_deepep::ops::detail::MakeMegaMoeHostTiling(home,tiling.m,tiling.k,tiling.n,
        tiling.topK,tiling.epWorldSize,descriptor.replica.home,descriptor.capacity,
        tiling.rankId,true,tiling.swigluLimit,true,"bf16");
    // Inactive placement changes physical experts, never the frozen save ABI.
    home.savedGateUpMode=tiling.savedGateUpMode;
    home.transW=tiling.transW;home.transW2=tiling.transW2;
    return home;
}

aclrtStream GetCurrentAclStream() {
    return c10_npu::getCurrentNPUStream().stream();
}

uint8_t* BytePtr(void* ptr) {
    return static_cast<uint8_t*>(ptr);
}

struct MegaMoeContext {
    uint32_t ep_rank_id;
    uint32_t rank_size_per_server;
    uint64_t kfc_context_addr;
    uint64_t ep_buffer[1024];
};
static_assert(sizeof(MegaMoeContext) == 8208, "MegaMoe context ABI changed");

struct MegaMoeW4A8Context {
    uint32_t ep_rank_id;
    uint32_t ep_rank_size;
    uint64_t win_size;
    uint64_t ep_buffer[1024];
};
static_assert(
    sizeof(MegaMoeW4A8Context) == 8208,
    "MegaMoe W4A8 context ABI changed");

auto ResolveMegaMoeKernel() -> decltype(&mega_moe_kernel_do) {
    static auto kernel = []() -> decltype(&mega_moe_kernel_do) {
        void* symbol = dlsym(RTLD_DEFAULT, "mega_moe_kernel_do");
        if (symbol == nullptr) {
            void* handle = dlopen(
                "libascend_deepep_mega_moe.so",
                RTLD_NOW | RTLD_GLOBAL);
            TORCH_CHECK(
                handle != nullptr,
                "MegaMoe kernel library is not loaded. Run "
                "scripts/build_mega_moe.sh first: ",
                dlerror());
            symbol = dlsym(handle, "mega_moe_kernel_do");
        }
        TORCH_CHECK(
            symbol != nullptr,
            "mega_moe_kernel_do was not exported by "
            "libascend_deepep_mega_moe.so");
        return reinterpret_cast<decltype(&mega_moe_kernel_do)>(symbol);
    }();
    return kernel;
}

auto ResolveMegaMoeW4A8Kernel() -> decltype(&mega_moe_w4a8_kernel_do) {
    static auto kernel = []() -> decltype(&mega_moe_w4a8_kernel_do) {
        void* symbol = dlsym(RTLD_DEFAULT, "mega_moe_w4a8_kernel_do");
        if (symbol == nullptr) {
            void* handle = dlopen(
                "libascend_deepep_mega_moe_w4a8.so",
                RTLD_NOW | RTLD_GLOBAL);
            TORCH_CHECK(
                handle != nullptr,
                "MegaMoe W4A8 kernel library is not loaded. Run "
                "scripts/build_mega_moe_w4a8.sh first: ",
                dlerror());
            symbol = dlsym(handle, "mega_moe_w4a8_kernel_do");
        }
        TORCH_CHECK(
            symbol != nullptr,
            "mega_moe_w4a8_kernel_do was not exported by "
            "libascend_deepep_mega_moe_w4a8.so");
        return reinterpret_cast<decltype(&mega_moe_w4a8_kernel_do)>(symbol);
    }();
    return kernel;
}

auto ResolveMegaMoeBF16Kernel() -> decltype(&mega_moe_bf16_kernel_do) {
    static auto kernel = []() -> decltype(&mega_moe_bf16_kernel_do) {
        void* symbol = dlsym(RTLD_DEFAULT, "mega_moe_bf16_kernel_do");
        if (symbol == nullptr) {
            void* handle = dlopen(
                "libascend_deepep_mega_moe_bf16.so",
                RTLD_NOW | RTLD_GLOBAL);
            TORCH_CHECK(
                handle != nullptr,
                "MegaMoeBF16 kernel library is not loaded. Run "
                "scripts/build_mega_moe_bf16.sh first: ",
                dlerror());
            symbol = dlsym(handle, "mega_moe_bf16_kernel_do");
        }
        TORCH_CHECK(
            symbol != nullptr,
            "mega_moe_bf16_kernel_do was not exported by "
            "libascend_deepep_mega_moe_bf16.so");
        return reinterpret_cast<decltype(&mega_moe_bf16_kernel_do)>(symbol);
    }();
    return kernel;
}

auto ResolveMegaMoeBF16MGroupKernel()
    -> decltype(&mega_moe_bf16_mgroup_kernel_do) {
    static auto kernel = []() -> decltype(&mega_moe_bf16_mgroup_kernel_do) {
        void* symbol = dlsym(RTLD_DEFAULT, "mega_moe_bf16_mgroup_kernel_do");
        if (symbol == nullptr) {
            void* handle = dlopen(
                "libascend_deepep_mega_moe_bf16_mgroup.so",
                RTLD_NOW | RTLD_GLOBAL);
            TORCH_CHECK(
                handle != nullptr,
                "MegaMoeBF16MGroup kernel library is not loaded. Run "
                "scripts/build_mega_moe_bf16_mgroup.sh first: ",
                dlerror());
            symbol = dlsym(handle, "mega_moe_bf16_mgroup_kernel_do");
        }
        TORCH_CHECK(
            symbol != nullptr,
            "mega_moe_bf16_mgroup_kernel_do was not exported by "
            "libascend_deepep_mega_moe_bf16_mgroup.so");
        return reinterpret_cast<
            decltype(&mega_moe_bf16_mgroup_kernel_do)>(symbol);
    }();
    return kernel;
}

auto ResolveMegaMoeBF16UDMAKernel() -> decltype(&mega_moe_bf16_udma_kernel_do) {
    static auto kernel = []() -> decltype(&mega_moe_bf16_udma_kernel_do) {
        void* symbol = dlsym(RTLD_DEFAULT, "mega_moe_bf16_udma_kernel_do");
        if (symbol == nullptr) {
            void* handle = dlopen(
                "libascend_deepep_mega_moe_bf16_udma.so",
                RTLD_NOW | RTLD_GLOBAL);
            TORCH_CHECK(
                handle != nullptr,
                "MegaMoeBF16UDMA kernel library is not loaded. Run "
                "scripts/build_mega_moe_bf16_udma.sh first: ",
                dlerror());
            symbol = dlsym(handle, "mega_moe_bf16_udma_kernel_do");
        }
        TORCH_CHECK(
            symbol != nullptr,
            "mega_moe_bf16_udma_kernel_do was not exported by "
            "libascend_deepep_mega_moe_bf16_udma.so");
        return reinterpret_cast<decltype(&mega_moe_bf16_udma_kernel_do)>(symbol);
    }();
    return kernel;
}

const std::array<uint8_t, 512>& MegaMoeW4A8Gamma() {
    static const auto gamma = [] {
        constexpr std::array<uint8_t, 16> kFp4ToFp8 = {
            0, 48, 56, 60, 64, 68, 72, 76,
            128, 176, 184, 188, 192, 196, 200, 204};
        std::array<uint8_t, 512> table{};
        for (std::size_t value = 0; value < 256; ++value) {
            table[value] = kFp4ToFp8[value & 0x0f];
            table[value + 256] = kFp4ToFp8[(value >> 4) & 0x0f];
        }
        return table;
    }();
    return gamma;
}

}  // namespace

void LaunchDispatchKernel(
    torch::Tensor x,
    torch::Tensor topk_idx,
    const c10::optional<torch::Tensor>& topk_weights,
    torch::Tensor recv_x,
    torch::Tensor recv_topk_idx,
    const c10::optional<torch::Tensor>& recv_topk_weights,
    torch::Tensor handle,
    const ascend_deepep::DispatchTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    void* topk_weights_ptr = topk_weights.has_value() ? topk_weights->data_ptr() : nullptr;
    void* recv_topk_weights_ptr =
        recv_topk_weights.has_value() ? recv_topk_weights->data_ptr() : nullptr;

    dispatch_kernel_do(
        tiling.block_dim,
        stream,
        BytePtr(x.data_ptr()),
        BytePtr(topk_idx.data_ptr()),
        BytePtr(topk_weights_ptr),
        BytePtr(symmetric_buffer),
        BytePtr(recv_x.data_ptr()),
        BytePtr(recv_topk_idx.data_ptr()),
        BytePtr(recv_topk_weights_ptr),
        BytePtr(handle.data_ptr()),
        ffts_addr);

    // The baseline API is synchronous, which guarantees safe reuse of the
    // op-owned symmetric buffer.
    // shmem.Synchronize(stream);
}

void LaunchCombineKernel(
    torch::Tensor expert_output,
    torch::Tensor handle,
    const c10::optional<torch::Tensor>& topk_weights,
    torch::Tensor out,
    const ascend_deepep::CombineTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    aclrtStream stream = GetCurrentAclStream();
    void* topk_weights_ptr = topk_weights.has_value() ? topk_weights->data_ptr() : nullptr;

    combine_kernel_do(
        tiling.block_dim,
        stream,
        BytePtr(expert_output.data_ptr()),
        BytePtr(handle.data_ptr()),
        BytePtr(topk_weights_ptr),
        BytePtr(symmetric_buffer),
        BytePtr(out.data_ptr()),
        ffts_addr);

    // shmem.Synchronize(stream);
}

void LaunchAllGatherMatmulKernel(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    const ascend_deepep::AllGatherMatmulTilingData& tiling,
    void* symmetric_buffer,
    uint64_t ffts_addr) {
    aclrtStream stream = GetCurrentAclStream();
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");

    allgather_matmul_kernel_do(
        stream,
        BytePtr(a.data_ptr()),
        BytePtr(b.data_ptr()),
        BytePtr(c.data_ptr()),
        BytePtr(symmetric_buffer),
        reinterpret_cast<uint8_t*>(const_cast<ascend_deepep::AllGatherMatmulTilingData*>(&tiling)),
        ffts_addr);
}

void LaunchMegaMoeKernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor weight_scales1,
    torch::Tensor weight_scales2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr) {
    TORCH_CHECK(
        symmetric_buffer != nullptr,
        "symmetric_buffer must not be null");
    TORCH_CHECK(
        world_size <= 1024,
        "MegaMoe supports at most 1024 EP ranks");
    if (timer.defined()) {
        TORCH_CHECK(
            timer.device().type() == c10::DeviceType::PrivateUse1 &&
                timer.is_contiguous() && timer.scalar_type() == torch::kInt64,
            "MegaMoe W8A8 timer must be a contiguous NPU int64 tensor");
        TORCH_CHECK(
            timer.numel() >= AscendTimer::TOTAL_BUFFER_SIZE,
            "MegaMoe W8A8 timer requires at least ",
            AscendTimer::TOTAL_BUFFER_SIZE,
            " int64 elements");
    }

    const std::size_t context_bytes =
        static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(
        context_bytes >= sizeof(MegaMoeContext),
        "MegaMoe context tensor is too small");

    MegaMoeContext host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.rank_size_per_server = static_cast<uint32_t>(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_ptr = aclshmem_ptr(symmetric_buffer, peer);
        TORCH_CHECK(
            peer_ptr != nullptr,
            "aclshmem_ptr failed for peer rank ",
            peer);
        host_context.ep_buffer[peer] =
            reinterpret_cast<uint64_t>(peer_ptr);
    }

    aclrtStream stream = GetCurrentAclStream();
    const auto copy_status = aclrtMemcpy(
        context.data_ptr(),
        context_bytes,
        &host_context,
        sizeof(host_context),
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(
        copy_status == ACL_SUCCESS,
        "failed to copy MegaMoe communication context, acl error ",
        copy_status);

    const auto pre_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(
        pre_sync_status == ACL_SUCCESS,
        "failed to synchronize before MegaMoe, acl error ",
        pre_sync_status);
    aclshmem_barrier_all();

    auto kernel = ResolveMegaMoeKernel();
    if (timer.defined()) {
        auto timer_enabled = reinterpret_cast<int (*)()>(
            dlsym(RTLD_DEFAULT, "mega_moe_timer_enabled"));
        TORCH_CHECK(
            timer_enabled != nullptr && timer_enabled() != 0,
            "MegaMoe W8A8 timer was requested, but the kernel was built "
            "without timer instrumentation. Rebuild with "
            "scripts/build_mega_moe.sh --enable-timer");
    }
    kernel(
        block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        BytePtr(weight_scales1.data_ptr()),
        BytePtr(weight_scales2.data_ptr()),
        nullptr,
        nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(workspace.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling),
        ffts_addr);

    const auto post_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(
        post_sync_status == ACL_SUCCESS,
        "MegaMoe kernel execution failed, acl error ",
        post_sync_status);
    aclshmem_barrier_all();

    (void)symmetric_buffer_bytes;
}

void LaunchMegaMoeW4A8Kernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor weight_scales1,
    torch::Tensor weight_scales2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor workspace,
    const MegaMoeW4A8TilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(world_size <= 1024, "MegaMoe W4A8 supports at most 1024 EP ranks");
    TORCH_CHECK(
        static_cast<std::size_t>(context.numel()) * context.element_size() >=
            sizeof(MegaMoeW4A8Context),
        "MegaMoe W4A8 context tensor is too small");
    TORCH_CHECK(
        static_cast<std::size_t>(workspace.numel()) * workspace.element_size() >=
            MegaMoeW4A8Gamma().size(),
        "MegaMoe W4A8 workspace is too small for the gamma table");
    if (timer.defined()) {
        TORCH_CHECK(
            timer.device().type() == c10::DeviceType::PrivateUse1 &&
                timer.is_contiguous() && timer.scalar_type() == torch::kInt64,
            "MegaMoe W4A8 timer must be a contiguous NPU int64 tensor");
        TORCH_CHECK(
            timer.numel() >= AscendTimer::TOTAL_BUFFER_SIZE,
            "MegaMoe W4A8 timer requires at least ",
            AscendTimer::TOTAL_BUFFER_SIZE,
            " int64 elements");
    }

    MegaMoeW4A8Context host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.ep_rank_size = static_cast<uint32_t>(world_size);
    host_context.win_size = symmetric_buffer_bytes;
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_ptr = aclshmem_ptr(symmetric_buffer, peer);
        TORCH_CHECK(peer_ptr != nullptr, "aclshmem_ptr failed for peer rank ", peer);
        host_context.ep_buffer[peer] = reinterpret_cast<uint64_t>(peer_ptr);
    }

    aclrtStream stream = GetCurrentAclStream();
    const auto context_status = aclrtMemcpy(
        context.data_ptr(),
        context.numel() * context.element_size(),
        &host_context,
        sizeof(host_context),
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(
        context_status == ACL_SUCCESS,
        "failed to copy MegaMoe W4A8 communication context, acl error ",
        context_status);
    const auto gamma_status = aclrtMemcpy(
        workspace.data_ptr(),
        workspace.numel() * workspace.element_size(),
        MegaMoeW4A8Gamma().data(),
        MegaMoeW4A8Gamma().size(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(
        gamma_status == ACL_SUCCESS,
        "failed to initialize MegaMoe W4A8 gamma table, acl error ",
        gamma_status);

    const auto pre_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(
        pre_sync_status == ACL_SUCCESS,
        "failed to synchronize before MegaMoe W4A8, acl error ",
        pre_sync_status);
    aclshmem_barrier_all();

    ResolveMegaMoeW4A8Kernel()(
        block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        BytePtr(weight_scales1.data_ptr()),
        BytePtr(weight_scales2.data_ptr()),
        nullptr,
        nullptr,
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        ffts_addr);

    const auto post_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(
        post_sync_status == ACL_SUCCESS,
        "MegaMoe W4A8 kernel execution failed, acl error ",
        post_sync_status);
    aclshmem_barrier_all();
}

bool MegaMoeBF16SwigluDecoupled(uint64_t hidden, uint32_t saved_mode) {
    using SavedModeQuery = int (*)(uint64_t, uint32_t);
    static SavedModeQuery saved_mode_query = []() -> SavedModeQuery {
        ResolveMegaMoeBF16Kernel();
        return reinterpret_cast<SavedModeQuery>(
            dlsym(RTLD_DEFAULT, "mega_moe_bf16_swiglu_decoupled_for_saved_mode"));
    }();
    if (saved_mode_query != nullptr) {
        return saved_mode_query(hidden, saved_mode) != 0;
    }
    using Query = int (*)(uint64_t);
    static Query query = []() -> Query {
        // Load the same ordinary-forward SO before looking up its policy.
        ResolveMegaMoeBF16Kernel();
        return reinterpret_cast<Query>(
            dlsym(RTLD_DEFAULT, "mega_moe_bf16_swiglu_decoupled"));
    }();
    // An older SO implements only the original UB path and needs no extra GM.
    return query != nullptr && query(hidden) != 0;
}

bool MegaMoeBF16WorkerSwigluDecoupled(uint64_t hidden, uint32_t saved_mode) {
    static void* library = dlopen("libascend_deepep_megaflux_early_bf16.so", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(library, "load selected worker for SwiGLU workspace policy: ", dlerror());
    auto query = reinterpret_cast<int (*)()>(dlsym(library, "megaflux_worker_swiglu_decoupling_mode"));
    if (!query) {
        // Historical S9 supplied only the unused _MODE macro, so its actual
        // kernel policy was off for all saved modes. Preserve that ABI.
        return false;
    }
    const int mode = query();
    TORCH_CHECK(mode >= 0 && mode <= 2, "unknown worker SwiGLU policy");
    return MegaMoeBF16Training::ForwardUsesGmSwiglu(hidden, saved_mode, mode);
}

namespace {
int SharedForwardWorkerQuery(const char* symbol, int fallback) {
    static void* library = dlopen("libascend_deepep_megaflux_early_bf16.so", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(library, "load selected worker policy: ", dlerror());
    auto query = reinterpret_cast<int (*)()>(dlsym(library, symbol));
    return query ? query() : fallback;
}
}

bool MegaMoeBF16WorkerGmTileReady() {
    return SharedForwardWorkerQuery("megaflux_worker_gm_swiglu_tile_ready", 0) == 1;
}
bool MegaMoeBF16WorkerBackwardNoneGmSupported() {
    // Use the same S9 image as descriptor 26/selector 1 or 3; another
    // RTLD_GLOBAL library must not authorize the encoded launch policy.
    static void* library = dlopen("libascend_deepep_megaflux_early_bf16.so", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(library, "load selected worker for none GM backward policy: ", dlerror());
    auto query = reinterpret_cast<int (*)()>(dlsym(library, "megaflux_backward_none_gm_version"));
    return query != nullptr && query() == 1;
}
int MegaMoeBF16WorkerSharedForwardTimerSchema() {
    return SharedForwardWorkerQuery("megaflux_shared_forward_timer_schema", 1);
}
void CheckMegaMoeBF16RoutedRedispatchCapability(int world) {
    TORCH_CHECK((world==2 || world==4 || world==8 || world==16 || world==32 || world==64) &&
                SharedForwardWorkerQuery("megaflux_routed_redispatch_version", 0)==2,
                "routed original-X redispatch requires worker capability2 with read-lease fence and EP<=64");
}
void CheckMegaMoeBF16SharedWorldSize(int world) {
    TORCH_CHECK((world==2 || world==4 || world==8 || world==16 || world==32 || world==64) &&
                world<=SharedForwardWorkerQuery("megaflux_shared_max_ep",8),
                "shared EP exceeds selected worker capability; rebuild matching Host/worker");
}
bool MegaMoeBF16WorkerSharedTilePipeline() {
    return SharedForwardWorkerQuery("megaflux_shared_expert_tile_pipeline_version", 0) == 1;
}

void CheckMegaMoeBF16SharedSavedMode(uint32_t saved_mode, bool backward) {
    const auto bit = MegaMoeBF16Training::SharedSavedModeBit(saved_mode);
    const int mask = SharedForwardWorkerQuery(backward ?
        "megaflux_shared_expert_backward_saved_modes" : "megaflux_shared_expert_forward_saved_modes", 1);
    TORCH_CHECK(bit && (static_cast<uint32_t>(mask) & bit),
        "shared routed saved format is not supported by the selected worker; rebuild matched Host/worker");
}

void InitializeMegaMoeBF16Context(
    torch::Tensor context,
    void* symmetric_buffer,
    int rank,
    int world_size) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    const std::size_t context_bytes =
        static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext),
                "MegaMoe BF16 context tensor is too small");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    MegaMoeContext host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.rank_size_per_server = static_cast<uint32_t>(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_ptr = aclshmem_ptr(symmetric_buffer, peer);
        TORCH_CHECK(peer_ptr != nullptr,
                    "aclshmem_ptr failed for MegaMoe BF16 peer rank ", peer);
        host_context.ep_buffer[peer] = reinterpret_cast<uint64_t>(peer_ptr);
    }
    const auto copy_status = aclrtMemcpy(
        context.data_ptr(), context_bytes, &host_context, sizeof(host_context),
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copy_status == ACL_SUCCESS,
                "failed to initialize MegaMoe BF16 communication context, acl error ",
                copy_status);
}

void LaunchMegaMoeBF16Kernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor dispatch_output,
    torch::Tensor forward_transient_workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr,
    const std::vector<int64_t>& replica) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(
        world_size <= 1024,
        "MegaMoe BF16 supports at most 1024 EP ranks");
    if (timer.defined()) {
        TORCH_CHECK(
            timer.device().type() == c10::DeviceType::PrivateUse1 &&
                timer.is_contiguous() && timer.scalar_type() == torch::kInt64,
            "MegaMoe BF16 timer must be a contiguous NPU int64 tensor");
        TORCH_CHECK(
            timer.numel() >= AscendTimer::TOTAL_BUFFER_SIZE,
            "MegaMoe BF16 timer requires at least ",
            AscendTimer::TOTAL_BUFFER_SIZE,
            " int64 elements");
    }

    const std::size_t context_bytes =
        static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(
        context_bytes >= sizeof(MegaMoeContext),
        "MegaMoe BF16 context tensor is too small");

    aclrtStream stream = GetCurrentAclStream();

    if (replica.size() == 40) {
        CheckMegaMoeBF16SharedWorldSize(world_size);
        TORCH_CHECK(replica[25] == 1 && tiling.transW && block_dim <= 64,
                    "shared expert forward requires canonical S9 selector 1 and at most 64 AIC blocks");
        MegaMoeDeviceReplicaDescriptor descriptor{};
        MegaMoeFullDescriptor pipeline{};
        MegaMoeSharedExpertDescriptor shared{};
        std::memcpy(&descriptor, replica.data(), sizeof(descriptor));
        std::memcpy(&pipeline, replica.data(), sizeof(pipeline));
        std::memcpy(&shared, replica.data() + 26, sizeof(shared));
        TORCH_CHECK(descriptor.replica.home == weight1.size(0) &&
                        descriptor.replica.slots >= 0 &&
                        descriptor.replica.home + descriptor.replica.slots == tiling.expertPerRank &&
                        descriptor.replica.home1 == reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
                        descriptor.replica.home2 == reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
                        descriptor.replica.hidden == tiling.k &&
                        descriptor.replica.ffn * 2 == tiling.n &&
                        descriptor.replica.rank == rank && descriptor.replica.ranks == world_size,
                    "shared expert replica descriptor shape/owner/runtime mismatch");
        TORCH_CHECK(pipeline.forward.ready && pipeline.forward.stripes &&
                        pipeline.grad_ready && pipeline.forward.mode == 2,
                    "shared expert requires initialized S9 readiness storage");
        TORCH_CHECK(shared.abi == 1 &&
                        (shared.flags & ~uint64_t{3}) == 0 && shared.reserved == 0 &&
                        shared.tokens == static_cast<uint64_t>(tiling.m) &&
                        shared.hidden == static_cast<uint64_t>(tiling.k) &&
                        shared.ffn > 0 && shared.ffn % 256 == 0 &&
                        shared.weight_gate && shared.weight_up && shared.weight_down &&
                        shared.raw_gate && shared.raw_up && shared.activation && shared.ready &&
                        (timer.defined() == (shared.timer != 0)),
                    "invalid shared expert descriptor ABI/shape/storage");
        static void* library = dlopen(
            "libascend_deepep_megaflux_early_bf16.so", RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(library, "load shared expert mixed kernel: ", dlerror());
        auto abi = reinterpret_cast<int (*)()>(
            dlsym(library, "megaflux_shared_expert_forward_abi"));
        TORCH_CHECK(abi && abi() == 1,
                    "shared expert forward requires a shared-enabled S9 ABI1 library; rebuild the isolated worker");
        const auto savedMode = MegaMoeBF16Training::ResolveSavedGateUpMode(
            tiling.savedGateUpMode, tiling.swigluLimit);
        CheckMegaMoeBF16SharedSavedMode(savedMode, false);
        TORCH_CHECK(!(shared.flags & uint64_t{1}) ||
                        MegaMoeBF16WorkerSwigluDecoupled(tiling.k, savedMode),
                    "shared tile pipeline requires routed GM SwiGLU for the actual saved format");
        TORCH_CHECK(!(shared.flags & uint64_t{1}) || MegaMoeBF16WorkerSharedTilePipeline(),
                    "selected worker does not support shared tile pipeline");
        TORCH_CHECK(!(shared.flags & uint64_t{2}) || MegaMoeBF16WorkerSharedForwardTimerSchema() == 2,
                    "selected worker does not support forward detail Timer schema");
        if (timer.defined()) {
            auto enabled = reinterpret_cast<int (*)()>(
                dlsym(library, "megaflux_early_bf16_timer_enabled"));
            TORCH_CHECK(enabled && enabled(),
                        "shared expert Timer capture requires a Timer-ON shared-enabled worker");
        }
        auto entry = reinterpret_cast<decltype(&megaflux_early_bf16_shared_kernel_do)>(
            dlsym(library, "megaflux_early_bf16_shared_kernel_do"));
        TORCH_CHECK(entry, "shared-enabled worker does not export its ABI1 kernel entry");
        const auto homeTiling = MakeDeviceHomeTiling(tiling, descriptor);
        entry(block_dim, stream, BytePtr(context.data_ptr()), BytePtr(x.data_ptr()),
              BytePtr(topk_ids.data_ptr()), BytePtr(topk_weights.data_ptr()),
              BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
              timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
              BytePtr(y.data_ptr()), BytePtr(expert_token_nums.data_ptr()),
              BytePtr(training_saved_state.data_ptr()), BytePtr(dispatch_output.data_ptr()),
              BytePtr(forward_transient_workspace.data_ptr()),
              reinterpret_cast<const uint8_t*>(&tiling),
              reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, pipeline, shared);
        return;
    }

    auto kernel = ResolveMegaMoeBF16Kernel();
    if (timer.defined() && replica.size()!=25 && replica.size()!=26) {
        auto timer_enabled = reinterpret_cast<int (*)()>(
            dlsym(RTLD_DEFAULT, "mega_moe_bf16_timer_enabled"));
        TORCH_CHECK(
            timer_enabled != nullptr && timer_enabled() != 0,
            "MegaMoe BF16 timer was requested, but the kernel was built "
            "without timer instrumentation. Rebuild with "
            "scripts/build_mega_moe_bf16.sh --enable-timer");
    }
    if (replica.size()==25 || replica.size()==26) {
        const auto selector=replica.size()==26?replica[25]:0;
        TORCH_CHECK(selector>=0 && selector<=3,"invalid full executor selector");
        // Explicit opt-in selector3: local GEMMs read framework tensors while
        // descriptor.home1/2 remain SHMEM export sources. Selectors0..2 keep
        // their original pointer-identity contract unchanged.
        const bool externalHome=selector==3;
        const bool early=selector==1 || externalHome,prefetch=selector==2;
        MegaMoeDeviceReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            (externalHome ? (descriptor.replica.base && descriptor.replica.home1 && descriptor.replica.home2) :
             (descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
              descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()))) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        TORCH_CHECK(tiling.transW,"full pipeline requires canonical BF16 weights");
        TORCH_CHECK(block_dim<=128,"pipeline worker capacity exceeded");
        MegaMoeFullDescriptor pipeline{};
        std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
        TORCH_CHECK(pipeline.forward.ready && pipeline.forward.stripes,"pipeline ready storage missing");
        TORCH_CHECK(pipeline.grad_ready,"gradient publication storage missing");
        TORCH_CHECK(prefetch?((pipeline.forward.mode&255)==2 && (pipeline.forward.mode>>8)>=1 && (pipeline.forward.mode>>8)<=4):pipeline.forward.mode==2,"invalid full executor mode");
        void* library=nullptr;
        if(prefetch){static void* selected=dlopen("libascend_deepep_megaflux_prefetch_bf16.so",RTLD_NOW|RTLD_LOCAL);library=selected;}
        else if(early){static void* selected=dlopen("libascend_deepep_megaflux_early_bf16.so",RTLD_NOW|RTLD_LOCAL);library=selected;}
        else {static void* selected=dlopen("libascend_deepep_megaflux_full_bf16.so",RTLD_NOW|RTLD_LOCAL);library=selected;}
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        if(timer.defined()) {
            auto enabled=reinterpret_cast<int(*)()>(dlsym(library,prefetch?"megaflux_prefetch_bf16_timer_enabled":early?"megaflux_early_bf16_timer_enabled":"megaflux_full_bf16_timer_enabled"));
            TORCH_CHECK(enabled && enabled(),"Full replica Timer capture requires a Timer-ON Full SO");
        }
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoeFullDescriptor);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,prefetch?"megaflux_prefetch_bf16_kernel_do":early?"megaflux_early_bf16_kernel_do":"megaflux_full_bf16_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, pipeline);
    } else if (replica.size()==23) {
        TORCH_CHECK(replica.size()==23,"invalid replica descriptor ABI");
        MegaMoeDeviceReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        TORCH_CHECK(block_dim<=128,"pipeline worker capacity exceeded");
        MegaMoeStreamDescriptor pipeline{};
        std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
        TORCH_CHECK(pipeline.ready && pipeline.stripes,"pipeline ready storage missing");
        TORCH_CHECK(pipeline.mode<=2,"invalid stream mode");
        static void* library=dlopen("libascend_deepep_megaflux_stream_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoeStreamDescriptor);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_stream_bf16_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, pipeline);
    } else if (replica.size()==22) {
        TORCH_CHECK(replica.size()==22,"invalid replica descriptor ABI");
        MegaMoeDeviceReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        TORCH_CHECK(block_dim<=128,"pipeline worker capacity exceeded");
        MegaMoeStripeDescriptor pipeline{};
        std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
        TORCH_CHECK(pipeline.ready && pipeline.stripes,"pipeline ready storage missing");
        static void* library=dlopen("libascend_deepep_megaflux_stripe_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoeStripeDescriptor);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_stripe_bf16_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, pipeline);
    } else if (replica.size()==21) {
        TORCH_CHECK(replica.size()==21,"invalid replica descriptor ABI");
        MegaMoeDeviceReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        TORCH_CHECK(block_dim<=128,"pipeline worker capacity exceeded");
        MegaMoePipelineDescriptor pipeline{};
        std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
        TORCH_CHECK(pipeline.ready,"pipeline ready storage missing");
        static void* library=dlopen("libascend_deepep_megaflux_pipeline_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoePipelineDescriptor);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_pipeline_bf16_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, pipeline);
    } else if (replica.size()==19) {
        TORCH_CHECK(replica.size()==19,"invalid replica descriptor ABI");
        MegaMoeDeviceReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        static void* library=dlopen("libascend_deepep_megaflux_device_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoeDeviceReplicaDescriptor);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_device_bf16_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, descriptor);
    } else if (!replica.empty()) {
        TORCH_CHECK(replica.size()==14,"invalid replica descriptor ABI");
        MegaMoeReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.home==weight1.size(0) && descriptor.slots>=0 &&
            descriptor.home+descriptor.slots==tiling.expertPerRank &&
            descriptor.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.hidden==tiling.k && descriptor.ffn*2==tiling.n &&
            descriptor.rank==rank && descriptor.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        static void* library=dlopen("libascend_deepep_megaflux_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,uint64_t, MegaMoeReplicaDescriptor);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_bf16_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        ffts_addr, descriptor);
    } else {
    kernel(
        block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        dispatch_output.defined() ? BytePtr(dispatch_output.data_ptr()) : nullptr,
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        ffts_addr);
    }

    (void)symmetric_buffer_bytes;
}

void LaunchMegaMoeBF16MGroupKernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor forward_transient_workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(
        world_size <= 1024,
        "MegaMoe BF16 MGroup supports at most 1024 EP ranks");
    if (timer.defined()) {
        TORCH_CHECK(
            timer.device().type() == c10::DeviceType::PrivateUse1 &&
                timer.is_contiguous() && timer.scalar_type() == torch::kInt64,
            "MegaMoe BF16 MGroup timer must be a contiguous NPU int64 tensor");
        TORCH_CHECK(
            timer.numel() >= AscendTimer::TOTAL_BUFFER_SIZE,
            "MegaMoe BF16 MGroup timer requires at least ",
            AscendTimer::TOTAL_BUFFER_SIZE,
            " int64 elements");
    }

    const std::size_t context_bytes =
        static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(
        context_bytes >= sizeof(MegaMoeContext),
        "MegaMoe BF16 MGroup context tensor is too small");

    MegaMoeContext host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.rank_size_per_server = static_cast<uint32_t>(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_ptr = aclshmem_ptr(symmetric_buffer, peer);
        TORCH_CHECK(
            peer_ptr != nullptr,
            "aclshmem_ptr failed for MegaMoe BF16 MGroup peer rank ",
            peer);
        host_context.ep_buffer[peer] = reinterpret_cast<uint64_t>(peer_ptr);
    }

    aclrtStream stream = GetCurrentAclStream();
    const auto copy_status = aclrtMemcpy(
        context.data_ptr(),
        context_bytes,
        &host_context,
        sizeof(host_context),
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(
        copy_status == ACL_SUCCESS,
        "failed to copy MegaMoe BF16 MGroup communication context, acl error ",
        copy_status);

    const auto pre_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(
        pre_sync_status == ACL_SUCCESS,
        "failed to synchronize before MegaMoeBF16MGroup, acl error ",
        pre_sync_status);
    aclshmem_barrier_all();

    auto kernel = ResolveMegaMoeBF16MGroupKernel();
    if (timer.defined()) {
        auto timer_enabled = reinterpret_cast<int (*)()>(
            dlsym(RTLD_DEFAULT, "mega_moe_bf16_mgroup_timer_enabled"));
        TORCH_CHECK(
            timer_enabled != nullptr && timer_enabled() != 0,
            "MegaMoe BF16 MGroup timer was requested, but the kernel was built "
            "without timer instrumentation. Rebuild with "
            "scripts/build_mega_moe_bf16_mgroup.sh --enable-timer");
    }
    kernel(
        block_dim,
        stream,
        BytePtr(context.data_ptr()),
        BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()),
        BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),
        BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()),
        BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        ffts_addr);

    const auto post_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(
        post_sync_status == ACL_SUCCESS,
        "MegaMoeBF16MGroup kernel execution failed, acl error ",
        post_sync_status);
    aclshmem_barrier_all();
    (void)symmetric_buffer_bytes;
}

auto ResolveMegaMoeBF16BackwardKernel()
    -> decltype(&mega_moe_bf16_backward_ready_k_kernel_do) {
    static auto kernel = []() -> decltype(&mega_moe_bf16_backward_ready_k_kernel_do) {
        void* symbol = dlsym(RTLD_DEFAULT, "mega_moe_bf16_backward_ready_k_kernel_do");
        if (symbol == nullptr) {
            void* handle = dlopen(
                "libascend_deepep_mega_moe_bf16.so",
                RTLD_NOW | RTLD_GLOBAL);
            TORCH_CHECK(
                handle != nullptr,
                "MegaMoeBF16 kernel library is not loaded. Run "
                "scripts/build_mega_moe_bf16.sh first: ",
                dlerror());
            symbol = dlsym(handle, "mega_moe_bf16_backward_ready_k_kernel_do");
        }
        TORCH_CHECK(
            symbol != nullptr,
            "mega_moe_bf16_backward_ready_k_kernel_do was not exported by "
            "libascend_deepep_mega_moe_bf16.so");
        return reinterpret_cast<decltype(&mega_moe_bf16_backward_ready_k_kernel_do)>(symbol);
    }();
    return kernel;
}

void LaunchMegaMoeBF16BackwardKernel(
    torch::Tensor context,
    torch::Tensor grad_y,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor grad_x,
    torch::Tensor grad_topk_weights,
    torch::Tensor grad_weight1,
    torch::Tensor grad_weight2,
    torch::Tensor training_saved_state,
    torch::Tensor dispatch_output,
    torch::Tensor backward_workspace,
    torch::Tensor timer,
    const MegaMoeTilingData& tiling,
    uint32_t splitExperts,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr,
    const std::vector<int64_t>& replica) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(world_size <= 1024, "MegaMoe BF16 supports at most 1024 EP ranks");
    const bool noneGm = MegaMoeBF16Training::BackwardNoneGmRequested(splitExperts);
    TORCH_CHECK(!noneGm || (tiling.savedGateUpMode == MegaMoeBF16Training::kSaveNone &&
                    replica.size() == 26 && (replica[25] == 1 || replica[25] == 3)),
                "encoded none GM policy requires save=none and S9 worker selector 1/3");
    if (timer.defined()) {
        TORCH_CHECK(
            timer.device().type() == c10::DeviceType::PrivateUse1 &&
                timer.is_contiguous() && timer.scalar_type() == torch::kInt64,
            "MegaMoe BF16 backward timer must be a contiguous NPU int64 tensor");
        TORCH_CHECK(
            timer.numel() >= AscendTimer::TOTAL_BUFFER_SIZE,
            "MegaMoe BF16 backward timer requires at least ",
            AscendTimer::TOTAL_BUFFER_SIZE,
            " int64 elements");
    }
    const std::size_t context_bytes =
        static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext),
                "MegaMoe BF16 context tensor is too small");

    aclrtStream stream = GetCurrentAclStream();
    auto kernel = ResolveMegaMoeBF16BackwardKernel();
    if (timer.defined() && replica.size()!=25 && replica.size()!=26) {
        auto timer_enabled = reinterpret_cast<int (*)()>(
            dlsym(RTLD_DEFAULT, "mega_moe_bf16_backward_timer_enabled"));
        TORCH_CHECK(
            timer_enabled != nullptr && timer_enabled() != 0,
            "MegaMoe BF16 backward timer was requested, but the kernel was "
            "built without timer instrumentation. Rebuild with "
            "scripts/build_mega_moe_bf16.sh --enable-timer");
    }
    if (replica.size()==25 || replica.size()==26) {
        const auto selector=replica.size()==26?replica[25]:0;
        TORCH_CHECK(selector>=0 && selector<=3,"invalid full executor selector");
        const bool externalHome=selector==3;
        const bool early=selector==1 || externalHome,prefetch=selector==2;
        MegaMoeFullDescriptor pipeline{};
        std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
        const auto& descriptor=pipeline.forward.device;
        TORCH_CHECK(pipeline.grad_ready && tiling.transW,"full backward requires epochs and canonical weights");
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            (externalHome ? (descriptor.replica.base && descriptor.replica.home1 && descriptor.replica.home2) :
             (descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
              descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()))) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        TORCH_CHECK(prefetch?((pipeline.forward.mode&255)==2 && (pipeline.forward.mode>>8)>=1 && (pipeline.forward.mode>>8)<=4):pipeline.forward.mode==2,"invalid full backward mode");
        void* library=nullptr;
        if(prefetch){static void* selected=dlopen("libascend_deepep_megaflux_prefetch_bf16.so",RTLD_NOW|RTLD_LOCAL);library=selected;}
        else if(early){static void* selected=dlopen("libascend_deepep_megaflux_early_bf16.so",RTLD_NOW|RTLD_LOCAL);library=selected;}
        else {static void* selected=dlopen("libascend_deepep_megaflux_full_bf16.so",RTLD_NOW|RTLD_LOCAL);library=selected;}
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        if (noneGm) {
            auto capability = reinterpret_cast<int (*)()>(dlsym(library, "megaflux_backward_none_gm_version"));
            TORCH_CHECK(capability && capability() == 1,
                        "selected backward SO does not support the encoded none GM policy");
        }
        if(timer.defined()) {
            auto enabled=reinterpret_cast<int(*)()>(dlsym(library,prefetch?"megaflux_prefetch_bf16_backward_timer_enabled":early?"megaflux_early_bf16_backward_timer_enabled":"megaflux_full_bf16_backward_timer_enabled"));
            TORCH_CHECK(enabled && enabled(),"Full replica backward Timer capture requires a Timer-ON Full SO");
        }
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoeFullDescriptor, uint32_t);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,prefetch?"megaflux_prefetch_bf16_backward_ready_k_kernel_do":early?"megaflux_early_bf16_backward_ready_k_kernel_do":"megaflux_full_bf16_backward_ready_k_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim, stream, BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()), BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()), BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),
        BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling), reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, pipeline, splitExperts);
    } else if (replica.size()==19) {
        TORCH_CHECK(replica.size()==19,"invalid replica descriptor ABI");
        MegaMoeDeviceReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.replica.home==weight1.size(0) && descriptor.replica.slots>=0 &&
            descriptor.replica.home+descriptor.replica.slots==tiling.expertPerRank &&
            descriptor.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.replica.hidden==tiling.k && descriptor.replica.ffn*2==tiling.n &&
            descriptor.replica.rank==rank && descriptor.replica.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        static void* library=dlopen("libascend_deepep_megaflux_device_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        const auto homeTiling=MakeDeviceHomeTiling(tiling,descriptor);
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,const uint8_t*,uint64_t, MegaMoeDeviceReplicaDescriptor, uint32_t);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_device_bf16_backward_ready_k_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim, stream, BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()), BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()), BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),
        BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling), reinterpret_cast<const uint8_t*>(&homeTiling), ffts_addr, descriptor, splitExperts);
    } else if (!replica.empty()) {
        TORCH_CHECK(replica.size()==14,"invalid replica descriptor ABI");
        MegaMoeReplicaDescriptor descriptor{};
        std::memcpy(&descriptor,replica.data(),sizeof(descriptor));
        TORCH_CHECK(descriptor.home==weight1.size(0) && descriptor.slots>=0 &&
            descriptor.home+descriptor.slots==tiling.expertPerRank &&
            descriptor.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
            descriptor.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()) &&
            descriptor.hidden==tiling.k && descriptor.ffn*2==tiling.n &&
            descriptor.rank==rank && descriptor.ranks==world_size,
            "replica descriptor shape/owner/runtime mismatch");
        static void* library=dlopen("libascend_deepep_megaflux_bf16.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(library,"load MegaFlux mixed kernels: ",dlerror());
        using ReplicaKernel=void(*)(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,const uint8_t*,uint64_t, MegaMoeReplicaDescriptor, uint32_t);
        auto entry=reinterpret_cast<ReplicaKernel>(dlsym(library,"megaflux_bf16_backward_ready_k_kernel_do"));
        TORCH_CHECK(entry,"missing MegaFlux kernel entry");
        entry(block_dim, stream, BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()), BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()), BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),
        BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling), ffts_addr, descriptor, splitExperts);
    } else {
    kernel(
        block_dim, stream, BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()), BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()), BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),
        BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling), ffts_addr, splitExperts);
    }
    (void)symmetric_buffer_bytes;
}

namespace {
void* SharedBackwardLibrary() {
    static void* library=dlopen("libascend_deepep_megaflux_early_bf16.so",RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(library,"load shared backward worker: ",dlerror());
    return library;
}
}

void CheckMegaMoeBF16SharedBackwardCapability() {
    auto library=SharedBackwardLibrary();
    const auto require=[&](const char* symbol,int expected) {
        auto getter=reinterpret_cast<int(*)()>(dlsym(library,symbol));
        TORCH_CHECK(getter && getter()==expected,
                    "shared backward requires ",symbol," == ",expected,
                    "; rebuild/select the isolated matching worker SO");
    };
    require("megaflux_shared_expert_backward_version",1);
    require("megaflux_worker_pipeline_flags",0);
    require("megaflux_worker_owner_rows",16);
    require("megaflux_worker_owner_compact",0);
    require("megaflux_backward_restore_mode",2);
    require("megaflux_restore_operand_ready",1);
    require("megaflux_restore_delay_cycles",0);
    require("megaflux_backward_replicated_dw1_early",1);
    require("megaflux_early_bf16_timer_enabled",0);
    require("megaflux_early_bf16_backward_timer_enabled",0);
    TORCH_CHECK(dlsym(library,"megaflux_worker_bf16_backward_shared"),
                "selected worker lacks shared backward ABI1 entry");
}

void LaunchMegaMoeBF16BackwardSharedKernel(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
    torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
    torch::Tensor grad_weight2, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, torch::Tensor backward_workspace,
    const MegaMoeTilingData& tiling, uint32_t split_experts, uint32_t block_dim,
    void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
    int rank, int world_size, uint64_t ffts_addr,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgs& shared) {
    CheckMegaMoeBF16SharedBackwardCapability();
    TORCH_CHECK(symmetric_buffer && replica.size()==26 &&
                (replica[25]==1 || replica[25]==3),
                "shared backward requires isolated early/external-home S9 descriptor");
    TORCH_CHECK(world_size==2 || world_size==4 || world_size==8,
                "shared backward supports EP2/4/8");
    TORCH_CHECK(context.device()==grad_y.device() && context.is_contiguous() &&
                context.numel()*context.element_size()>=sizeof(MegaMoeContext),
                "shared backward context mismatch");
    TORCH_CHECK(shared.version==1 && shared.bytes==sizeof(shared) && shared.flags==0 &&
                shared.m==tiling.m && shared.hidden==tiling.k && shared.epoch>0,
                "shared backward descriptor ABI/geometry mismatch");
    MegaMoeFullDescriptor pipeline{};std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
    const auto& d=pipeline.forward.device;
    const bool external=replica[25]==3;
    TORCH_CHECK(pipeline.grad_ready && tiling.transW && pipeline.forward.mode==2 &&
                d.replica.home==weight1.size(0) && d.replica.slots>=0 &&
                d.replica.home+d.replica.slots==tiling.expertPerRank &&
                d.replica.hidden==tiling.k && d.replica.ffn*2==tiling.n &&
                d.replica.rank==rank && d.replica.ranks==world_size &&
                (external ? (d.replica.base && d.replica.home1 && d.replica.home2) :
                 (d.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
                  d.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()))),
                "shared backward replica owner/runtime/shape mismatch");
    const auto home=MakeDeviceHomeTiling(tiling,d);
    using Entry=void(*)(uint32_t,void*,
        uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
        uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
        const uint8_t*,const uint8_t*,uint64_t,MegaMoeFullDescriptor,uint32_t,
        MegaMoeSharedExpert::SharedExpertBackwardArgs);
    auto entry=reinterpret_cast<Entry>(dlsym(SharedBackwardLibrary(),"megaflux_worker_bf16_backward_shared"));
    entry(block_dim,GetCurrentAclStream(),BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()),BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()),BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),nullptr,
        reinterpret_cast<const uint8_t*>(&tiling),reinterpret_cast<const uint8_t*>(&home),
        ffts_addr,pipeline,split_experts,shared);
    (void)symmetric_buffer_bytes;
}

void CheckMegaMoeBF16SharedBackwardCapabilityV2(bool timer) {
    auto library=SharedBackwardLibrary();
    const auto require=[&](const char* symbol,int expected) {
        auto getter=reinterpret_cast<int(*)()>(dlsym(library,symbol));
        TORCH_CHECK(getter && getter()==expected,
                    "shared backward requires ",symbol," == ",expected,
                    "; rebuild/select the isolated matching worker SO");
    };
    require("megaflux_shared_expert_backward_v2_version",2);
    require("megaflux_worker_pipeline_flags",0);
    require("megaflux_worker_owner_rows",16);
    require("megaflux_worker_owner_compact",0);
    require("megaflux_backward_restore_mode",2);
    require("megaflux_restore_operand_ready",1);
    require("megaflux_restore_delay_cycles",0);
    require("megaflux_backward_replicated_dw1_early",1);
    require("megaflux_early_bf16_timer_enabled",timer ? 1 : 0);
    require("megaflux_early_bf16_backward_timer_enabled",timer ? 1 : 0);
    TORCH_CHECK(dlsym(library,"megaflux_worker_bf16_backward_shared_v2"),
                "selected worker lacks shared backward V2 entry");
}

void LaunchMegaMoeBF16BackwardSharedKernelV2(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
    torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
    torch::Tensor grad_weight2, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, torch::Tensor backward_workspace, torch::Tensor timer,
    const MegaMoeTilingData& tiling, uint32_t split_experts, uint32_t block_dim,
    void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
    int rank, int world_size, uint64_t ffts_addr,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgsV2& shared) {
    CheckMegaMoeBF16SharedBackwardCapabilityV2(timer.defined());
    TORCH_CHECK(symmetric_buffer && replica.size()==26 &&
                (replica[25]==1 || replica[25]==3),
                "shared backward requires isolated early/external-home S9 descriptor");
    CheckMegaMoeBF16SharedWorldSize(world_size);
    TORCH_CHECK(context.device()==grad_y.device() && context.is_contiguous() &&
                context.numel()*context.element_size()>=sizeof(MegaMoeContext),
                "shared backward context mismatch");
    TORCH_CHECK(shared.version==2 && shared.recompute<=1 && shared.raw_gate && shared.raw_up &&
                shared.base.version==1 && shared.base.bytes==sizeof(shared.base) && shared.base.flags==0 &&
                shared.base.m==tiling.m && shared.base.hidden==tiling.k && shared.base.epoch>0 &&
                block_dim>0 && block_dim<=32 && (timer.defined()==(shared.timer!=0)),
                "shared backward descriptor ABI/geometry mismatch");
    MegaMoeFullDescriptor pipeline{};std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
    const auto& d=pipeline.forward.device;
    const bool external=replica[25]==3;
    TORCH_CHECK(pipeline.grad_ready && tiling.transW && pipeline.forward.mode==2 &&
                d.replica.home==weight1.size(0) && d.replica.slots>=0 &&
                d.replica.home+d.replica.slots==tiling.expertPerRank &&
                d.replica.hidden==tiling.k && d.replica.ffn*2==tiling.n &&
                d.replica.rank==rank && d.replica.ranks==world_size &&
                (external ? (d.replica.base && d.replica.home1 && d.replica.home2) :
                 (d.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
                  d.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()))),
                "shared backward replica owner/runtime/shape mismatch");
    const auto home=MakeDeviceHomeTiling(tiling,d);
    using Entry=decltype(&megaflux_worker_bf16_backward_shared_v2);
    auto entry=reinterpret_cast<Entry>(dlsym(SharedBackwardLibrary(),"megaflux_worker_bf16_backward_shared_v2"));
    entry(block_dim,GetCurrentAclStream(),BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()),BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()),BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling),reinterpret_cast<const uint8_t*>(&home),
        ffts_addr,pipeline,split_experts,shared);
    (void)symmetric_buffer_bytes;
}

void CheckMegaMoeBF16SharedBackwardCapabilityV3(bool timer) {
    // The V3 image also preserves the entire V2 numerical implementation.
    CheckMegaMoeBF16SharedBackwardCapabilityV2(timer);
    auto library=SharedBackwardLibrary();
    const auto require=[&](const char* symbol,int expected) {
        auto getter=reinterpret_cast<int(*)()>(dlsym(library,symbol));
        TORCH_CHECK(getter && getter()==expected,
                    "X redispatch backward requires ",symbol," == ",expected,
                    "; rebuild/select the matching V3 worker SO");
    };
    require("megaflux_shared_expert_backward_v3_version",3);
    require("megaflux_shared_expert_redispatch_timer_phases",
            MegaMoeSharedExpert::kBackwardRedispatchTimerPhases);
    require("megaflux_shared_expert_redispatch_timer_lanes",
            MegaMoeSharedExpert::kBackwardRedispatchTimerLanes);
    TORCH_CHECK(dlsym(library,"megaflux_worker_bf16_backward_shared_v3"),
                "selected worker lacks X redispatch backward V3 entry");
}

void LaunchMegaMoeBF16BackwardSharedKernelV3(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor grad_x,
    torch::Tensor grad_topk_weights, torch::Tensor grad_weight1,
    torch::Tensor grad_weight2, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, torch::Tensor backward_workspace, torch::Tensor timer,
    const MegaMoeTilingData& tiling, uint32_t split_experts, uint32_t block_dim,
    void* symmetric_buffer, std::size_t symmetric_buffer_bytes,
    int rank, int world_size, uint64_t ffts_addr,
    const std::vector<int64_t>& replica,
    const MegaMoeSharedExpert::SharedExpertBackwardArgsV3& extended) {
    CheckMegaMoeBF16SharedBackwardCapabilityV3(timer.defined());
    TORCH_CHECK(symmetric_buffer && replica.size()==26 &&
                (replica[25]==1 || replica[25]==3),
                "X redispatch backward requires isolated early/external-home S9 descriptor");
    const bool routedOnly=extended.flags==MegaMoeSharedExpert::kBackwardRoutedOnlyRedispatchFlag;
    if(routedOnly)CheckMegaMoeBF16RoutedRedispatchCapability(world_size);
    else CheckMegaMoeBF16SharedWorldSize(world_size);
    TORCH_CHECK(context.device()==grad_y.device() && context.is_contiguous() &&
                context.numel()*context.element_size()>=sizeof(MegaMoeContext),
                "X redispatch backward context mismatch");
    const auto& shared=extended.shared;
    TORCH_CHECK(extended.version==3 &&
                (routedOnly || extended.flags==MegaMoeSharedExpert::kBackwardRedispatchFlag) &&
                shared.version==2 && shared.recompute<=1 &&
                (routedOnly ? (replica[25]==1 && shared.base.control && shared.base.x) :
                              (shared.raw_gate && shared.raw_up)) &&
                shared.base.version==1 && shared.base.bytes==sizeof(shared.base) && shared.base.flags==0 &&
                shared.base.m==tiling.m && shared.base.hidden==tiling.k && shared.base.epoch>0 &&
                block_dim>0 && block_dim<=32 &&
                (routedOnly ? shared.timer==0 : timer.defined()==(shared.timer!=0)) &&
                (timer.defined()==(extended.timer!=0)),
                "X redispatch backward descriptor ABI/geometry mismatch");
    CheckMegaMoeBF16SharedSavedMode(MegaMoeBF16Training::ResolveSavedGateUpMode(
        tiling.savedGateUpMode, tiling.swigluLimit), true);
    MegaMoeFullDescriptor pipeline{};std::memcpy(&pipeline,replica.data(),sizeof(pipeline));
    const auto& d=pipeline.forward.device;
    const bool external=replica[25]==3;
    TORCH_CHECK(pipeline.grad_ready && tiling.transW && pipeline.forward.mode==2 &&
                d.replica.home==weight1.size(0) && d.replica.slots>=0 &&
                d.replica.home+d.replica.slots==tiling.expertPerRank &&
                d.replica.hidden==tiling.k && d.replica.ffn*2==tiling.n &&
                d.replica.rank==rank && d.replica.ranks==world_size &&
                (external ? (d.replica.base && d.replica.home1 && d.replica.home2) :
                 (d.replica.home1==reinterpret_cast<uint64_t>(weight1.data_ptr()) &&
                  d.replica.home2==reinterpret_cast<uint64_t>(weight2.data_ptr()))),
                "X redispatch backward replica owner/runtime/shape mismatch");
    const auto home=MakeDeviceHomeTiling(tiling,d);
    using Entry=decltype(&megaflux_worker_bf16_backward_shared_v3);
    auto entry=reinterpret_cast<Entry>(dlsym(SharedBackwardLibrary(),"megaflux_worker_bf16_backward_shared_v3"));
    entry(block_dim,GetCurrentAclStream(),BytePtr(context.data_ptr()),
        BytePtr(grad_y.data_ptr()),BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()),BytePtr(weight2.data_ptr()),
        BytePtr(grad_x.data_ptr()),BytePtr(grad_topk_weights.data_ptr()),
        BytePtr(grad_weight1.data_ptr()),BytePtr(grad_weight2.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),BytePtr(dispatch_output.data_ptr()),
        BytePtr(backward_workspace.data_ptr()),timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(&tiling),reinterpret_cast<const uint8_t*>(&home),
        ffts_addr,pipeline,split_experts,extended);
    (void)symmetric_buffer_bytes;
}

void LaunchMegaMoeBF16UDMAKernel(
    torch::Tensor context,
    torch::Tensor x,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor weight1,
    torch::Tensor weight2,
    torch::Tensor timer,
    torch::Tensor y,
    torch::Tensor expert_token_nums,
    torch::Tensor training_saved_state,
    torch::Tensor forward_transient_workspace,
    const MegaMoeTilingData& tiling,
    uint32_t block_dim,
    void* symmetric_buffer,
    std::size_t symmetric_buffer_bytes,
    int rank,
    int world_size,
    uint64_t ffts_addr) {
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(world_size <= 1024,
                "MegaMoe BF16 UDMA supports at most 1024 EP ranks");
    if (timer.defined()) {
        TORCH_CHECK(
            timer.device().type() == c10::DeviceType::PrivateUse1 &&
                timer.is_contiguous() && timer.scalar_type() == torch::kInt64,
            "MegaMoe BF16 UDMA timer must be a contiguous NPU int64 tensor");
        TORCH_CHECK(
            timer.numel() >= AscendTimer::TOTAL_BUFFER_SIZE,
            "MegaMoe BF16 UDMA timer requires at least ",
            AscendTimer::TOTAL_BUFFER_SIZE,
            " int64 elements");
    }
    const std::size_t context_bytes =
        static_cast<std::size_t>(context.numel()) * context.element_size();
    TORCH_CHECK(context_bytes >= sizeof(MegaMoeContext),
                "MegaMoe BF16 UDMA context tensor is too small");

    MegaMoeContext host_context{};
    host_context.ep_rank_id = static_cast<uint32_t>(rank);
    host_context.rank_size_per_server = static_cast<uint32_t>(world_size);
    for (int peer = 0; peer < world_size; ++peer) {
        void* peer_ptr = aclshmem_ptr(symmetric_buffer, peer);
        TORCH_CHECK(peer_ptr != nullptr,
                    "aclshmem_ptr failed for MegaMoe BF16 UDMA peer rank ", peer);
        host_context.ep_buffer[peer] = reinterpret_cast<uint64_t>(peer_ptr);
    }

    aclrtStream stream = GetCurrentAclStream();
    const auto copy_status = aclrtMemcpy(
        context.data_ptr(), context_bytes, &host_context, sizeof(host_context),
        ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copy_status == ACL_SUCCESS,
                "failed to copy MegaMoe BF16 UDMA communication context, acl error ",
                copy_status);
    const auto pre_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(pre_sync_status == ACL_SUCCESS,
                "failed to synchronize before MegaMoeBF16UDMA, acl error ",
                pre_sync_status);
    aclshmem_barrier_all();

    auto kernel = ResolveMegaMoeBF16UDMAKernel();
    if (timer.defined()) {
        auto timer_enabled = reinterpret_cast<int (*)()>(
            dlsym(RTLD_DEFAULT, "mega_moe_bf16_udma_timer_enabled"));
        TORCH_CHECK(
            timer_enabled != nullptr && timer_enabled() != 0,
            "MegaMoe BF16 UDMA timer was requested, but the kernel was built "
            "without timer instrumentation. Rebuild with "
            "scripts/build_mega_moe_bf16_udma.sh --enable-timer");
    }
    kernel(
        block_dim, stream, BytePtr(context.data_ptr()), BytePtr(x.data_ptr()),
        BytePtr(topk_ids.data_ptr()), BytePtr(topk_weights.data_ptr()),
        BytePtr(weight1.data_ptr()), BytePtr(weight2.data_ptr()),
        timer.defined() ? BytePtr(timer.data_ptr()) : nullptr,
        BytePtr(y.data_ptr()), BytePtr(expert_token_nums.data_ptr()),
        BytePtr(training_saved_state.data_ptr()),
        BytePtr(forward_transient_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(&tiling),
        ffts_addr);
    const auto post_sync_status = aclrtSynchronizeStream(stream);
    TORCH_CHECK(post_sync_status == ACL_SUCCESS,
                "MegaMoeBF16UDMA kernel execution failed, acl error ",
                post_sync_status);
    aclshmem_barrier_all();
    (void)symmetric_buffer_bytes;
}
