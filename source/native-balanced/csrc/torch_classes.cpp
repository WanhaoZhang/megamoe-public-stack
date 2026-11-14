#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>

#include <acl/acl.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "shmem.h"

#include "csrc/common/check.h"
#include "csrc/launcher.h"
#include "csrc/ops/allgather_matmul.h"
#include "csrc/ops/combine.h"
#include "csrc/ops/dispatch.h"
#include "csrc/ops/mega_moe.h"
#include "csrc/ops/mega_moe_bf16.h"
#include "csrc/ops/mega_moe_bf16_mgroup.h"
#include "csrc/ops/mega_moe_bf16_udma.h"
#include "csrc/ops/mega_moe_w4a8.h"
#include "kernels/common/timer/AscendTimerV2.hpp"
#include "kernels/mega_moe_bf16/include/mega_moe_shared_expert_types.h"
#include "csrc/shmem_instance.h"

namespace ascend_deepep::torch_classes {
namespace {

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;
constexpr int64_t kMegaMoeContextBytes =
    static_cast<int64_t>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025);

std::size_t CheckedLocalMemSize(int64_t local_mem_size) {
    TORCH_CHECK(
        local_mem_size > 0,
        "local_mem_size must be greater than zero");
    TORCH_CHECK(
        local_mem_size % static_cast<int64_t>(kDeviceLargePageBytes) == 0,
        "local_mem_size must be aligned to DEVICE_LARGE_PAGE_SIZE (",
        kDeviceLargePageBytes,
        " bytes)");
    return static_cast<std::size_t>(local_mem_size);
}

void CheckShmem(const c10::intrusive_ptr<ShmemInstance>& shmem) {
    TORCH_CHECK(shmem != nullptr, "shmem must not be None");
    TORCH_CHECK(shmem->Initialized(), "ShmemInstance is not initialized");
}

}  // namespace

class ShmemWorkspaceOp : public torch::jit::CustomClassHolder {
public:
    explicit ShmemWorkspaceOp(int64_t local_mem_size)
        : shmem_ptr_(nullptr),
          local_mem_size_(CheckedLocalMemSize(local_mem_size)),
          owns_memory_(true) {
        shmem_ptr_ = aclshmem_malloc(local_mem_size_);
        if (shmem_ptr_ == nullptr) {
            throw std::runtime_error("aclshmem_malloc returned nullptr for op local memory");
        }
        ASCEND_DEEPEP_CHECK_ACL(aclrtMemset(shmem_ptr_, local_mem_size_, 0, local_mem_size_));
    }

    ~ShmemWorkspaceOp() override {
        destroy();
    }

    void destroy() {
        if (shmem_ptr_ != nullptr && owns_memory_) {
            aclshmem_free(shmem_ptr_);
        }
        shmem_ptr_ = nullptr;
    }

    int64_t local_mem_size() const {
        return static_cast<int64_t>(local_mem_size_);
    }

protected:
    void* SymmetricBuffer() const {
        return RawPtr();
    }

    std::size_t SymmetricBufferBytes() const {
        return local_mem_size_;
    }

protected:
    // ``external_sym_ptr`` == 0 allocates from the active heap. Otherwise it is
    // a symmetric address the caller already registered; destroy must not free it.
    ShmemWorkspaceOp(int64_t local_mem_size, int64_t external_sym_ptr)
        : shmem_ptr_(nullptr),
          local_mem_size_(CheckedLocalMemSize(local_mem_size)),
          owns_memory_(external_sym_ptr == 0) {
        if (owns_memory_) {
            shmem_ptr_ = aclshmem_malloc(local_mem_size_);
            if (shmem_ptr_ == nullptr) {
                throw std::runtime_error("aclshmem_malloc returned nullptr for op local memory");
            }
        } else {
            shmem_ptr_ = reinterpret_cast<void*>(static_cast<uintptr_t>(external_sym_ptr));
        }
        ASCEND_DEEPEP_CHECK_ACL(aclrtMemset(shmem_ptr_, local_mem_size_, 0, local_mem_size_));
    }

private:
    void* RawPtr() const {
        TORCH_CHECK(shmem_ptr_ != nullptr, "op local SHMEM memory has been destroyed");
        return shmem_ptr_;
    }

    void* shmem_ptr_;
    std::size_t local_mem_size_;
    bool owns_memory_;
};

class Dispatch : public ShmemWorkspaceOp {
public:
    explicit Dispatch(int64_t local_mem_size) : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, torch::Tensor> compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_idx,
        c10::optional<torch::Tensor> topk_weights,
        int64_t num_experts,
        int64_t num_max_tokens_per_rank,
        int64_t expert_alignment) {
        CheckShmem(shmem);
        return ascend_deepep::ops::Dispatch(
            x,
            topk_idx,
            topk_weights,
            num_experts,
            num_max_tokens_per_rank,
            expert_alignment,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            *shmem,
            SymmetricBuffer());
    }
};

class Combine : public ShmemWorkspaceOp {
public:
    explicit Combine(int64_t local_mem_size) : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    std::tuple<torch::Tensor> compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor expert_output,
        torch::Tensor handle,
        c10::optional<torch::Tensor> topk_weights) {
        CheckShmem(shmem);
        return ascend_deepep::ops::Combine(
            expert_output,
            handle,
            topk_weights,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            *shmem,
            SymmetricBuffer());
    }
};

class AllGatherMatmul : public ShmemWorkspaceOp {
public:
    explicit AllGatherMatmul(int64_t local_mem_size) : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    torch::Tensor compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor a,
        torch::Tensor b) {
        CheckShmem(shmem);
        return ascend_deepep::ops::AllGatherMatmul(
            a,
            b,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),
            *shmem,
            SymmetricBuffer());
    }

    torch::Tensor compute_out(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor out) {
        CheckShmem(shmem);
        return ascend_deepep::ops::AllGatherMatmulOut(
            a,
            b,
            out,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),
            *shmem,
            SymmetricBuffer());
    }
};

class MegaMoe : public ShmemWorkspaceOp {
public:
    explicit MegaMoe(int64_t local_mem_size)
        : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor weight_scales1,
        torch::Tensor weight_scales2,
        int64_t max_recv_tokens) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoe(
            x,
            topk_ids,
            topk_weights,
            weight1,
            weight2,
            weight_scales1,
            weight_scales2,
            max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),
            *shmem,
            SymmetricBuffer(),
            torch::Tensor());
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor weight_scales1,
        torch::Tensor weight_scales2,
        int64_t max_recv_tokens) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            x.options().dtype(torch::kInt64));
        auto [y, expert_token_nums] = ascend_deepep::ops::MegaMoe(
            x, topk_ids, topk_weights, weight1, weight2, weight_scales1,
            weight_scales2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), timer);
        return {y, expert_token_nums, timer};
    }
};

class MegaMoeBF16 : public ShmemWorkspaceOp {
public:
    explicit MegaMoeBF16(int64_t local_mem_size)
        : ShmemWorkspaceOp(local_mem_size) {
        // Ordinary BF16 keeps the generation header across launches. No rank
        // may publish its first device-side signal until every peer has
        // completed the one-time workspace memset.
        aclshmem_barrier_all();
    }

    // external_sym_ptr == 0 allocates with aclshmem_malloc. A non-zero value is
    // a symmetric address already registered in the active heap.
    MegaMoeBF16(int64_t local_mem_size, int64_t external_sym_ptr)
        : ShmemWorkspaceOp(local_mem_size, external_sym_ptr) {
        aclshmem_barrier_all();
    }

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    // Training ABI: packed Dispatch output is independent from compact saved
    // state so the framework can decide whether to retain it for backward.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    compute_train_replica(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output, std::vector<int64_t> replica) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16Forward(
            Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), swiglu_limit, save_gmm1_output, replica);
    }

    ascend_deepep::ops::MegaMoeBF16SharedForwardResult compute_train_replica_shared(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor shared_weight_gate,
        torch::Tensor shared_weight_up, torch::Tensor shared_weight_down) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16SharedForward(
            Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), swiglu_limit,
            save_gmm1_output, replica, shared_weight_gate,
            shared_weight_up, shared_weight_down);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
    compute_train_replica_shared_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor shared_weight_gate,
        torch::Tensor shared_weight_up, torch::Tensor shared_weight_down) {
        CheckShmem(shmem);
        auto timer = torch::zeros({AscendTimer::TOTAL_BUFFER_SIZE},
                                  x.options().dtype(torch::kInt64));
        auto result = ascend_deepep::ops::MegaMoeBF16SharedForward(
            Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), timer, swiglu_limit,
            save_gmm1_output, replica, shared_weight_gate,
            shared_weight_up, shared_weight_down);
        return std::tuple_cat(result, std::make_tuple(timer));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backward_replica(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor expert_token_nums,
        torch::Tensor training_saved_state,
        torch::Tensor dispatch_output,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output, std::vector<int64_t> replica) {
        CheckShmem(shmem);
        auto [grad_x, grad_topk_weights, grad_weight1, grad_weight2,
              unused_backward_workspace] =
            ascend_deepep::ops::MegaMoeBF16Backward(
                Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2, expert_token_nums,
                training_saved_state, dispatch_output,
                max_recv_tokens,
                static_cast<int>(shmem->Rank()),
                static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
                *shmem, SymmetricBuffer(), torch::Tensor(), swiglu_limit, save_gmm1_output, replica);
        (void)unused_backward_workspace;
        return {grad_x, grad_topk_weights, grad_weight1, grad_weight2};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
    backward_replica_shared(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2,
        torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
        torch::Tensor dispatch_output, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor x,
        torch::Tensor shared_wg, torch::Tensor shared_wu,
        torch::Tensor shared_wd, torch::Tensor raw_gu,
        torch::Tensor activation, double shared_limit) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16BackwardShared(
            Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2,
            expert_token_nums, training_saved_state, dispatch_output,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), swiglu_limit, save_gmm1_output, replica,
            x, shared_wg, shared_wu, shared_wd, raw_gu, activation, shared_limit);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
    backward_replica_shared_v2(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2,
        torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
        torch::Tensor dispatch_output, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor x,
        torch::Tensor shared_wg, torch::Tensor shared_wu,
        torch::Tensor shared_wd, torch::Tensor raw_gate, torch::Tensor raw_up,
        torch::Tensor activation, double shared_limit, bool recompute) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16BackwardSharedV2(
            Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2,
            expert_token_nums, training_saved_state, dispatch_output,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), swiglu_limit, save_gmm1_output, replica,
            x, shared_wg, shared_wu, shared_wd, raw_gate, raw_up, activation, shared_limit, recompute);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_replica_shared_v2_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2,
        torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
        torch::Tensor dispatch_output, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor x,
        torch::Tensor shared_wg, torch::Tensor shared_wu,
        torch::Tensor shared_wd, torch::Tensor raw_gate, torch::Tensor raw_up,
        torch::Tensor activation, double shared_limit, bool recompute) {
        CheckShmem(shmem);
        auto timer = torch::zeros({AscendTimer::TOTAL_BUFFER_SIZE}, grad_y.options().dtype(torch::kInt64));
        auto shared_timer = torch::zeros({10,128,2}, grad_y.options().dtype(torch::kInt64));
        auto result = ascend_deepep::ops::MegaMoeBF16BackwardSharedV2(
            Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2,
            expert_token_nums, training_saved_state, dispatch_output,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), swiglu_limit, save_gmm1_output, replica,
            x, shared_wg, shared_wu, shared_wd, raw_gate, raw_up, activation, shared_limit, recompute, timer, shared_timer);
        return std::tuple_cat(result, std::make_tuple(timer, shared_timer));
    }

    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
    backward_replica_redispatch(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad, torch::Tensor prob, torch::Tensor w1, torch::Tensor w2,
        torch::Tensor counts, torch::Tensor state, int64_t capacity,
        double limit, std::string save_mode, std::vector<int64_t> replica,
        torch::Tensor original_x) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16BackwardRedispatch(
            Context(shmem,grad),grad,prob,w1,w2,counts,state,capacity,
            static_cast<int>(shmem->Rank()),static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),*shmem,SymmetricBuffer(),limit,save_mode,replica,original_x);
    }

    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
               torch::Tensor,torch::Tensor>
    backward_replica_redispatch_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad, torch::Tensor prob, torch::Tensor w1, torch::Tensor w2,
        torch::Tensor counts, torch::Tensor state, int64_t capacity,
        double limit, std::string save_mode, std::vector<int64_t> replica,
        torch::Tensor original_x) {
        CheckShmem(shmem);
        auto timer=torch::zeros({AscendTimer::TOTAL_BUFFER_SIZE},grad.options().dtype(torch::kInt64));
        auto transport_timer=torch::zeros({4,128,2},grad.options().dtype(torch::kInt64));
        auto result=ascend_deepep::ops::MegaMoeBF16BackwardRedispatch(
            Context(shmem,grad),grad,prob,w1,w2,counts,state,capacity,
            static_cast<int>(shmem->Rank()),static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),*shmem,SymmetricBuffer(),limit,save_mode,replica,
            original_x,timer,transport_timer);
        return std::tuple_cat(result,std::make_tuple(timer,transport_timer));
    }

    ascend_deepep::ops::MegaMoeBF16SharedBackwardV3Result
    backward_replica_shared_v3(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2,
        torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
        torch::Tensor dispatch_output, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor x,
        torch::Tensor shared_wg, torch::Tensor shared_wu,
        torch::Tensor shared_wd, torch::Tensor raw_gate, torch::Tensor raw_up,
        torch::Tensor activation, double shared_limit, bool recompute) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16BackwardSharedV3(
            Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2,
            expert_token_nums, training_saved_state, dispatch_output,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), swiglu_limit, save_gmm1_output, replica,
            x, shared_wg, shared_wu, shared_wd, raw_gate, raw_up, activation,
            shared_limit, recompute);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
    backward_replica_shared_v3_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2,
        torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
        torch::Tensor dispatch_output, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica, torch::Tensor x,
        torch::Tensor shared_wg, torch::Tensor shared_wu,
        torch::Tensor shared_wd, torch::Tensor raw_gate, torch::Tensor raw_up,
        torch::Tensor activation, double shared_limit, bool recompute) {
        CheckShmem(shmem);
        auto timer=torch::zeros({AscendTimer::TOTAL_BUFFER_SIZE},grad_y.options().dtype(torch::kInt64));
        auto shared_timer=torch::zeros({10,128,2},grad_y.options().dtype(torch::kInt64));
        auto redispatch_timer=torch::zeros(
            {MegaMoeSharedExpert::kBackwardRedispatchTimerPhases,
             MegaMoeSharedExpert::kBackwardRedispatchTimerLanes,2},
            grad_y.options().dtype(torch::kInt64));
        auto result=ascend_deepep::ops::MegaMoeBF16BackwardSharedV3(
            Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2,
            expert_token_nums, training_saved_state, dispatch_output,
            max_recv_tokens, static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), swiglu_limit, save_gmm1_output, replica,
            x, shared_wg, shared_wu, shared_wd, raw_gate, raw_up, activation,
            shared_limit, recompute,timer,shared_timer,redispatch_timer);
        return std::tuple_cat(result,std::make_tuple(timer,shared_timer,redispatch_timer));
    }

    // Diagnostic-only entry points preserve the replica descriptor and expose
    // the existing AscendTimerV2 buffer. Production methods remain unchanged.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    compute_train_replica_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica) {
        CheckShmem(shmem);
        auto timer = torch::zeros({AscendTimer::TOTAL_BUFFER_SIZE},
                                  x.options().dtype(torch::kInt64));
        auto [y, counts, state, dispatch, workspace] =
            ascend_deepep::ops::MegaMoeBF16Forward(
                Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2,
                max_recv_tokens, static_cast<int>(shmem->Rank()),
                static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
                *shmem, SymmetricBuffer(), timer, swiglu_limit,
                save_gmm1_output, replica);
        return {y, counts, state, dispatch, workspace, timer};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    backward_replica_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2,
        torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
        torch::Tensor dispatch_output, int64_t max_recv_tokens,
        double swiglu_limit, std::string save_gmm1_output,
        std::vector<int64_t> replica) {
        CheckShmem(shmem);
        auto timer = torch::zeros({AscendTimer::TOTAL_BUFFER_SIZE},
                                  grad_y.options().dtype(torch::kInt64));
        auto [dx, dp, dw1, dw2, workspace] =
            ascend_deepep::ops::MegaMoeBF16Backward(
                Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2,
                expert_token_nums, training_saved_state, dispatch_output,
                max_recv_tokens, static_cast<int>(shmem->Rank()),
                static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
                *shmem, SymmetricBuffer(), timer, swiglu_limit,
                save_gmm1_output, replica);
        (void)workspace;
        return {dx, dp, dw1, dw2, timer};
    }

    void prepare_context(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        const torch::Tensor& like) {
        CheckShmem(shmem);
        (void)Context(shmem, like);
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    ascend_deepep::ops::MegaMoeBF16ForwardResult compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16(
            Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), save_gmm1_output);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    compute_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            x.options().dtype(torch::kInt64));
        auto [y, expert_token_nums, training_saved_state] =
            ascend_deepep::ops::MegaMoeBF16(
            Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), timer, save_gmm1_output);
        return {y, expert_token_nums, training_saved_state, timer};
    }

    // Training ABI: packed Dispatch output is independent from compact saved
    // state so the framework can decide whether to retain it for backward.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    compute_train(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16Forward(
            Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), swiglu_limit, save_gmm1_output);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    compute_train_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            x.options().dtype(torch::kInt64));
        auto [y, expert_token_nums, training_saved_state, dispatch_output,
              forward_transient_workspace] =
            ascend_deepep::ops::MegaMoeBF16Forward(
                Context(shmem, x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
                static_cast<int>(shmem->Rank()),
                static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
                *shmem, SymmetricBuffer(), timer, swiglu_limit,
                save_gmm1_output);
        return {y, expert_token_nums, training_saved_state, dispatch_output,
                forward_transient_workspace, timer};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backward(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor expert_token_nums,
        torch::Tensor training_saved_state,
        torch::Tensor dispatch_output,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        auto [grad_x, grad_topk_weights, grad_weight1, grad_weight2,
              unused_backward_workspace] =
            ascend_deepep::ops::MegaMoeBF16Backward(
                Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2, expert_token_nums,
                training_saved_state, dispatch_output,
                max_recv_tokens,
                static_cast<int>(shmem->Rank()),
                static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
                *shmem, SymmetricBuffer(), torch::Tensor(), swiglu_limit, save_gmm1_output);
        (void)unused_backward_workspace;
        return {grad_x, grad_topk_weights, grad_weight1, grad_weight2};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor> backward_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor expert_token_nums,
        torch::Tensor training_saved_state,
        torch::Tensor dispatch_output,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            grad_y.options().dtype(torch::kInt64));
        auto [grad_x, grad_topk_weights, grad_weight1, grad_weight2,
              unused_backward_workspace] =
            ascend_deepep::ops::MegaMoeBF16Backward(
                Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2, expert_token_nums,
                training_saved_state, dispatch_output,
                max_recv_tokens,
                static_cast<int>(shmem->Rank()),
                static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
                *shmem, SymmetricBuffer(), timer, swiglu_limit, save_gmm1_output);
        (void)unused_backward_workspace;
        return {grad_x, grad_topk_weights, grad_weight1, grad_weight2, timer};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor> backward_debug(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor grad_y,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor expert_token_nums,
        torch::Tensor training_saved_state,
        torch::Tensor dispatch_output,
        int64_t max_recv_tokens,
        double swiglu_limit,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16Backward(
            Context(shmem, grad_y), grad_y, topk_weights, weight1, weight2, expert_token_nums,
            training_saved_state, dispatch_output, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), swiglu_limit, save_gmm1_output);
    }

private:
    torch::Tensor Context(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        const torch::Tensor& like) {
        TORCH_CHECK(
            like.defined(),
            "MegaMoe BF16 context reference tensor must be defined");
        TORCH_CHECK(
            like.device().type() == c10::DeviceType::PrivateUse1,
            "MegaMoe BF16 context reference tensor must be an NPU tensor");
        TORCH_CHECK(
            like.is_contiguous(),
            "MegaMoe BF16 context reference tensor must be contiguous");
        TORCH_CHECK(
            like.scalar_type() == torch::kBFloat16,
            "MegaMoe BF16 context reference tensor must be bfloat16");
        const int rank = static_cast<int>(shmem->Rank());
        const int world_size = static_cast<int>(shmem->WorldSize());
        if (!context_.defined()) {
            auto new_context = torch::empty(
                {kMegaMoeContextBytes},
                like.options().dtype(torch::kUInt8));
            InitializeMegaMoeBF16Context(
                new_context, SymmetricBuffer(), rank, world_size);
            // Publish the persistent state only after initialization succeeds.
            // A failed first call must remain retryable instead of leaving a
            // defined but partially initialized context behind.
            context_ = new_context;
            context_rank_ = rank;
            context_world_size_ = world_size;
        } else {
            TORCH_CHECK(
                context_.device() == like.device(),
                "MegaMoe BF16 context device changed across calls");
            TORCH_CHECK(
                context_rank_ == rank && context_world_size_ == world_size,
                "MegaMoe BF16 SHMEM rank/world size changed across calls");
        }
        return context_;
    }

    torch::Tensor context_;
    int context_rank_ = -1;
    int context_world_size_ = -1;
};

class MegaMoeBF16MGroup : public ShmemWorkspaceOp {
public:
    explicit MegaMoeBF16MGroup(int64_t local_mem_size)
        : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    ascend_deepep::ops::MegaMoeBF16ForwardResult compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16MGroup(
            x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), save_gmm1_output);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    compute_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            x.options().dtype(torch::kInt64));
        auto [y, expert_token_nums, training_saved_state] =
            ascend_deepep::ops::MegaMoeBF16MGroup(
            x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), timer, save_gmm1_output);
        return {y, expert_token_nums, training_saved_state, timer};
    }
};

class MegaMoeBF16UDMA : public ShmemWorkspaceOp {
public:
    explicit MegaMoeBF16UDMA(int64_t local_mem_size)
        : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    ascend_deepep::ops::MegaMoeBF16ForwardResult compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeBF16UDMA(
            x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), torch::Tensor(), save_gmm1_output);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    compute_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        int64_t max_recv_tokens,
        std::string save_gmm1_output) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            x.options().dtype(torch::kInt64));
        auto [y, expert_token_nums, training_saved_state] =
            ascend_deepep::ops::MegaMoeBF16UDMA(
            x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()), SymmetricBufferBytes(),
            *shmem, SymmetricBuffer(), timer, save_gmm1_output);
        return {y, expert_token_nums, training_saved_state, timer};
    }
};

class MegaMoeW4A8 : public ShmemWorkspaceOp {
public:
    explicit MegaMoeW4A8(int64_t local_mem_size)
        : ShmemWorkspaceOp(local_mem_size) {}

    int64_t local_mem_size() const {
        return ShmemWorkspaceOp::local_mem_size();
    }

    void destroy() {
        ShmemWorkspaceOp::destroy();
    }

    std::tuple<torch::Tensor, torch::Tensor> compute(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor weight_scales1,
        torch::Tensor weight_scales2,
        int64_t max_recv_tokens) {
        CheckShmem(shmem);
        return ascend_deepep::ops::MegaMoeW4A8(
            x,
            topk_ids,
            topk_weights,
            weight1,
            weight2,
            weight_scales1,
            weight_scales2,
            max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),
            *shmem,
            SymmetricBuffer(),
            torch::Tensor());
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_with_timer(
        const c10::intrusive_ptr<ShmemInstance>& shmem,
        torch::Tensor x,
        torch::Tensor topk_ids,
        torch::Tensor topk_weights,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor weight_scales1,
        torch::Tensor weight_scales2,
        int64_t max_recv_tokens) {
        CheckShmem(shmem);
        auto timer = torch::zeros(
            {AscendTimer::TOTAL_BUFFER_SIZE},
            x.options().dtype(torch::kInt64));
        auto [y, expert_token_nums] = ascend_deepep::ops::MegaMoeW4A8(
            x,
            topk_ids,
            topk_weights,
            weight1,
            weight2,
            weight_scales1,
            weight_scales2,
            max_recv_tokens,
            static_cast<int>(shmem->Rank()),
            static_cast<int>(shmem->WorldSize()),
            SymmetricBufferBytes(),
            *shmem,
            SymmetricBuffer(),
            timer);
        return {y, expert_token_nums, timer};
    }
};

}  // namespace ascend_deepep::torch_classes

static auto shmem_instance_registry =
    torch::jit::class_<ShmemInstance>("AscendDeepEP", "ShmemInstance")
        .def(torch::jit::init<int64_t, int64_t, std::string>())
        .def("init", static_cast<void (ShmemInstance::*)(int64_t)>(&ShmemInstance::Init))
        .def("attach", &ShmemInstance::Attach)
        .def("set_data_op_engine_type", &ShmemInstance::SetDataOpEngineType)
        .def("barrier", &ShmemInstance::Barrier)
        .def("destroy", &ShmemInstance::Destroy)
        .def("is_available", &ShmemInstance::IsAvailable)
        .def("rank", &ShmemInstance::Rank)
        .def("world_size", &ShmemInstance::WorldSize);

static auto dispatch_registry =
    torch::jit::class_<ascend_deepep::torch_classes::Dispatch>("AscendDeepEP", "Dispatch")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_deepep::torch_classes::Dispatch::compute)
        .def("local_mem_size", &ascend_deepep::torch_classes::Dispatch::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::Dispatch::destroy);

static auto combine_registry =
    torch::jit::class_<ascend_deepep::torch_classes::Combine>("AscendDeepEP", "Combine")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_deepep::torch_classes::Combine::compute)
        .def("local_mem_size", &ascend_deepep::torch_classes::Combine::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::Combine::destroy);

static auto allgather_matmul_registry =
    torch::jit::class_<ascend_deepep::torch_classes::AllGatherMatmul>("AscendDeepEP", "AllGatherMatmul")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_deepep::torch_classes::AllGatherMatmul::compute)
        .def("compute_out", &ascend_deepep::torch_classes::AllGatherMatmul::compute_out)
        .def("local_mem_size", &ascend_deepep::torch_classes::AllGatherMatmul::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::AllGatherMatmul::destroy);

static auto mega_moe_registry =
    torch::jit::class_<ascend_deepep::torch_classes::MegaMoe>(
        "AscendDeepEP",
        "MegaMoe")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_deepep::torch_classes::MegaMoe::compute)
        .def(
            "compute_with_timer",
            &ascend_deepep::torch_classes::MegaMoe::compute_with_timer)
        .def(
            "local_mem_size",
            &ascend_deepep::torch_classes::MegaMoe::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::MegaMoe::destroy);

static auto mega_moe_bf16_registry =
    torch::jit::class_<ascend_deepep::torch_classes::MegaMoeBF16>(
        "AscendDeepEP",
        "MegaMoeBF16")
        .def(torch::jit::init<int64_t, int64_t>(),
             "",
             {torch::arg("local_mem_size"), torch::arg("external_sym_ptr") = int64_t(0)})
        .def("prepare_context",
             &ascend_deepep::torch_classes::MegaMoeBF16::prepare_context)
        .def("compute", &ascend_deepep::torch_classes::MegaMoeBF16::compute, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("save_gmm1_output") = "bf16"})
        .def("compute_with_timer",
             &ascend_deepep::torch_classes::MegaMoeBF16::compute_with_timer, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("save_gmm1_output") = "bf16"})
        .def("compute_train_replica", &ascend_deepep::torch_classes::MegaMoeBF16::compute_train_replica)
        .def("compute_train_replica_shared", &ascend_deepep::torch_classes::MegaMoeBF16::compute_train_replica_shared)
        .def("compute_train_replica_shared_with_timer", &ascend_deepep::torch_classes::MegaMoeBF16::compute_train_replica_shared_with_timer)
        .def("backward_replica", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica)
        .def("backward_replica_shared", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_shared)
        .def("backward_replica_shared_v2", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_shared_v2)
        .def("backward_replica_shared_v2_with_timer", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_shared_v2_with_timer)
        .def("backward_replica_shared_v3", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_shared_v3)
        .def("backward_replica_redispatch", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_redispatch)
        .def("backward_replica_redispatch_with_timer", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_redispatch_with_timer)
        .def("backward_replica_shared_v3_with_timer", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_shared_v3_with_timer)
        .def("compute_train_replica_with_timer", &ascend_deepep::torch_classes::MegaMoeBF16::compute_train_replica_with_timer)
        .def("backward_replica_with_timer", &ascend_deepep::torch_classes::MegaMoeBF16::backward_replica_with_timer)
        .def("compute_train",
             &ascend_deepep::torch_classes::MegaMoeBF16::compute_train, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("swiglu_limit"), torch::arg("save_gmm1_output") = "bf16"})
        .def("compute_train_with_timer",
             &ascend_deepep::torch_classes::MegaMoeBF16::compute_train_with_timer, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("swiglu_limit"), torch::arg("save_gmm1_output") = "bf16"})
        .def("backward",
             &ascend_deepep::torch_classes::MegaMoeBF16::backward, "",
             {torch::arg("shmem"), torch::arg("grad_y"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("expert_token_nums"), torch::arg("training_saved_state"), torch::arg("dispatch_output"), torch::arg("max_recv_tokens"), torch::arg("swiglu_limit"), torch::arg("save_gmm1_output") = "bf16"})
        .def("backward_with_timer",
             &ascend_deepep::torch_classes::MegaMoeBF16::backward_with_timer, "",
             {torch::arg("shmem"), torch::arg("grad_y"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("expert_token_nums"), torch::arg("training_saved_state"), torch::arg("dispatch_output"), torch::arg("max_recv_tokens"), torch::arg("swiglu_limit"), torch::arg("save_gmm1_output") = "bf16"})
        .def("backward_debug",
             &ascend_deepep::torch_classes::MegaMoeBF16::backward_debug, "",
             {torch::arg("shmem"), torch::arg("grad_y"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("expert_token_nums"), torch::arg("training_saved_state"), torch::arg("dispatch_output"), torch::arg("max_recv_tokens"), torch::arg("swiglu_limit"), torch::arg("save_gmm1_output") = "bf16"})
        .def(
            "local_mem_size",
            &ascend_deepep::torch_classes::MegaMoeBF16::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::MegaMoeBF16::destroy);

static auto mega_moe_bf16_udma_registry =
    torch::jit::class_<ascend_deepep::torch_classes::MegaMoeBF16UDMA>(
        "AscendDeepEP", "MegaMoeBF16UDMA")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_deepep::torch_classes::MegaMoeBF16UDMA::compute, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("save_gmm1_output") = "bf16"})
        .def("compute_with_timer",
             &ascend_deepep::torch_classes::MegaMoeBF16UDMA::compute_with_timer, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("save_gmm1_output") = "bf16"})
        .def("local_mem_size",
             &ascend_deepep::torch_classes::MegaMoeBF16UDMA::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::MegaMoeBF16UDMA::destroy);

static auto mega_moe_bf16_mgroup_registry =
    torch::jit::class_<ascend_deepep::torch_classes::MegaMoeBF16MGroup>(
        "AscendDeepEP", "MegaMoeBF16MGroup")
        .def(torch::jit::init<int64_t>())
        .def("compute",
             &ascend_deepep::torch_classes::MegaMoeBF16MGroup::compute, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("save_gmm1_output") = "bf16"})
        .def("compute_with_timer",
             &ascend_deepep::torch_classes::MegaMoeBF16MGroup::compute_with_timer, "",
             {torch::arg("shmem"), torch::arg("x"), torch::arg("topk_ids"), torch::arg("topk_weights"), torch::arg("weight1"), torch::arg("weight2"), torch::arg("max_recv_tokens"), torch::arg("save_gmm1_output") = "bf16"})
        .def("local_mem_size",
             &ascend_deepep::torch_classes::MegaMoeBF16MGroup::local_mem_size)
        .def("destroy",
             &ascend_deepep::torch_classes::MegaMoeBF16MGroup::destroy);

static auto mega_moe_w4a8_registry =
    torch::jit::class_<ascend_deepep::torch_classes::MegaMoeW4A8>(
        "AscendDeepEP",
        "MegaMoeW4A8")
        .def(torch::jit::init<int64_t>())
        .def("compute", &ascend_deepep::torch_classes::MegaMoeW4A8::compute)
        .def(
            "compute_with_timer",
            &ascend_deepep::torch_classes::MegaMoeW4A8::compute_with_timer)
        .def(
            "local_mem_size",
            &ascend_deepep::torch_classes::MegaMoeW4A8::local_mem_size)
        .def("destroy", &ascend_deepep::torch_classes::MegaMoeW4A8::destroy);
