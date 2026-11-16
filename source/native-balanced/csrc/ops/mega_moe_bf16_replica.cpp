#include <acl/acl.h>
#include <chrono>
#include <cstdlib>
#include <string>
#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/custom_class.h>
#include "shmem.h"
#include "csrc/shmem_instance.h"
#include "kernels/mega_moe_bf16/include/mega_moe_replica_plan.h"
#include "include/mega_moe_large_ep_limits.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {
using Tensor=torch::Tensor;
using Runtime=c10::intrusive_ptr<ShmemInstance>;
using Native=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int64_t,int64_t,int64_t,int64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,int32_t,int32_t,int32_t);
using Route=void(*)(void*,uint8_t*,uint8_t*,uint8_t*,int64_t);
using Planner=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,int,int,int);
void* Resolve(const char* symbol) {
    static void* lib=dlopen("libascend_deepep_replica.so",RTLD_NOW|RTLD_GLOBAL);
    TORCH_CHECK(lib,"Cannot load staged replica kernel: ",dlerror());
    void* p=dlsym(lib,symbol);TORCH_CHECK(p,"Missing replica symbol ",symbol);return p;
}
void* ResolveControl() {
    static void* lib=dlopen("libascend_deepep_replica_control.so",RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(lib,"Cannot load optimized replica planner: ",dlerror());
    void* p=dlsym(lib,"mega_moe_replica_planner_control_do");
    TORCH_CHECK(p,"Missing optimized planner entry");return p;
}
void Check(const Tensor& t,torch::ScalarType dtype) {
    TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1 && t.is_contiguous() && t.scalar_type()==dtype,"invalid replica tensor device/layout/dtype");
}
void* ResolveValidator(const char* symbol) {
    static void* lib=dlopen("libascend_deepep_plan_validator.so",RTLD_NOW|RTLD_LOCAL);
    TORCH_CHECK(lib,"fused validation requires libascend_deepep_plan_validator.so: ",dlerror());
    using Version=int(*)();
    auto version=reinterpret_cast<Version>(dlsym(lib,"megaflux_plan_validator_version"));
    TORCH_CHECK(version && version()==2,"fused validator native ABI mismatch; rebuild validator SO and binding together");
    void* p=dlsym(lib,symbol);TORCH_CHECK(p,"missing fused validator entry ",symbol);return p;
}
class MegaMoeReplicaWorkspace:public torch::jit::CustomClassHolder {
    MegaMoeReplicaLayout layout_;void* base_=nullptr;
    bool split_=false;
    uint64_t replica1_=0,replica2_=0;
    void* gradientReady_=nullptr;int64_t gradientBytes_=0;
    void* controlHost_=nullptr;
    aclrtEvent controlEvent_=nullptr;
    void* largePlanner_=nullptr;
    int64_t largeRanks_=0;
    uint64_t largePlannerBytes_=0;
    void Launch(const Runtime& shmem,const Tensor& plan,int mode,void* a=nullptr,void* b=nullptr,void* c=nullptr,void* d=nullptr) {
        TORCH_CHECK(base_ && shmem && shmem->Initialized(),"replica workspace/runtime closed");
        Check(plan,torch::kInt32);
        TORCH_CHECK(plan.dim()==2 && plan.size(0)==shmem->WorldSize() && plan.size(1)==layout_.slots &&
            MegaMoeLargeEP::SupportedShape(shmem->WorldSize(),layout_.home,layout_.slots),"replica plan shape/capacity mismatch");
        auto stream=c10_npu::getCurrentNPUStream().stream();
        reinterpret_cast<Native>(Resolve("mega_moe_replica_do"))(stream,shmem->FftsAddr(),(uint8_t*)base_,(uint8_t*)plan.data_ptr(),(uint8_t*)a,(uint8_t*)b,(uint8_t*)c,(uint8_t*)d,
            layout_.home,layout_.slots,layout_.hidden,layout_.ffn,layout_.w1,layout_.w2,layout_.dw1,layout_.dw2,layout_.counts,shmem->Rank(),shmem->WorldSize(),mode);
    }
public:
    static c10::intrusive_ptr<MegaMoeReplicaWorkspace> split(int64_t home,int64_t slots,int64_t hidden,int64_t ffn) {
        return c10::make_intrusive<MegaMoeReplicaWorkspace>(home,slots,hidden,ffn,true);
    }
    MegaMoeReplicaWorkspace(int64_t home,int64_t slots,int64_t hidden,int64_t ffn)
      :MegaMoeReplicaWorkspace(home,slots,hidden,ffn,false) {}
    MegaMoeReplicaWorkspace(int64_t home,int64_t slots,int64_t hidden,int64_t ffn,bool split)
      :layout_(home,slots,hidden,ffn),split_(split) {
        TORCH_CHECK(home>0 && slots>=0 && home+slots<=MegaMoeLargeEP::MaxLocalSlots && hidden>0 && hidden%256==0 && ffn>0 && ffn%256==0,"invalid replica capacity/shape");
        if(split_) {
            // Owner matrices and replica matrices are independent spans. There
            // is no padded owner tensor or concatenation in the compute ABI.
            const uint64_t p1=2*ffn*hidden*2,p2=hidden*ffn*2;
            layout_.w1=0;layout_.w2=MegaMoeReplicaLayout::Align(home*p1);
            replica1_=layout_.w2+MegaMoeReplicaLayout::Align(home*p2);
            replica2_=replica1_+MegaMoeReplicaLayout::Align(slots*p1);
            layout_.dw1=replica2_+MegaMoeReplicaLayout::Align(slots*p2);
            layout_.dw2=layout_.dw1+MegaMoeReplicaLayout::Align(slots*p1);
            const auto old=layout_.counts;
            layout_.counts=layout_.dw2+MegaMoeReplicaLayout::Align(slots*p2);
            const auto delta=old-layout_.counts;
            layout_.histogram-=delta;layout_.planner_meta-=delta;layout_.planner_errors-=delta;
            layout_.bytes=(layout_.planner_errors+32*128+2*32*256*4+2097151)/2097152*2097152;
        }
        base_=aclshmem_malloc(layout_.bytes);TORCH_CHECK(base_,"replica SHMEM allocation failed");
        TORCH_CHECK(aclrtMemset(base_,layout_.bytes,0,layout_.bytes)==ACL_SUCCESS,"replica initialization failed");
    }
    ~MegaMoeReplicaWorkspace() override {
        if(largePlanner_)aclshmem_free(largePlanner_);
        if(controlEvent_)aclrtDestroyEvent(controlEvent_);
        if(controlHost_)aclrtFreeHost(controlHost_);
        if(gradientReady_)aclshmem_free(gradientReady_);
        if(base_)aclshmem_free(base_);
    }
    std::tuple<Tensor,Tensor> weights(Tensor like) {
        Check(like,torch::kBFloat16);TORCH_CHECK(base_,"closed workspace");
        auto options=like.options().dtype(torch::kBFloat16);
        const auto e=split_?layout_.home:layout_.home+layout_.slots;
        auto a=torch::from_blob((uint8_t*)base_+layout_.w1,{e,2*layout_.ffn,layout_.hidden},[](void*){},options);
        auto b=torch::from_blob((uint8_t*)base_+layout_.w2,{e,layout_.hidden,layout_.ffn},[](void*){},options);
        return {a,b};
    }
    std::tuple<Tensor,Tensor> replicas(Tensor like) {
        Check(like,torch::kBFloat16);TORCH_CHECK(split_,"requires split workspace");
        auto a=torch::from_blob((uint8_t*)base_+replica1_,{layout_.slots,2*layout_.ffn,layout_.hidden},[](void*){},like.options());
        auto b=torch::from_blob((uint8_t*)base_+replica2_,{layout_.slots,layout_.hidden,layout_.ffn},[](void*){},like.options());
        return {a,b};
    }
    std::vector<int64_t> descriptor(Runtime shmem,Tensor plan) {
        Check(plan,torch::kInt32);
        TORCH_CHECK(split_ && shmem && shmem->Initialized() && plan.dim()==2 &&
            plan.size(0)==shmem->WorldSize() && plan.size(1)==layout_.slots &&
            MegaMoeLargeEP::SupportedShape(shmem->WorldSize(),layout_.home,layout_.slots),
            "invalid split replica descriptor");
        auto addr=[&](uint64_t off){return reinterpret_cast<int64_t>((uint8_t*)base_+off);};
        return {addr(0),reinterpret_cast<int64_t>(plan.data_ptr()),addr(layout_.w1),addr(layout_.w2),
            addr(replica1_),addr(replica2_),addr(layout_.dw1),addr(layout_.dw2),
            layout_.home,layout_.slots,layout_.hidden,layout_.ffn,shmem->Rank(),shmem->WorldSize()};
    }
    std::vector<int64_t> gradient_pipe() {
        TORCH_CHECK(base_ && split_,"gradient pipe requires an open split workspace");
        if(!gradientReady_) {
            const auto tiles=(2*layout_.ffn/128)*(layout_.hidden/256)+(layout_.hidden/128)*(layout_.ffn/256);
            gradientBytes_=MegaMoeReplicaLayout::Align((layout_.home+layout_.slots)*tiles*64);
            gradientReady_=aclshmem_malloc(gradientBytes_);
            TORCH_CHECK(gradientReady_,"gradient epoch allocation failed");
            TORCH_CHECK(aclrtMemset(gradientReady_,gradientBytes_,0,gradientBytes_)==ACL_SUCCESS,"gradient epochs init failed");
        }
        return {reinterpret_cast<int64_t>(gradientReady_),gradientBytes_};
    }
    int64_t bytes() const {return layout_.bytes+gradientBytes_+largePlannerBytes_;}
    std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> online_plan(Runtime shmem,Tensor ids,
            int64_t beta,int64_t penalty,int64_t gain,int64_t epoch) {
        Check(ids,torch::kInt32);
        TORCH_CHECK(shmem && shmem->Initialized() && base_,"closed runtime/workspace");
        const auto ranks=shmem->WorldSize(),experts=layout_.home*ranks;
        TORCH_CHECK(ranks<=8 && experts<=256 && ids.dim()==2 && ids.size(0)>0 && ids.size(1)>0 && ids.size(1)<=experts && ids.numel()*ranks<2147483520LL && beta>0 && beta<=2147483647 && penalty>=0 && penalty<=1048576 && gain>=0 && gain<=1048576 && epoch>0 && epoch<=2147483647,"unsupported online planner capacity/policy/epoch");
        auto copies=torch::empty({ranks,layout_.slots},ids.options());
        auto physical=torch::empty_like(ids);
        auto logical=torch::empty({experts},ids.options());
        auto q=torch::empty({experts,ranks},ids.options());
        auto header=torch::empty({8},ids.options());
        reinterpret_cast<Planner>(Resolve("mega_moe_replica_planner_do"))(
            c10_npu::getCurrentNPUStream().stream(),shmem->FftsAddr(),(uint8_t*)base_,
            (uint8_t*)ids.data_ptr(),(uint8_t*)copies.data_ptr(),(uint8_t*)physical.data_ptr(),
            (uint8_t*)logical.data_ptr(),(uint8_t*)q.data_ptr(),(uint8_t*)header.data_ptr(),
            layout_.histogram,layout_.planner_meta,layout_.planner_errors,ids.numel(),ids.size(1),
            layout_.home,layout_.slots,shmem->Rank(),ranks,beta,penalty,gain,epoch);
        return {copies,physical,logical,q,header};
    }
    std::vector<int64_t> online_plan_control(Runtime shmem,Tensor ids,
            int64_t beta,int64_t penalty,int64_t gain,int64_t epoch,
            Tensor copies,Tensor physical,Tensor logical,Tensor q,Tensor header) {
        Check(ids,torch::kInt32);
        TORCH_CHECK(shmem && shmem->Initialized() && base_,"closed runtime/workspace");
        const auto ranks=shmem->WorldSize(),experts=layout_.home*ranks;
        TORCH_CHECK(ranks>=2 && ranks<=8 && experts<=256 && ids.dim()==2 && ids.size(0)>0 && ids.size(1)>0 && ids.size(1)<=experts && ids.numel()*ranks<2147483520LL && beta>0 && beta<=2147483647 && penalty>=0 && penalty<=1048576 && gain>=0 && gain<=1048576 && epoch>0 && epoch<=2147483647,"unsupported optimized planner capacity/policy/epoch");
        std::vector<Tensor> buffers{copies,physical,logical,q,header};
        for(const auto& t:buffers) {
            Check(t,torch::kInt32);TORCH_CHECK(t.device()==ids.device(),"planner buffer device mismatch");
            TORCH_CHECK(t.numel()==0 || t.data_ptr()!=ids.data_ptr(),"planner output must not alias routes");
        }
        for(size_t i=0;i<buffers.size();++i)for(size_t j=0;j<i;++j)
            TORCH_CHECK(!buffers[i].numel() || !buffers[j].numel() || buffers[i].data_ptr()!=buffers[j].data_ptr(),"planner outputs must not alias");
        TORCH_CHECK(copies.dim()==2 && copies.size(0)==ranks && copies.size(1)==layout_.slots && physical.sizes()==ids.sizes() && logical.dim()==1 && logical.numel()==experts && q.dim()==2 && q.size(0)==experts && q.size(1)==ranks && header.dim()==1 && header.numel()==8,"optimized planner buffer shape mismatch");
        if(!controlHost_)TORCH_CHECK(aclrtMallocHost(&controlHost_,32)==ACL_SUCCESS,"planner pinned allocation failed");
        if(!controlEvent_)TORCH_CHECK(aclrtCreateEvent(&controlEvent_)==ACL_SUCCESS,"planner event allocation failed");
        auto stream=c10_npu::getCurrentNPUStream().stream();
        reinterpret_cast<Planner>(ResolveControl())(
            stream,shmem->FftsAddr(),(uint8_t*)base_,(uint8_t*)ids.data_ptr(),
            (uint8_t*)copies.data_ptr(),(uint8_t*)physical.data_ptr(),(uint8_t*)logical.data_ptr(),
            (uint8_t*)q.data_ptr(),(uint8_t*)header.data_ptr(),layout_.histogram,layout_.planner_meta,
            layout_.planner_errors,ids.numel(),ids.size(1),layout_.home,layout_.slots,
            shmem->Rank(),ranks,beta,penalty,gain,epoch);
        // Deliberate host dependency, included in every online call. Reuse pinned
        // storage/event instead of Tensor.cpu dispatch and a fresh CPU tensor.
        TORCH_CHECK(aclrtMemcpyAsync(controlHost_,32,header.data_ptr(),32,ACL_MEMCPY_DEVICE_TO_HOST,stream)==ACL_SUCCESS,"planner header copy failed");
        TORCH_CHECK(aclrtRecordEvent(controlEvent_,stream)==ACL_SUCCESS,"planner event record failed");
        // The short planner otherwise pays the runtime's blocking-wait wakeup
        // latency. Poll for at most 1 ms, then use the regular blocking wait.
        // The CPU cost and host dependency remain part of every online call.
        const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(1);
        aclrtEventRecordedStatus status=ACL_EVENT_RECORDED_STATUS_NOT_READY;
        do {
            TORCH_CHECK(aclrtQueryEventStatus(controlEvent_,&status)==ACL_SUCCESS,"planner control query failed");
        } while(status!=ACL_EVENT_RECORDED_STATUS_COMPLETE && std::chrono::steady_clock::now()<until);
        if(status!=ACL_EVENT_RECORDED_STATUS_COMPLETE)
            TORCH_CHECK(aclrtSynchronizeEvent(controlEvent_)==ACL_SUCCESS,"planner control wait failed");
        const auto* p=static_cast<const int32_t*>(controlHost_);
        return std::vector<int64_t>(p,p+8);
    }
    void online_plan_device(Runtime shmem,Tensor ids,
            int64_t beta,int64_t penalty,int64_t gain,int64_t epoch,
            Tensor copies,Tensor physical,Tensor logical,Tensor q,Tensor header,int64_t capacity) {
        online_plan_impl(shmem,ids,beta,penalty,gain,epoch,copies,physical,logical,q,header,capacity,-1);
    }
    void online_plan_optimized(Runtime shmem,Tensor ids,
            int64_t beta,int64_t penalty,int64_t gain,int64_t epoch,
            Tensor copies,Tensor physical,Tensor logical,Tensor q,Tensor header,int64_t capacity,int64_t algorithm) {
        TORCH_CHECK(algorithm>=0 && algorithm<=2,"invalid optimized planner algorithm");
        online_plan_impl(shmem,ids,beta,penalty,gain,epoch,copies,physical,logical,q,header,capacity,algorithm);
    }
    void online_plan_moonep(Runtime shmem,Tensor ids,int64_t epoch,
            Tensor copies,Tensor physical,Tensor logical,Tensor q,Tensor header,int64_t capacity) {
        TORCH_CHECK(layout_.slots==layout_.home && layout_.home<=MegaMoeLargeEP::MaxLocalSlots/2,
                    "MoonEP policy requires slots=home within compiled capacity");
        online_plan_impl(shmem,ids,1,0,0,epoch,copies,physical,logical,q,header,capacity,-2);
    }
    void online_plan_impl(Runtime shmem,Tensor ids,
            int64_t beta,int64_t penalty,int64_t gain,int64_t epoch,
            Tensor copies,Tensor physical,Tensor logical,Tensor q,Tensor header,int64_t capacity,int64_t algorithm) {
        Check(ids,torch::kInt32);
        TORCH_CHECK(shmem && shmem->Initialized() && base_,"closed runtime/workspace");
        const auto ranks=shmem->WorldSize(),experts=layout_.home*ranks;
        TORCH_CHECK(MegaMoeLargeEP::SupportedShape(ranks,layout_.home,layout_.slots) && ids.dim()==2 && ids.size(0)>0 && ids.size(1)>0 && ids.size(1)<=experts && ids.numel()*ranks<2147483520LL && beta>0 && beta<=2147483647 && penalty>=0 && penalty<=1048576 && gain>=0 && gain<=1048576 && epoch>0 && epoch<=2147483647,"unsupported device planner capacity/policy/epoch");
        std::vector<Tensor> buffers{copies,physical,logical,q,header};
        for(const auto& t:buffers) {
            Check(t,torch::kInt32);TORCH_CHECK(t.device()==ids.device(),"planner buffer device mismatch");
            TORCH_CHECK(t.numel()==0 || t.data_ptr()!=ids.data_ptr(),"planner output must not alias routes");
        }
        for(size_t i=0;i<buffers.size();++i)for(size_t j=0;j<i;++j)
            TORCH_CHECK(!buffers[i].numel() || !buffers[j].numel() || buffers[i].data_ptr()!=buffers[j].data_ptr(),"planner outputs must not alias");
        TORCH_CHECK(copies.dim()==2 && copies.size(0)==ranks && copies.size(1)==layout_.slots && physical.sizes()==ids.sizes() && logical.dim()==1 && logical.numel()==experts && q.dim()==2 && q.size(0)==experts && q.size(1)==ranks && header.dim()==1 && header.numel()==8,"optimized planner buffer shape mismatch");
        TORCH_CHECK(capacity>0 && capacity<=2147483520LL && capacity%128==0,"device receive capacity must be positive, aligned128, and fit int32");
        const char* force=std::getenv("ASCEND_DEEPEP_FORCE_GM_PLANNER");
        if(ranks>8 || (force && std::string(force)=="1")) {
            MegaMoeLargeEP::PlannerLayout scratch(experts,ranks);
            if(!largePlanner_) {
                largePlanner_=aclshmem_malloc(scratch.bytes);
                TORCH_CHECK(largePlanner_,"large EP planner SHMEM allocation failed");
                largeRanks_=ranks;largePlannerBytes_=scratch.bytes;
                TORCH_CHECK(aclrtMemset(largePlanner_,scratch.bytes,0,scratch.bytes)==ACL_SUCCESS,"large EP scratch init failed");
            }
            TORCH_CHECK(largeRanks_==ranks,"large EP workspace cannot change topology");
            static void* large=dlopen("libascend_deepep_large_ep_planner.so",RTLD_NOW|RTLD_LOCAL);
            TORCH_CHECK(large,"Cannot load GM-backed large EP planner: ",dlerror());
            using LargePlanner=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,int,int,int,int,int);
            auto entry=reinterpret_cast<LargePlanner>(dlsym(large,"mega_moe_large_ep_planner_do"));
            TORCH_CHECK(entry,"Missing large EP planner entry");
            entry(c10_npu::getCurrentNPUStream().stream(),shmem->FftsAddr(),(uint8_t*)largePlanner_,
                (uint8_t*)ids.data_ptr(),(uint8_t*)copies.data_ptr(),(uint8_t*)physical.data_ptr(),
                (uint8_t*)logical.data_ptr(),(uint8_t*)q.data_ptr(),(uint8_t*)header.data_ptr(),
                scratch.histogram,scratch.meta,scratch.errors,ids.numel(),ids.size(1),layout_.home,layout_.slots,
                shmem->Rank(),ranks,beta,penalty,gain,epoch,capacity,algorithm);
            return;
        }
        // Do not load or mutate the old planner when selecting the new SO.
        void* lib=nullptr;
        if(algorithm==-2) {static void* moon=dlopen("libascend_deepep_moonep_planner.so",RTLD_NOW|RTLD_LOCAL);lib=moon;}
        else if(algorithm<0) {static void* old=dlopen("libascend_deepep_replica_device.so",RTLD_NOW|RTLD_LOCAL);lib=old;}
        else {static void* optimized=dlopen("libascend_deepep_replica_optimized.so",RTLD_NOW|RTLD_LOCAL);lib=optimized;}
        TORCH_CHECK(lib,"Cannot load device planner: ",dlerror());
        using DevicePlanner=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,int,int,int,int);
        auto symbol=dlsym(lib,algorithm==-2?"mega_moe_moonep_planner_do":
                         algorithm<0?"mega_moe_replica_planner_device_do":"mega_moe_replica_planner_optimized_do");
        auto entry=reinterpret_cast<DevicePlanner>(symbol);
        TORCH_CHECK(entry,"Missing device planner entry");
        auto stream=c10_npu::getCurrentNPUStream().stream();
        if(algorithm>=0) {
            using OptimizedPlanner=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,int,int,int,int,int);
            reinterpret_cast<OptimizedPlanner>(symbol)(stream,shmem->FftsAddr(),(uint8_t*)base_,(uint8_t*)ids.data_ptr(),
                (uint8_t*)copies.data_ptr(),(uint8_t*)physical.data_ptr(),(uint8_t*)logical.data_ptr(),(uint8_t*)q.data_ptr(),(uint8_t*)header.data_ptr(),
                layout_.histogram,layout_.planner_meta,layout_.planner_errors,ids.numel(),ids.size(1),layout_.home,layout_.slots,
                shmem->Rank(),ranks,beta,penalty,gain,epoch,capacity,algorithm);
            return;
        }
        entry(stream,shmem->FftsAddr(),(uint8_t*)base_,
            (uint8_t*)ids.data_ptr(),(uint8_t*)copies.data_ptr(),(uint8_t*)physical.data_ptr(),
            (uint8_t*)logical.data_ptr(),(uint8_t*)q.data_ptr(),(uint8_t*)header.data_ptr(),
            layout_.histogram,layout_.planner_meta,layout_.planner_errors,ids.numel(),ids.size(1),
            layout_.home,layout_.slots,shmem->Rank(),ranks,beta,penalty,gain,epoch,capacity);
    }
    void online_plan_unified(Runtime shmem,Tensor ids,int64_t epoch,Tensor copies,
            Tensor physical,Tensor logical,Tensor q,Tensor header,Tensor sourceQuota,
            Tensor costs,int64_t capacity,int64_t policy) {
        TORCH_CHECK(shmem && shmem->Initialized() && base_,"closed planner workspace");
        const int ep=shmem->WorldSize(),home=layout_.home,slots=layout_.slots,experts=ep*home;
        TORCH_CHECK((ep==2||ep==4||ep==8) && home>0 && slots>=0 &&
            home+slots<=MEGAMOE_MAX_LOCAL_EXPERTS && experts<=256 && ep*slots<=256,
            "unsupported unified planner layout");
        std::vector<Tensor> tensors{ids,copies,physical,logical,q,header,sourceQuota};
        for(const auto& t:tensors){Check(t,torch::kInt32);TORCH_CHECK(t.device()==ids.device(),"planner device mismatch");}
        Check(costs,torch::kFloat32);TORCH_CHECK(costs.device()==ids.device() && costs.numel()>=1,"invalid planner cost model");
        for(size_t i=0;i<tensors.size();++i)for(size_t j=0;j<i;++j)
            TORCH_CHECK(!tensors[i].numel()||!tensors[j].numel()||tensors[i].data_ptr()!=tensors[j].data_ptr(),"planner tensors alias");
        TORCH_CHECK(ids.dim()==2 && ids.size(0)>0 && ids.size(0)<=16384 && ids.size(1)>0 &&
            ids.size(1)<=experts && copies.dim()==2 && copies.size(0)==ep && copies.size(1)==slots &&
            physical.sizes()==ids.sizes() && logical.dim()==1 && logical.numel()==experts &&
            q.dim()==2 && q.size(0)==experts && q.size(1)==ep && header.dim()==1 && header.numel()==8 &&
            sourceQuota.dim()==3 && sourceQuota.size(0)==ep && sourceQuota.size(1)==experts && sourceQuota.size(2)==ep,
            "unified proposal/output shape mismatch");
        TORCH_CHECK(epoch>0 && epoch<=2147483647 && capacity>0 && capacity<=2147483520LL &&
            capacity%128==0 && ((policy>=0 && policy<=6) ||
            (policy>=100 && policy<=100+ep*slots)),"invalid unified planner epoch/capacity/policy");
        static void* lib=dlopen("libascend_deepep_unified_planner.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(lib,"Cannot load unified planner: ",dlerror());
        using Get=int(*)();auto get=reinterpret_cast<Get>(dlsym(lib,"megaflux_max_local_experts"));
        auto abi=reinterpret_cast<Get>(dlsym(lib,"megaflux_unified_planner_abi"));
        TORCH_CHECK(get && get()==MEGAMOE_MAX_LOCAL_EXPERTS && abi && abi()==1,"unified planner SO capability/ABI mismatch");
        using Entry=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,
            uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,int,int);
        auto fn=reinterpret_cast<Entry>(dlsym(lib,"mega_moe_unified_planner_do"));
        TORCH_CHECK(fn,"Missing unified planner entry");
        fn(c10_npu::getCurrentNPUStream().stream(),shmem->FftsAddr(),(uint8_t*)base_,
            (uint8_t*)ids.data_ptr(),(uint8_t*)copies.data_ptr(),(uint8_t*)physical.data_ptr(),
            (uint8_t*)logical.data_ptr(),(uint8_t*)q.data_ptr(),(uint8_t*)header.data_ptr(),
            (uint8_t*)sourceQuota.data_ptr(),(uint8_t*)costs.data_ptr(),layout_.histogram,layout_.planner_meta,layout_.planner_errors,
            ids.numel(),ids.size(1),home,slots,shmem->Rank(),ep,epoch,capacity,policy);
    }
    void validate_plan_local(Runtime shmem,Tensor ids,Tensor physical,Tensor copies,
            Tensor scratch,Tensor packed,int64_t epoch,int64_t capacity) {
        TORCH_CHECK(shmem && shmem->Initialized() && base_,"closed runtime/workspace");
        const int ep=shmem->WorldSize(),home=layout_.home,slots=layout_.slots;
        const int experts=ep*home,pc=ep*(home+slots),size=experts+pc+ep*slots+3;
        TORCH_CHECK(MegaMoeLargeEP::SupportedShape(ep,home,slots),"unsupported fused validator topology");
        std::vector<Tensor> tensors{ids,physical,copies,scratch,packed};
        for(const auto& t:tensors){Check(t,torch::kInt32);TORCH_CHECK(t.device()==ids.device(),"validator device mismatch");}
        for(size_t i=0;i<tensors.size();++i)for(size_t j=0;j<i;++j)
            TORCH_CHECK(!tensors[i].numel() || !tensors[j].numel() || tensors[i].data_ptr()!=tensors[j].data_ptr(),"validator buffers must not alias");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(scratch.data_ptr())%128==0,"validator scratch must start on a 128-byte cache line");
        TORCH_CHECK(ids.dim()==2 && ids.size(0)>0 && ids.size(0)<=16384 && ids.size(1)>0 && ids.size(1)<=experts &&
                    ids.numel()*ep<2147483392LL && physical.sizes()==ids.sizes(),"invalid validator routes");
        TORCH_CHECK(copies.dim()==2 && copies.size(0)==ep && copies.size(1)==slots &&
                    scratch.numel()==32*((experts+pc+1+31)/32*32) && packed.numel()==size,"invalid validator scratch shape");
        TORCH_CHECK(epoch>0 && epoch<=2147483647 && capacity>0 && capacity<=2147483520LL && capacity%128==0,"invalid validator epoch/capacity");
        using Local=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int,int,int,int,int,int);
        reinterpret_cast<Local>(ResolveValidator("megaflux_plan_validate_local_do"))(
            c10_npu::getCurrentNPUStream().stream(),shmem->FftsAddr(),(uint8_t*)ids.data_ptr(),
            (uint8_t*)physical.data_ptr(),(uint8_t*)copies.data_ptr(),(uint8_t*)scratch.data_ptr(),
            (uint8_t*)packed.data_ptr(),ids.numel(),ep,home,slots,epoch,capacity);
    }
    void validate_plan_global(Runtime shmem,Tensor gathered,Tensor nativeHeader,Tensor counts,
            Tensor header,int64_t epoch,int64_t capacity) {
        TORCH_CHECK(shmem && shmem->Initialized() && base_,"closed runtime/workspace");
        const int ep=shmem->WorldSize(),home=layout_.home,slots=layout_.slots;
        const int experts=ep*home,size=experts+ep*(home+slots)+ep*slots+3;
        TORCH_CHECK(MegaMoeLargeEP::SupportedShape(ep,home,slots),"unsupported fused validator topology");
        std::vector<Tensor> tensors{gathered,nativeHeader,counts,header};
        for(const auto& t:tensors){Check(t,torch::kInt32);TORCH_CHECK(t.device()==gathered.device(),"validator device mismatch");}
        for(size_t i=0;i<tensors.size();++i)for(size_t j=0;j<i;++j)
            TORCH_CHECK(tensors[i].data_ptr()!=tensors[j].data_ptr(),"validator buffers must not alias");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(counts.data_ptr())%128==0,"validator reduction scratch must start on a 128-byte cache line");
        TORCH_CHECK(gathered.numel()==ep*size && nativeHeader.numel()==8 && counts.numel()==((experts+31)/32*32)+((ep*(home+slots)+31)/32*32)+32*32 && header.numel()==8,"invalid validator global shape");
        TORCH_CHECK(epoch>0 && epoch<=2147483647 && capacity>0 && capacity<=2147483520LL && capacity%128==0,"invalid validator epoch/capacity");
        using Global=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,int,int,int,int,int);
        reinterpret_cast<Global>(ResolveValidator("megaflux_plan_validate_global_do"))(
            c10_npu::getCurrentNPUStream().stream(),shmem->FftsAddr(),(uint8_t*)gathered.data_ptr(),(uint8_t*)nativeHeader.data_ptr(),
            (uint8_t*)counts.data_ptr(),(uint8_t*)header.data_ptr(),ep,home,slots,epoch,capacity);
    }
    std::vector<int64_t> finish_device(Tensor header,int64_t epoch,int64_t capacity) {
        return finish_device_impl(header,epoch,capacity,true);
    }
    std::vector<int64_t> finish_device_stream(Tensor header,int64_t epoch,int64_t capacity) {
        return finish_device_impl(header,epoch,capacity,false);
    }
    std::vector<int64_t> finish_device_impl(Tensor header,int64_t epoch,int64_t capacity,bool useEvent) {
        Check(header,torch::kInt32);TORCH_CHECK(header.numel()==8,"invalid device status");
        if(!controlHost_)TORCH_CHECK(aclrtMallocHost(&controlHost_,32)==ACL_SUCCESS,"device status pinned allocation failed");
        if(useEvent && !controlEvent_)TORCH_CHECK(aclrtCreateEvent(&controlEvent_)==ACL_SUCCESS,"device status event allocation failed");
        auto stream=c10_npu::getCurrentNPUStream().stream();
        TORCH_CHECK(aclrtMemcpyAsync(controlHost_,32,header.data_ptr(),32,ACL_MEMCPY_DEVICE_TO_HOST,stream)==ACL_SUCCESS,"device final status copy failed");
        if(useEvent) {
            TORCH_CHECK(aclrtRecordEvent(controlEvent_,stream)==ACL_SUCCESS,"device final event record failed");
            TORCH_CHECK(aclrtSynchronizeEvent(controlEvent_)==ACL_SUCCESS,"device final wait failed");
        } else {
            // The status DMA is on the original execution stream. Waiting
            // that stream therefore drains both compute and the host copy;
            // reading/error-checking the header below remains mandatory.
            TORCH_CHECK(aclrtSynchronizeStream(stream)==ACL_SUCCESS,"device final stream wait failed");
        }
        const auto* p=static_cast<const int32_t*>(controlHost_);
        TORCH_CHECK(p[2]==epoch && p[7]==capacity,"device plan epoch/capacity mismatch");
        TORCH_CHECK(p[3]==0,"device plan failed (1=invalid routes,2=collective policy mismatch,3=capacity exceeded), status=",p[3]," required=",p[1]," capacity=",capacity);
        return std::vector<int64_t>(p,p+8);
    }
    void copy(Runtime shmem,Tensor plan) { TORCH_CHECK(!split_,"split copy is inside mixed kernel");Launch(shmem,plan,0); }
    void restore_weights(Runtime shmem,Tensor plan) {
        TORCH_CHECK(split_,"backward restore requires split home/replica storage");
        using Version=int(*)();
        TORCH_CHECK(reinterpret_cast<Version>(Resolve("megaflux_replica_restore_version"))()==1,
            "backward restore requires the integration replica SO");
        Launch(shmem,plan,4,nullptr,nullptr,
            static_cast<uint8_t*>(base_)+replica1_,static_cast<uint8_t*>(base_)+replica2_);
    }
    Tensor map_routes(Tensor blocks,Tensor table) {
        Check(blocks,torch::kInt32);Check(table,torch::kInt32);
        auto out=torch::empty_like(blocks);
        reinterpret_cast<Route>(Resolve("mega_moe_replica_route_do"))(c10_npu::getCurrentNPUStream().stream(),(uint8_t*)blocks.data_ptr(),(uint8_t*)table.data_ptr(),(uint8_t*)out.data_ptr(),blocks.numel());
        return out;
    }
    std::tuple<Tensor,Tensor> reduce(Runtime shmem,Tensor plan,Tensor a,Tensor b) {
        Check(a,torch::kBFloat16);Check(b,torch::kBFloat16);
        TORCH_CHECK(a.dim()==3 && b.dim()==3 && a.size(0)==layout_.home+layout_.slots && a.size(1)==2*layout_.ffn && a.size(2)==layout_.hidden && b.size(0)==a.size(0) && b.size(1)==layout_.hidden && b.size(2)==layout_.ffn,"partial gradient shape mismatch");
        auto x=torch::empty({layout_.home,2*layout_.ffn,layout_.hidden},a.options());
        auto y=torch::empty({layout_.home,layout_.hidden,layout_.ffn},b.options());
        Launch(shmem,plan,1,a.data_ptr(),b.data_ptr(),x.data_ptr(),y.data_ptr());return {x,y};
    }
    std::tuple<Tensor,Tensor> reduce_sparse(Runtime shmem,Tensor plan,Tensor a,Tensor b) {
        Check(a,torch::kBFloat16);Check(b,torch::kBFloat16);
        TORCH_CHECK(a.dim()==3 && b.dim()==3 && a.size(0)==layout_.home+layout_.slots && a.size(1)==2*layout_.ffn && a.size(2)==layout_.hidden && b.size(0)==a.size(0) && b.size(1)==layout_.hidden && b.size(2)==layout_.ffn,"partial gradient shape mismatch");
        // Export every remote replica first, then amend home partials in place.
        // Nonreplicated experts remain the original output without extra I/O.
        auto x=a.narrow(0,0,layout_.home);
        auto y=b.narrow(0,0,layout_.home);
        Launch(shmem,plan,3,a.data_ptr(),b.data_ptr(),x.data_ptr(),y.data_ptr());return {x,y};
    }
    Tensor counts(Runtime shmem,Tensor plan,Tensor counts) {
        Check(counts,torch::kInt32);TORCH_CHECK(counts.numel()==layout_.home+layout_.slots,"count shape mismatch");
        auto out=torch::empty({layout_.home},counts.options());Launch(shmem,plan,2,counts.data_ptr(),nullptr,out.data_ptr());return out;
    }
};
}
// Called by PyInit after the original TORCH_LIBRARY static initializer has
// registered ShmemInstance. Cross-translation-unit static order is undefined.
void RegisterMegaMoeReplicaWorkspace() {
static auto registry=torch::jit::class_<MegaMoeReplicaWorkspace>("AscendDeepEP","MegaMoeReplicaWorkspace")
 .def(torch::jit::init<int64_t,int64_t,int64_t,int64_t>())
 .def_static("split",&MegaMoeReplicaWorkspace::split)
 .def("weights",&MegaMoeReplicaWorkspace::weights).def("copy",&MegaMoeReplicaWorkspace::copy)
 .def("replicas",&MegaMoeReplicaWorkspace::replicas).def("descriptor",&MegaMoeReplicaWorkspace::descriptor)
 .def("restore_weights",&MegaMoeReplicaWorkspace::restore_weights)
 .def("map_routes",&MegaMoeReplicaWorkspace::map_routes).def("reduce",&MegaMoeReplicaWorkspace::reduce)
 .def("reduce_sparse",&MegaMoeReplicaWorkspace::reduce_sparse)
 .def("online_plan",&MegaMoeReplicaWorkspace::online_plan)
 .def("online_plan_control",&MegaMoeReplicaWorkspace::online_plan_control)
 .def("online_plan_device",&MegaMoeReplicaWorkspace::online_plan_device)
 .def("online_plan_moonep",&MegaMoeReplicaWorkspace::online_plan_moonep)
 .def("online_plan_optimized",&MegaMoeReplicaWorkspace::online_plan_optimized)
 .def("online_plan_unified",&MegaMoeReplicaWorkspace::online_plan_unified)
 .def("validate_plan_local",&MegaMoeReplicaWorkspace::validate_plan_local)
 .def("validate_plan_global",&MegaMoeReplicaWorkspace::validate_plan_global)
 .def("gradient_pipe",&MegaMoeReplicaWorkspace::gradient_pipe)
 .def("finish_device",&MegaMoeReplicaWorkspace::finish_device)
 .def("finish_device_stream",&MegaMoeReplicaWorkspace::finish_device_stream)
 .def("counts",&MegaMoeReplicaWorkspace::counts).def("bytes",&MegaMoeReplicaWorkspace::bytes);
}
