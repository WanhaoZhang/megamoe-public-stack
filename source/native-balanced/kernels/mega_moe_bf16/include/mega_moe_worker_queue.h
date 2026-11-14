#pragma once
#include "kernel_operator.h"
#include "mega_moe_replica_mixed.h"
#include "mega_moe_stage_timer.h"

// Candidate-private envelope v3. The public route/owner descriptor stays ABI2.
// No SHMEM runtime or arithmetic atomic is added: atomics only claim/counter work.
namespace MegaFluxWorker {
using namespace AscendC;
constexpr int WORDS=8,HEADER=32,LINE=16,WORKERS=128,CHUNK=32768;
__aicore__ inline uint64_t Field(uint64_t side,int i) {
    return *((__gm__ uint64_t*)side+i);
}
__aicore__ inline int Read(__gm__ int32_t* p) {return ReadGmBypassDCache(p);}
__aicore__ inline void Release(__gm__ int32_t* p,int value) {
    *p=value;GlobalTensor<int32_t> g;g.SetGlobalBuffer(p);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(g);
    DataSyncBarrier<MemDsbT::ALL>();
}
__aicore__ inline uint64_t Trace(uint64_t side) {return Field(side,4);}
struct Copy {
    LocalTensor<bfloat16_t> buffer[2];bool pending[2]={false,false};int next=0;
    __aicore__ inline Copy():buffer{
        LocalTensor<bfloat16_t>(TPosition::VECCALC,224*1024,4096),
        LocalTensor<bfloat16_t>(TPosition::VECCALC,232*1024,4096)} {}
    __aicore__ inline void Push(uint64_t src,uint64_t dst,int n) {
        int s=next;next^=1;const auto id=s?EVENT_ID5:EVENT_ID4;
        if(pending[s])WaitFlag<HardEvent::MTE3_MTE2>(id);
        GlobalTensor<bfloat16_t> from,to;from.SetGlobalBuffer((__gm__ bfloat16_t*)src);to.SetGlobalBuffer((__gm__ bfloat16_t*)dst);
        from.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);to.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{1,(uint32_t)(n*2),0,0,0};DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
        DataCopyPad(buffer[s],from,cp,pad);
        SetFlag<HardEvent::MTE2_MTE3>(id);WaitFlag<HardEvent::MTE2_MTE3>(id);
        DataCopyPad(to,buffer[s],cp);SetFlag<HardEvent::MTE3_MTE2>(id);pending[s]=true;
    }
    __aicore__ inline void Drain() {for(int s=0;s<2;++s)if(pending[s]) {
        WaitFlag<HardEvent::MTE3_MTE2>(s?EVENT_ID5:EVENT_ID4);pending[s]=false;
    }}
};
#if MEGAFLUX_WORKER_EXTERNAL_REFRESH
// Runs after parameter-only op.Init and before routing/TPipe allocation.
// Raw upper UB belongs to this loan until both copy queues are retired.
// Process()'s existing cross-rank entry rendezvous publishes all home writes
// before any peer reaches its replica weight consumption/copy phase.
__aicore__ inline void RefreshHome(const MegaMoeReplicaDescriptor& d,uint64_t w1,uint64_t w2) {
    if constexpr(g_coreType==AIV) {
        int exported[32],mask[32]={},used=0;
        for(int i=0;i<d.ranks*d.slots;++i) {
            const int e=Read((__gm__ int32_t*)d.plan+i);
            if(e>=0 && e/d.home==d.rank)mask[e%d.home]=1;
        }
        for(int e=0;e<d.home;++e)if(mask[e])exported[used++]=e;
        Copy copy;const int worker=GetBlockIdx(),workers=GetBlockNum()*2;
        for(int p=0;p<2;++p) {
            const int64_t plane=p?d.hidden*d.ffn:2*d.ffn*d.hidden;
            const uint64_t src=p?w2:w1,dst=p?d.home2:d.home1;
            const int64_t chunks=(plane+4095)/4096;
            for(int64_t task=worker;task<used*chunks;task+=workers) {
                const int e=exported[task/chunks];const int64_t begin=task%chunks*4096;
                const int count=plane-begin<4096?plane-begin:4096;
                copy.Push(src+(e*plane+begin)*2,dst+(e*plane+begin)*2,count);
            }
        }
        copy.Drain();
        // Drain retires buffer reuse on MTE2. The following scalar/global
        // rendezvous also needs explicit MTE3 completion before publishing.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID4);WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
        PipeBarrier<PIPE_ALL>();DataSyncBarrier<MemDsbT::ALL>();
    }
    SyncAll<false>();
}
#endif
struct Queue {
    uint64_t side=0,ready=0;int epoch=0;
    __aicore__ inline void Bind(uint64_t s,uint64_t r,int e) {side=s;ready=r;epoch=e;}
    __aicore__ inline void Init(const MegaMoeReplicaDescriptor& d) {
        auto* q=(__gm__ int32_t*)Field(side,1);int entries=0,tasks=0;
        int active[32],logical[32],used=0;
        for(int s=0;s<d.slots;++s) {const int e=Read((__gm__ int32_t*)d.plan+d.rank*d.slots+s);
            if(e>=0){active[used]=s;logical[used++]=e;}}
        for(int p=0;p<2;++p)for(int stripe=0;stripe<(p?d.hidden:d.ffn)/256;++stripe)
            for(int a=0;a<used;++a) {
                const int width=p?d.ffn:d.hidden;
                const int bandTasks=(256*width+CHUNK-1)/CHUNK,n=(p?1:2)*bandTasks;
                auto* t=q+HEADER+entries*WORDS;
                t[0]=active[a];t[1]=logical[a];t[2]=p;t[3]=stripe;t[4]=tasks;t[5]=n;t[6]=bandTasks;t[7]=width;
                Release(t,t[0]);
                const int idx=p?d.slots*(d.ffn/256)+active[a]*(d.hidden/256)+stripe:active[a]*(d.ffn/256)+stripe;
                Release((__gm__ int32_t*)ready+idx*WORKERS*LINE,0);
                Release((__gm__ int32_t*)ready+idx*WORKERS*LINE+LINE,0);
                ++entries;tasks+=n;
            }
        for(int i=0;i<HEADER;++i)q[i]=0;
        q[1]=entries;q[2]=tasks;q[4]=epoch;q[6]=used;
        PipeBarrier<PIPE_ALL>();DataSyncBarrier<MemDsbT::ALL>();
        Release(q+LINE,0);Release(q,0); // published before the existing all-AIV initialization rendezvous
    }
    __aicore__ inline void Service(const MegaMoeReplicaDescriptor& d) {
        if(!side)return;
        auto* q=(__gm__ int32_t*)Field(side,1);const int count=Read(q+1),total=Read(q+2);
        Copy copy;
        while(true) {
            const int task=AscendC::AtomicAdd(q,int32_t(1));
            if(task>=total)break;
            int lo=0,hi=count;
            while(lo+1<hi){const int mid=(lo+hi)/2;if(Read(q+HEADER+mid*WORDS+4)<=task)lo=mid;else hi=mid;}
            auto* t=q+HEADER+lo*WORDS;
            const int slot=Read(t),logical=Read(t+1),p=Read(t+2),stripe=Read(t+3);
            const int ordinal=task-Read(t+4),expected=Read(t+5),bands=Read(t+6),width=Read(t+7);
            const int inner=(ordinal%bands)*CHUNK,band=ordinal/bands;
            const int elements=256*width-inner<CHUNK?256*width-inner:CHUNK;
            const int64_t plane=p?d.hidden*d.ffn:2*d.ffn*d.hidden;
            const int64_t offset=((int64_t)stripe*256+(p?0:band*d.ffn))*width+inner;
            const uint64_t peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,logical/d.home);
            const uint64_t src=peer+(p?d.home2:d.home1)-d.base+((logical%d.home)*plane+offset)*2;
            const uint64_t dst=(p?d.replica2:d.replica1)+(slot*plane+offset)*2;
            // Raw asynchronous queue residence, not a pure device busy span.
            const auto copyTimer=MegaFluxStageTimer::BeginAsync(Trace(side),false,
                MegaFluxStageTimer::WORKER_COPY_RESIDENCY,logical,p,stripe,ordinal,logical/d.home,slot);
            for(int done=0;done<elements;done+=4096)copy.Push(src+done*2,dst+done*2,elements-done<4096?elements-done:4096);
            copy.Drain();
            // MTE3_MTE2 retires a UB reader for the next MTE2 operation; its
            // wait is queued on MTE2 and does not hold the scalar claimant.
            // The completion counter is scalar: acquire MTE3 on S as well.
            SetFlag<HardEvent::MTE3_S>(EVENT_ID4);WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
            DataSyncBarrier<MemDsbT::ALL>();
            MegaFluxStageTimer::EndAsync(copyTimer);
            const int idx=p?d.slots*(d.ffn/256)+slot*(d.hidden/256)+stripe:slot*(d.ffn/256)+stripe;
            auto* flag=(__gm__ int32_t*)ready+idx*WORKERS*LINE;
            const int retired=AscendC::AtomicAdd(flag+LINE,int32_t(1))+1;
            if(retired==expected) {
                MEGAFLUX_STAGE_BEGIN(publishTimer,Trace(side),false,MegaFluxStageTimer::WEIGHT_PUBLISH,logical,p,stripe,-1,logical/d.home,slot);
                DataSyncBarrier<MemDsbT::ALL>();Release(flag,epoch);
                MEGAFLUX_STAGE_END(publishTimer);
                MegaFluxStageTimer::Mark(Trace(side),false,MegaFluxStageTimer::WEIGHT_RELEASE,logical,p,stripe,-1,logical/d.home,slot);
            }
            AscendC::AtomicAdd(q+3,int32_t(1));
        }
        copy.Drain();
    }
};
} // namespace MegaFluxWorker
