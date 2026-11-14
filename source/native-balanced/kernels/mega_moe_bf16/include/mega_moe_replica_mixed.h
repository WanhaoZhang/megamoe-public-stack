#pragma once
#include "mega_moe_replica_descriptor.h"
#include "shmem.h"
#include "mega_moe_large_ep_limits.h"

// All paired AIVs participate; AICs meet them at SyncAll<false> outside this
// helper. No pure-AIV aclshmemx_barrier_all_vec is used inside the mixed grid.
namespace MegaFluxMixed {
using namespace AscendC;
constexpr int CHUNK=4096;
__aicore__ inline int Fresh(__gm__ int32_t* p) {return ReadGmBypassDCache(p);}
__aicore__ inline void Load(LocalTensor<bfloat16_t> ub,GM_ADDR addr,int n) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    GlobalTensor<bfloat16_t> gm;gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams p(1,n*2,0,0,0);DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
    DataCopyPad(ub,gm,p,pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void Store(GM_ADDR addr,LocalTensor<bfloat16_t> ub,int n) {
    GlobalTensor<bfloat16_t> gm;gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams p(1,n*2,0,0,0);DataCopyPad(gm,ub,p);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
template<class Op>
__aicore__ inline void Copy(const MegaMoeReplicaDescriptor& d,Op& op) {
    if constexpr(g_coreType==AIV) {
        op.ReplicaBarrier();
        TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,CHUNK*2);
        auto b=buffer.Get<bfloat16_t>();const int worker=GetBlockIdx(),workers=GetBlockNum()*2;
        auto* plan=(__gm__ int32_t*)d.plan;
        for(int projection=0;projection<2;++projection) {
            const int64_t plane=projection==0?2*d.ffn*d.hidden:d.hidden*d.ffn;
            const uint64_t home=projection==0?d.home1:d.home2;
            const uint64_t replica=projection==0?d.replica1:d.replica2;
            const int64_t chunks=(plane+CHUNK-1)/CHUNK;
            for(int64_t task=worker;task<d.slots*chunks;task+=workers) {
                const auto slot=task/chunks,begin=task%chunks*CHUNK;
                const int logical=Fresh(plan+d.rank*d.slots+slot);if(logical<0)continue;
                const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,logical/d.home);
                const int n=plane-begin<CHUNK?plane-begin:CHUNK;
                Load(b,(GM_ADDR)(peer+home-d.base+(logical%d.home*plane+begin)*2),n);
                Store((GM_ADDR)(replica+(slot*plane+begin)*2),b,n);
            }
        }
        PipeBarrier<PIPE_ALL>();pipe.Destroy();op.ReplicaBarrier();
    }
    SyncAll<false>();
}
template<class Op>
__aicore__ inline void Reduce(const MegaMoeReplicaDescriptor& d,Op& op,GM_ADDR dw1,GM_ADDR dw2) {
    // AIC FIX output and AIV empty-expert clearing must both finish before
    // any AIV exports or reads a partial. Drain all pipelines before rendezvous.
    PipeBarrier<PIPE_ALL>();SyncAll<false>();
    if constexpr(g_coreType==AIV) {
        TPipe pipe;TBuf<TPosition::VECCALC> bbuf,tbuf,abuf;
        pipe.InitBuffer(bbuf,CHUNK*2);pipe.InitBuffer(tbuf,CHUNK*4);pipe.InitBuffer(abuf,CHUNK*4);
        auto b=bbuf.Get<bfloat16_t>();auto tmp=tbuf.Get<float>();auto accum=abuf.Get<float>();
        const int worker=GetBlockIdx(),workers=GetBlockNum()*2;
        auto* plan=(__gm__ int32_t*)d.plan;int copies[MegaMoeLargeEP::MaxCopies];
        for(int i=0;i<d.ranks*d.slots;++i)copies[i]=Fresh(plan+i);
        for(int projection=0;projection<2;++projection) {
            const int64_t plane=projection==0?2*d.ffn*d.hidden:d.hidden*d.ffn;
            const auto input=projection==0?dw1:dw2;
            const uint64_t partial=projection==0?d.partial1:d.partial2;
            const int64_t chunks=(plane+CHUNK-1)/CHUNK;
            for(int64_t task=worker;task<d.slots*chunks;task+=workers) {
                const auto slot=task/chunks,begin=task%chunks*CHUNK;
                if(copies[d.rank*d.slots+slot]<0)continue;
                const int n=plane-begin<CHUNK?plane-begin:CHUNK;
                Load(b,input+((d.home+slot)*plane+begin)*2,n);
                Store((GM_ADDR)(partial+(slot*plane+begin)*2),b,n);
            }
        }
        PipeBarrier<PIPE_ALL>();op.ReplicaBarrier();
        for(int projection=0;projection<2;++projection) {
            const int64_t plane=projection==0?2*d.ffn*d.hidden:d.hidden*d.ffn;
            const auto input=projection==0?dw1:dw2;
            const uint64_t partial=projection==0?d.partial1:d.partial2;
            const int64_t chunks=(plane+CHUNK-1)/CHUNK;
            for(int64_t task=worker;task<d.home*chunks;task+=workers) {
                const auto expert=task/chunks,begin=task%chunks*CHUNK;
                const int logical=d.rank*d.home+expert;bool remote=false;
                for(int i=0;i<d.ranks*d.slots;++i)if(copies[i]==logical)remote=true;
                if(!remote)continue;
                const int n=plane-begin<CHUNK?plane-begin:CHUNK;
                Load(b,input+(expert*plane+begin)*2,n);
                Cast(accum,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
                for(int rank=0;rank<d.ranks;++rank)for(int slot=0;slot<d.slots;++slot) {
                    if(copies[rank*d.slots+slot]!=logical)continue;
                    const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,rank);
                    Load(b,(GM_ADDR)(peer+partial-d.base+(slot*plane+begin)*2),n);
                    Cast(tmp,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
                    Add(accum,accum,tmp,n);PipeBarrier<PIPE_V>();
                }
                Cast(b,accum,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
                Store(input+(expert*plane+begin)*2,b,n);
            }
        }
        PipeBarrier<PIPE_ALL>();pipe.Destroy();op.ReplicaBarrier();
    }
}
}
