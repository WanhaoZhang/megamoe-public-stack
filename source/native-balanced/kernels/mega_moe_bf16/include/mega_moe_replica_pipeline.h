#pragma once
#include "mega_moe_replica_mixed.h"
#include "mega_moe_detail_trace.h"
#include "mega_moe_stage_timer.h"

namespace MegaFluxPipeline {
using namespace AscendC;
constexpr int MAX_WORKERS=128;
constexpr int LINE=16; // int32: independent 64-byte publication lines
constexpr int COPY_UB_OFFSET=224*1024;
#if MEGAFLUX_OPT_COPY
// Two high-UB 8 KiB slots; low Fixpipe/Vector tiles are unchanged. Events 4/5
// are local to this copy scope and fully consumed before stripe publication.
struct CopyPingPong {
    LocalTensor<bfloat16_t> b0,b1;
    bool pending[2]={false,false};
    int next=0;
    __aicore__ inline CopyPingPong():
        b0(TPosition::VECCALC,COPY_UB_OFFSET,MegaFluxMixed::CHUNK),
        b1(TPosition::VECCALC,COPY_UB_OFFSET+MegaFluxMixed::CHUNK*2,MegaFluxMixed::CHUNK) {
        static_assert(COPY_UB_OFFSET+4*MegaFluxMixed::CHUNK<=248*1024,"copy UB exceeds budget");
    }
    __aicore__ inline TEventID Event(int s) {return s?EVENT_ID5:EVENT_ID4;}
    __aicore__ inline void Push(uint64_t src,uint64_t dst,int n) {
        const int s=next;next^=1;const auto event=Event(s);
        if(pending[s])WaitFlag<HardEvent::MTE3_MTE2>(event);
        auto b=s?b1:b0;
        GlobalTensor<bfloat16_t> from,to;
        from.SetGlobalBuffer((__gm__ bfloat16_t*)src);to.SetGlobalBuffer((__gm__ bfloat16_t*)dst);
        from.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);to.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyExtParams cp{1,(uint32_t)(n*2),0,0,0};
        DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
        DataCopyPad(b,from,cp,pad);
        SetFlag<HardEvent::MTE2_MTE3>(event);WaitFlag<HardEvent::MTE2_MTE3>(event);
        DataCopyPad(to,b,cp);
        SetFlag<HardEvent::MTE3_MTE2>(event);pending[s]=true;
    }
    __aicore__ inline void Drain() {
        for(int s=0;s<2;++s)if(pending[s]) {
            WaitFlag<HardEvent::MTE3_MTE2>(Event(s));pending[s]=false;
        }
    }
};
#endif
#if defined(DEEPEP_BF16_REPLICA_STREAM)
constexpr int TRACE_FIELDS=333;
#elif defined(DEEPEP_BF16_REPLICA_STRIPES)
constexpr int TRACE_FIELDS=327;
#else
constexpr int TRACE_FIELDS=325;
#endif
// copy4 + expert64 + 128 MMAD spans + last-task stamp (+ first replica span)

__aicore__ inline void Stamp(uint64_t trace,int worker,int field) {
#if defined(MEGAFLUX_SUMMARY_TIMER)
    return; // Legacy per-MMAD stamps are intentionally absent in summary capture.
#endif
    if(!trace)return;
    auto* ptr=(__gm__ int64_t*)trace+worker*TRACE_FIELDS*8+field*8;
    *ptr=GetSystemCycle();
    GlobalTensor<int64_t> gm;gm.SetGlobalBuffer(ptr);
    DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(gm);
}

__aicore__ inline void Publish(uint64_t ready,int projection,int worker,int epoch) {
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    auto* ptr=(__gm__ int32_t*)ready+(projection*MAX_WORKERS+worker)*LINE;
    *ptr=epoch;
    GlobalTensor<int32_t> gm;gm.SetGlobalBuffer(ptr);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
}

__aicore__ inline void Wait(uint64_t ready,int projection,int workers,int epoch) {
    auto* flags=(__gm__ int32_t*)ready+projection*MAX_WORKERS*LINE;
    for(int w=0;w<workers;++w)
        while(ReadGmBypassDCache(flags+w*LINE)!=epoch){}
    DataSyncBarrier<MemDsbT::ALL>();
}

// Half the AIVs copy each projection; their peers keep making progress. Scratch
// is above the two 192x256 BF16 Fixpipe tiles, never TPipe's default low UB.
__aicore__ inline void CopyProjection(const MegaMoeReplicaDescriptor& d,
        uint64_t ready,uint64_t trace,int projection,int epoch) {
    if constexpr(g_coreType==AIV) {
        const int worker=GetBlockIdx()/GetTaskRation(),workers=GetBlockNum();
        static_assert(COPY_UB_OFFSET>=2*192*256*2,"copy aliases GMM1 Fixpipe UB");
        static_assert(COPY_UB_OFFSET+MegaFluxMixed::CHUNK*2<=248*1024,"copy exceeds UB budget");
        LocalTensor<bfloat16_t> b(TPosition::VECCALC,COPY_UB_OFFSET,MegaFluxMixed::CHUNK);
        Stamp(trace,worker,projection*2);
        const int64_t plane=projection==0?2*d.ffn*d.hidden:d.hidden*d.ffn;
        const uint64_t home=projection==0?d.home1:d.home2;
        const uint64_t replica=projection==0?d.replica1:d.replica2;
        const int64_t chunks=(plane+MegaFluxMixed::CHUNK-1)/MegaFluxMixed::CHUNK;
        for(int64_t task=worker;task<d.slots*chunks;task+=workers) {
            const auto slot=task/chunks,begin=task%chunks*MegaFluxMixed::CHUNK;
            const int logical=MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+d.rank*d.slots+slot);
            if(logical<0)continue;
            const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,logical/d.home);
            const int n=plane-begin<MegaFluxMixed::CHUNK?plane-begin:MegaFluxMixed::CHUNK;
            MegaFluxMixed::Load(b,(GM_ADDR)(peer+home-d.base+(logical%d.home*plane+begin)*2),n);
            MegaFluxMixed::Store((GM_ADDR)(replica+(slot*plane+begin)*2),b,n);
        }
        PipeBarrier<PIPE_ALL>();
        Stamp(trace,worker,projection*2+1);
        Publish(ready,projection,worker,epoch);
    }
}

#if defined(DEEPEP_BF16_REPLICA_STRIPES)
constexpr int STRIPE_N=256;

__aicore__ inline void WaitStripe(uint64_t ready,int slot,int stripe,int stripes,
                                 int workers,int epoch) {
    auto* flags=(__gm__ int32_t*)ready+(slot*stripes+stripe)*MAX_WORKERS*LINE;
#if MEGAFLUX_WORKER_FORWARD
    while(ReadGmBypassDCache(flags)!=epoch){}
#else
    for(int w=0;w<workers;++w)
        while(ReadGmBypassDCache(flags+w*LINE)!=epoch){}
#endif
    DataSyncBarrier<MemDsbT::ALL>();
}

#if MEGAFLUX_OPT_CACHE
// Direct-mapped positive acquire cache. Collisions only cause another wait.
// GroupMatmulParam is constructed anew inside each forward, once per AIC and
// projection. Rebinding address/epoch/projection also invalidates every entry.
struct ReadyCache {
    uint64_t ready=0;
    int epoch=0,projection=-1;
    int acquired[64]={};
    __aicore__ inline bool Has(uint64_t ptr,int ep,int proj,int key) {
        if(ready!=ptr || epoch!=ep || projection!=proj) {
            ready=ptr;epoch=ep;projection=proj;
            for(int i=0;i<64;++i)acquired[i]=0;
        }
        return acquired[key&63]==key;
    }
    __aicore__ inline void Add(int key) {acquired[key&63]=key;}
};
#endif

// Independent AIV1 producer after Dispatch retires. No Cube handshake here:
// AIC and AIV0 may consume early stripes while later stripes are copied.
__aicore__ inline void CopyW1Stripes(const MegaMoeReplicaDescriptor& d,
        uint64_t ready,uint64_t trace,int epoch,uint64_t cache=0) {
    if constexpr(g_coreType==AIV) {
        const int worker=GetBlockIdx()/GetTaskRation(),workers=GetBlockNum();
        const int stripes=(d.ffn+STRIPE_N-1)/STRIPE_N;
        LocalTensor<bfloat16_t> b(TPosition::VECCALC,COPY_UB_OFFSET,MegaFluxMixed::CHUNK);
#if MEGAFLUX_OPT_COPY
        CopyPingPong copy;
#endif
        const auto phaseTimer=MegaFluxStageTimer::BeginAsync(trace,false,MegaFluxStageTimer::FWD_COPY_PHASE,-1,0);
        Stamp(trace,worker,0);
        for(int slot=0;slot<d.slots;++slot) {
            const int logical=MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+d.rank*d.slots+slot);
            if(logical<0)continue;
            const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,logical/d.home);
            const int64_t plane=2*d.ffn*d.hidden;
            for(int stripe=0;stripe<stripes;++stripe) {
                MEGAFLUX_STAGE_BEGIN(stageCopy,trace,false,MegaFluxStageTimer::COPY,logical,0,stripe,-1,logical/d.home,slot);
                const auto copyStart=MegaFluxDetail::Now(trace);
                int64_t copiedBytes=0;
                const int64_t first=stripe*STRIPE_N;
                const int64_t rows=d.ffn-first<STRIPE_N?d.ffn-first:STRIPE_N;
                const int64_t elements=rows*d.hidden;
                const int64_t chunks=(elements+MegaFluxMixed::CHUNK-1)/MegaFluxMixed::CHUNK;
                for(int64_t task=worker;task<2*chunks;task+=workers) {
                    const int64_t begin=task%chunks*MegaFluxMixed::CHUNK;
                    const int64_t offset=(first+(task/chunks)*d.ffn)*d.hidden+begin;
                    const int n=elements-begin<MegaFluxMixed::CHUNK?elements-begin:MegaFluxMixed::CHUNK;
#if MEGAFLUX_OPT_COPY
                    const uint64_t cached=cache?*((__gm__ uint64_t*)cache+slot*2):0;
                    copy.Push(cached?cached+offset*2:peer+d.home1-d.base+(logical%d.home*plane+offset)*2,
                        d.replica1+(slot*plane+offset)*2,n);
#else
                    const uint64_t cached=cache?*((__gm__ uint64_t*)cache+slot*2):0;
                    MegaFluxMixed::Load(b,(GM_ADDR)(cached?cached+offset*2:peer+d.home1-d.base+(logical%d.home*plane+offset)*2),n);
                    MegaFluxMixed::Store((GM_ADDR)(d.replica1+(slot*plane+offset)*2),b,n);
#endif
                    copiedBytes+=n*2;
                }
                // Same release fence/cache-line publication as matrix readiness.
#if MEGAFLUX_OPT_COPY
                copy.Drain(); // includes zero-task workers, before release
#endif
                MEGAFLUX_STAGE_END(stageCopy);
                const auto copied=MegaFluxDetail::Now(trace);
                MEGAFLUX_STAGE_BEGIN(stagePublish,trace,false,MegaFluxStageTimer::WEIGHT_PUBLISH,logical,0,stripe,-1,logical/d.home,slot);
                Publish(ready,slot*stripes+stripe,worker,epoch);
                MEGAFLUX_STAGE_END(stagePublish);
                const auto published=MegaFluxDetail::Now(trace);
                MegaFluxStageTimer::Mark(trace,false,MegaFluxStageTimer::WEIGHT_RELEASE,logical,0,stripe,-1,logical/d.home,slot);
                MegaFluxDetail::Record(trace,false,MegaFluxDetail::COPY,copyStart,copied,logical,0,stripe,-1,logical/d.home,slot,copiedBytes);
                MegaFluxDetail::Record(trace,false,MegaFluxDetail::WEIGHT_RELEASE,published,published,logical,0,stripe,-1,logical/d.home,slot);
            }
        }
        PipeBarrier<PIPE_ALL>();
        Stamp(trace,worker,1);MegaFluxStageTimer::EndAsync(phaseTimer);
    }
}

#if defined(DEEPEP_BF16_REPLICA_STREAM)
// W2 is [H,F]: one GMM2 output-N stripe is contiguous 256 rows, full K=F.
__aicore__ inline void CopyW2StripeRange(const MegaMoeReplicaDescriptor& d,
        uint64_t ready,uint64_t trace,int epoch,int firstStripe,int endStripe,uint64_t cache=0) {
    if constexpr(g_coreType==AIV) {
        if(firstStripe>=endStripe)return;
        const int worker=GetBlockIdx()/GetTaskRation(),workers=GetBlockNum();
        const int stripes=(d.hidden+STRIPE_N-1)/STRIPE_N;
        const int offset=d.slots*((d.ffn+STRIPE_N-1)/STRIPE_N);
        LocalTensor<bfloat16_t> b(TPosition::VECCALC,COPY_UB_OFFSET,MegaFluxMixed::CHUNK);
#if MEGAFLUX_OPT_COPY
        CopyPingPong copy;
#endif
        const auto phaseTimer=(firstStripe==0&&endStripe==stripes)?MegaFluxStageTimer::BeginAsync(trace,false,MegaFluxStageTimer::FWD_COPY_PHASE,-1,1):0;
        if(firstStripe==0)Stamp(trace,worker,2);
        for(int slot=0;slot<d.slots;++slot) {
            const int logical=MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+d.rank*d.slots+slot);
            if(logical<0)continue;
            const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,logical/d.home);
            const int64_t plane=d.hidden*d.ffn;
            for(int stripe=firstStripe;stripe<endStripe;++stripe) {
                MEGAFLUX_STAGE_BEGIN(stageCopy,trace,false,MegaFluxStageTimer::COPY,logical,1,stripe,-1,logical/d.home,slot);
                const auto copyStart=MegaFluxDetail::Now(trace);
                int64_t copiedBytes=0;
                const int64_t first=stripe*STRIPE_N;
                const int64_t rows=d.hidden-first<STRIPE_N?d.hidden-first:STRIPE_N;
                const int64_t elements=rows*d.ffn;
                const int64_t chunks=(elements+MegaFluxMixed::CHUNK-1)/MegaFluxMixed::CHUNK;
                for(int64_t task=worker;task<chunks;task+=workers) {
                    const int64_t begin=task*MegaFluxMixed::CHUNK;
                    const int64_t position=first*d.ffn+begin;
                    const int n=elements-begin<MegaFluxMixed::CHUNK?elements-begin:MegaFluxMixed::CHUNK;
#if MEGAFLUX_OPT_COPY
                    const uint64_t cached=cache?*((__gm__ uint64_t*)cache+slot*2+1):0;
                    copy.Push(cached?cached+position*2:peer+d.home2-d.base+(logical%d.home*plane+position)*2,
                        d.replica2+(slot*plane+position)*2,n);
#else
                    const uint64_t cached=cache?*((__gm__ uint64_t*)cache+slot*2+1):0;
                    MegaFluxMixed::Load(b,(GM_ADDR)(cached?cached+position*2:peer+d.home2-d.base+(logical%d.home*plane+position)*2),n);
                    MegaFluxMixed::Store((GM_ADDR)(d.replica2+(slot*plane+position)*2),b,n);
#endif
                    copiedBytes+=n*2;
                }
#if MEGAFLUX_OPT_COPY
                copy.Drain();
#endif
                MEGAFLUX_STAGE_END(stageCopy);
                const auto copied=MegaFluxDetail::Now(trace);
                MEGAFLUX_STAGE_BEGIN(stagePublish,trace,false,MegaFluxStageTimer::WEIGHT_PUBLISH,logical,1,stripe,-1,logical/d.home,slot);
                Publish(ready,offset+slot*stripes+stripe,worker,epoch);
                MEGAFLUX_STAGE_END(stagePublish);
                const auto published=MegaFluxDetail::Now(trace);
                MegaFluxStageTimer::Mark(trace,false,MegaFluxStageTimer::WEIGHT_RELEASE,logical,1,stripe,-1,logical/d.home,slot);
                MegaFluxDetail::Record(trace,false,MegaFluxDetail::COPY,copyStart,copied,logical,1,stripe,-1,logical/d.home,slot,copiedBytes);
                MegaFluxDetail::Record(trace,false,MegaFluxDetail::WEIGHT_RELEASE,published,published,logical,1,stripe,-1,logical/d.home,slot);
            }
        }
        PipeBarrier<PIPE_ALL>();if(endStripe==stripes)Stamp(trace,worker,3);MegaFluxStageTimer::EndAsync(phaseTimer);
    }
}
__aicore__ inline void CopyW2Stripes(const MegaMoeReplicaDescriptor& d,
        uint64_t ready,uint64_t trace,int epoch,uint64_t cache=0) {
    CopyW2StripeRange(d,ready,trace,epoch,0,(d.hidden+STRIPE_N-1)/STRIPE_N,cache);
}
#endif
#endif
}
