#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
using namespace AscendC;

namespace {
constexpr int32_t CHUNK = 4096;
__aicore__ inline int32_t Fresh(__gm__ int32_t* p) {
    return ReadGmBypassDCache(p);
}
__aicore__ inline void PublishScalarWrites() {
    GlobalTensor<uint8_t> gm;
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE,
        DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
}
__aicore__ inline int32_t Worker() {
    return GetBlockIdx() / (GetSubBlockNum() > 0 ? GetSubBlockNum() : 1);
}
__aicore__ inline void Load(LocalTensor<bfloat16_t> ub, GM_ADDR addr, int32_t n) {
    // The previous cast or MTE3 transfer may still consume this UB buffer.
    // MTE3_V at Store's end gates V; this V_MTE2 edge also gates the next DMA.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    GlobalTensor<bfloat16_t> gm; gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams p(1, n * 2, 0, 0, 0);
    DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
    DataCopyPad(ub, gm, p, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void Store(GM_ADDR addr, LocalTensor<bfloat16_t> ub, int32_t n) {
    GlobalTensor<bfloat16_t> gm; gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams p(1,n*2,0,0,0); DataCopyPad(gm,ub,p);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
}

// mode 0: fresh replica weights, 1: publish BF16 partials + owner FP32 SUM,
// mode 2: publish exec counts + return actual logical owner counts.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void mega_moe_replica_kernel(uint64_t ffts, GM_ADDR base, GM_ADDR plan,
    GM_ADDR in1, GM_ADDR in2, GM_ADDR out1, GM_ADDR out2,
    int64_t eh, int64_t slots, int64_t h, int64_t f,
    uint64_t w1, uint64_t w2, uint64_t dw1, uint64_t dw2, uint64_t counts,
    int32_t rank, int32_t ranks, int32_t mode) {
    util_set_ffts_config(ffts);
    if ASCEND_IS_AIC { return; }
    const int32_t worker=Worker(), workers=GetBlockNum();
    const int64_t es=eh+slots, p1=2*f*h, p2=h*f;
    auto* copies=(__gm__ int32_t*)plan;
    int32_t copyTable[2048];
    for(int32_t i=0;i<ranks*slots;++i)copyTable[i]=Fresh(copies+i);
    if (mode==2) {
        // A single writer owns the small count header: different scalar writers
        // sharing a cache line would race when each flushes its private DCache.
        if (worker==0) {
            for (int64_t e=0;e<es;++e)
                ((__gm__ int32_t*)(base+counts))[e]=Fresh((__gm__ int32_t*)in1+e);
            PublishScalarWrites();
        }
        PipeBarrier<PIPE_ALL>(); aclshmemx_barrier_all_vec();
        if (worker==0) for (int64_t e=0;e<eh;++e) {
            int32_t sum=Fresh((__gm__ int32_t*)(base+counts)+e);
            const int32_t logical=rank*eh+e;
            for(int32_t r=0;r<ranks;++r) for(int32_t s=0;s<slots;++s) {
                if(copyTable[r*slots+s]==logical) {
                    auto* peer=(__gm__ int32_t*)aclshmem_ptr(base+counts,r);
                    sum+=Fresh(peer+eh+s);
                }
            }
            ((__gm__ int32_t*)out1)[e]=sum;
        }
        if (worker==0) PublishScalarWrites();
        PipeBarrier<PIPE_ALL>(); aclshmemx_barrier_all_vec(); return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> bbuf, fbuf, abuf;
    pipe.InitBuffer(bbuf,CHUNK*2); pipe.InitBuffer(fbuf,CHUNK*4); pipe.InitBuffer(abuf,CHUNK*4);
    auto b=bbuf.Get<bfloat16_t>();auto tmp=fbuf.Get<float>();auto accum=abuf.Get<float>();
    // Split storage restore for a network backward. Homes were republished
    // from this layer's live FSDP Parameter; outputs are separate replica spans.
    if(mode==0 || mode==4) {
        aclshmemx_barrier_all_vec();
        for(int32_t projection=0;projection<2;++projection) {
            const int64_t plane=projection==0?p1:p2;
            const uint64_t off=projection==0?w1:w2;
            const int64_t chunks=(plane+CHUNK-1)/CHUNK;
            for(int64_t task=worker;task<slots*chunks;task+=workers) {
                const int64_t s=task/chunks, begin=(task%chunks)*CHUNK;
                const int32_t logical=copyTable[rank*slots+s];
                if(logical<0) continue;
                const int32_t owner=logical/eh, local=logical%eh;
                GM_ADDR peer=(GM_ADDR)aclshmem_ptr(base,owner);
                const int32_t n=plane-begin<CHUNK?plane-begin:CHUNK;
                Load(b,peer+off+(local*plane+begin)*2,n);
                GM_ADDR target=mode==4
                    ? (projection==0?out1:out2)+(s*plane+begin)*2
                    : base+off+((eh+s)*plane+begin)*2;
                Store(target,b,n);
            }
        }
        PipeBarrier<PIPE_ALL>(); aclshmemx_barrier_all_vec(); return;
    }
    // mode 3 publishes only active remote replicas. The S3 mode 1 remains an
    // independently measurable full-plane control.
    for(int32_t projection=0;projection<2;++projection) {
        const int64_t plane=projection==0?p1:p2;
        const uint64_t off=projection==0?dw1:dw2;
        GM_ADDR input=projection==0?in1:in2;
        const int64_t total=es*plane;
        for(int64_t begin=worker*CHUNK;begin<total;begin+=workers*CHUNK) {
            if(mode==3) {
                const int64_t slot=begin/plane-eh;
                if(slot<0 || copyTable[rank*slots+slot]<0)continue;
            }
            const int32_t n=total-begin<CHUNK?total-begin:CHUNK;
            Load(b,input+begin*2,n);Store(base+off+begin*2,b,n);
        }
    }
    PipeBarrier<PIPE_ALL>();aclshmemx_barrier_all_vec();
    for(int32_t projection=0;projection<2;++projection) {
        const int64_t plane=projection==0?p1:p2;
        const uint64_t off=projection==0?dw1:dw2;
        GM_ADDR output=projection==0?out1:out2;
        const int64_t chunks=(plane+CHUNK-1)/CHUNK;
        for(int64_t task=worker;task<eh*chunks;task+=workers) {
            const int64_t local=task/chunks,begin=task%chunks*CHUNK;
            const int32_t n=plane-begin<CHUNK?plane-begin:CHUNK;
            const int32_t logical=rank*eh+local;
            bool remote=false;
            if(mode==3) for(int32_t r=0;r<ranks;++r) for(int32_t s=0;s<slots;++s)
                if(copyTable[r*slots+s]==logical)remote=true;
            if(mode==3 && !remote)continue;
            GM_ADDR homeInput=(mode==3?(projection==0?in1:in2):base+off);
            Load(b,homeInput+(local*plane+begin)*2,n);
            Cast(accum,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
            for(int32_t r=0;r<ranks;++r) for(int32_t s=0;s<slots;++s) {
                if(copyTable[r*slots+s]!=logical)continue;
                GM_ADDR peer=(GM_ADDR)aclshmem_ptr(base,r);
                Load(b,peer+off+((eh+s)*plane+begin)*2,n);
                Cast(tmp,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
                Add(accum,accum,tmp,n);PipeBarrier<PIPE_V>();
            }
            Cast(b,accum,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
            Store(output+(local*plane+begin)*2,b,n);
        }
    }
    PipeBarrier<PIPE_ALL>();aclshmemx_barrier_all_vec();
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void mega_moe_replica_route_kernel(GM_ADDR blocks,GM_ADDR table,GM_ADDR output,int64_t n) {
    if ASCEND_IS_AIC {return;}
    auto* b=(__gm__ int32_t*)blocks;auto* t=(__gm__ int32_t*)table;auto* o=(__gm__ int32_t*)output;
    // Partition whole 128-byte regions, so no two scalar writers share a
    // cache line (including the final partial region).
    for(int64_t begin=Worker()*32;begin<n;begin+=GetBlockNum()*32)
        for(int64_t i=begin;i<begin+32 && i<n;++i) o[i]=Fresh(t+Fresh(b+i));
    PublishScalarWrites();
    PipeBarrier<PIPE_ALL>();
}

extern "C" __attribute__((visibility("default"))) int megaflux_replica_restore_version() {
    return 1;
}

extern "C" __attribute__((visibility("default"))) void mega_moe_replica_do(
    void* stream,uint64_t ffts,uint8_t* base,uint8_t* plan,uint8_t* in1,uint8_t* in2,
    uint8_t* out1,uint8_t* out2,int64_t eh,int64_t slots,int64_t h,int64_t f,
    uint64_t w1,uint64_t w2,uint64_t dw1,uint64_t dw2,uint64_t counts,
    int32_t rank,int32_t ranks,int32_t mode) {
    mega_moe_replica_kernel<<<32,nullptr,stream>>>(ffts,base,plan,in1,in2,out1,out2,
        eh,slots,h,f,w1,w2,dw1,dw2,counts,rank,ranks,mode);
}
extern "C" __attribute__((visibility("default"))) void mega_moe_replica_route_do(
    void* stream,uint8_t* blocks,uint8_t* table,uint8_t* output,int64_t n) {
    mega_moe_replica_route_kernel<<<32,nullptr,stream>>>(blocks,table,output,n);
}
