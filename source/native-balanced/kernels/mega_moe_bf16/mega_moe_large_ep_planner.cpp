#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
#include "mega_moe_large_ep_policy.h"
using namespace AscendC;
#include "mega_moe_planner_stage_timer.h"
namespace {
constexpr int Workers=32, MaxE=2048, MaxCopies=2048, Tile=1024;
__aicore__ inline int Read(__gm__ int32_t* p) { return ReadGmBypassDCache(p); }
__aicore__ inline int Min(int a,int b) { return a<b?a:b; }
__aicore__ inline void Publish() {
    GlobalTensor<uint8_t> gm;
    DataCacheCleanAndInvalid<uint8_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
}
__aicore__ inline void Load(LocalTensor<int32_t> tile,GlobalTensor<int32_t> gm,int pos,int size) {
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopyExtParams p(1,size*4,0,0,0);
    DataCopyPadExtParams<int32_t> pad{false,0,0,0};DataCopyPad(tile,gm[pos],p,pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}
}

// Separate large-EP entry: the measured <=8 kernels and their scratch ABI stay
// intact. The public assignment buffer is working GM Q, not a local matrix.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mega_moe_large_ep_planner_kernel(uint64_t ffts,GM_ADDR base,GM_ADDR ids,
    GM_ADDR copies,GM_ADDR physical,GM_ADDR logical,GM_ADDR assignment,GM_ADDR header,
    uint64_t histoff,uint64_t metaoff,uint64_t erroroff,int n,int topk,
    int home,int slots,int rank,int ranks,int beta,int penalty,int gain,int epoch,int capacity,int algorithm) {
    util_set_ffts_config(ffts);if ASCEND_IS_AIC {return;}
    const int worker=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);
    LARGE_PLAN_START(histogramTimer,0);
    const int experts=home*ranks,es=home+slots,stride=(experts+31)/32*32;
    const int chunk=((n+Workers-1)/Workers+31)/32*32;
    const int begin=Min(n,worker*chunk),end=Min(n,begin+chunk);
    auto hist=(__gm__ int32_t*)(base+histoff),meta=(__gm__ int32_t*)(base+metaoff);
    auto errors=(__gm__ int32_t*)(base+erroroff);
    auto prefixes=errors+Workers*32,totals=prefixes+experts*Workers;
    auto quota=totals+experts*Workers,cumulative=quota+ranks*ranks;
    auto cpg=(__gm__ int32_t*)copies,q=(__gm__ int32_t*)assignment;
    auto out=(__gm__ int32_t*)physical,lc=(__gm__ int32_t*)logical,hdr=(__gm__ int32_t*)header;
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,Tile*4);
    auto tile=buffer.Get<int32_t>();GlobalTensor<int32_t> input;
    input.SetGlobalBuffer((__gm__ int32_t*)ids);input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    int local[MaxE];for(int e=0;e<experts;++e)local[e]=0;
    int bad=0;
    for(int pos=begin;pos<end;pos+=Tile) {
        int size=Min(Tile,end-pos);Load(tile,input,pos,size);
        for(int i=0;i<size;++i) {int e=tile.GetValue(i);if(e<0||e>=experts)bad=1;else ++local[e];}
    }
    for(int e=0;e<experts;++e)hist[worker*stride+e]=local[e];
    errors[worker*32]=bad;
    if(worker==0) {
        meta[0]=epoch;meta[1]=n;meta[2]=topk;meta[3]=experts;meta[4]=slots;
        meta[5]=beta;meta[6]=penalty;meta[7]=gain;meta[9]=capacity;meta[10]=algorithm;
    }
    LARGE_PLAN_END(histogramTimer);
    LARGE_PLAN_START(exchangeTimer,1);
    Publish();
#if MEGAFLUX_MOONEP_TIMER
    const auto localBefore = GetSystemCycle();
#endif
    SyncAll<true>();
#if MEGAFLUX_MOONEP_TIMER
    const auto localAfter = GetSystemCycle();
#endif
    for(int e=worker;e<experts;e+=Workers) {
        int sum=0;
        for(int w=0;w<Workers;++w) {prefixes[e*32+w]=sum;sum+=Read(hist+w*stride+e);}
        totals[e*32]=sum;
    }
    if(worker==0) {int error=0;for(int w=0;w<Workers;++w)if(Read(errors+w*32))error=1;meta[8]=error;}
    Publish();
#if MEGAFLUX_MOONEP_TIMER
    const auto epBefore = GetSystemCycle();
#endif
    aclshmemx_barrier_all_vec();
#if MEGAFLUX_MOONEP_TIMER
    const auto epAfter = GetSystemCycle();
#endif
    for(int e=worker;e<experts;e+=Workers) {
        int total=0,prefix=0;
        for(int r=0;r<ranks;++r) {
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);
            auto counts=(__gm__ int32_t*)(peer+erroroff+Workers*128+experts*Workers*4);
            int v=Read(counts+e*32);total+=v;if(r<rank)prefix+=v;
        }
        totals[e*32+1]=total;totals[e*32+2]=prefix;
    }
#if MEGAFLUX_MOONEP_TIMER
    const auto countsAfter = GetSystemCycle();
#endif
    if(worker==0) {
        int error=0;
        for(int r=0;r<ranks;++r) {
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);auto pm=(__gm__ int32_t*)(peer+metaoff);
            if(Read(pm+8)&&error<1)error=1;
            for(int f=0;f<11;++f)if(f!=8&&Read(pm+f)!=Read(meta+f))error=2;
        }
        if(algorithm!=-2&&algorithm!=-1&&algorithm!=0)error=2;
        errors[1]=error;
    }
    Publish();
#if MEGAFLUX_MOONEP_TIMER
    const auto finalSyncBefore = GetSystemCycle();
#endif
    SyncAll<true>();
    LARGE_PLAN_END(exchangeTimer);
#if MEGAFLUX_MOONEP_TIMER
    if (exchangeTimer) {
        // Keep the original pair at [0, 1]. Store the split after the window
        // so recording adds no memory traffic or pipe fence to the barrier.
        auto* split = reinterpret_cast<__gm__ int64_t*>(exchangeTimer);
        split[2] = localBefore;
        split[3] = localAfter;
        split[4] = epBefore;
        split[5] = epAfter;
        split[6] = countsAfter;
        split[7] = finalSyncBefore;
    }
#endif
    LARGE_PLAN_START(policyTimer,2);
    if(worker==0) {
        // Reuse the histogram array after its publication. Separate E-sized
        // caches made the binary exceed the runtime's 32 KiB stack limit.
        int* counts=local;
        int before=0,error=Read(errors+1),active=0,rows[256];
        for(int e=0;e<experts;++e) {counts[e]=Read(totals+e*32+1);lc[e]=counts[e];}
        for(int i=0;i<experts*ranks;++i)q[i]=0;
        for(int i=0;i<ranks*slots;++i)cpg[i]=-1;
        if(!error) {
            bool ok=algorithm==-2 ? MegaMoeLargePolicy::Moon(counts,experts,ranks,home,n,q,cpg,quota) :
                MegaMoeLargePolicy::Native(counts,experts,ranks,home,slots,beta,penalty,gain,q,cpg,before);
            if(!ok)error=1;
        }
        for(int r=0;r<ranks;++r)rows[r]=0;
        for(int e=0;e<experts;++e) {
            int blocks=0,offset=0;
            for(int r=0;r<ranks;++r) {
                int value=q[e*ranks+r];if(value<0&&!error)error=1;
                if(value>0&&r!=e/home) {
                    bool found=false;for(int s=0;s<slots;++s)if(cpg[r*slots+s]==e)found=true;
                    if(!found&&!error)error=1;
                }
                blocks+=value;cumulative[e*ranks+r]=blocks;
                int64_t next=static_cast<int64_t>(blocks)*beta;
                int clipped=next>counts[e]?counts[e]:static_cast<int>(next);
                rows[r]+=clipped-offset;offset=clipped;
            }
            if(!error&&blocks!=(static_cast<int64_t>(counts[e])+beta-1)/beta)error=1;
        }
        for(int i=0;i<ranks*slots;++i)if(cpg[i]>=0)++active;
        int required=1;for(int r=0;r<ranks;++r)if(rows[r]>required)required=rows[r];
        required=(required+127)/128*128;if(!error&&required>capacity)error=3;
        hdr[0]=active;hdr[1]=required;hdr[2]=epoch;hdr[3]=error;
        hdr[4]=before;hdr[5]=active>0;hdr[6]=active;hdr[7]=capacity;Publish();
    }
    aclshmemx_barrier_all_vec();
    LARGE_PLAN_END(policyTimer);
    LARGE_PLAN_START(routeTimer,3);
    bad=0;
    if(!Read(hdr+3)) {
        for(int e=0;e<experts;++e)local[e]=Read(prefixes+e*32+worker)+Read(totals+e*32+2);
        for(int pos=begin;pos<end;pos+=Tile) {
            int size=Min(Tile,end-pos);Load(tile,input,pos,size);
            for(int i=0;i<size;++i) {
                int e=tile.GetValue(i),block=local[e]++/beta,lo=0,hi=ranks;
                while(lo<hi) {int mid=(lo+hi)/2;if(Read(cumulative+e*ranks+mid)<=block)lo=mid+1;else hi=mid;}
                if(lo>=ranks) {bad=1;out[pos+i]=-1;continue;}
                int slot=e%home;
                if(lo!=e/home) {
                    bool found=false;
                    for(int s=0;s<slots;++s)if(Read(cpg+lo*slots+s)==e){slot=home+s;found=true;break;}
                    if(!found) {bad=1;out[pos+i]=-1;continue;}
                }
                out[pos+i]=lo*es+slot;
            }
        }
        Publish();
    }
    LARGE_PLAN_END(routeTimer);
    LARGE_PLAN_START(retireTimer,4);
    // A route inconsistency must fail on all ranks before the fused executor
    // can consume physical IDs, rather than leaving one peer in a barrier.
    errors[worker*32]=bad;Publish();SyncAll<true>();
    if(worker==0) {
        int error=Read(hdr+3);
        for(int w=0;w<Workers;++w)if(Read(errors+w*32)&&!error)error=1;
        meta[8]=error;Publish();
    }
    aclshmemx_barrier_all_vec();
    if(worker==0) {
        int error=Read(hdr+3);
        for(int r=0;r<ranks;++r) {
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);
            int value=Read((__gm__ int32_t*)(peer+metaoff)+8);if(value>error)error=value;
        }
        hdr[3]=error;Publish();
    }
    aclshmemx_barrier_all_vec();
    pipe.Destroy();
    LARGE_PLAN_END(retireTimer);
}
extern "C" __attribute__((visibility("default"))) int megaflux_moonep_timer_schema() {
    return MEGAFLUX_MOONEP_TIMER ? 1 : 0;
}
extern "C" __attribute__((visibility("default"))) void mega_moe_large_ep_planner_do(
    void* stream,uint64_t ffts,uint8_t* base,uint8_t* ids,uint8_t* copies,uint8_t* physical,
    uint8_t* logical,uint8_t* assignment,uint8_t* header,uint64_t histogram,uint64_t meta,uint64_t errors,
    int n,int topk,int home,int slots,int rank,int ranks,int beta,int penalty,int gain,int epoch,int capacity,int algorithm) {
    mega_moe_large_ep_planner_kernel<<<Workers,nullptr,stream>>>(ffts,base,ids,copies,physical,logical,assignment,header,
        histogram,meta,errors,n,topk,home,slots,rank,ranks,beta,penalty,gain,epoch,capacity,algorithm);
}

extern "C" __attribute__((visibility("default"))) int megaflux_max_local_experts(){return MEGAMOE_MAX_LOCAL_EXPERTS;}
