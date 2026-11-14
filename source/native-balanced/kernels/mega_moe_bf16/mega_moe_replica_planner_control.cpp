#include "mega_moe_capacity.h"
#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
#if defined(DEEPEP_MOONEP_PLANNER)
#include "mega_moe_moonep_algorithm.h"
#define MEGAFLUX_DIRECT_INLINE __aicore__ inline
#include "mega_moe_moonep_direct.h"
#endif
#if defined(DEEPEP_OPTIMIZED_PLANNER)
#include "include/mega_moe_search_algorithm.h"
#endif
using namespace AscendC;
#if defined(DEEPEP_MOONEP_STAGE_TIMER)
#include "../common/timer/AscendTimerV2_device.hpp"
__aicore__ inline uint64_t PlannerStart(GM_ADDR header,int worker,int stage) {
    auto* h=(__gm__ int64_t*)header+4;
    if(h[0]!=0x4d46504c0001LL)return 0;
    auto address=(uint64_t)(h+16+(worker*5+stage)*8);
    AscendTimerDevice::TikPair((GM_ADDR)address);return address;
}
__aicore__ inline void PlannerEnd(uint64_t address) {
    if(address)AscendTimerDevice::TokPair((GM_ADDR)address);
}
#define PLAN_START(name,stage) const auto name=PlannerStart(header,worker,stage)
#define PLAN_END(name) PlannerEnd(name)
#else
#define PLAN_START(name,stage) do{}while(0)
#define PLAN_END(name) do{}while(0)
#endif

namespace {
constexpr int MAX_E=256, MAX_P=8, WORKERS=32;
constexpr int INPUT_TILE=1024;
__aicore__ inline int Read(__gm__ int32_t* p) {return ReadGmBypassDCache(p);}
__aicore__ inline int Min(int a,int b) {return a<b?a:b;}
__aicore__ inline int Max(int a,int b) {return a>b?a:b;}
__aicore__ inline void LoadInput(LocalTensor<int32_t> tile,GlobalTensor<int32_t> gm,int pos,int size) {
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopyExtParams p(1,size*4,0,0,0);
    DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(tile,gm[pos],p,pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}
__aicore__ inline void Publish() {
    GlobalTensor<uint8_t> gm;
    DataCacheCleanAndInvalid<uint8_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
}
__aicore__ inline void Sort(int* order,int n,int* values,bool descending) {
    for(int i=0;i<n;++i)order[i]=i;
    for(int i=1;i<n;++i) {
        int v=order[i],j=i-1;
        while(j>=0 && (descending?values[order[j]]<values[v]:values[order[j]]>values[v])) {
            order[j+1]=order[j];--j;
        }
        order[j+1]=v;
    }
}
}


// Local aggregation reduces remote histogram reads by WORKERS. Each expert
// owns a full cache line in totals/prefixes; there are no concurrent writers.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mega_moe_replica_planner_control_kernel(uint64_t ffts,GM_ADDR base,GM_ADDR ids,
    GM_ADDR copies,GM_ADDR physical,GM_ADDR logical,GM_ADDR assignment,GM_ADDR header,
    uint64_t histoff,uint64_t metaoff,uint64_t erroroff,int n,int topk,
    int home,int slots,int rank,int ranks,int beta,int penalty,int gain,int epoch
#if defined(DEEPEP_DEVICE_PLANNER)
    ,int capacity
#endif
#if defined(DEEPEP_OPTIMIZED_PLANNER)
    ,int algorithm
#endif
    ) {
    util_set_ffts_config(ffts);
    if ASCEND_IS_AIC {return;}
    const int worker=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);
    const int experts=home*ranks,es=home+slots;
    const int chunk=((n+WORKERS-1)/WORKERS+31)/32*32;
    const int begin=Min(n,worker*chunk),end=Min(n,begin+chunk);
    auto hist=(__gm__ int32_t*)(base+histoff);
    auto meta=(__gm__ int32_t*)(base+metaoff);
    auto errors=(__gm__ int32_t*)(base+erroroff);
    auto prefixes=(__gm__ int32_t*)(base+erroroff+WORKERS*128);
    auto totals=prefixes+MAX_E*WORKERS;
    auto input=(__gm__ int32_t*)ids;
    auto cpg=(__gm__ int32_t*)copies;auto qout=(__gm__ int32_t*)assignment;
    auto out=(__gm__ int32_t*)physical;auto lc=(__gm__ int32_t*)logical;
    auto hdr=(__gm__ int32_t*)header;
    TPipe pipe;TBuf<TPosition::VECCALC> inputBuffer;
    pipe.InitBuffer(inputBuffer,INPUT_TILE*4);
    auto inputTile=inputBuffer.Get<int32_t>();
    GlobalTensor<int32_t> inputGm;inputGm.SetGlobalBuffer(input);
    inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    PLAN_START(histTimer,0);
    int local[MAX_E];for(int e=0;e<experts;++e)local[e]=0;
    int bad=0;
    for(int pos=begin;pos<end;pos+=INPUT_TILE) {
        int size=Min(INPUT_TILE,end-pos);LoadInput(inputTile,inputGm,pos,size);
        for(int i=0;i<size;++i) {int e=inputTile.GetValue(i);if(e<0||e>=experts)bad=1;else ++local[e];}
    }
    for(int e=0;e<experts;++e)hist[worker*MAX_E+e]=local[e];
    errors[worker*32]=bad;
    if(worker==0) {
        meta[0]=epoch;meta[1]=n;meta[2]=topk;meta[3]=experts;
        meta[4]=slots;meta[5]=beta;meta[6]=penalty;meta[7]=gain;
#if defined(DEEPEP_DEVICE_PLANNER)
        meta[9]=capacity;
#endif
#if defined(DEEPEP_OPTIMIZED_PLANNER)
        meta[10]=algorithm;
#endif
    }
    PLAN_END(histTimer);
    Publish();SyncAll<true>();
    PLAN_START(prefixTimer,1);
    for(int e=worker;e<experts;e+=WORKERS) {
        int sum=0;
        for(int w=0;w<WORKERS;++w) {
            prefixes[e*32+w]=sum;sum+=Read(hist+w*MAX_E+e);
        }
        totals[e*32]=sum;
    }
    if(worker==0) {
        int error=0;for(int w=0;w<WORKERS;++w)error=Max(error,Read(errors+w*32));
        meta[8]=error;
    }
    PLAN_END(prefixTimer);
    Publish();aclshmemx_barrier_all_vec();
    PLAN_START(exchangeTimer,2);
    // Local counts remain unchanged until every rank has consumed them.
    for(int e=worker;e<experts;e+=WORKERS) {
        int total=0,prefix=0;
        for(int r=0;r<ranks;++r) {
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);
            auto counts=(__gm__ int32_t*)(peer+erroroff+WORKERS*128+MAX_E*WORKERS*4);
            int v=Read(counts+e*32);total+=v;if(r<rank)prefix+=v;
        }
        // Same cache-line owner updates fields 1/2, preserving local count 0.
        totals[e*32+1]=total;totals[e*32+2]=prefix;
    }
    if(worker==0) {
        int error=0;
        for(int r=0;r<ranks;++r) {
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);
            auto pm=(__gm__ int32_t*)(peer+metaoff);
            if(Read(pm+8))error=Max(error,1);
            for(int field=0;field<8;++field)if(Read(pm+field)!=Read(meta+field))error=2;
#if defined(DEEPEP_DEVICE_PLANNER)
            if(Read(pm+9)!=capacity)error=2;
#endif
#if defined(DEEPEP_OPTIMIZED_PLANNER)
            if(Read(pm+10)!=algorithm)error=2;
#endif
        }
        errors[1]=error;
    }
    PLAN_END(exchangeTimer);
    Publish();SyncAll<true>();
    if(worker==0) {
        PLAN_START(policyTimer,3);
        int counts[MAX_E],blocks[MAX_E],q[MAX_E*MAX_P],load[MAX_P],original[MAX_P],cp[MAX_E];
        int donororder[MAX_P],receiverorder[MAX_P],expertorder[MAX_E],values[MAX_E];
        for(int e=0;e<experts;++e)counts[e]=Read(totals+e*32+1);
        int error=Read(errors+1);
        for(int r=0;r<ranks;++r)load[r]=original[r]=0;
        for(int i=0;i<ranks*slots;++i)cp[i]=-1;
        int total=0;
        for(int e=0;e<experts;++e) {
#if defined(DEEPEP_OPTIMIZED_PLANNER)
            blocks[e]=(static_cast<int64_t>(counts[e])+beta-1)/beta;
#else
            blocks[e]=(counts[e]+beta-1)/beta;
#endif
            lc[e]=counts[e];total+=blocks[e];
            for(int r=0;r<ranks;++r)q[e*ranks+r]=r==e/home?blocks[e]:0;
            load[e/home]+=blocks[e];original[e/home]+=blocks[e];
        }
        const int target=(total+ranks-1)/ranks;
#if defined(DEEPEP_MOONEP_PLANNER)
        // Host entry enforces beta=1, slots=home and equal per-rank M*K.
        bool keepHome=false;
#if MEGAFLUX_MOONEP_DIRECT_PERMILLE
        int ownerMaximum=0;
        for(int r=0;r<ranks;++r)ownerMaximum=Max(ownerMaximum,original[r]);
        keepHome=MegaFluxMoonEPDirect::KeepHome(ownerMaximum,n,ranks,MEGAFLUX_MOONEP_DIRECT_PERMILLE);
#endif
        // q already contains the original owner assignment and cp is empty.
        // Keep the existing route encoding, capacity and collective checks.
        if(!error && !keepHome && !MegaFluxMoonEP::Run(counts,experts,ranks,home,n,q,cp))error=1;
#else
        bool runGreedy=!error;
#if defined(DEEPEP_OPTIMIZED_PLANNER)
        // ceil(total/P) is a lower bound for any nonnegative training proxy.
        // Keep old penalty=gain=0 semantics in bounded-greedy mode.
        int originalMax=0;for(int r=0;r<ranks;++r)originalMax=Max(originalMax,original[r]);
        if(!slots || ((algorithm>0 || penalty || gain) &&
                static_cast<int64_t>(target)+gain>=originalMax))runGreedy=false;
        if(!error && algorithm>0) {
            MegaFluxSearch::Run(blocks,experts,ranks,slots,penalty,gain,
                                algorithm==1,q,cp);
            for(int r=0;r<ranks;++r)load[r]=0;
            for(int e=0;e<experts;++e)for(int r=0;r<ranks;++r)load[r]+=q[e*ranks+r];
            runGreedy=false;
        }
#endif
        if(runGreedy)for(int round=0;round<=total;++round) {
            bool moved=false;Sort(donororder,ranks,load,true);
            for(int di=0;di<ranks;++di) {
                int d=donororder[di];if(load[d]<=target)continue;
                for(int e=0;e<experts;++e)values[e]=q[e*ranks+d];
                Sort(expertorder,experts,values,true);
                for(int ei=0;ei<experts;++ei) {
                    int e=expertorder[ei];if(!q[e*ranks+d])continue;
                    Sort(receiverorder,ranks,load,false);
                    for(int ri=0;ri<ranks;++ri) {
                        int r=receiverorder[ri];if(r==d||load[r]>=target)continue;
                        int existing=-1,free=-1;
                        for(int s=0;s<slots;++s) {
                            int v=cp[r*slots+s];
                            if(v==e && existing<0)existing=s;
                            if(v<0 && free<0)free=s;
                        }
                        if(r!=e/home && existing<0 && free<0)continue;
                        int take=Min(q[e*ranks+d],Min(target-load[r],load[d]-target));
                        if(take<=0)continue;
                        if(r!=e/home && existing<0)cp[r*slots+free]=e;
                        q[e*ranks+d]-=take;q[e*ranks+r]+=take;load[d]-=take;load[r]+=take;moved=true;
                        if(load[d]<=target)break;
                    }
                    if(load[d]<=target)break;
                }
            }
            if(!moved)break;
        }
        if((penalty||gain)
#if defined(DEEPEP_OPTIMIZED_PLANNER)
            && algorithm==0
#endif
            ) {
            for(int iteration=0;iteration<ranks*slots;++iteration) {
                int aug[MAX_P];for(int r=0;r<ranks;++r)aug[r]=load[r];
                for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                    int e=cp[r*slots+s];if(e>=0 && q[e*ranks+r]>0)aug[e/home]+=penalty;
                }
                int d=0;for(int r=1;r<ranks;++r)if(aug[r]>aug[d])d=r;
                int bestv=0x7fffffff,beste=-1,bestr=-1,bests=-1;
                for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                    int e=cp[r*slots+s];if(e<0||e/home!=d)continue;
                    int v=q[e*ranks+r];if(v<=0)continue;
                    if(v<bestv || (v==bestv && e<beste)) {bestv=v;beste=e;bestr=r;bests=s;}
                }
                if(beste<0||bestv>=penalty)break;
                q[beste*ranks+bestr]=0;q[beste*ranks+d]+=bestv;
                load[bestr]-=bestv;load[d]+=bestv;cp[bestr*slots+bests]=-1;
            }
            int aug[MAX_P];for(int r=0;r<ranks;++r)aug[r]=load[r];
            for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                int e=cp[r*slots+s];if(e>=0 && q[e*ranks+r]>0)aug[e/home]+=penalty;
            }
            int before=0,after=0;
            for(int r=0;r<ranks;++r) {before=Max(before,original[r]);after=Max(after,aug[r]);}
            if(
#if defined(DEEPEP_OPTIMIZED_PLANNER)
                static_cast<int64_t>(after)+gain>=before
#else
                after+gain>=before
#endif
                ) {
                for(int e=0;e<experts;++e)for(int r=0;r<ranks;++r)q[e*ranks+r]=r==e/home?blocks[e]:0;
                for(int i=0;i<ranks*slots;++i)cp[i]=-1;
            }
        }
#endif
#if defined(DEEPEP_MOONEP_PLANNER) && defined(DEEPEP_DEVICE_PLANNER)
        // Native provider contract: check the generated small tables in the
        // existing policy stage, before publishing one common error status.
        if(!error) {
            for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                int e=cp[r*slots+s];
                if(e < -1 || e>=experts || (e>=0 && e/home==r))error=4;
                for(int j=0;j<s;++j)if(e>=0 && cp[r*slots+j]==e)error=4;
            }
            for(int e=0;e<experts;++e) {
                int64_t sum=0;
                for(int r=0;r<ranks;++r) {
                    int quota=q[e*ranks+r];sum+=quota;
                    if(quota<0)error=4;
                    if(quota>0 && r!=e/home) {
                        bool found=false;for(int s=0;s<slots;++s)if(cp[r*slots+s]==e)found=true;
                        if(!found)error=4;
                    }
                }
                if(sum!=counts[e])error=4;
            }
            if(error)for(int i=0;i<ranks*slots;++i)cp[i]=-1;
        }
#endif
        int active=0;for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
            int e=cp[r*slots+s];
            if(e>=0 && q[e*ranks+r]==0)cp[r*slots+s]=-1;
            else if(e>=0)++active;
        }
        for(int i=0;i<ranks*slots;++i)cpg[i]=cp[i];
        int rows[MAX_P],homeRows[MAX_P];for(int r=0;r<ranks;++r)rows[r]=homeRows[r]=0;
        for(int e=0;e<experts;++e) {
            homeRows[e/home]+=counts[e];int offset=0;
            for(int r=0;r<ranks;++r) {
                qout[e*ranks+r]=q[e*ranks+r];
#if defined(DEEPEP_OPTIMIZED_PLANNER)
                // Saturate the token prefix at the actual count. A final
                // partial block can extend beyond INT32 even for valid input.
                int64_t next=static_cast<int64_t>(offset)+static_cast<int64_t>(q[e*ranks+r])*beta;
                int clipped=next>counts[e]?counts[e]:static_cast<int>(next);
                rows[r]+=clipped-offset;offset=clipped;
#else
                int next=offset+q[e*ranks+r]*beta;
                rows[r]+=Min(counts[e],next)-Min(counts[e],offset);offset=next;
#endif
            }
        }
        int maxrecv=1,before=0;
        for(int r=0;r<ranks;++r){maxrecv=Max(maxrecv,Max(rows[r],homeRows[r]));before=Max(before,original[r]);}
#if defined(DEEPEP_DEVICE_PLANNER)
        // Only execution rows matter for this fixed physical-layout executor.
        maxrecv=1;for(int r=0;r<ranks;++r)maxrecv=Max(maxrecv,rows[r]);
        if(!error && (maxrecv+127)/128*128>capacity)error=3;
#endif
        hdr[0]=active;hdr[1]=(maxrecv+127)/128*128;hdr[2]=epoch;hdr[3]=error;
        hdr[4]=before;hdr[5]=active>0;hdr[6]=active;hdr[7]=0;
#if defined(DEEPEP_DEVICE_PLANNER)
        hdr[7]=capacity;
#endif
        PLAN_END(policyTimer);
        Publish();
    }

    // Both publishes and cross-rank histogram consumption finish here. After
    // this barrier the active mapping uses only local prefixes and outputs.
    aclshmemx_barrier_all_vec();
    PLAN_START(routesTimer,4);
    if(!Read(hdr+3)
#if !defined(DEEPEP_MOONEP_PLANNER)
        && Read(hdr)>0
#endif
        ) {
        int cachedQ[MAX_E*MAX_P],cachedCopies[MAX_E];
        for(int i=0;i<experts*ranks;++i)cachedQ[i]=Read(qout+i);
        for(int i=0;i<ranks*slots;++i)cachedCopies[i]=Read(cpg+i);
        for(int e=0;e<experts;++e)local[e]=Read(prefixes+e*32+worker)+Read(totals+e*32+2);
        for(int pos=begin;pos<end;pos+=INPUT_TILE) {
            int size=Min(INPUT_TILE,end-pos);LoadInput(inputTile,inputGm,pos,size);
            for(int i=0;i<size;++i) {
                int e=inputTile.GetValue(i),block=local[e]++/beta,r=0;
                for(;r<ranks-1;++r) {int v=cachedQ[e*ranks+r];if(block<v)break;block-=v;}
                int slot=e%home;
                if(r!=e/home)for(int s=0;s<slots;++s)if(cachedCopies[r*slots+s]==e){slot=home+s;break;}
                out[pos+i]=r*es+slot;
            }
        }
        Publish();
    }
    // Zero replicas deliberately leave physical scratch untouched. All calls
    // execute in one stream; the next invocation cannot overwrite this rank's
    // local prefixes while its current mapping is still running.
    PLAN_END(routesTimer);
    pipe.Destroy();
}
extern "C" __attribute__((visibility("default"))) void mega_moe_replica_planner_control_do(
    void* stream,uint64_t ffts,uint8_t* base,uint8_t* ids,uint8_t* copies,uint8_t* physical,
    uint8_t* logical,uint8_t* assignment,uint8_t* header,uint64_t histogram,uint64_t meta,uint64_t errors,
    int n,int topk,int home,int slots,int rank,int ranks,int beta,int penalty,int gain,int epoch
#if defined(DEEPEP_DEVICE_PLANNER)
    ,int capacity
#endif
#if defined(DEEPEP_OPTIMIZED_PLANNER)
    ,int algorithm
#endif
    ) {
    mega_moe_replica_planner_control_kernel<<<WORKERS,nullptr,stream>>>(ffts,base,ids,copies,physical,logical,assignment,header,
        histogram,meta,errors,n,topk,home,slots,rank,ranks,beta,penalty,gain,epoch
#if defined(DEEPEP_DEVICE_PLANNER)
        ,capacity
#endif
#if defined(DEEPEP_OPTIMIZED_PLANNER)
        ,algorithm
#endif
        );
}

extern "C" __attribute__((visibility("default"))) int megaflux_max_local_experts(){return MEGAMOE_MAX_LOCAL_EXPERTS;}
