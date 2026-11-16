// Histogram/exchange adapted from frozen 48ef24a6 + accepted EP8 patches.
// Policies only return copies and quotas; common compilation owns all routing.
#include "mega_moe_capacity.h"
#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
#define UP_INLINE __aicore__ inline
#include "mega_moe_unified_policy.h"
#ifndef UP_POLICY_RUN
#define UP_POLICY_RUN(state,policy,target,costs,base,hist,meta,errors,capacity) UnifiedPlanner::run(state,policy,target)
#endif
#define DEEPEP_DEVICE_PLANNER 1
#define DEEPEP_OPTIMIZED_PLANNER 1
#define PLAN_START(name,stage) do{}while(0)
#define PLAN_END(name) do{}while(0)
using namespace AscendC;
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


extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mega_moe_unified_planner_kernel(uint64_t ffts,GM_ADDR base,GM_ADDR ids,
    GM_ADDR copies,GM_ADDR physical,GM_ADDR logical,GM_ADDR assignment,GM_ADDR header,
    GM_ADDR sourcequota,GM_ADDR costs,uint64_t histoff,uint64_t metaoff,uint64_t erroroff,
    int n,int topk,int home,int slots,int rank,int ranks,int epoch,int capacity,int algorithm) {
    const int beta=1,penalty=0,gain=0;
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

    // Policy stage: identical logical/source statistics on all EP ranks.
    if(worker==0) {
        UnifiedPlanner::State a;a.p=ranks;a.h=home;a.s=slots;a.e=experts;a.source_policy=algorithm;
        a.evaluations=a.rounds=a.adopted=a.model_flags=0;a.predicted_before=a.predicted_after=0.0f;
        for(int e=0;e<experts;++e)a.c[e]=Read(totals+e*32+1);
        a.home();int error=Read(errors+1);
        if(algorithm!=6 && !error&&!UP_POLICY_RUN(a,algorithm,n,costs,base,histoff,metaoff,erroroff,capacity))error=4;
        meta[13]=a.evaluations;meta[14]=a.rounds;meta[15]=a.adopted;meta[16]=a.source_policy;
        union {float f;int32_t i;} before,after;before.f=a.predicted_before;after.f=a.predicted_after;
        meta[17]=before.i;meta[18]=after.i;meta[19]=a.model_flags;meta[20]=capacity;
        int active=0,rows[MAX_P];for(int d=0;d<ranks;++d)rows[d]=0;
        if(algorithm!=6)for(int d=0;d<ranks;++d)for(int s=0;s<slots;++s) {
            int e=a.cp[d*slots+s];
            if(e < -1 || e>=experts || (e>=0 && (e/home==d||a.q[e*ranks+d]<=0)))error=4;
            for(int j=0;j<s;++j)if(e>=0&&a.cp[d*slots+j]==e)error=4;
            if(e>=0)++active;cpg[d*slots+s]=e;
        }
        for(int e=0;e<experts;++e){int sum=0;lc[e]=a.c[e];
            if(algorithm==6)continue;
            for(int d=0;d<ranks;++d){int v=a.q[e*ranks+d];if(v<0)error=4;
                if(v>0&&d!=e/home){bool found=false;for(int s=0;s<slots;++s)if(a.cp[d*slots+s]==e)found=true;if(!found)error=4;}
                qout[e*ranks+d]=a.source_policy==4?a.instances[e*ranks+d]:v;sum+=v;rows[d]+=v;}
            if(sum!=a.c[e])error=4;
        }
        int maxrecv=1;for(int d=0;d<ranks;++d)maxrecv=Max(maxrecv,rows[d]);
        int required=(maxrecv+127)/128*128;
        hdr[0]=active;hdr[1]=required;hdr[2]=epoch;hdr[3]=error;
        hdr[4]=a.source_policy;hdr[5]=active>0;hdr[6]=active;hdr[7]=capacity;
        Publish();
    }
    aclshmemx_barrier_all_vec();
    // Final policy stage: construct the source-aware quota proposal. Internal
    // destination tables and mode decisions do not reach the common compiler.
    // Code6 accepts an external provider's copies/quota without rewriting it.
    auto qsg=(__gm__ int32_t*)sourcequota;
    int perline=32/ranks;
    int echunk=((experts+WORKERS-1)/WORKERS+perline-1)/perline*perline;
    int eb=Min(experts,worker*echunk),ee=Min(experts,eb+echunk),qerror=0;
    if(algorithm!=6 && !Read(hdr+3))for(int e=eb;e<ee;++e) {
        int source[MAX_P],dest[MAX_P],matrix[MAX_P*MAX_P];
        for(int s=0;s<ranks;++s){
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,s);
            auto counts=(__gm__ int32_t*)(peer+erroroff+WORKERS*128+MAX_E*WORKERS*4);
            source[s]=Read(counts+e*32);dest[s]=Read(qout+e*ranks+s);
        }
        int sourcePolicy=Read(hdr+4);
        int mode=sourcePolicy==2?1:sourcePolicy==3?2:sourcePolicy==4?3:0;
        if(!UnifiedPlanner::source_quota(source,dest,ranks,e/home,mode,matrix))qerror=4;
        for(int s=0;s<ranks;++s)for(int d=0;d<ranks;++d)qsg[(s*experts+e)*ranks+d]=matrix[s*ranks+d];
    }
    errors[worker*32]=qerror;Publish();SyncAll<true>();
    // Public compiler/validator: only source counts, copies and source quota.
    // No policy name/code, global destination heuristic or gradient policy.
    uint32_t hash1=0,hash2=0;
    if(!Read(hdr+3))for(int e=eb;e<ee;++e){
        int destination[MAX_P];for(int d=0;d<ranks;++d)destination[d]=0;
        for(int s=0;s<ranks;++s){int sum=0;
            for(int d=0;d<ranks;++d){int index=(s*experts+e)*ranks+d,v=Read(qsg+index);if(v<0||v>n)qerror=4;sum+=v;destination[d]+=v;
                uint32_t key=(uint32_t)(index+1)*2654435761u;hash1+=(uint32_t)v*key;
                uint32_t mixed=(uint32_t)v^key;mixed^=mixed>>16;mixed*=2246822519u;mixed^=mixed>>13;hash2+=mixed;}
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,s);
            auto source=(__gm__ int32_t*)(peer+erroroff+WORKERS*128+MAX_E*WORKERS*4);
            if(sum!=Read(source+e*32))qerror=4;
        }
        for(int d=0;d<ranks;++d){
            int matches=0;for(int s=0;s<slots;++s)if(Read(cpg+d*slots+s)==e)++matches;
            if(d==e/home ? matches!=0 : (destination[d]>0 ? matches!=1 : matches!=0))qerror=4;
            totals[e*32+3+d]=destination[d];
        }
    }
    errors[worker*32]=qerror;errors[worker*32+1]=(int32_t)hash1;errors[worker*32+2]=(int32_t)hash2;Publish();SyncAll<true>();
    if(worker==0){int error=Read(hdr+3),active=0,rows[MAX_P];uint32_t h1=0,h2=0;for(int d=0;d<ranks;++d)rows[d]=0;
        for(int w=0;w<WORKERS;++w){h1+=(uint32_t)Read(errors+w*32+1);h2+=(uint32_t)Read(errors+w*32+2);}meta[11]=(int32_t)h1;meta[12]=(int32_t)h2;
        for(int w=0;w<WORKERS;++w)error=Max(error,Read(errors+w*32));
        for(int d=0;d<ranks;++d)for(int s=0;s<slots;++s){int e=Read(cpg+d*slots+s);if(e>=0)++active;
            if(e < -1 || e>=experts || (e>=0&&e/home==d))error=4;
            for(int j=0;j<s;++j)if(e>=0&&Read(cpg+d*slots+j)==e)error=4;}
        if(!error)for(int e=0;e<experts;++e)for(int d=0;d<ranks;++d)rows[d]+=Read(totals+e*32+3+d);
        int required=1;for(int d=0;d<ranks;++d)required=Max(required,rows[d]);required=(required+127)/128*128;
        if(!error&&required>capacity)error=3;
        hdr[0]=active;hdr[1]=required;hdr[3]=error;hdr[5]=active>0;hdr[6]=active;Publish();}
    aclshmemx_barrier_all_vec();
    if(worker==0){for(int r=0;r<ranks;++r){auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);auto pm=(__gm__ int32_t*)(peer+metaoff);
        if(Read(pm+11)!=Read(meta+11)||Read(pm+12)!=Read(meta+12))hdr[3]=2;}Publish();}
    SyncAll<true>();
    // One deterministic physical assignment for every original assignment,
    // including empty-replica/home-only plans and repeated TopK expert IDs.
    if(!Read(hdr+3)) {
        int cachedQ[MAX_E*MAX_P],cachedCopies[UnifiedPlanner::CMAX];
        for(int e=0;e<experts;++e)for(int d=0;d<ranks;++d)cachedQ[e*ranks+d]=Read(qsg+(rank*experts+e)*ranks+d);
        for(int i=0;i<ranks*slots;++i)cachedCopies[i]=Read(cpg+i);
        for(int e=0;e<experts;++e)local[e]=Read(prefixes+e*32+worker);
        for(int pos=begin;pos<end;pos+=INPUT_TILE){
            int size=Min(INPUT_TILE,end-pos);LoadInput(inputTile,inputGm,pos,size);
            for(int i=0;i<size;++i){int e=inputTile.GetValue(i),ordinal=local[e]++,d=0;
                for(;d<ranks-1;++d){int v=cachedQ[e*ranks+d];if(ordinal<v)break;ordinal-=v;}
                int slot=e%home;if(d!=e/home)for(int s=0;s<slots;++s)if(cachedCopies[d*slots+s]==e){slot=home+s;break;}
                out[pos+i]=d*es+slot;
            }
        }
    }else for(int pos=begin;pos<end;++pos)out[pos]=-1;
    if(worker==0 && experts*ranks>=8){for(int i=0;i<8;++i)qout[i]=Read(meta+13+i);}
    Publish();pipe.Destroy();
}
#if !defined(UP_PLUGIN_BUILD)
using PolicyExtension=void(*)(void*,uint64_t,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int,int,int);
static PolicyExtension extensionPolicy=nullptr;
extern "C" __attribute__((visibility("default"))) int megaflux_unified_budget_abi(){return 1;}
extern "C" __attribute__((visibility("default"))) void megaflux_install_policy_extension(void* entry){extensionPolicy=reinterpret_cast<PolicyExtension>(entry);}
#endif
extern "C" __attribute__((visibility("default"))) void mega_moe_unified_planner_do(
    void* stream,uint64_t ffts,uint8_t* base,uint8_t* ids,uint8_t* copies,uint8_t* physical,
    uint8_t* logical,uint8_t* assignment,uint8_t* header,uint8_t* sourcequota,uint8_t* costs,
    uint64_t histogram,uint64_t meta,uint64_t errors,
    int n,int topk,int home,int slots,int rank,int ranks,int epoch,int capacity,int policy) {
#if !defined(UP_PLUGIN_BUILD)
    if(policy==5&&extensionPolicy){extensionPolicy(stream,ffts,base,ids,copies,physical,logical,assignment,header,sourcequota,costs,histogram,meta,errors,n,topk,home,slots,rank,ranks,epoch,capacity,policy);return;}
#endif
    mega_moe_unified_planner_kernel<<<WORKERS,nullptr,stream>>>(ffts,base,ids,copies,physical,logical,
        assignment,header,sourcequota,costs,histogram,meta,errors,n,topk,home,slots,rank,ranks,epoch,capacity,policy);
}
extern "C" __attribute__((visibility("default"))) int megaflux_max_local_experts(){return MEGAMOE_MAX_LOCAL_EXPERTS;}
extern "C" __attribute__((visibility("default"))) int megaflux_unified_planner_abi(){return 1;}
