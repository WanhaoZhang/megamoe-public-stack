#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
using namespace AscendC;
#define MEGAFLUX_VALIDATOR_INLINE __aicore__ inline
#include "include/mega_moe_plan_validator_logic.h"
namespace {
constexpr int Workers=32, MaxValues=2048, Tile=1024;
__aicore__ inline int Read(__gm__ int32_t* p) {return ReadGmBypassDCache(p);}
struct Reader {
    __gm__ int32_t* p;
    __aicore__ inline int operator()(int index) const {return Read(p+index);}
};
__aicore__ inline void Publish() {
    GlobalTensor<uint8_t> gm;
    DataCacheCleanAndInvalid<uint8_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
}
}

// All scratch is ordinary per-executor GM; the only cross-rank operation is
// the following framework all-gather. No replica/weight/runtime ABI is changed.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void megaflux_plan_validate_local_kernel(uint64_t ffts,GM_ADDR ids,GM_ADDR physical,
        GM_ADDR copies,GM_ADDR scratch,GM_ADDR packed,int n,int ep,int home,int slots,
        int epoch,int capacity) {
    util_set_ffts_config(ffts);if ASCEND_IS_AIC {return;}
    const int w=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);
    const int experts=ep*home,width=home+slots,physicalCount=ep*width;
    const int stride=(experts+physicalCount+1+31)/32*32;
    const int chunk=(n+Workers-1)/Workers,begin=w*chunk,end=(begin+chunk<n?begin+chunk:n);
    auto* cp=(__gm__ int32_t*)copies;auto* tmp=(__gm__ int32_t*)scratch;
    auto* output=(__gm__ int32_t*)packed;
    int logicalCount[MaxValues],executionCount[MaxValues];
    for(int e=0;e<experts;++e)logicalCount[e]=0;
    for(int e=0;e<physicalCount;++e)executionCount[e]=0;
    TPipe pipe;TBuf<TPosition::VECCALC> il,ip;pipe.InitBuffer(il,Tile*4);pipe.InitBuffer(ip,Tile*4);
    auto l=il.Get<int32_t>(),p=ip.Get<int32_t>();
    GlobalTensor<int32_t> li,pi;li.SetGlobalBuffer((__gm__ int32_t*)ids);pi.SetGlobalBuffer((__gm__ int32_t*)physical);
    li.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);pi.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    int bad=0;
    for(int pos=begin;pos<end;pos+=Tile) {
        const int size=(end-pos<Tile?end-pos:Tile);
        DataCopyExtParams dp{1,(uint32_t)(size*4),0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(l,li[pos],dp,pad);DataCopyPad(p,pi[pos],dp,pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for(int i=0;i<size;++i) {
            const int lid=l.GetValue(i),pid=p.GetValue(i);
            // Match generic clamp-before-gather/count semantics even on bad values.
            const int le=lid<0?0:(lid>=experts?experts-1:lid);
            const int pe=pid<0?0:(pid>=physicalCount?physicalCount-1:pid);
            ++logicalCount[le];++executionCount[pe];
            const int owner=pe/width,slot=pe%width;
            const int mapped=slot<home?owner*home+slot:Read(cp+owner*slots+slot-home);
            if(lid<0||lid>=experts||pid<0||pid>=physicalCount||mapped!=lid)bad=1;
        }
    }
    if(w==0)for(int r=0;r<ep;++r)for(int s=0;s<slots;++s) {
        const int e=Read(cp+r*slots+s);
        if(e < -1 || e>=experts || (e>=0 && e/home==r))bad=1;
        if(e>=0)for(int t=s+1;t<slots;++t)if(e==Read(cp+r*slots+t))bad=1;
    }
    for(int e=0;e<experts;++e)tmp[w*stride+e]=logicalCount[e];
    for(int e=0;e<physicalCount;++e)tmp[w*stride+experts+e]=executionCount[e];
    tmp[w*stride+experts+physicalCount]=bad;
    Publish();SyncAll<true>();
    // Packed scalars share cache lines: a single writer avoids concurrent
    // dirty-cache writeback losing other AIVs' neighboring int32 stores.
    // Per-worker histogram scratch remains line-aligned and parallel above.
    if(w==0) {
        for(int e=0;e<experts+physicalCount;++e) {
            int sum=0;for(int r=0;r<Workers;++r)sum+=Read(tmp+r*stride+e);output[e]=sum;
        }
        for(int i=0;i<ep*slots;++i)output[experts+physicalCount+i]=Read(cp+i);
        int error=0;for(int r=0;r<Workers;++r)error|=Read(tmp+r*stride+experts+physicalCount);
        const int meta=experts+physicalCount+ep*slots;
        output[meta]=error;output[meta+1]=epoch;output[meta+2]=capacity;
    }
    Publish();pipe.Destroy();
}

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void megaflux_plan_validate_global_kernel(uint64_t ffts,GM_ADDR gathered,GM_ADDR nativeHeader,
        GM_ADDR logicalCounts,GM_ADDR header,int ep,int home,int slots,int epoch,int capacity) {
    util_set_ffts_config(ffts);if ASCEND_IS_AIC {return;}
    const int w=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);
    const int experts=ep*home,width=home+slots,pc=ep*width;
    const int placement=experts+pc,meta=placement+ep*slots,stride=meta+3;
    auto* in=(__gm__ int32_t*)gathered;auto* reduced=(__gm__ int32_t*)logicalCounts;
    const int ls=(experts+31)/32*32,ps=(pc+31)/32*32;
    // Each worker exclusively writes complete 128B lines. Interleaved scalar
    // stores from different AIV caches could otherwise lose adjacent updates.
    for(int block=w*32;block<experts;block+=Workers*32)
        for(int e=block;e<block+32 && e<experts;++e) {
            int sum=0;for(int r=0;r<ep;++r)sum+=Read(in+r*stride+e);
            reduced[e]=sum;
        }
    for(int block=w*32;block<pc;block+=Workers*32)
        for(int e=block;e<block+32 && e<pc;++e) {
            int sum=0;for(int r=0;r<ep;++r)sum+=Read(in+r*stride+experts+e);
            reduced[ls+e]=sum;
        }
    Publish();SyncAll<true>();
    MegaFluxPlanValidation::Summary summary;
    for(int r=w;r<ep;r+=Workers) {
        MegaFluxPlanValidation::CheckSource(Reader{in},r,ep,home,slots,epoch,capacity,summary);
        MegaFluxPlanValidation::CheckDestination(Reader{in},Reader{reduced+ls},r,ep,home,slots,summary);
    }
    auto* stats=reduced+ls+ps+w*32;
    stats[0]=summary.error;stats[1]=summary.mismatch;stats[2]=summary.active;stats[3]=summary.required;
    Publish();SyncAll<true>();
    if(w==0) {
        MegaFluxPlanValidation::Summary total;
        for(int i=0;i<Workers;++i) {
            auto* row=reduced+ls+ps+i*32;
            total.error|=Read(row);total.mismatch|=Read(row+1);total.active+=Read(row+2);
            const int required=Read(row+3);if(required>total.required)total.required=required;
        }
        const auto result=MegaFluxPlanValidation::Finish(total,Read((__gm__ int32_t*)nativeHeader+3),capacity);
        auto* out=(__gm__ int32_t*)header;
        out[0]=result.active;out[1]=result.required;out[2]=epoch;out[3]=result.status;
        out[4]=0;out[5]=result.active>0;out[6]=result.active;out[7]=capacity;
    }
    Publish();
}

extern "C" __attribute__((visibility("default"))) int megaflux_plan_validator_version(){return 2;}
extern "C" __attribute__((visibility("default"))) void megaflux_plan_validate_local_do(void* stream,
        uint64_t ffts,uint8_t* ids,uint8_t* physical,uint8_t* copies,uint8_t* scratch,uint8_t* packed,
        int n,int ep,int home,int slots,int epoch,int capacity) {
    megaflux_plan_validate_local_kernel<<<Workers,nullptr,stream>>>(ffts,ids,physical,copies,scratch,packed,n,ep,home,slots,epoch,capacity);
}
extern "C" __attribute__((visibility("default"))) void megaflux_plan_validate_global_do(void* stream,uint64_t ffts,
        uint8_t* gathered,uint8_t* nativeHeader,uint8_t* logicalCounts,uint8_t* header,
        int ep,int home,int slots,int epoch,int capacity) {
    megaflux_plan_validate_global_kernel<<<Workers,nullptr,stream>>>(ffts,gathered,nativeHeader,logicalCounts,header,ep,home,slots,epoch,capacity);
}
