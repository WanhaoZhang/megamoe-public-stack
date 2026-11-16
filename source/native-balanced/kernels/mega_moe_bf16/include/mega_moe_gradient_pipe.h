#pragma once
#include "kernel_operator.h"
#include "mega_moe_replica_mixed.h"
#include "mega_moe_detail_trace.h"
#include "mega_moe_stage_timer.h"
#include "mega_moe_owner_schedule.h"
#define MEGAFLUX_PRIORITY_INLINE __aicore__ inline
#include "mega_moe_gradient_priority.h"
#ifndef DEEPEP_BF16_OWNER_RATIO
#define DEEPEP_BF16_OWNER_RATIO 1
#endif
#ifndef DEEPEP_BF16_OWNER_SCAN
#define DEEPEP_BF16_OWNER_SCAN 0
#endif
#ifndef MEGAFLUX_OWNER_ROWS
#define MEGAFLUX_OWNER_ROWS 16
#endif
#ifndef MEGAFLUX_OWNER_COMPACT
#define MEGAFLUX_OWNER_COMPACT 0
#endif
#if defined(DEEPEP_BF16_OWNER_DB)
#include "mega_moe_owner_reduce_db.h"
#endif

// AIC writes replica partials directly to their symmetric spans. Each unique
// 128x256 GEMM tile releases one 64B epoch line after FIX has drained. AIVs
// consume independent output tiles while AICs produce other tiles.
namespace MegaFluxGradient {
using namespace AscendC;
constexpr int TM=128,TN=256,LINE=16,ROWS=16;
// Only the standalone S9 consumer is widened. CooperativeOwner may lend its
// low UB between producers and keeps the original 16-row resource contract.
constexpr int REDUCE_ROWS=MEGAFLUX_OWNER_ROWS;
static_assert(REDUCE_ROWS==16 || REDUCE_ROWS==32,"qualified owner row counts are 16 and 32");
struct Context {
    MegaMoeReplicaDescriptor replica{};
    uint64_t ready=0,trace=0;
    int epoch=0;
};
__aicore__ inline int Columns(const Context& c,int p) {
    return p==0?c.replica.hidden:c.replica.ffn;
}
__aicore__ inline int Rows(const Context& c,int p) {
    return p==0?2*c.replica.ffn:c.replica.hidden;
}
__aicore__ inline int Tiles(const Context& c,int p) {
    return ((Rows(c,p)+TM-1)/TM)*((Columns(c,p)+TN-1)/TN);
}
__aicore__ inline int64_t Index(const Context& c,int e,int p,int row,int col) {
    return (int64_t)e*(Tiles(c,0)+Tiles(c,1))+(p?Tiles(c,0):0)
        +(row/TM)*((Columns(c,p)+TN-1)/TN)+col/TN;
}
__aicore__ inline void Publish(const Context& c,int e,int p,int row,int col) {
    if(!c.ready)return;
    PipeBarrier<PIPE_ALL>(); // MMAD's FIX -> GM must precede the epoch release
    const auto drained=MegaFluxDetail::Now(c.trace,true);
    MegaFluxStageTimer::Mark(c.trace,true,MegaFluxStageTimer::PARTIAL_DRAIN,e,p,row,col);
    MEGAFLUX_STAGE_BEGIN(stagePublish,c.trace,true,MegaFluxStageTimer::PARTIAL_PUBLISH,e,p,row,col);
    auto* ptr=(__gm__ int32_t*)c.ready+Index(c,e,p,row,col)*LINE;
    *ptr=c.epoch;
    GlobalTensor<int32_t> gm;gm.SetGlobalBuffer(ptr);
    DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
    MEGAFLUX_STAGE_END(stagePublish);
    const auto released=MegaFluxDetail::Now(c.trace,true);
    MegaFluxStageTimer::Mark(c.trace,true,MegaFluxStageTimer::PARTIAL_RELEASE,e,p,row,col);
    MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::PARTIAL_DRAIN,drained,drained,e,p,row,col);
    MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::PARTIAL_RELEASE,released,released,e,p,row,col);
}
__aicore__ inline void Wait(const Context& c,uint64_t ready,int e,int p,int row,int col) {
    auto* ptr=(__gm__ int32_t*)ready+Index(c,e,p,row,col)*LINE;
    while(ReadGmBypassDCache(ptr)!=c.epoch){}
    DataSyncBarrier<MemDsbT::ALL>();
}
__aicore__ inline void Stamp(const Context& c,int worker,int field) {
    if(!c.trace)return;
    auto* ptr=(__gm__ int64_t*)c.trace+(worker*8+field)*8;
    *ptr=GetSystemCycle();GlobalTensor<int64_t> gm;gm.SetGlobalBuffer(ptr);
    DataCacheCleanAndInvalid<int64_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(gm);
}
// Readiness probe has no ownership side effects. The chosen tile still executes
// the original acquire waits/fence before reading any partial payload.
__aicore__ inline bool TaskReady(const Context& c,const int* copies,
        const int32_t* localRows,int p,int64_t task,int tiles,int nc,const int* expertOrder) {
    const auto& d=c.replica;
    const int e=expertOrder[task/tiles],tile=task%tiles,logical=d.rank*d.home+e;
    const int row=tile/nc*TM,col=tile%nc*TN;
    bool remote=false;
    for(int i=0;i<d.ranks*d.slots;++i)if(copies[i]==logical)remote=true;
    if(!remote)return true;
    if(localRows[e]>0 && ReadGmBypassDCache((__gm__ int32_t*)c.ready+Index(c,e,p,row,col)*LINE)!=c.epoch)return false;
    for(int r=0;r<d.ranks;++r)for(int s=0;s<d.slots;++s) {
        if(copies[r*d.slots+s]!=logical)continue;
        const auto flags=(__gm__ int32_t*)aclshmem_ptr((GM_ADDR)c.ready,r);
        if(ReadGmBypassDCache(flags+Index(c,d.home+s,p,row,col)*LINE)!=c.epoch)return false;
    }
    return true;
}
__aicore__ inline void LoadRows(LocalTensor<bfloat16_t> b,uint64_t addr,
        int row,int col,int rows,int cols,int stride) {
    GlobalTensor<bfloat16_t> gm;gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams cp{(uint16_t)rows,(uint32_t)(cols*2),(uint32_t)((stride-cols)*2),0,0};
    DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopyPad(b,gm[(int64_t)row*stride+col],cp,pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void StoreRows(LocalTensor<bfloat16_t> b,uint64_t addr,
        int row,int col,int rows,int cols,int stride) {
    GlobalTensor<bfloat16_t> gm;gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams cp{(uint16_t)rows,(uint32_t)(cols*2),0,(uint32_t)((stride-cols)*2),0};
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(gm[(int64_t)row*stride+col],b,cp);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
// The base pipeline enters after empty-owner clears and before the P5
// cross-rank barrier. The early GM pipeline partitions consumers across P1
// (odd workers) and P5 (even workers); shared empty owners are initialized
// here, not by the later plane clear. No AIC depends on this consumer phase,
// so waits cannot hold a producer resource. Partial buffers and epochs live
// until final completion.
template<bool TakePairedTail = false>
__aicore__ inline void Reduce(const Context& c,GM_ADDR dw1,GM_ADDR dw2,const int32_t* localRows) {
    if constexpr(g_coreType==AIV) {
        if(!c.ready)return;
        const auto& d=c.replica;
        const int worker=GetBlockIdx(),workers=GetBlockNum()*2;
        TPipe pipe;
#if defined(DEEPEP_BF16_OWNER_DB)
        static_assert(REDUCE_ROWS==MegaFluxOwnerDb::ROWS && TN==MegaFluxOwnerDb::COLS,"owner tile mismatch");
        MegaFluxOwnerDb::Buffers db;db.Init(pipe);
#else
        TBuf<TPosition::VECCALC> bb,tb,ab;
        // 32 rows use 16KiB BF16 + 32KiB temp + 32KiB FP32 accumulator.
        // Odd S9 workers enter only after their last P1 SwiGLU/FIX handshake;
        // later GM dX uses AIV0. Even workers enter after production at P5.
        pipe.InitBuffer(bb,REDUCE_ROWS*TN*2);pipe.InitBuffer(tb,REDUCE_ROWS*TN*4);pipe.InitBuffer(ab,REDUCE_ROWS*TN*4);
        auto b=bb.Get<bfloat16_t>();auto tmp=tb.Get<float>();auto acc=ab.Get<float>();
#endif
        int copies[MegaMoeLargeEP::MaxCopies];for(int i=0;i<d.ranks*d.slots;++i)copies[i]=MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+i);
        int expertOrder[32];for(int e=0;e<d.home;++e)expertOrder[e]=e;
#if MEGAFLUX_GRADIENT_PRIORITY
        int priority[32];
        for(int e=0;e<d.home;++e) {
            int count=0;for(int i=0;i<d.ranks*d.slots;++i)count+=copies[i]==d.rank*d.home+e;
            priority[e]=MegaFluxPriority::Key(d.rank*d.home+e,count,d.ranks);
        }
        for(int i=1;i<d.home;++i) {
            int value=expertOrder[i],j=i-1;
            while(j>=0 && priority[expertOrder[j]]>priority[value]) {
                expertOrder[j+1]=expertOrder[j];--j;
            }
            expertOrder[j+1]=value;
        }
#endif
#if MEGAFLUX_OWNER_COMPACT
        // Scalar-only, per-call cache: retain ascending rank/slot order.
        // No UB allocation, global task counter or numeric atomic is added.
        int offsets[33],contributors[256],contributorCount=0;
        for(int e=0;e<d.home;++e) {
            offsets[e]=contributorCount;
            for(int i=0;i<d.ranks*d.slots;++i)
                if(copies[i]==d.rank*d.home+e)contributors[contributorCount++]=i;
        }
        offsets[d.home]=contributorCount;
#endif
        Stamp(c,worker,0);bool first=true,firstDone=true;
        MEGAFLUX_STAGE_BEGIN(stageWorker,c.trace,true,MegaFluxStageTimer::REDUCE_WORKER);
        const auto reduceStart=MegaFluxDetail::Now(c.trace,true);
        for(int phase=0;phase<2;++phase) {
            const int p=1-phase;
#if defined(DEEPEP_BF16_STAGE_TIMER)
            const auto stagePhase=TakePairedTail?0:MegaFluxStageTimer::Begin(
                c.trace,true,MegaFluxStageTimer::REDUCE_PHASE,-1,p);
            uint64_t expertPhase=0;
            int timedExpert=-1;
#endif
            const auto phaseStart=MegaFluxDetail::Now(c.trace,true);
            const int rows=Rows(c,p),cols=Columns(c,p),nc=(cols+TN-1)/TN,tiles=Tiles(c,p);
            const uint64_t result=(uint64_t)(p==0?dw1:dw2);
            const uint64_t partial=p==0?d.partial1:d.partial2;
            const int64_t plane=(int64_t)rows*cols;
            static_assert(DEEPEP_BF16_OWNER_RATIO>=1 && DEEPEP_BF16_OWNER_RATIO<=3,"invalid owner share");
            static_assert(DEEPEP_BF16_OWNER_SCAN==0 || DEEPEP_BF16_OWNER_SCAN==8,"invalid lookahead");
            uint32_t completed=0;
            // AIV1 consumes the union of both static shares in ascending tile
            // order. Finish the entire projection before advancing to dW1.
            for(int64_t head=0;MegaFluxOwnerSchedule::Task<TakePairedTail>(head,worker,workers,DEEPEP_BF16_OWNER_RATIO)<d.home*tiles;) {
                int chosen=0;
#if DEEPEP_BF16_OWNER_SCAN
                const auto oldest=MegaFluxOwnerSchedule::Task<TakePairedTail>(head,worker,workers,DEEPEP_BF16_OWNER_RATIO);
                if(!TaskReady(c,copies,localRows,p,oldest,tiles,nc,expertOrder)) {
                    for(int ahead=1;ahead<=DEEPEP_BF16_OWNER_SCAN;++ahead) {
                        const auto future=MegaFluxOwnerSchedule::Task<TakePairedTail>(head+ahead,worker,workers,DEEPEP_BF16_OWNER_RATIO);
                        if(future>=d.home*tiles || (TakePairedTail && future/tiles!=oldest/tiles))break;
                        if(!(completed&(1u<<ahead)) && TaskReady(c,copies,localRows,p,future,tiles,nc,expertOrder)) {chosen=ahead;break;}
                    }
                }
#endif
                const auto task=MegaFluxOwnerSchedule::Task<TakePairedTail>(head+chosen,worker,workers,DEEPEP_BF16_OWNER_RATIO);
                // No other worker can take this task. Mark before the existing
                // no-contributor continue; head advances only over executed work.
                completed|=1u<<chosen;
                while(completed&1u){++head;completed>>=1;}
                const int expert=expertOrder[task/tiles],tile=task%tiles,logical=d.rank*d.home+expert;
                int contributorBegin=0,contributorEnd=d.ranks*d.slots;
#if MEGAFLUX_OWNER_COMPACT
                contributorBegin=offsets[expert];contributorEnd=offsets[expert+1];
                const bool remote=contributorBegin!=contributorEnd;
#else
                bool remote=false;for(int i=0;i<d.ranks*d.slots;++i)if(copies[i]==logical)remote=true;
#endif
#if defined(DEEPEP_BF16_STAGE_TIMER)
                if constexpr(TakePairedTail) {
                    if(expert!=timedExpert) {
                        MegaFluxStageTimer::End(expertPhase);
                        expertPhase=remote?MegaFluxStageTimer::Begin(
                            c.trace,true,MegaFluxStageTimer::OWNER_EXPERT_PHASE,expert,p):0;
                        timedExpert=expert;
                    }
                }
#endif
                if(!remote)continue;
                const int row=tile/nc*TM,col=tile%nc*TN;
                MEGAFLUX_STAGE_BEGIN(stageTileWait,c.trace,true,MegaFluxStageTimer::OWNER_TILE_WAIT,expert,p,row,col);
                if(localRows[expert]>0) {
                    const auto begin=MegaFluxDetail::Now(c.trace,true);
                    MEGAFLUX_STAGE_BEGIN(stageWait,c.trace,true,MegaFluxStageTimer::CONTRIBUTOR_WAIT,expert,p,row,col,d.rank,-1);
                    Wait(c,c.ready,expert,p,row,col);
                    MEGAFLUX_STAGE_END(stageWait);
                    MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::CONTRIBUTOR_WAIT,begin,MegaFluxDetail::Now(c.trace,true),logical,p,row,col,d.rank,-1);
                }
#if MEGAFLUX_OWNER_COMPACT
                for(int i=contributorBegin;i<contributorEnd;++i) {
                    const int index=contributors[i];
                    const int r=index/d.slots,s=index%d.slots;
#else
                for(int r=0;r<d.ranks;++r)for(int s=0;s<d.slots;++s) {
                    if(copies[r*d.slots+s]!=logical)continue;
#endif
                    const auto flags=(uint64_t)aclshmem_ptr((GM_ADDR)c.ready,r);
                    const auto begin=MegaFluxDetail::Now(c.trace,true);
                    MEGAFLUX_STAGE_BEGIN(stageWait,c.trace,true,MegaFluxStageTimer::CONTRIBUTOR_WAIT,expert,p,row,col,r,s);
                    Wait(c,flags,d.home+s,p,row,col);
                    MEGAFLUX_STAGE_END(stageWait);
                    MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::CONTRIBUTOR_WAIT,begin,MegaFluxDetail::Now(c.trace,true),logical,p,row,col,r,s);
                }
                MEGAFLUX_STAGE_END(stageTileWait);
                MEGAFLUX_STAGE_BEGIN(stageService,c.trace,true,MegaFluxStageTimer::OWNER_TILE_SERVICE,expert,p,row,col);
                if(first){Stamp(c,worker,1);first=false;}
                const int width=cols-col<TN?cols-col:TN,height=rows-row<TM?rows-row:TM;
                const auto dst=result+expert*plane*2;
#if defined(DEEPEP_BF16_OWNER_DB)
                MegaFluxOwnerDb::Tile(db,d,dst,partial,plane,copies,logical,
                    row,col,height,width,cols,localRows[expert]>0,c.trace,expert,p);
#else
                for(int offset=0;offset<height;offset+=REDUCE_ROWS) {
                    const int count=height-offset<REDUCE_ROWS?height-offset:REDUCE_ROWS,n=count*width;
                    auto detailStart=MegaFluxDetail::Now(c.trace,true);
#if defined(DEEPEP_BF16_EARLY_REDUCE)
                    if(localRows[expert]<=0) {
                        MEGAFLUX_STAGE_BEGIN(stageInit,c.trace,true,MegaFluxStageTimer::OWNER_CAST,expert,p,row+offset,col,d.rank,-2);
                        Duplicate(acc,0.0f,n);PipeBarrier<PIPE_V>();
                        MEGAFLUX_STAGE_END(stageInit);
                        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::OWNER_CAST,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,d.rank,-2,n*4);
                    } else
#endif
                    {
                        MEGAFLUX_STAGE_BEGIN(stageLoad,c.trace,true,MegaFluxStageTimer::OWNER_LOAD,expert,p,row+offset,col,d.rank,-1);
                        LoadRows(b,dst,row+offset,col,count,width,cols);
                        MEGAFLUX_STAGE_END(stageLoad);
                        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::OWNER_LOAD,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,d.rank,-1,n*2);
                        detailStart=MegaFluxDetail::Now(c.trace,true);
                        MEGAFLUX_STAGE_BEGIN(stageCast,c.trace,true,MegaFluxStageTimer::OWNER_CAST,expert,p,row+offset,col,d.rank,-1);
                        Cast(acc,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
                        MEGAFLUX_STAGE_END(stageCast);
                        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::OWNER_CAST,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,d.rank,-1,n*4);
                    }
#if MEGAFLUX_OWNER_COMPACT
                    for(int i=contributorBegin;i<contributorEnd;++i) {
                        const int index=contributors[i];
                        const int r=index/d.slots,s=index%d.slots;
#else
                    for(int r=0;r<d.ranks;++r)for(int s=0;s<d.slots;++s) {
                        if(copies[r*d.slots+s]!=logical)continue;
#endif
                        const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,r);
                        detailStart=MegaFluxDetail::Now(c.trace,true);
                        MEGAFLUX_STAGE_BEGIN(stageLoad,c.trace,true,MegaFluxStageTimer::PARTIAL_LOAD,expert,p,row+offset,col,r,s);
                        LoadRows(b,peer+partial-d.base+s*plane*2,row+offset,col,count,width,cols);
                        MEGAFLUX_STAGE_END(stageLoad);
                        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::PARTIAL_LOAD,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,r,s,n*2);
                        detailStart=MegaFluxDetail::Now(c.trace,true);
                        MEGAFLUX_STAGE_BEGIN(stageCast,c.trace,true,MegaFluxStageTimer::PARTIAL_CAST,expert,p,row+offset,col,r,s);
                        Cast(tmp,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
                        MEGAFLUX_STAGE_END(stageCast);
                        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::PARTIAL_CAST,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,r,s,n*4);
                        detailStart=MegaFluxDetail::Now(c.trace,true);
                        MEGAFLUX_STAGE_BEGIN(stageSum,c.trace,true,MegaFluxStageTimer::OWNER_SUM,expert,p,row+offset,col,r,s);
                        Add(acc,acc,tmp,n);PipeBarrier<PIPE_V>();
                        MEGAFLUX_STAGE_END(stageSum);
                        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::OWNER_SUM,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,r,s,n*4);
                    }
                    detailStart=MegaFluxDetail::Now(c.trace,true);
                    MEGAFLUX_STAGE_BEGIN(stageCast,c.trace,true,MegaFluxStageTimer::RESULT_CAST,expert,p,row+offset,col,d.rank,-1);
                    Cast(b,acc,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
                    MEGAFLUX_STAGE_END(stageCast);
                    MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::RESULT_CAST,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,d.rank,-1,n*2);
                    detailStart=MegaFluxDetail::Now(c.trace,true);
                    MEGAFLUX_STAGE_BEGIN(stageStore,c.trace,true,MegaFluxStageTimer::RESULT_STORE,expert,p,row+offset,col,d.rank,-1);
                    StoreRows(b,dst,row+offset,col,count,width,cols);
                    MEGAFLUX_STAGE_END(stageStore);
                    MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::RESULT_STORE,detailStart,MegaFluxDetail::Now(c.trace,true),logical,p,row+offset,col,d.rank,-1,n*2);
                }
#endif
                MEGAFLUX_STAGE_END(stageService);
                MegaFluxStageTimer::Mark(c.trace,true,MegaFluxStageTimer::OWNER_TILE_DONE,expert,p,row,col);
                if(firstDone){Stamp(c,worker,7);firstDone=false;}
            }
#if defined(DEEPEP_BF16_STAGE_TIMER)
            MegaFluxStageTimer::End(expertPhase);
            MegaFluxStageTimer::End(stagePhase);
#endif
            Stamp(c,worker,2+phase);
            MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::REDUCE_PHASE,phaseStart,MegaFluxDetail::Now(c.trace,true),-1,p);
        }
        PipeBarrier<PIPE_ALL>();pipe.Destroy();Stamp(c,worker,4);
        MEGAFLUX_STAGE_END(stageWorker);
        MegaFluxDetail::Record(c.trace,true,MegaFluxDetail::REDUCE_WORKER,reduceStart,MegaFluxDetail::Now(c.trace,true));
    }
}
#if MEGAFLUX_WORKER_BACKWARD
// Persistent scalar state only; UB is borrowed and fully retired at each call.
// Task residues are static, so readiness probing never changes tile ownership.
struct CooperativeOwner {
    Context c{};uint64_t outputs[2]={};int local[32]={},offsets[33]={},contributors[256]={};
    int64_t head[2]={};uint32_t done[2]={};int active[32]={},activeCount=0,nextPhase=0;bool initialized=false;
    __aicore__ inline void Init(const Context& context,GM_ADDR w1,GM_ADDR w2,const int32_t* rows) {
        c=context;outputs[0]=(uint64_t)w1;outputs[1]=(uint64_t)w2;int count=0;
        const auto& d=c.replica;
        for(int e=0;e<d.home;++e) {
            local[e]=rows[e];offsets[e]=count;
            for(int r=0;r<d.ranks;++r)for(int s=0;s<d.slots;++s)
                if(MegaFluxMixed::Fresh((__gm__ int32_t*)d.plan+r*d.slots+s)==d.rank*d.home+e)
                    contributors[count++]=r*d.slots+s;
        }
        offsets[d.home]=count;
        for(int e=0;e<d.home;++e)if(offsets[e]!=offsets[e+1])active[activeCount++]=e;
#if MEGAFLUX_GRADIENT_PRIORITY
        for(int i=1;i<activeCount;++i) {
            const int value=active[i];int j=i-1;
            const int key=MegaFluxPriority::Key(d.rank*d.home+value,offsets[value+1]-offsets[value],d.ranks);
            while(j>=0 && MegaFluxPriority::Key(d.rank*d.home+active[j],offsets[active[j]+1]-offsets[active[j]],d.ranks)>key) {
                active[j+1]=active[j];--j;
            }
            active[j+1]=value;
        }
#endif
        initialized=true;
    }
    __aicore__ inline bool Ready(int e,int p,int row,int col) {
        if(offsets[e]==offsets[e+1])return true;
        if(local[e]>0 && ReadGmBypassDCache((__gm__ int32_t*)c.ready+Index(c,e,p,row,col)*LINE)!=c.epoch)return false;
        for(int i=offsets[e];i<offsets[e+1];++i) {
            const int r=contributors[i]/c.replica.slots,s=contributors[i]%c.replica.slots;
            auto* flags=(__gm__ int32_t*)aclshmem_ptr((GM_ADDR)c.ready,r);
            if(ReadGmBypassDCache(flags+Index(c,c.replica.home+s,p,row,col)*LINE)!=c.epoch)return false;
        }
        DataSyncBarrier<MemDsbT::ALL>();return true;
    }
#if defined(MEGAFLUX_SUMMARY_TIMER) && defined(DEEPEP_BF16_STAGE_TIMER)
    uint64_t summaryPhase[2]={0,0};bool summaryStarted[2]={false,false},summaryEnded[2]={false,false};
#endif
    __aicore__ inline void Tile(int e,int p,int row,int col) {
#if defined(MEGAFLUX_SUMMARY_TIMER) && defined(DEEPEP_BF16_STAGE_TIMER)
        if(!summaryStarted[p]) {
            summaryPhase[p]=MegaFluxStageTimer::BeginAsync(c.trace,true,MegaFluxStageTimer::OWNER_SERVICE_PHASE,-1,p);
            summaryStarted[p]=true;
        }
#endif
        MEGAFLUX_STAGE_BEGIN(tileService,c.trace,true,MegaFluxStageTimer::OWNER_TILE_SERVICE,e,p,row,col);
        const auto& d=c.replica;const int cols=Columns(c,p),rows=Rows(c,p);
        const int width=cols-col<TN?cols-col:TN,height=rows-row<TM?rows-row:TM;
        const int64_t plane=(int64_t)rows*cols;
        const uint64_t dst=outputs[p]+e*plane*2,partial=p?d.partial2:d.partial1;
        // Every preceding stage retired MTE/Vector before this low-UB loan.
        LocalTensor<bfloat16_t> b(TPosition::VECCALC,0,ROWS*TN);
        LocalTensor<float> tmp(TPosition::VECCALC,ROWS*TN*2,ROWS*TN);
        LocalTensor<float> acc(TPosition::VECCALC,ROWS*TN*6,ROWS*TN);
        for(int offset=0;offset<height;offset+=ROWS) {
            const int count=height-offset<ROWS?height-offset:ROWS,n=count*width;
            if(local[e]>0){
                MEGAFLUX_STAGE_BEGIN(load,c.trace,true,MegaFluxStageTimer::OWNER_LOAD,e,p,row+offset,col,d.rank,-1);
                LoadRows(b,dst,row+offset,col,count,width,cols);
                MEGAFLUX_STAGE_END(load);
            }
            MEGAFLUX_STAGE_BEGIN(init,c.trace,true,MegaFluxStageTimer::OWNER_CAST,e,p,row+offset,col,d.rank,-1);
            if(local[e]>0)Cast(acc,b,RoundMode::CAST_NONE,n);else Duplicate(acc,0.0f,n);
            PipeBarrier<PIPE_V>();MEGAFLUX_STAGE_END(init);
            for(int i=offsets[e];i<offsets[e+1];++i) {
                const int r=contributors[i]/d.slots,s=contributors[i]%d.slots;
                const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,r);
                MEGAFLUX_STAGE_BEGIN(load,c.trace,true,MegaFluxStageTimer::PARTIAL_LOAD,e,p,row+offset,col,r,s);
                LoadRows(b,peer+partial-d.base+s*plane*2,row+offset,col,count,width,cols);MEGAFLUX_STAGE_END(load);
                MEGAFLUX_STAGE_BEGIN(cast,c.trace,true,MegaFluxStageTimer::PARTIAL_CAST,e,p,row+offset,col,r,s);
                Cast(tmp,b,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();MEGAFLUX_STAGE_END(cast);
                MEGAFLUX_STAGE_BEGIN(sum,c.trace,true,MegaFluxStageTimer::OWNER_SUM,e,p,row+offset,col,r,s);
                Add(acc,acc,tmp,n);PipeBarrier<PIPE_V>();MEGAFLUX_STAGE_END(sum);
            }
            MEGAFLUX_STAGE_BEGIN(cast,c.trace,true,MegaFluxStageTimer::RESULT_CAST,e,p,row+offset,col,d.rank,-1);
            Cast(b,acc,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();MEGAFLUX_STAGE_END(cast);
            MEGAFLUX_STAGE_BEGIN(store,c.trace,true,MegaFluxStageTimer::RESULT_STORE,e,p,row+offset,col,d.rank,-1);
            StoreRows(b,dst,row+offset,col,count,width,cols);MEGAFLUX_STAGE_END(store);
        }
        PipeBarrier<PIPE_ALL>();MEGAFLUX_STAGE_END(tileService);
        MegaFluxStageTimer::Mark(c.trace,true,MegaFluxStageTimer::OWNER_TILE_DONE,e,p,row,col);
    }
    __aicore__ inline bool Complete() {
        for(int p=0;p<2;++p)if(head[p]*GetBlockNum()*2+GetBlockIdx()<activeCount*Tiles(c,p))return false;
        return true;
    }
    __aicore__ inline void Service() {
        if(!initialized)return;int serviced=0,probed=0;
#if MEGAFLUX_WORKER_SHARED_PREFIX
        if(activeCount==0 || Complete())return;
#endif
        PipeBarrier<PIPE_ALL>();
        const int firstPhase=nextPhase;nextPhase^=1;
        for(int phase=0;phase<2 && serviced<2 && probed<8;++phase) {
            const int p=1-((firstPhase+phase)&1),tiles=Tiles(c,p),nc=(Columns(c,p)+TN-1)/TN;
            // Alternate the first projection so newly produced dW1 cannot
            // starve behind the entire dW2 backlog. SUM order inside a tile
            // remains identical; the two projections have disjoint writers.
            for(int ahead=0;ahead<4 && serviced<2 && probed<8;++ahead) {
                if(done[p]&(1u<<ahead))continue;
                const int64_t task=(head[p]+ahead)*GetBlockNum()*2+GetBlockIdx();
                if(task>=activeCount*tiles)break;
                ++probed;
                const int e=active[task/tiles],t=task%tiles,row=t/nc*TM,col=t%nc*TN;
                if(!Ready(e,p,row,col))continue;
                if(offsets[e]!=offsets[e+1]){Tile(e,p,row,col);++serviced;}
                done[p]|=1u<<ahead;
            }
            while(done[p]&1u){++head[p];done[p]>>=1;}
#if defined(MEGAFLUX_SUMMARY_TIMER) && defined(DEEPEP_BF16_STAGE_TIMER)
            if(summaryStarted[p]&&!summaryEnded[p]&&head[p]*GetBlockNum()*2+GetBlockIdx()>=activeCount*tiles) {
                MegaFluxStageTimer::EndAsync(summaryPhase[p]);summaryEnded[p]=true;
            }
#endif
        }
    }
    __aicore__ inline void Drain() {while(!Complete())Service();PipeBarrier<PIPE_ALL>();}
};
#endif

}
