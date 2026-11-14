#pragma once
#include "kernel_operator.h"
#include "mega_moe_replica_descriptor.h"
#include "shmem.h"
#include "mega_moe_stage_timer.h"

// Optional S9 owner consumer. Event pairs are balanced within one tile;
// inputs and outputs never alias. Placement, producer epochs and SUM order
// remain the responsibility of the existing executor.
namespace MegaFluxOwnerDb {
using namespace AscendC;
constexpr int ROWS=16,COLS=256;
struct Buffers {
    TBuf<TPosition::VECCALC> ib0,ib1,ob0,ob1,tb,ab;
    LocalTensor<bfloat16_t> input[2],output[2];
    LocalTensor<float> tmp,acc;
    __aicore__ inline void Init(TPipe& pipe) {
        pipe.InitBuffer(ib0,ROWS*COLS*2);pipe.InitBuffer(ib1,ROWS*COLS*2);
        pipe.InitBuffer(ob0,ROWS*COLS*2);pipe.InitBuffer(ob1,ROWS*COLS*2);
        pipe.InitBuffer(tb,ROWS*COLS*4);pipe.InitBuffer(ab,ROWS*COLS*4);
        input[0]=ib0.Get<bfloat16_t>();input[1]=ib1.Get<bfloat16_t>();
        output[0]=ob0.Get<bfloat16_t>();output[1]=ob1.Get<bfloat16_t>();
        tmp=tb.Get<float>();acc=ab.Get<float>();
    }
};
__aicore__ inline TEventID InputEvent(int slot) {return slot?EVENT_ID1:EVENT_ID0;}
__aicore__ inline TEventID OutputEvent(int slot) {return slot?EVENT_ID3:EVENT_ID2;}

__aicore__ inline void LoadAsync(LocalTensor<bfloat16_t> buffer,uint64_t addr,
        int row,int col,int rows,int width,int stride,int slot,bool timed=false) {
    const auto event=InputEvent(slot);
    // The previous Vector read of this slot must retire before overwrite.
    SetFlag<HardEvent::V_MTE2>(event);WaitFlag<HardEvent::V_MTE2>(event);
    GlobalTensor<bfloat16_t> gm;gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams cp{(uint16_t)rows,(uint32_t)(width*2),(uint32_t)((stride-width)*2),0,0};
    DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
    DataCopyPad(buffer,gm[(int64_t)row*stride+col],cp,pad);
    SetFlag<HardEvent::MTE2_V>(event); // consumed only at the actual Vector use
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(timed)SetFlag<HardEvent::MTE2_S>(event);
#endif
}
__aicore__ inline void StoreAsync(LocalTensor<bfloat16_t> buffer,uint64_t addr,
        int row,int col,int rows,int width,int stride,int slot,bool timed=false) {
    const auto event=OutputEvent(slot);
    GlobalTensor<bfloat16_t> gm;gm.SetGlobalBuffer((__gm__ bfloat16_t*)addr);
    gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    DataCopyExtParams cp{(uint16_t)rows,(uint32_t)(width*2),0,(uint32_t)((stride-width)*2),0};
    SetFlag<HardEvent::V_MTE3>(event);WaitFlag<HardEvent::V_MTE3>(event);
    DataCopyPad(gm[(int64_t)row*stride+col],buffer,cp);
    SetFlag<HardEvent::MTE3_V>(event); // wait at reuse or final tile completion
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(timed)SetFlag<HardEvent::MTE3_S>(event);
#endif
}
__aicore__ inline uint64_t VectorBegin(uint64_t trace,int kind,int e,int p,
        int row,int col,int rank,int slot) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
    // Event 6 belongs to this owner scope; only drain Vector, never the
    // independent MTE2 prefetch or MTE3 store currently in flight.
    if(trace){SetFlag<HardEvent::V_S>(EVENT_ID6);WaitFlag<HardEvent::V_S>(EVENT_ID6);}
    return MegaFluxStageTimer::BeginAsync(trace,true,kind,e,p,row,col,rank,slot);
#else
    return 0;
#endif
}
__aicore__ inline void VectorEnd(uint64_t span) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(span){SetFlag<HardEvent::V_S>(EVENT_ID6);WaitFlag<HardEvent::V_S>(EVENT_ID6);}
    MegaFluxStageTimer::EndAsync(span);
#endif
}
__aicore__ inline int Next(const int* copies,int count,int logical,int& cursor) {
    while(cursor<count) {const int i=cursor++;if(copies[i]==logical)return i;}
    return -1;
}
__aicore__ inline uint64_t Remote(const MegaMoeReplicaDescriptor& d,
        uint64_t partial,int64_t plane,int index) {
    const auto peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,index/d.slots);
    return peer+partial-d.base+(index%d.slots)*plane*2;
}
__aicore__ inline void Tile(Buffers& b,const MegaMoeReplicaDescriptor& d,
        uint64_t dst,uint64_t partial,int64_t plane,const int* copies,int logical,
        int row,int col,int height,int width,int cols,bool haveHome,
        uint64_t trace=0,int expert=0,int projection=0) {
#if defined(DEEPEP_BF16_STAGE_TIMER) && defined(MEGAFLUX_SUMMARY_TIMER)
    // Outer expert/phase scopes provide summary timestamps. Disabled subtile
    // slots must not retain the diagnostic-only Vector/MTE-to-Scalar fences.
    trace=0;
#endif
    bool outputPending[2]={false,false};
#if defined(DEEPEP_BF16_STAGE_TIMER)
    uint64_t inputSpan[2]={0,0},outputSpan[2]={0,0};
#endif
    for(int offset=0;offset<height;offset+=ROWS) {
        const int count=height-offset<ROWS?height-offset:ROWS,n=count*width;
        int cursor=0,remote=Next(copies,d.ranks*d.slots,logical,cursor),current=0;
        if(haveHome) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
            inputSpan[0]=MegaFluxStageTimer::BeginAsync(trace,true,MegaFluxStageTimer::OWNER_LOAD,expert,projection,row+offset,col,d.rank,-1);
#endif
            LoadAsync(b.input[0],dst,row+offset,col,count,width,cols,0,trace!=0);
            // Prefetch the first remote while the Vector initializes acc.
            if(remote>=0) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
                inputSpan[1]=MegaFluxStageTimer::BeginAsync(trace,true,MegaFluxStageTimer::PARTIAL_LOAD,expert,projection,row+offset,col,remote/d.slots,remote%d.slots);
#endif
                LoadAsync(b.input[1],Remote(d,partial,plane,remote),row+offset,col,count,width,cols,1,trace!=0);
            }
            WaitFlag<HardEvent::MTE2_V>(InputEvent(0));
#if defined(DEEPEP_BF16_STAGE_TIMER)
            if(trace)WaitFlag<HardEvent::MTE2_S>(InputEvent(0));
            MegaFluxStageTimer::EndAsync(inputSpan[0]);
#endif
            const auto homeCast=VectorBegin(trace,MegaFluxStageTimer::OWNER_CAST,expert,projection,row+offset,col,d.rank,-1);
            Cast(b.acc,b.input[0],RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
            VectorEnd(homeCast);
            current=1;
        } else {
            const auto emptyCast=VectorBegin(trace,MegaFluxStageTimer::OWNER_CAST,expert,projection,row+offset,col,d.rank,-2);
            Duplicate(b.acc,0.0f,n);PipeBarrier<PIPE_V>();
            VectorEnd(emptyCast);
            if(remote>=0) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
                inputSpan[0]=MegaFluxStageTimer::BeginAsync(trace,true,MegaFluxStageTimer::PARTIAL_LOAD,expert,projection,row+offset,col,remote/d.slots,remote%d.slots);
#endif
                LoadAsync(b.input[0],Remote(d,partial,plane,remote),row+offset,col,count,width,cols,0,trace!=0);
            }
        }
        while(remote>=0) {
            const int next=Next(copies,d.ranks*d.slots,logical,cursor);
            if(next>=0) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
                inputSpan[current^1]=MegaFluxStageTimer::BeginAsync(trace,true,MegaFluxStageTimer::PARTIAL_LOAD,expert,projection,row+offset,col,next/d.slots,next%d.slots);
#endif
                LoadAsync(b.input[current^1],Remote(d,partial,plane,next),row+offset,col,count,width,cols,current^1,trace!=0);
            }
            WaitFlag<HardEvent::MTE2_V>(InputEvent(current));
#if defined(DEEPEP_BF16_STAGE_TIMER)
            if(trace)WaitFlag<HardEvent::MTE2_S>(InputEvent(current));
            MegaFluxStageTimer::EndAsync(inputSpan[current]);
#endif
            const auto partialCast=VectorBegin(trace,MegaFluxStageTimer::PARTIAL_CAST,expert,projection,row+offset,col,remote/d.slots,remote%d.slots);
            Cast(b.tmp,b.input[current],RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
            VectorEnd(partialCast);
            const auto sum=VectorBegin(trace,MegaFluxStageTimer::OWNER_SUM,expert,projection,row+offset,col,remote/d.slots,remote%d.slots);
            Add(b.acc,b.acc,b.tmp,n);PipeBarrier<PIPE_V>();
            VectorEnd(sum);
            remote=next;current^=1;
        }
        const int out=(offset/ROWS)&1;
        if(outputPending[out]) {
            WaitFlag<HardEvent::MTE3_V>(OutputEvent(out));
#if defined(DEEPEP_BF16_STAGE_TIMER)
            if(trace)WaitFlag<HardEvent::MTE3_S>(OutputEvent(out));
            MegaFluxStageTimer::EndAsync(outputSpan[out]);
#endif
        }
        const auto resultCast=VectorBegin(trace,MegaFluxStageTimer::RESULT_CAST,expert,projection,row+offset,col,d.rank,-1);
        Cast(b.output[out],b.acc,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
        VectorEnd(resultCast);
#if defined(DEEPEP_BF16_STAGE_TIMER)
        outputSpan[out]=MegaFluxStageTimer::BeginAsync(trace,true,MegaFluxStageTimer::RESULT_STORE,expert,projection,row+offset,col,d.rank,-1);
#endif
        StoreAsync(b.output[out],dst,row+offset,col,count,width,cols,out,trace!=0);
        outputPending[out]=true;
    }
    // No output tile is complete until both stores have actually retired.
    for(int slot=0;slot<2;++slot)
        if(outputPending[slot]) {
            WaitFlag<HardEvent::MTE3_V>(OutputEvent(slot));
#if defined(DEEPEP_BF16_STAGE_TIMER)
            if(trace)WaitFlag<HardEvent::MTE3_S>(OutputEvent(slot));
            MegaFluxStageTimer::EndAsync(outputSpan[slot]);
#endif
        }
}
} // namespace MegaFluxOwnerDb
