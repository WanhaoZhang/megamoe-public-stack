#pragma once
#include "kernel_operator.h"

// Opt-in suffix to the existing sparse trace tensor. Each AIC/AIV lane is the
// sole writer of its counter and records; no inter-lane atomic or barrier.
namespace MegaFluxDetail {
using namespace AscendC;
constexpr int64_t MAGIC=0x4d4644540001LL;
constexpr int FW_PREFIX=128*333*8,BW_PREFIX=256*8*8,HEADER=16,LANES=96,WORDS=10;
enum Kind {COPY=1,WEIGHT_RELEASE,WEIGHT_WAIT,WEIGHT_USE,PARTIAL_DRAIN,
    PARTIAL_RELEASE,CONTRIBUTOR_WAIT,OWNER_LOAD,OWNER_CAST,PARTIAL_LOAD,
    PARTIAL_CAST,OWNER_SUM,RESULT_CAST,RESULT_STORE,REDUCE_PHASE,REDUCE_WORKER};
__aicore__ inline int64_t Now(uint64_t trace,bool backward=false) {
#if defined(DEEPEP_BF16_DETAIL_TRACE)
    if(trace && ((__gm__ int64_t*)trace)[backward?BW_PREFIX:FW_PREFIX]==MAGIC)
        return GetSystemCycle();
#endif
    return 0;
}
__aicore__ inline void Record(uint64_t trace,bool backward,int kind,int64_t start,
        int64_t end,int expert=-1,int projection=-1,int row=-1,int col=-1,
        int peer=-1,int slot=-1,int64_t bytes=0) {
#if defined(DEEPEP_BF16_DETAIL_TRACE)
    if(!start || !trace)return;
    auto* h=(__gm__ int64_t*)trace+(backward?BW_PREFIX:FW_PREFIX);
    if(h[0]!=MAGIC)return;
    int group=GetBlockIdx(),sub=0;
    if ASCEND_IS_AIV {group=GetBlockIdx()/GetTaskRation();sub=GetSubBlockIdx()+1;}
    const int lane=group*3+sub;
    if(lane>=LANES)return;
    auto* counter=h+HEADER+lane*8;
    const int64_t ca=h[1],cv=h[2],cap=sub?cv:ca,seq=counter[0];
    if(seq>=cap){counter[1]=counter[1]+1;return;}
    const int64_t base=group*(ca+2*cv)+(sub?ca+(sub-1)*cv:0);
    auto* p=h+HEADER+LANES*8+(base+seq)*WORDS;
    p[0]=start;p[1]=end;p[2]=kind;p[3]=expert;p[4]=projection;
    p[5]=row;p[6]=col;p[7]=peer;p[8]=slot;p[9]=bytes;
    counter[0]=seq+1;
#endif
}
}
