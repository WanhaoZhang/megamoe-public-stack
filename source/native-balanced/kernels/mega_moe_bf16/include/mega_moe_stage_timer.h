#pragma once
#include "kernel_operator.h"
#if defined(DEEPEP_BF16_STAGE_TIMER)
#include "../../common/timer/AscendTimerV2_device.hpp"
#define MEGAFLUX_STAGE_BEGIN(name, ...) const auto name=MegaFluxStageTimer::Begin(__VA_ARGS__)
#define MEGAFLUX_STAGE_END(name) MegaFluxStageTimer::End(name)
#else
// Erase arguments too: the base train structs have no replica trace fields.
#define MEGAFLUX_STAGE_BEGIN(name, ...) do {} while (0)
#define MEGAFLUX_STAGE_END(name) do {} while (0)
#endif

// AscendTimerV2 task-coordinate slots, not an append/event logger. Each pair
// has its own cache line, one writer, and a deterministic host-decodable index.
namespace MegaFluxStageTimer {
using namespace AscendC;
constexpr int64_t MAGIC=0x4d4654560003LL;
constexpr int HEADER=16,LANES=96,WORDS=8;
enum Kind {COPY=1,WEIGHT_RELEASE,WEIGHT_WAIT,WEIGHT_USE,PARTIAL_DRAIN,
    PARTIAL_RELEASE,CONTRIBUTOR_WAIT,OWNER_LOAD,OWNER_CAST,PARTIAL_LOAD,
    PARTIAL_CAST,OWNER_SUM,RESULT_CAST,RESULT_STORE,REDUCE_PHASE,REDUCE_WORKER,
    WEIGHT_PUBLISH,PARTIAL_PUBLISH,OWNER_TILE_WAIT,OWNER_TILE_SERVICE,OWNER_TILE_DONE,
    FWD_GEMM_TILE,FWD_VECTOR_WAIT,FWD_VECTOR_TILE,FWD_CUBE_WAIT_VECTOR,FWD_INPUT_WAIT,
    WGRAD_TILE,BW_SYNC,BW_EMPTY_GRAD_CLEAR,BWD_DGRAD_TILE,BWD_VECTOR_TILE,
    WORKER_COPY_RESIDENCY,RESTORE_COPY,RESTORE_READY,
    FWD_SOURCE_SUBMIT,FWD_BLOCK_RELEASE_REQUEST,FWD_SOURCE_RELEASE_REQUEST,
    BWD_SOURCE_COPY_TO_RETIRE,BWD_SOURCE_RELEASE_REQUEST,
    FWD_COPY_PHASE,BWD_RESTORE_PHASE,OWNER_SERVICE_PHASE,OWNER_EXPERT_PHASE};

__aicore__ inline uint64_t Slot(uint64_t trace,bool backward,int kind,int e=-1,
        int p=-1,int row=-1,int col=-1,int peer=-1,int slot=-1) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
#if defined(MEGAFLUX_SUMMARY_TIMER)
    // Constant kind folds unselected per-tile call sites away before any GM read.
    if(kind!=REDUCE_PHASE&&kind!=REDUCE_WORKER&&kind!=BW_SYNC&&kind!=BW_EMPTY_GRAD_CLEAR&&kind!=FWD_COPY_PHASE&&kind!=BWD_RESTORE_PHASE&&kind!=OWNER_SERVICE_PHASE&&kind!=OWNER_EXPERT_PHASE
#if defined(MEGAFLUX_CALIBRATION_TIMER)
       &&kind!=PARTIAL_RELEASE&&kind!=OWNER_TILE_WAIT&&kind!=OWNER_TILE_SERVICE
#endif
       )return 0;
#endif
    if(!trace)return 0;
    auto* h=(__gm__ int64_t*)trace+(backward?256*8*8:128*333*8);
    if(h[0]!=MAGIC)return 0;
    const int ep=h[1],home=h[2],slots=h[3],hidden=h[4],ffn=h[5],capacity=h[6];
    int group=GetBlockIdx(),sub=0;
    if ASCEND_IS_AIV {group=GetBlockIdx()/GetTaskRation();sub=GetSubBlockIdx()+1;}
    const int lane=group*3+sub,worker=group*2+sub-1;
    if(lane>=LANES)return 0;
    int64_t index=-1;
#if defined(MEGAFLUX_SUMMARY_TIMER)
    if(h[9]==7 || h[9]==8 || h[9]==9) {
        const int kinds[7]={REDUCE_PHASE,REDUCE_WORKER,BW_SYNC,BW_EMPTY_GRAD_CLEAR,FWD_COPY_PHASE,BWD_RESTORE_PHASE,OWNER_SERVICE_PHASE};
        const int counts[7]={2,1,8,1,2,2,2};
        int base=0;
        for(int i=0;i<7;++i) {
            if(kind==kinds[i]) {
                const int coordinate=kind==BW_SYNC?row:(counts[i]==2?p:0);
                if(coordinate<0||coordinate>=counts[i])return 0;
                index=base+coordinate*LANES+lane;break;
            }
            base+=counts[i]*LANES;
        }
        if(kind==OWNER_EXPERT_PHASE && (h[9]==8||h[9]==9) && backward && e>=0 && e<home && p>=0 && p<2)
            index=base+(p*home+e)*LANES+lane;
#if defined(MEGAFLUX_CALIBRATION_TIMER)
        if(h[9]==9 && backward && p>=0 && p<2 && row>=0 && row%128==0 && col==0) {
            const int bins=((2*ffn>hidden?2*ffn:hidden)+127)/128;
            const int64_t sampleBase=base+2*home*LANES;
            if(kind==PARTIAL_RELEASE && e>=0 && e<home+slots)
                index=sampleBase+((int64_t)p*(home+slots)+e)*bins+row/128;
            if((kind==OWNER_TILE_WAIT||kind==OWNER_TILE_SERVICE) && e>=0 && e<home)
                index=sampleBase+2*(home+slots)*bins+(kind-OWNER_TILE_WAIT)*2*home*bins+((int64_t)p*home+e)*bins+row/128;
        }
#endif
        // Unselected fine-grained sites do not timestamp or touch counters.
        if(index<0)return 0;
    } else
#endif
    if(!backward) {
        const int s1=(ffn+255)/256,s2=(hidden+255)/256,s=s1+s2;
        const int mb=(capacity+127)/128;
        const int64_t copyPairs=(int64_t)slots*s*32,taskPairs=(int64_t)(home+slots)*mb*s;
        if(kind==COPY||kind==WEIGHT_RELEASE)
            index=(kind-COPY)*copyPairs+((int64_t)slot*s+(p?s1:0)+row)*32+group;
        if(kind==WEIGHT_WAIT||kind==WEIGHT_USE)
            index=2*copyPairs+(kind-WEIGHT_WAIT)*taskPairs+((int64_t)e*mb+row/128)*s+(p?s1:0)+col/256;
        const int64_t legacy=2*copyPairs+2*taskPairs;
        if(kind==WEIGHT_PUBLISH)
            index=legacy+((int64_t)slot*s+(p?s1:0)+row)*32+group;
        if(kind>=FWD_GEMM_TILE&&kind<=FWD_INPUT_WAIT)
            index=legacy+copyPairs+(kind-FWD_GEMM_TILE)*taskPairs+
                ((int64_t)e*mb+row/128)*s+(p?s1:0)+col/256;
        if(kind==WORKER_COPY_RESIDENCY && (h[9]==4||h[9]==5||h[9]==6)) {
            const int chunks1=2*((256*hidden+32767)/32768),chunks2=(256*ffn+32767)/32768;
            index=3*copyPairs+7*taskPairs+(p?(int64_t)slots*s1*chunks1:0)
                +((int64_t)slot*(p?s2:s1)+row)*(p?chunks2:chunks1)+col;
        }
    } else {
        const int t1=((2*ffn+127)/128)*((hidden+255)/256),t2=((hidden+127)/128)*((ffn+255)/256),t=t1+t2;
        const int peers=ep*slots,ops=4+3*peers;
        const int64_t producer=(int64_t)(home+slots)*t,owner=(int64_t)home*t;
        const int nc=((p?ffn:hidden)+255)/256;
        const int64_t pt=(int64_t)e*t+(p?t1:0)+(row/128)*nc+col/256;
        if(kind==PARTIAL_DRAIN||kind==PARTIAL_RELEASE)index=(kind-PARTIAL_DRAIN)*producer+pt;
        const int64_t waitBase=2*producer,opBase=waitBase+owner*(1+peers);
        if(kind==CONTRIBUTOR_WAIT)index=waitBase+pt*(1+peers)+(slot<0?0:1+peer*slots+slot);
        int op=-1;
        if(kind==OWNER_LOAD)op=0;
        if(kind==OWNER_CAST)op=1;
        if(kind>=PARTIAL_LOAD&&kind<=OWNER_SUM)op=2+(peer*slots+slot)*3+(kind-PARTIAL_LOAD);
        if(kind==RESULT_CAST)op=2+3*peers;
        if(kind==RESULT_STORE)op=3+3*peers;
        if(op>=0)index=opBase+(pt*8+(row%128)/16)*ops+op;
        const int64_t phaseBase=opBase+owner*8*ops;
        if(kind==REDUCE_PHASE)index=phaseBase+worker*2+p;
        if(kind==REDUCE_WORKER)index=phaseBase+128+worker;
        const int64_t legacy=phaseBase+192;
        if(kind==PARTIAL_PUBLISH)index=legacy+pt;
        if(kind>=OWNER_TILE_WAIT&&kind<=OWNER_TILE_DONE)
            index=legacy+producer+(kind-OWNER_TILE_WAIT)*owner+pt;
        if(kind==WGRAD_TILE)index=legacy+producer+3*owner+pt;
        if(kind==BW_SYNC)index=legacy+2*producer+3*owner+row*LANES+lane;
        if(kind==BW_EMPTY_GRAD_CLEAR)index=legacy+2*producer+3*owner+8*LANES+lane;
        if(kind==BWD_DGRAD_TILE||kind==BWD_VECTOR_TILE) {
            const int s1=(ffn+255)/256,s2=(hidden+255)/256,s=s1+s2;
            const int mb=(capacity+127)/128;
            const int64_t tasks=(int64_t)(home+slots)*mb*s;
            index=legacy+2*producer+3*owner+9*LANES+(kind-BWD_DGRAD_TILE)*tasks+
                ((int64_t)e*mb+row/128)*s+(p?s1:0)+col/256;
        }
        if((kind==RESTORE_COPY||kind==RESTORE_READY) && (h[9]==5||h[9]==6)) {
            const int w1Panels=(2*ffn+255)/256,w2Panels=(hidden+255)/256;
            const int s1=(ffn+255)/256,s2=(hidden+255)/256;
            const int64_t tasks=(int64_t)(home+slots)*((capacity+127)/128)*(s1+s2);
            const int64_t base=legacy+2*producer+3*owner+9*LANES+2*tasks;
            const int64_t copies=(int64_t)slots*(w1Panels+w2Panels)*32;
            index=base+(kind-RESTORE_COPY)*copies+
                ((int64_t)slot*(w1Panels+w2Panels)+(p?w1Panels:0)+row)*32+col;
        }
    }
    // Schema6 appends unique source/partition slots. No old slot moves.
    // A source can own several partitions on one core: key by partition, not lane.
    if(h[9]==6 && kind>=FWD_SOURCE_SUBMIT && kind<=BWD_SOURCE_RELEASE_REQUEST &&
            ep<=8 && e>=0 && e<home+slots && peer>=0 && peer<ep && slot>=0 && slot<64) {
        const int s1=(ffn+255)/256,s2=(hidden+255)/256;
        const int64_t sourceCount=(int64_t)(home+slots)*ep*64;
        const int64_t sourceIndex=((int64_t)e*ep+peer)*64+slot;
        if(!backward) {
            const int64_t cp=(int64_t)slots*(s1+s2)*32;
            const int64_t tasks=(int64_t)(home+slots)*((capacity+127)/128)*(s1+s2);
            const int c1=2*((256*hidden+32767)/32768),c2=(256*ffn+32767)/32768;
            const int64_t base=3*cp+7*tasks+(int64_t)slots*(s1*c1+s2*c2);
            const int mb=(capacity+255)/256;
            const int64_t blocks=(int64_t)(home+slots)*mb*ep*64;
            if(kind==FWD_SOURCE_SUBMIT)index=base+sourceIndex;
            if(kind==FWD_BLOCK_RELEASE_REQUEST && row>=0 && row<mb)
                index=base+sourceCount+(((int64_t)e*mb+row)*ep+peer)*64+slot;
            if(kind==FWD_SOURCE_RELEASE_REQUEST)index=base+sourceCount+blocks+sourceIndex;
        } else {
            const int t1=((2*ffn+127)/128)*((hidden+255)/256),t2=((hidden+127)/128)*((ffn+255)/256);
            const int64_t producer=(int64_t)(home+slots)*(t1+t2),owner=(int64_t)home*(t1+t2);
            const int peers=ep*slots,ops=4+3*peers;
            const int64_t phaseBase=2*producer+owner*(1+peers)+owner*8*ops;
            const int64_t tasks=(int64_t)(home+slots)*((capacity+127)/128)*(s1+s2);
            const int64_t restore=(int64_t)slots*((2*ffn+255)/256+(hidden+255)/256)*32;
            const int64_t base=phaseBase+192+2*producer+3*owner+9*LANES+2*tasks+2*restore;
            if(kind==BWD_SOURCE_COPY_TO_RETIRE)index=base+sourceIndex;
            if(kind==BWD_SOURCE_RELEASE_REQUEST)index=base+sourceCount+sourceIndex;
        }
    }
    auto* stats=h+HEADER+lane*8;
    if(index<0||index>=h[8]){stats[2]=stats[2]+1;return 0;}
    auto* pair=h+HEADER+LANES*8+index*WORDS;
    if(pair[0])stats[1]=stats[1]+1;
    pair[2]=lane;pair[3]=pair[3]+1;pair[4]=row;pair[5]=col;stats[0]=stats[0]+1;
    return (uint64_t)pair;
#endif
    return 0;
}
__aicore__ inline uint64_t Begin(uint64_t trace,bool backward,int kind,int e=-1,
        int p=-1,int row=-1,int col=-1,int peer=-1,int slot=-1) {
    const auto address=Slot(trace,backward,kind,e,p,row,col,peer,slot);
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(address)AscendTimerDevice::TikPair((GM_ADDR)address);
#endif
    return address;
}
__aicore__ inline void End(uint64_t address) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(address)AscendTimerDevice::TokPair((GM_ADDR)address);
#endif
}
__aicore__ inline uint64_t BeginAsync(uint64_t trace,bool backward,int kind,int e=-1,
        int p=-1,int row=-1,int col=-1,int peer=-1,int slot=-1) {
    const auto address=Slot(trace,backward,kind,e,p,row,col,peer,slot);
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(address)AscendTimerDevice::TikPairAsync((GM_ADDR)address);
#endif
    return address;
}
__aicore__ inline void EndAsync(uint64_t address) {
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(address)AscendTimerDevice::TokPairAsync((GM_ADDR)address);
#endif
}
__aicore__ inline void MarkAsync(uint64_t trace,bool backward,int kind,int e=-1,
        int p=-1,int row=-1,int col=-1,int peer=-1,int slot=-1) {
    const auto address=Slot(trace,backward,kind,e,p,row,col,peer,slot);
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(address) {
        AscendTimerDevice::TikPairAsync((GM_ADDR)address);
        AscendTimerDevice::TokPairAsync((GM_ADDR)address);
    }
#endif
}
__aicore__ inline void Mark(uint64_t trace,bool backward,int kind,int e=-1,
        int p=-1,int row=-1,int col=-1,int peer=-1,int slot=-1) {
    const auto address=Slot(trace,backward,kind,e,p,row,col,peer,slot);
#if defined(DEEPEP_BF16_STAGE_TIMER)
    if(address)AscendTimerDevice::MarkPair((GM_ADDR)address);
#endif
}
}
