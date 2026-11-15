// Bounded ready-frontier selection. No model parameters or future routes.
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

namespace {
__aicore__ inline bool LexBefore(int candidate,int previous,int tasks) {
    int left=0,right=0;
    while(true) {
        while(left<tasks && !(candidate&(1<<left)))++left;
        while(right<tasks && !(previous&(1<<right)))++right;
        if(left==tasks || right==tasks)return left<tasks && right==tasks;
        if(left!=right)return left<right;
        ++left;++right;
    }
}
}

extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mega_moe_temporal_scheduler_kernel(GM_ADDR loads,GM_ADDR metadata,GM_ADDR output,
                                       int tasks,int ranks,int window,int policy) {
    if ASCEND_IS_AIC {return;}
    if(GetBlockIdx()!=0)return;
    auto cost=(__gm__ int32_t*)loads;
    auto meta=(__gm__ int32_t*)metadata;
    auto out=(__gm__ int32_t*)output;
    int values[32],layers[4],arrival[4],forced=0;
    int64_t taskWork[4];
    bool bad=tasks<1||tasks>4||ranks!=8||tasks>window||
        (window!=1&&window!=2&&window!=4)||policy<0||policy>2;
    if(!bad)for(int task=0;task<tasks;++task) {
        layers[task]=ReadGmBypassDCache(meta+task*3);
        int age=ReadGmBypassDCache(meta+task*3+1);
        arrival[task]=ReadGmBypassDCache(meta+task*3+2);
        if(arrival[task]<0||arrival[task]>=tasks)bad=true;
        for(int previous=0;previous<task;++previous)if(arrival[task]==arrival[previous])bad=true;
        if(age<0)bad=true;
        if(age>=2)forced|=1<<task;
        taskWork[task]=0;
        for(int rank=0;rank<ranks;++rank) {
            int value=ReadGmBypassDCache(cost+task*ranks+rank);
            // Integer cross-products below stay within int64 under this bound.
            if(value<0||value>16777216)bad=true;
            values[task*8+rank]=value;taskWork[task]+=value;
        }
    }
    int selected=0;
    if(!bad && policy==0) {
        int minimum=(window+1)/2;if(minimum>tasks)minimum=tasks;
        selected=forced;int count=0;
        for(int task=0;task<tasks;++task)if(forced&(1<<task))++count;
        for(int order=0;order<tasks&&count<minimum;++order)
            for(int task=0;task<tasks;++task)
                if(arrival[task]==order&&!(selected&(1<<task))){selected|=1<<task;++count;}
    }
    if(!bad && policy==1) {
        int first=0;
        if(forced)while(!(forced&(1<<first)))++first;
        for(int task=0;task<tasks;++task)
            if(layers[task]==layers[first]||(forced&(1<<task)))selected|=1<<task;
    }
    if(!bad && policy==2) {
        int minimum=(window+1)/2;if(minimum>tasks)minimum=tasks;
        int64_t bestNumerator=0,bestDenominator=1,bestWork=-1;
        for(int mask=1;mask<(1<<tasks);++mask) {
            if((mask&forced)!=forced)continue;
            int count=0;int64_t work=0;
            for(int task=0;task<tasks;++task)if(mask&(1<<task)){++count;work+=taskWork[task];}
            if(count<minimum)continue;
            int unique=0;
            for(int task=0;task<tasks;++task)if(mask&(1<<task)) {
                bool seen=false;
                for(int previous=0;previous<task;++previous)
                    if((mask&(1<<previous))&&layers[previous]==layers[task])seen=true;
                if(!seen)++unique;
            }
            if(unique>2)continue;
            int64_t peak=0;
            for(int rank=0;rank<ranks;++rank) {
                int64_t sum=0;
                for(int task=0;task<tasks;++task)if(mask&(1<<task))sum+=values[task*8+rank];
                if(sum>peak)peak=sum;
            }
            int64_t numerator=peak?work:1,denominator=peak?peak*ranks:1;
            int64_t delta=numerator*bestDenominator-bestNumerator*denominator;
            if(!selected||delta>0||(delta==0&&(work>bestWork||
                (work==bestWork&&LexBefore(mask,selected,tasks))))) {
                selected=mask;bestNumerator=numerator;bestDenominator=denominator;bestWork=work;
            }
        }
    }
    if(!selected)bad=true;
    int work=0,peak=0,serial=0;
    if(!bad) {
        for(int task=0;task<tasks;++task)if(selected&(1<<task)) {
            int taskPeak=0;
            for(int rank=0;rank<ranks;++rank)
                if(values[task*8+rank]>taskPeak)taskPeak=values[task*8+rank];
            serial+=taskPeak;
        }
        for(int rank=0;rank<ranks;++rank) {
            int sum=0;
            for(int task=0;task<tasks;++task)if(selected&(1<<task))sum+=values[task*8+rank];
            work+=sum;if(sum>peak)peak=sum;
        }
    }
    out[0]=bad?0:selected;out[1]=bad?1:0;out[2]=work;out[3]=peak;out[4]=serial;
    GlobalTensor<int32_t> publish;publish.SetGlobalBuffer(out);
    DataCacheCleanAndInvalid<int32_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(publish);
    DataSyncBarrier<MemDsbT::ALL>();
}

extern "C" __attribute__((visibility("default"))) void mega_moe_temporal_scheduler_do(
    void* stream,uint8_t* loads,uint8_t* metadata,uint8_t* output,
    int tasks,int ranks,int window,int policy) {
    mega_moe_temporal_scheduler_kernel<<<1,nullptr,stream>>>(loads,metadata,output,tasks,ranks,window,policy);
}
extern "C" __attribute__((visibility("default"))) int megaflux_temporal_scheduler_abi(){return 3;}
