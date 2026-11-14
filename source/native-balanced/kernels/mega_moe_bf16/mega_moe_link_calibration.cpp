// Isolated diagnostic: same SHMEM peer address translation and MTE transport.
// Never linked into or loaded by formal training processes.
#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
using namespace AscendC;
#include "../common/timer/AscendTimerV2_device.hpp"
using namespace AscendC;
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void planner_link_calibration(uint64_t ffts,GM_ADDR base,GM_ADDR timing,uint64_t sourceoff,
        uint64_t targetoff,int source,int dest,int rank,int bytes,int repeats) {
    util_set_ffts_config(ffts);if ASCEND_IS_AIC {return;}
    int w=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);
    TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,32768);
    auto ub=buffer.Get<uint8_t>();
    GlobalTensor<uint8_t> input,output;
    input.SetGlobalBuffer((__gm__ uint8_t*)aclshmem_ptr(base,source)+sourceoff);
    output.SetGlobalBuffer((__gm__ uint8_t*)base+targetoff);
    input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    const int chunk=(bytes+31)/32;
    for(int it=0;it<repeats;++it){
        aclshmemx_barrier_all_vec();
        if(rank==dest || (dest<0 && rank!=source)){
            auto pair=(GM_ADDR)((__gm__ uint64_t*)timing+(it*32+w)*16);
            AscendTimerDevice::TikPair(pair);
            int begin=w*chunk,end=begin+chunk;if(end>bytes)end=bytes;
            for(int offset=begin;offset<end;offset+=32768){
                int size=end-offset;if(size>32768)size=32768;
                DataCopyExtParams p(1,size,0,0,0);DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                DataCopyPad(ub,input[offset],p,pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopyPad(output[offset],ub,p);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
            AscendTimerDevice::TokPair(pair);
        }
        aclshmemx_barrier_all_vec();
    }
    pipe.Destroy();
}
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void planner_clock_calibration(GM_ADDR timing,uint64_t ticks) {
    if ASCEND_IS_AIC {return;}
    int w=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);if(w)return;
    AscendTimerDevice::TikPair(timing);auto start=GetSystemCycle();
    while(GetSystemCycle()-start<ticks){}
    AscendTimerDevice::TokPair(timing);
}
extern "C" __attribute__((visibility("default"))) void planner_link_calibration_do(
        void* stream,uint64_t ffts,uint8_t* base,uint8_t* timing,uint64_t sourceoff,uint64_t targetoff,
        int source,int dest,int rank,int bytes,int repeats){
    planner_link_calibration<<<32,nullptr,stream>>>(ffts,base,timing,sourceoff,targetoff,source,dest,rank,bytes,repeats);
}
extern "C" __attribute__((visibility("default"))) void planner_clock_calibration_do(void* stream,uint8_t* timing,uint64_t ticks){
    planner_clock_calibration<<<1,nullptr,stream>>>(timing,ticks);
}
