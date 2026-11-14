#pragma once
// Same AscendTimerV2 pairs and bounded 32-worker/5-stage schema as small EP.
#if MEGAFLUX_MOONEP_TIMER
#include "../../common/timer/AscendTimerV2_device.hpp"
__aicore__ inline uint64_t LargePlannerStart(GM_ADDR header,int worker,int stage) {
    auto* h=(__gm__ int64_t*)header+4;
    if(h[0]!=0x4d46504c0001LL)return 0;
    auto address=(uint64_t)(h+16+(worker*5+stage)*8);
    AscendTimerDevice::TikPair((GM_ADDR)address);
    return address;
}
__aicore__ inline void LargePlannerEnd(uint64_t address) {
    if(address)AscendTimerDevice::TokPair((GM_ADDR)address);
}
#define LARGE_PLAN_START(name,stage) const auto name=LargePlannerStart(header,worker,stage)
#define LARGE_PLAN_END(name) LargePlannerEnd(name)
#else
#define LARGE_PLAN_START(name,stage) do{}while(0)
#define LARGE_PLAN_END(name) do{}while(0)
#endif
