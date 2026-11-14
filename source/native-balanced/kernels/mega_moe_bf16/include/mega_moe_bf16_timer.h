#ifndef DEEPEP_MEGA_MOE_BF16_TIMER_H
#define DEEPEP_MEGA_MOE_BF16_TIMER_H

// Keep every BF16 timer reference behind the device compile-time switch.
// Without -DENABLE_TIMER the timer object, cycle reads, barriers and GM writes
// are all removed by preprocessing.
#if defined(ENABLE_TIMER)
#include "../../common/timer/AscendTimerV2_device.hpp"
#define MEGA_MOE_BF16_TIMER_PARAM(name) , AscendTimerDevice &name
#define MEGA_MOE_BF16_TIMER_ARG(name) , name
#define MEGA_MOE_BF16_TIMER_BLOCK(code) TIMER_BLOCK(code)
#else
#define MEGA_MOE_BF16_TIMER_PARAM(name)
#define MEGA_MOE_BF16_TIMER_ARG(name)
#define MEGA_MOE_BF16_TIMER_BLOCK(code) do { } while (0)
#endif

#endif  // DEEPEP_MEGA_MOE_BF16_TIMER_H
