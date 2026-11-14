#ifndef DEEPEP_MEGA_MOE_TIMER_H
#define DEEPEP_MEGA_MOE_TIMER_H

// Timer instrumentation is compiled out completely unless ENABLE_TIMER is
// supplied by the build. AscendTimerDevice also accepts a null address, so a
// timer-enabled kernel remains usable through the ordinary compute API.
#if defined(ENABLE_TIMER)
#include "../../common/timer/AscendTimerV2_device.hpp"
#define MEGA_MOE_TIMER_PARAM(name) , AscendTimerDevice &name
#define MEGA_MOE_TIMER_ARG(name) , name
#define MEGA_MOE_TIMER_BLOCK(code) TIMER_BLOCK(code)
#else
#define MEGA_MOE_TIMER_PARAM(name)
#define MEGA_MOE_TIMER_ARG(name)
#define MEGA_MOE_TIMER_BLOCK(code) do { } while (0)
#endif

#endif  // DEEPEP_MEGA_MOE_TIMER_H
