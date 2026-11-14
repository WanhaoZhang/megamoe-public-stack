// Reuse only histogram exchange and stable per-route prefix machinery.
#ifndef MEGAFLUX_MOONEP_DIRECT_PERMILLE
#define MEGAFLUX_MOONEP_DIRECT_PERMILLE 0
#endif
#define DEEPEP_DEVICE_PLANNER 1
#define DEEPEP_MOONEP_PLANNER 1
#if MEGAFLUX_MOONEP_TIMER
#define DEEPEP_MOONEP_STAGE_TIMER 1
#endif
#define mega_moe_replica_planner_control_kernel mega_moe_moonep_planner_kernel
#define mega_moe_replica_planner_control_do mega_moe_moonep_planner_do
#include "mega_moe_replica_planner_control.cpp"
extern "C" __attribute__((visibility("default"))) int megaflux_moonep_direct_permille() {
    return MEGAFLUX_MOONEP_DIRECT_PERMILLE;
}
extern "C" __attribute__((visibility("default"))) int megaflux_moonep_timer_schema() {
    return MEGAFLUX_MOONEP_TIMER ? 1 : 0;
}
