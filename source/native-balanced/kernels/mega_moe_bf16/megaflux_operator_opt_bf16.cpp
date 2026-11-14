// Isolated, explicitly configured single-operator experiment; never replaces S9.
#if MEGAFLUX_OPT_TIMER
#define DEEPEP_BF16_STAGE_TIMER 1
#endif
#include "megaflux_early_bf16.cpp"
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_operator_opt_version() {return 1;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_operator_opt_flags() {
    return MEGAFLUX_OPT_DB | (DEEPEP_BF16_OWNER_RATIO<<4)
        | (DEEPEP_BF16_OWNER_SCAN<<8) | (MEGAFLUX_OPT_COPY<<16)
        | (MEGAFLUX_OPT_CACHE<<17) | (MEGAFLUX_OPT_TIMER<<18);
}
#if MEGAFLUX_OPT_TIMER
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_stage_timer_version() {return 3;}
#endif
