#if MEGAFLUX_OPT_DB
#define DEEPEP_BF16_OWNER_DB 1
#endif
#if MEGAFLUX_OPT_TIMER
#define DEEPEP_BF16_STAGE_TIMER 1
#endif
#include "megaflux_early_bf16_backward.cpp"
