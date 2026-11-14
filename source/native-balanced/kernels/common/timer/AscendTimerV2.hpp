#ifndef ASCEND_DEEPEP_COMMON_ASCEND_TIMER_V2_H
#define ASCEND_DEEPEP_COMMON_ASCEND_TIMER_V2_H

#include <stdint.h>

namespace AscendTimer {

enum FixedTiming {
    KERNEL_TIMING_IDX = 0,
    AIV_INIT_PARAM,
    AIV_INITROUTING_SORT,
    AIV_INITROUTING_EXPERT_TOKEN_COUNT,
    AIV_INITROUTING_GATHEROUT_QUANT,
    AIV_ALLGATHER,
    AIV_WAIT_COMB,
    AIV_UNPERMUTE_WAIT,
    AIV_UNPERMUTE,
    AIC_WAIT_ALLGATHER,
    FIXED_TIMING_COUNT
};

enum DynamicTimingType {
    AIV_DISPATCH,
    AIV_SWIGLU_WAIT,
    AIV_SWIGLU,
    AIV_COMBINE_WAIT,
    AIV_COMBINE,
    AIC_GMM1_WAIT,
    AIC_GMM1,
    AIC_GMM2_WAIT,
    AIC_GMM2,
    AIV_GMM1_PROLOGUE,
    AIV_GMM2_PROLOGUE,
    // Backward per-expert stages (the dynamic iteration is expertIdx).  The
    // BW_ infix avoids colliding with the forward entries above; the trace
    // parsers display them without it (AIC_GMM2GRAD_0..N etc.).
    AIV_BW_DISPATCH,
    AIC_BW_RECOMPUTE,
    AIC_BW_DO_WAIT,
    AIC_BW_GMM2GRAD,
    AIV_BW_SWIGLU_GRAD,
    AIC_BW_GMM1GRAD,
    AIV_BW_COMBINE,
    AIC_BW_DW1,
    AIC_BW_DW2,
    DYNAMIC_TYPE_COUNT_ENUM
};

static constexpr int MAX_DYNAMIC_ITER = 64;
static constexpr int N_TIMING_ITEM_PER_CORE =
    FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE =
    N_TIMING_ITEM_PER_CORE * 2;
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN =
    (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
static constexpr int N_CORE_COUNT = 96;
static constexpr int N_TIMING_COUNTER =
    N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN =
    (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
static constexpr int DYNAMIC_ITER_TOTAL_SIZE =
    DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;
static constexpr int TOTAL_BUFFER_SIZE =
    N_TIMING_COUNTER + DYNAMIC_ITER_TOTAL_SIZE;

}  // namespace AscendTimer

#define TIMER_BLOCK(code) \
    do {                  \
        code;             \
    } while (0)

#endif  // ASCEND_DEEPEP_COMMON_ASCEND_TIMER_V2_H
