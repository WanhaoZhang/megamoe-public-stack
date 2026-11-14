#pragma once

#include <cstdint>

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
#include <kernel_operator.h>
#define BF16_GM_PROGRESS_HD __host_aicore__
#elif defined(__CCE__)
#include <kernel_operator.h>
#define BF16_GM_PROGRESS_HD __host__ __aicore__
#else
#define BF16_GM_PROGRESS_HD
#endif

#ifndef DEEPEP_BF16_GM_SWIGLU_TILE_READY
#define DEEPEP_BF16_GM_SWIGLU_TILE_READY 0
#endif
static_assert(DEEPEP_BF16_GM_SWIGLU_TILE_READY == 0 ||
              DEEPEP_BF16_GM_SWIGLU_TILE_READY == 1,
    "BF16 GM SwiGLU tile readiness must be off(0) or on(1)");

namespace MegaMoeBF16GmTileProgress {

// A private line per expert and AIC producer prevents unrelated writers from
// invalidating a consumer's counter. The paired AIV0 uses the same task order;
// a producer can run ahead without waiting for the consumer to acknowledge.
constexpr uint32_t kMaxProducers = 128;
constexpr uint64_t kSlotBytes = 128;
constexpr uint64_t kWordsPerSlot = kSlotBytes / sizeof(int32_t);

BF16_GM_PROGRESS_HD constexpr uint64_t SlotIndex(
    uint32_t expert, uint32_t producer)
{
    return static_cast<uint64_t>(expert) * kMaxProducers + producer;
}

BF16_GM_PROGRESS_HD constexpr uint64_t ProgressWordOffset(
    uint32_t expert, uint32_t producer)
{
    return SlotIndex(expert, producer) * kWordsPerSlot;
}

BF16_GM_PROGRESS_HD constexpr uint32_t ProducerTaskCount(
    uint32_t totalTasks, uint32_t firstTask, uint32_t producers)
{
    return producers == 0 || firstTask >= totalTasks ? 0U :
        1U + (totalTasks - 1U - firstTask) / producers;
}

BF16_GM_PROGRESS_HD constexpr bool CanStream(
    uint32_t totalTasks, uint32_t producers)
{
    return producers != 0 && producers <= kMaxProducers &&
        (totalTasks / producers + (totalTasks % producers != 0U)) <= 0x7FFFFFFFU;
}

// Producer-local ordinals start at one; zero is the per-call reset state.
// AIV waits for >= rather than == so it remains correct when AIC runs ahead.
BF16_GM_PROGRESS_HD constexpr uint32_t Ordinal(
    uint32_t task, uint32_t firstTask, uint32_t producers)
{
    return producers == 0 || task < firstTask ? 0U :
        1U + (task - firstTask) / producers;
}

BF16_GM_PROGRESS_HD constexpr bool Reached(int32_t published, uint32_t target)
{
    return published >= 0 && static_cast<uint32_t>(published) >= target;
}

}  // namespace MegaMoeBF16GmTileProgress

#undef BF16_GM_PROGRESS_HD
