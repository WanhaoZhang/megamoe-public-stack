/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#pragma once
#include <cstdint>

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
#define MEGA_MOE_X_REDISPATCH_HOST_DEVICE __host_aicore__
#elif defined(__CCE__)
#define MEGA_MOE_X_REDISPATCH_HOST_DEVICE __host__ __aicore__
#else
#define MEGA_MOE_X_REDISPATCH_HOST_DEVICE
#endif

namespace MegaMoeXRedispatch {
struct ProducerSpan {
    int64_t sourceRow = 0;
    int64_t destinationRow = 0;
    int32_t rows = 0;
};

// Same source/producer partition as native DispatchTokensBackward. Destination
// rows are relative to one local expert; its packed expert prefix is added by
// the caller. Empty producers still take the global read-retirement barrier.
MEGA_MOE_X_REDISPATCH_HOST_DEVICE constexpr ProducerSpan SplitProducer(int64_t sourceBase, int64_t destinationBase,
    int32_t sourceRows, uint32_t producer, uint32_t producers)
{
    if (sourceRows <= 0 || producers == 0 || producer >= producers) return {};
    const int64_t per = (static_cast<int64_t>(sourceRows) + producers - 1) / producers;
    const int64_t begin = static_cast<int64_t>(producer) * per;
    if (begin >= sourceRows) return {};
    const int64_t count = sourceRows - begin < per ? sourceRows - begin : per;
    return {sourceBase + begin, destinationBase + begin, static_cast<int32_t>(count)};
}
}  // namespace MegaMoeXRedispatch
#undef MEGA_MOE_X_REDISPATCH_HOST_DEVICE
