/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef MEGA_MOE_SHARED_EXPERT_TYPES_H
#define MEGA_MOE_SHARED_EXPERT_TYPES_H

#include <cstdint>

namespace MegaMoeSharedExpert {

constexpr uint32_t kBackwardVersion = 1;
constexpr uint32_t kMaxParticipants = 64;
constexpr uint32_t kControlPhases = 7;
constexpr uint32_t kControlLineWords = 16;
constexpr uint32_t kControlWords = kControlPhases * kMaxParticipants * kControlLineWords;
constexpr uint32_t kControlBytes = kControlWords * sizeof(int32_t);

// Device addresses only. Passed by value by the dedicated shared-backward
// entry, so no asynchronous host metadata allocation is required.
struct SharedExpertBackwardArgs {
    uint64_t x, gu, a, wg, wu, wd;
    uint64_t dwg, dwu, dwd;
    uint64_t da, dg, du, dxg, dxu, control;
    uint32_t m, hidden, ffn, epoch;
    float limit;
    uint32_t version, bytes, flags;
};
static_assert(sizeof(SharedExpertBackwardArgs) == 152, "shared backward ABI changed");

// V2 is a separate exported entry. Its prefix retains the complete ABI1
// descriptor (including version/bytes/flags); the ABI1 gu pointer is unused.
// Independent contiguous planes avoid a row-interleaved GU materialization.
constexpr uint32_t kBackwardVersionV2 = 2;
struct SharedExpertBackwardArgsV2 {
    SharedExpertBackwardArgs base;
    uint64_t raw_gate, raw_up;
    uint32_t version, recompute;
    uint64_t timer;
};
static_assert(sizeof(SharedExpertBackwardArgsV2) == 184, "shared backward V2 ABI changed");
constexpr uint32_t kBackwardTimerPhasesV2 = 10;
constexpr uint32_t kBackwardTimerLanesV2 = 128;

// Explicit original-X redispatch entry. ABI1/2 retain their complete layouts
// and flags==0 contract. Only the independent ABI3 entry interprets flags=1.
constexpr uint32_t kBackwardVersionV3 = 3;
constexpr uint32_t kBackwardRedispatchFlag = 1;
// Same integer X/prob transport, without shared expert math or shared planes.
constexpr uint32_t kBackwardRoutedOnlyRedispatchFlag = 2;
struct SharedExpertBackwardArgsV3 {
    SharedExpertBackwardArgsV2 shared;
    uint32_t version, flags;
    uint64_t timer;
};
static_assert(sizeof(SharedExpertBackwardArgsV3) == 200, "shared backward V3 ABI changed");
constexpr uint32_t kBackwardRedispatchTimerPhasesV3 = 4;
constexpr uint32_t kBackwardRedispatchTimerLanesV3 = 128;
constexpr uint32_t kBackwardRedispatchTimerPhases = kBackwardRedispatchTimerPhasesV3;
constexpr uint32_t kBackwardRedispatchTimerLanes = kBackwardRedispatchTimerLanesV3;

}  // namespace MegaMoeSharedExpert
#endif
