#pragma once
#include <cstdint>
#include "mega_moe_capacity.h"

// Existing BF16 compute limits. Planner storage is independent of these caps.
namespace MegaMoeLargeEP {
constexpr int MaxRanks = 256;
constexpr int MaxExperts = 2048;
constexpr int MaxCopies = 2048;
constexpr int MaxLocalSlots = MEGAMOE_MAX_LOCAL_EXPERTS;
constexpr int MaxPhysicalExperts = 2048;
constexpr int Workers = 32;
inline bool SupportedRanks(int64_t p) {
    return (p >= 2 && p <= 8) || p == 16 || p == 32 || p == 64 || p == 128 || p == 256;
}
inline bool SupportedShape(int64_t p, int64_t home, int64_t slots) {
    return SupportedRanks(p) && home > 0 && slots >= 0 && home + slots <= MaxLocalSlots &&
           p * home <= MaxExperts && p * slots <= MaxCopies &&
           p * (home + slots) <= MaxPhysicalExperts;
}
// One symmetric, lazily allocated planner scratch per workspace. All matrices
// are GM spans; no E*P or P*P matrix is placed on an AIV's local stack.
struct PlannerLayout {
    uint64_t histogram, meta, errors, prefixes, totals, quota, cumulative, bytes;
    static uint64_t Align(uint64_t x, uint64_t a = 512) { return (x + a - 1) / a * a; }
    PlannerLayout(int64_t experts, int64_t ranks) {
        histogram = 0;
        // Each AIV owns complete cache lines, including E=16.
        meta = Align(Workers * Align(experts, 32) * 4);
        errors = meta + 512;
        prefixes = errors + Workers * 128;
        totals = prefixes + experts * Workers * 4;
        quota = totals + experts * Workers * 4;
        cumulative = quota + ranks * ranks * 4;
        bytes = Align(cumulative + experts * ranks * 4, 2 * 1024 * 1024);
    }
};
}
