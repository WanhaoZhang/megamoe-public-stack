#pragma once
#include <cstdint>

// All byte offsets are rank-local offsets within one symmetric allocation.
// This is an additional native ABI; the original MegaMoE ABI is unchanged.
struct MegaMoeReplicaLayout {
    int64_t home, slots, hidden, ffn;
    uint64_t w1, w2, dw1, dw2, counts, histogram, planner_meta, planner_errors, bytes;
    static uint64_t Align(uint64_t x) { return (x + 511) / 512 * 512; }
    MegaMoeReplicaLayout(int64_t e, int64_t s, int64_t h, int64_t f)
        : home(e), slots(s), hidden(h), ffn(f) {
        const uint64_t es = e + s;
        w1 = 0;
        w2 = Align(es * 2 * f * h * 2);
        dw1 = w2 + Align(es * h * f * 2);
        dw2 = dw1 + Align(es * 2 * f * h * 2);
        counts = dw2 + Align(es * h * f * 2);
        histogram = counts + Align(es * 4);
        planner_meta = histogram + 32 * 256 * 4;
        planner_errors = planner_meta + 512;
        // Padded expert-major chunk prefixes and global totals, one writer per
        // complete cache line; planner.cpp derives these offsets after errors.
        bytes = (planner_errors + 32 * 128 + 2 * 32 * 256 * 4 + 2097151) / 2097152 * 2097152;
    }
};
