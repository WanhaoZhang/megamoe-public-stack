// Host-only layout and asynchronous schedule contract checks. This does not
// validate FIX->GM memory ordering; that requires the Ascend precision gate.
#include "../../kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace P = MegaMoeBF16GmTileProgress;
using namespace MegaMoeBF16Training;

static void CheckLayout()
{
    for (uint64_t cap : {0ULL, 1ULL, 193ULL, 20480ULL, 131072ULL}) {
        for (uint64_t n : {512ULL, 6144ULL, 8192ULL}) {
            for (uint64_t experts : {1ULL, 4ULL, 7ULL, 32ULL}) {
                const ForwardDispatchTileReadyLayout prefix(n, experts, cap);
                for (uint32_t saved : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
                    for (bool decoupled : {false, true}) {
                        for (bool allow : {false, true}) {
                            const ForwardGmTileProgressLayout layout(
                                n, experts, cap, saved, decoupled, allow);
                            const uint64_t oldTotal = prefix.totalBytes +
                                ForwardSwigluExtraBytes(n, cap, saved, decoupled);
                            const bool enabled = allow && decoupled && saved == kSaveBf16;
                            assert(layout.enabled == enabled);
                            assert(layout.offset == oldTotal && layout.offset % 128 == 0);
                            assert(layout.slotCount == (enabled ? experts * 128 : 0));
                            assert(layout.bytes == layout.slotCount * 128);
                            assert(layout.totalBytes == oldTotal + layout.bytes);
                            if (!enabled) { assert(layout.bytes == 0); continue; }
                            std::set<uint64_t> words;
                            for (uint32_t e = 0; e < experts; ++e) {
                                for (uint32_t c = 0; c < P::kMaxProducers; ++c) {
                                    const auto word = layout.ProgressWordOffset(e, c);
                                    const auto start = layout.offset + word * sizeof(int32_t);
                                    assert(start >= oldTotal && start % 128 == 0);
                                    assert(start + P::kSlotBytes <= layout.totalBytes);
                                    assert(words.insert(word).second);
                                }
                            }
                            assert(words.size() == layout.slotCount);
                        }
                    }
                }
            }
        }
    }
    const ForwardGmTileProgressLayout main(6144, 4, 20480, kSaveBf16, true, true);
    assert(main.bytes == 65536);  // four experts, 128 producer cache lines each
    const ForwardGmTileProgressLayout def(6144, 4, 20480, kSaveBf16, true);
    assert(def.enabled == (DEEPEP_BF16_GM_SWIGLU_TILE_READY != 0));
}

// Different producer/core progress and start rotations are interleaved. A fast
// producer may finish the next expert while a slow AIV still consumes its prior
// expert; separate expert slots and >= progress must keep all payloads valid.
static void CheckSchedule(uint32_t tasks, uint32_t cores, uint32_t rotation)
{
    constexpr uint32_t experts = 3;
    std::mt19937 random(42U + tasks * 3U + cores + rotation);
    std::vector<uint32_t> produced(experts * cores), consumed(experts * cores);
    std::vector<int32_t> flags(experts * P::kMaxProducers);
    std::vector<uint8_t> gate(experts * tasks), up(experts * tasks);
    std::vector<uint8_t> seen(experts * tasks);
    uint32_t remaining = experts * tasks;
    for (uint32_t pass = 0; remaining != 0 && pass < 1000000U; ++pass) {
        const uint32_t core = random() % cores;
        const uint32_t first = (core + cores - rotation) % cores;
        const uint32_t count = P::ProducerTaskCount(tasks, first, cores);
        for (uint32_t expert = 0; expert < experts; ++expert) {
            const uint32_t slot = expert * cores + core;
            const uint64_t flag = P::SlotIndex(expert, core);
            if (produced[slot] < count && (random() & 1U)) {
                // Reverse logical coordinates stand in for an immutable
                // swizzled schedule: producer and consumer use the same map.
                const uint32_t task = first + produced[slot] * cores;
                const uint32_t tile = tasks - 1U - task;
                const uint32_t element = expert * tasks + tile;
                gate[element] = 1;
                up[element] = 1;
                ++produced[slot];
                flags[flag] = static_cast<int32_t>(P::Ordinal(task, first, cores));
                assert(static_cast<uint32_t>(flags[flag]) == produced[slot]);
            }
            if (consumed[slot] < count &&
                P::Reached(flags[flag], consumed[slot] + 1U)) {
                const uint32_t task = first + consumed[slot] * cores;
                const uint32_t tile = tasks - 1U - task;
                const uint32_t element = expert * tasks + tile;
                assert(gate[element] && up[element] && !seen[element]);
                seen[element] = 1;
                ++consumed[slot];
                --remaining;
            }
        }
    }
    assert(remaining == 0);
    assert(std::all_of(seen.begin(), seen.end(), [](uint8_t x) { return x == 1; }));
}

int main()
{
    CheckLayout();
    assert(!P::CanStream(7, 0) && !P::CanStream(7, 129));
    assert(!P::CanStream(0xFFFFFFFFU, 1));
    assert(!P::CanStream(0xFFFFFFFFU, 2));
    assert(P::CanStream(0xFFFFFFFFU, 3));
    assert(P::Reached(7, 1) && P::Reached(7, 7) && !P::Reached(7, 8));
    assert(!P::Reached(-1, 1));
    for (uint32_t cores : {1U, 28U, 32U, 36U, 128U}) {
        for (uint32_t tasks : {0U, 1U, 31U, 32U, 33U, 88U, 384U, 1009U}) {
            assert(P::CanStream(tasks, cores));
            for (uint32_t rotation : {0U, cores / 2U, cores - 1U}) {
                uint32_t sum = 0;
                for (uint32_t c = 0; c < cores; ++c) {
                    const uint32_t first = (c + cores - rotation) % cores;
                    sum += P::ProducerTaskCount(tasks, first, cores);
                }
                assert(sum == tasks);
                CheckSchedule(tasks, cores, rotation);
            }
        }
    }
    std::cout << "BF16 GM producer tile progress layout/schedule PASS\n";
}
