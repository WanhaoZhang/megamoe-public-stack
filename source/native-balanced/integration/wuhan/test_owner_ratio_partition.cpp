#define MEGAFLUX_SCHEDULE_INLINE inline
#include "../../kernels/mega_moe_bf16/include/mega_moe_owner_schedule.h"
#include <cassert>
#include <iostream>
#include <vector>

void CheckAiv1Tail(int workers, int tasks) {
    constexpr int ratio = 2;
    std::vector<int> visits(tasks);
    for (int aiv1 = 1; aiv1 < workers; aiv1 += 2) {
        std::vector<int> original(tasks);
        for (int worker : {aiv1 - 1, aiv1}) {
            for (long long ordinal = 0; ; ++ordinal) {
                const auto task = MegaFluxOwnerSchedule::Task(ordinal, worker, workers, ratio);
                if (task >= tasks) break;
                ++original[task];
            }
        }
        long long previous = -1;
        for (long long ordinal = 0; ; ++ordinal) {
            const auto task = MegaFluxOwnerSchedule::Task<true>(ordinal, aiv1, workers, ratio);
            if (task >= tasks) break;
            assert(task > previous);
            previous = task;
            --original[task];
            ++visits[task];
        }
        for (int count : original) assert(count == 0);
    }
    for (int count : visits) assert(count == 1);
}

int main() {
    int cases = 0;
    int aiv1TailCases = 0;
    for (int workers : {2, 4, 8, 16, 32, 48, 64, 72}) {
        for (int ratio : {1, 2}) {
            for (int tasks : {0, 1, 2, 31, 32, 63, 64, 65, 95, 96, 97,
                              127, 128, 129, 1344, 2688, 4032, 5376, 4097}) {
                std::vector<int> seen(tasks, 0);
                for (int w = 0; w < workers; ++w) {
                    long long last = -1;
                    for (long long ordinal = 0;; ++ordinal) {
                        auto t = MegaFluxOwnerSchedule::Task(ordinal, w, workers, ratio);
                        assert(t > last);
                        last = t;
                        if (t >= tasks) break;
                        assert(t >= 0);
                        ++seen[t];
                    }
                }
                for (int count : seen) assert(count == 1);
                ++cases;
                if (ratio == 2) {
                    CheckAiv1Tail(workers, tasks);
                    ++aiv1TailCases;
                }
            }
            // Whole scheduling rounds have exactly the intended early share.
            const int n = (workers / 2) * (ratio + 1) * 7;
            int early = 0, late = 0;
            for (int w = 0; w < workers; ++w)
                for (int o = 0; MegaFluxOwnerSchedule::Task(o,w,workers,ratio) < n; ++o)
                    (w & 1 ? early : late)++;
            assert(early == ratio * late);
        }
    }
    std::cout << "OWNER_PARTITION_PASS " << cases << " exact coverage cases\n";
    std::cout << "OWNER_AIV1_TAIL_PASS " << aiv1TailCases << " ordered exact-coverage cases\n";
}
