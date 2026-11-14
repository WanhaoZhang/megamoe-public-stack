// CPU test of the production span and ABI headers; no Ascend/NPU dependency.
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>
#include "../../kernels/mega_moe_bf16/include/backward_x_redispatch_layout.h"
#include "../../kernels/mega_moe_bf16/include/mega_moe_shared_expert_types.h"

int main()
{
    using namespace MegaMoeXRedispatch;
    using namespace MegaMoeSharedExpert;
    static_assert(sizeof(SharedExpertBackwardArgs) == 152);
    static_assert(sizeof(SharedExpertBackwardArgsV2) == 184);
    static_assert(sizeof(SharedExpertBackwardArgsV3) == 200);
    static_assert(offsetof(SharedExpertBackwardArgsV3, shared) == 0);
    static_assert(offsetof(SharedExpertBackwardArgsV3, version) == 184);
    static_assert(offsetof(SharedExpertBackwardArgsV3, timer) == 192);
    static_assert(kBackwardRedispatchTimerPhasesV3 == 4);
    static_assert(kBackwardRedispatchTimerLanesV3 == 128);
    for (int rows : {0, 1, 3, 7, 31, 32, 33, 127, 128, 129, 257, 717, 8192, 20480}) {
        for (uint32_t producers : {1U, 2U, 3U, 8U, 16U, 32U, 64U, 128U}) {
            std::vector<int> visits(rows, 0);
            for (uint32_t worker = 0; worker < producers; ++worker) {
                auto p = SplitProducer(101, 37, rows, worker, producers);
                assert(p.rows >= 0);
                for (int i = 0; i < p.rows; ++i) {
                    const auto offset = p.destinationRow + i - 37;
                    assert(offset >= 0 && offset < rows);
                    assert(p.sourceRow + i == 101 + offset);
                    visits[offset]++;
                }
            }
            for (auto n : visits) assert(n == 1);
            assert(SplitProducer(0, 0, rows, producers, producers).rows == 0);
        }
    }
    assert(SplitProducer(0, 0, 4, 0, 0).rows == 0);
    // Independent source payload/probability tags prove both fields use the
    // same expert-major/source-major partition, including uneven producers.
    const int sourceCounts[4][3] = {{1, 0, 9}, {3, 2, 0}, {0, 4, 7}, {5, 1, 2}};
    int localBase = 0;
    for (int expert = 0; expert < 3; ++expert) {
        int destinationPrefix = 0;
        std::vector<int> actualX, actualP, expectedX, expectedP;
        int total = 0;
        for (int source = 0; source < 4; ++source) total += sourceCounts[source][expert];
        actualX.resize(total, -1); actualP.resize(total, -1);
        for (int source = 0; source < 4; ++source) {
            const int rows = sourceCounts[source][expert];
            int sourceBase = 0;
            for (int e = 0; e < expert; ++e) sourceBase += sourceCounts[source][e];
            for (int row = 0; row < rows; ++row) {
                expectedX.push_back(10000*source + 100*expert + row);
                expectedP.push_back(50000 + 1000*source + 10*expert + row);
            }
            for (uint32_t producer = 0; producer < 8; ++producer) {
                const auto p = SplitProducer(sourceBase, destinationPrefix, rows, producer, 8);
                for (int i = 0; i < p.rows; ++i) {
                    const int sourceRow = static_cast<int>(p.sourceRow+i-sourceBase);
                    actualX[p.destinationRow+i] = 10000*source + 100*expert + sourceRow;
                    actualP[p.destinationRow+i] = 50000 + 1000*source + 10*expert + sourceRow;
                }
            }
            destinationPrefix += rows;
        }
        assert(actualX == expectedX && actualP == expectedP);
        localBase += total;
    }
    assert(localBase == 34);
}
