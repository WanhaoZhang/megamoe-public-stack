#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#include "kernels/mega_moe_bf16_udma/include/mega_moe_bf16_udma_tiling_host.h"

using MegaMoeBF16Training::TrainingSavedStateLayout;
using MegaMoeBF16Training::UDMAForwardTransientWorkspaceLayout;

int main()
{
    constexpr UDMAForwardTransientWorkspaceLayout current(8192, 4, 65536);
    static_assert(current.totalBytes == 536879616);
    static_assert(537404928 - current.totalBytes == 513 * 1024);
    constexpr TrainingSavedStateLayout defaultSaved(4096, 7168, 16, 8, 4, 65536, 8192, false);
    static_assert(defaultSaved.dispatchedTopkWeightsOffset == 940053504);
    static_assert(defaultSaved.backwardProgressOffset == 940184576);
    static_assert(defaultSaved.legacyBytes == 940185600);
    static_assert(defaultSaved.gateOffset == 940185600);
    static_assert(defaultSaved.gateUpPlaneBytes == 512ULL * 1024 * 1024);
    static_assert(defaultSaved.upOffset == defaultSaved.gateOffset + defaultSaved.gateUpPlaneBytes);
    static_assert(defaultSaved.totalBytes == defaultSaved.legacyBytes + 1024ULL * 1024 * 1024);
    // Moving packed weights from ordinary/MGroup transient to saved state
    // changes lifetime, not total allocation size or the UDMA activation area.
    constexpr MegaMoeBF16Training::ForwardTransientWorkspaceLayout transient(8192, 4, 65536);
    static_assert(transient.totalBytes == 536871424);
    const auto extra = MegaMoeBF16UDMAHost::CalcExtraBufferSize(4096, 16, 8, 4, 65536, 7168);
    assert(extra.peerBytes == 939656192);

    // Partial row groups and odd expert counts exercise both rounding rules.
    for (uint64_t rows : {1ULL, 127ULL, 128ULL, 129ULL, 255ULL, 257ULL, 65536ULL}) {
        for (uint64_t experts : {1ULL, 3ULL, 4ULL, 16ULL}) {
            const UDMAForwardTransientWorkspaceLayout layout(8192, experts, rows);
            const uint64_t activationBytes = rows * 4096 * 2;
            const uint64_t flagBytes = experts * 16 * 4;
            const uint64_t readyBytes = experts * ((rows + 127) / 128) * 4;
            assert(layout.ptrA2Offset == 0);
            assert(layout.swigluToGmm2FlagOffset >= activationBytes);
            assert(layout.dispatchToGmm1FlagOffset >=
                   layout.swigluToGmm2FlagOffset + flagBytes);
            assert(layout.combineReadyOffset >=
                   layout.dispatchToGmm1FlagOffset + flagBytes);
            assert(layout.totalBytes >= layout.combineReadyOffset + readyBytes);
            assert(layout.swigluToGmm2FlagOffset % 128 == 0);
            assert(layout.dispatchToGmm1FlagOffset % 128 == 0);
            assert(layout.combineReadyOffset % 64 == 0);
            assert(layout.totalBytes - layout.combineReadyOffset - readyBytes < 512);
            const TrainingSavedStateLayout saved(33, 256, 4, 2, experts, rows, 8192, false);
            const uint64_t weightsEnd = saved.dispatchedTopkWeightsOffset + rows * 2;
            assert(saved.dispatchedTopkWeightsOffset >= saved.savedTokenCountsOffset +
                2 * MegaMoeBF16Training::AlignUp(2 * experts, 128) * 4);
            assert(saved.backwardProgressOffset >= weightsEnd);
            assert(saved.backwardProgressOffset - weightsEnd < 512);
            assert(saved.legacyBytes == saved.backwardProgressOffset + 1024);
            assert(saved.gateOffset >= saved.legacyBytes && saved.gateOffset % 512 == 0);
            assert(saved.upOffset == saved.gateOffset + saved.gateUpPlaneBytes);
            assert(saved.totalBytes == saved.upOffset + saved.gateUpPlaneBytes);
        }
    }

    // Routing table sizes may grow independently of received activation rows.
    // They must not increase the transient allocation once stored persistently.
    for (uint64_t m : {33ULL, 4096ULL, 16384ULL}) {
        for (uint64_t ep : {2ULL, 8ULL, 64ULL}) {
            const TrainingSavedStateLayout saved(m, 7168, 16, ep, 4, 65536, 8192, false);
            const uint64_t routingBytes = saved.savedTokenCountsOffset - saved.cumsumOffset;
            const uint64_t legacyTransient = current.totalBytes + routingBytes;
            assert(legacyTransient > current.totalBytes);
            assert(routingBytes >= 2 * m * 16 * sizeof(int32_t));
        }
    }
}
