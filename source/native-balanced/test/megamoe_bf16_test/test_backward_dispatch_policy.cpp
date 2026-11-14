// Host-only selection/layout checks. NPU tests cover publication visibility.
#include "../../kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"
#include <cassert>
#include <cstdint>
#include <iostream>

using namespace MegaMoeBF16Training;

int main()
{
    static_assert(SharedSavedModeBit(kSaveBf16) == 1);
    static_assert(SharedSavedModeBit(kSaveNone) == 2);
    static_assert(SharedSavedModeBit(kSaveMxFp8) == 4);
    static_assert(SharedSavedModeBit(kSaveAuto) == 0);
    constexpr uint64_t tileM = 128;
    static_assert(ForwardDispatchTileReadyLayout::kMaxReadyExperts == 7);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveBf16, 7) == 7);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveBf16, 8) == 0);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveMxFp8, 16) == 0);
    static_assert(DefaultBackwardDispatchSplitExperts(kSaveNone, 7) == 0);
    for (uint32_t mode : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
        for (uint32_t experts : {0U, 1U, 4U, 7U, 8U, 16U, 32U}) {
            const uint32_t k = DefaultBackwardDispatchSplitExperts(mode, experts);
            assert(k == (mode != kSaveNone && experts <= 7 ? experts : 0));
            assert(UseBackwardDispatchRowReady(mode, 0, k, 129, tileM) == (k > 0));
            assert(!UseBackwardDispatchRowReady(mode, 0, k, 128, tileM));
            for (uint64_t ep : {4ULL, 8ULL, 64ULL, 128ULL, 256ULL}) {
                constexpr uint64_t oldBytes = 4099;
                assert(BackwardDispatchProgressLayout::TotalBytes(
                    oldBytes, k, ep, 1, mode) ==
                    (k ? AlignUp(oldBytes, 64) + k * ep * 64 : oldBytes));
            }
        }
    }
    // Explicit debug k remains usable above the default expert cutoff.
    assert(UseBackwardDispatchRowReady(kSaveBf16, 7, 8, 4096, tileM));
    for (uint32_t mode : {kSaveNone, kSaveBf16, kSaveMxFp8}) {
        for (uint64_t rows : {0ULL, 1ULL, 127ULL, 128ULL, 129ULL, 256ULL,
                              16384ULL, 18725ULL, 131072ULL}) {
            for (uint32_t k : {0U, 1U, 4U, 7U}) {
                for (uint32_t ordinal = 0; ordinal < 8; ++ordinal) {
                    const bool expected = mode != kSaveNone && ordinal < k && rows > tileM;
                    assert(UseBackwardDispatchRowReady(mode, ordinal, k, rows, tileM) == expected);
                }
            }
        }
    }
    // A protected first non-empty expert still consumes ordinal 0. The next
    // expert does not get promoted into k=1 merely because the first is small.
    assert(!UseBackwardDispatchRowReady(kSaveBf16, 0, 1, 128, tileM));
    assert(!UseBackwardDispatchRowReady(kSaveBf16, 1, 1, 4096, tileM));
    assert(UseBackwardDispatchRowReady(kSaveBf16, 1, 7, 4096, tileM));
    for (uint64_t ep : {1ULL, 4ULL, 8ULL, 16ULL, 32ULL, 64ULL, 128ULL, 256ULL}) {
        for (uint64_t producers : {1ULL, 4ULL, 8ULL}) {
            for (uint64_t k : {0ULL, 1ULL, 4ULL, 7ULL}) {
                constexpr uint64_t oldBytes = 4099;
                const auto base = k == 0 ? oldBytes : AlignUp(oldBytes, 64);
                assert(BackwardDispatchProgressLayout::TotalBytes(
                    oldBytes, k, ep, producers, kSaveBf16) == base + k * ep * producers * 64);
                assert(BackwardDispatchProgressLayout::TotalBytes(
                    oldBytes, k, ep, producers, kSaveMxFp8) == base + k * ep * producers * 64);
                assert(BackwardDispatchProgressLayout::TotalBytes(
                    oldBytes, k, ep, producers, kSaveNone) == oldBytes);
            }
        }
    }
    // The opt-in is carried by the existing launch argument. Both old-Host /
    // new-SO and new-Host / old-SO combinations retain the legacy path unless
    // the Host has positively negotiated the capability and sets the bit.
    for (uint32_t experts : {0U, 1U, 4U, 7U, 8U, 16U, 32U}) {
        const uint32_t split = DefaultBackwardDispatchSplitExperts(kSaveNone, experts, true);
        assert(split == (experts <= 7 ? experts : 0));
        const uint32_t policy = EncodeBackwardDispatchPolicy(split, true);
        assert(BackwardSplitExperts(policy) == split);
        assert(BackwardNoneGmRequested(policy));
        assert(IsBackwardDispatchPolicyValid(policy, kSaveNone, experts, true));
        assert(!IsBackwardDispatchPolicyValid(policy, kSaveNone, experts, false));
        assert(!IsBackwardDispatchPolicyValid(policy, kSaveBf16, experts, true));
        assert(!IsBackwardDispatchPolicyValid(policy, kSaveMxFp8, experts, true));
        const auto legacy = EncodeBackwardDispatchPolicy(split, false);
        assert(legacy == split && !BackwardNoneGmRequested(legacy));
        assert(IsBackwardDispatchPolicyValid(legacy, kSaveNone, experts, false));
        assert(BackwardUsesGm(kSaveNone, true));
        assert(!BackwardUsesGm(kSaveNone));
        assert(UseBackwardDispatchRowReady(kSaveNone, 0, split, 129, tileM, true) == (split > 0));
        assert(!UseBackwardDispatchRowReady(kSaveNone, 0, split, 128, tileM, true));
        for (uint32_t saved : {kSaveBf16, kSaveMxFp8}) {
            assert(BackwardUsesGm(saved) == BackwardUsesGm(saved, true));
            assert(DefaultBackwardDispatchSplitExperts(saved, experts) ==
                DefaultBackwardDispatchSplitExperts(saved, experts, true));
        }
        for (uint64_t ep : {1ULL, 4ULL, 8ULL, 256ULL}) {
            constexpr uint64_t legacyBytes = 4099;
            const auto expected = split
                ? AlignUp(legacyBytes, 64) + split * ep * 32 * 64 : legacyBytes;
            assert(BackwardDispatchProgressLayout::TotalBytes(
                legacyBytes, split, ep, 32, kSaveNone, true) == expected);
        }
    }
    assert(!IsBackwardDispatchPolicyValid(
        EncodeBackwardDispatchPolicy(5, true), kSaveNone, 4, true));
    assert(BackwardRestoreProjection(0, false) == 1);
    assert(BackwardRestoreProjection(1, false) == 0);
    assert(BackwardRestoreProjection(0, true) == 0);
    assert(BackwardRestoreProjection(1, true) == 1);
    std::cout << "[BACKWARD_DISPATCH_POLICY] PASS single-M, ordinal, modes, EP256 layout, none-GM handshake\n";
}
