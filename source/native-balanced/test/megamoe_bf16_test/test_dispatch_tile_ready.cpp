// Host-only protocol/layout test; this does not validate Ascend memory ordering.
// Build with a C++17 compiler and run the resulting executable.
#include "../../kernels/mega_moe_bf16/include/mega_moe_training_workspace.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using MegaMoeBF16Training::ForwardDispatchTileReadyLayout;
using MegaMoeBF16Training::ForwardDispatchUbLayout;
using MegaMoeBF16Training::ForwardTransientWorkspaceLayout;
using MegaMoeBF16Training::DispatchOutputLayout;
using MegaMoeBF16Training::TrainingSavedStateLayout;
using MegaMoeBF16Training::CompactTokenCountExchangeLayout;
constexpr uint32_t kComputeRows = ForwardDispatchTileReadyLayout::kComputeRows;
constexpr uint32_t kReadyRows = ForwardDispatchTileReadyLayout::kReadyRows;

struct Fragment {
    uint32_t begin;
    uint32_t rows;
    uint32_t source;
    uint32_t sourceRow;
    uint32_t owner;
};

// Reproduce the existing source-rank/core partition, using the production
// fragment-boundary helper. The oracle below is an independent concatenation
// of source-rank rows, including rank-wide capacity clipping.
template<bool WaitForFullExpert>
void CheckExpert(const std::vector<uint32_t>& counts, uint32_t blocksPerRank,
                 uint32_t capacity, std::mt19937& rng)
{
    const uint32_t total = std::accumulate(counts.begin(), counts.end(), 0U);
    const uint32_t validRows = std::min(total, capacity);
    const uint32_t tiles = (validRows + kComputeRows - 1) / kComputeRows;
    const uint32_t readyBlocks = (validRows + kReadyRows - 1) / kReadyRows;
    std::vector<uint64_t> expected;
    for (uint32_t rank = 0; rank < counts.size(); ++rank) {
        for (uint32_t row = 0; row < counts[rank] && expected.size() < validRows; ++row) {
            expected.push_back((static_cast<uint64_t>(rank) << 32) | row);
        }
    }

    std::vector<Fragment> fragments;
    uint32_t rankBase = 0;
    for (uint32_t rank = 0; rank < counts.size(); ++rank) {
        const uint32_t rankRows = rankBase < validRows
            ? std::min(counts[rank], validRows - rankBase) : 0;
        const uint32_t rowsPerCore = (rankRows + blocksPerRank - 1) / blocksPerRank;
        for (uint32_t core = 0; core < blocksPerRank; ++core) {
            const uint32_t coreBegin = core * rowsPerCore;
            if (coreBegin >= rankRows) {
                continue;
            }
            const uint32_t coreRows = std::min(rowsPerCore, rankRows - coreBegin);
            for (uint32_t copied = 0; copied < coreRows;) {
                const uint32_t begin = rankBase + coreBegin + copied;
                const uint32_t rows = WaitForFullExpert ? coreRows - copied
                    : ForwardDispatchTileReadyLayout::FragmentRows(begin, coreRows - copied);
                assert(rows > 0);
                if constexpr (!WaitForFullExpert) {
                    assert(begin / kReadyRows == (begin + rows - 1) / kReadyRows);
                }
                fragments.push_back({begin, rows, rank, coreBegin + copied,
                                     rank * blocksPerRank + core});
                copied += rows;
            }
        }
        rankBase += counts[rank];
    }

    // Arbitrary arrival orders must never make a partially written tile ready.
    // Repeat with dirty counters/storage to verify the next-launch reset model.
    std::vector<uint32_t> ready(readyBlocks, 999);
    std::vector<uint64_t> payload(validRows, UINT64_MAX);
    std::vector<uint64_t> weights(validRows, UINT64_MAX);
    const uint32_t owners = static_cast<uint32_t>(counts.size()) * blocksPerRank;
    uint32_t expertDone = owners; // Dirty state from the preceding launch.
    for (uint32_t launch = 0; launch < 3; ++launch) {
        // Poison unused row counters in expert mode: neither publication nor
        // consumption may rely on them, including on a repeated invocation.
        std::fill(ready.begin(), ready.end(), WaitForFullExpert ? 999U : 0U);
        std::fill(payload.begin(), payload.end(), UINT64_MAX);
        std::fill(weights.begin(), weights.end(), UINT64_MAX);
        expertDone = 0;
        bool expertLatched = false;
        std::vector<uint32_t> remaining(owners, 0);
        for (const auto& fragment : fragments) {
            ++remaining[fragment.owner];
        }
        if constexpr (WaitForFullExpert) {
            for (uint32_t count : remaining) {
                assert(count <= 1); // One complete slice per nonempty owner.
            }
        }
        // Include zero-row partitions in the expert completion target. Retire
        // half before payload arrival and the others after all payload writes.
        uint32_t deferredEmpty = 0;
        for (uint32_t owner = 0; owner < owners; ++owner) {
            if (remaining[owner] == 0) {
                if (owner % 2 == 0) {
                    ++expertDone;
                } else {
                    ++deferredEmpty;
                }
            }
        }
        uint32_t flagReads = 0;
        // Model the AIC gate and verify it against actual payload, independently
        // of the flags. This is a protocol model, not a device execution test.
        const auto canCompute = [&](uint32_t tile) {
            if (!expertLatched) {
                ++flagReads;
                expertLatched = expertDone == owners;
                if (!expertLatched) {
                    if constexpr (WaitForFullExpert) {
                        return false;
                    }
                    ++flagReads;
                    const uint32_t readyBlock = tile * kComputeRows / kReadyRows;
                    const uint32_t target = ForwardDispatchTileReadyLayout::RowsInReadyBlock(
                        readyBlock, validRows);
                    assert(target == std::min(kReadyRows, validRows - readyBlock * kReadyRows));
                    if (ready[readyBlock] != target) {
                        return false;
                    }
                }
            }
            const uint32_t end = std::min(validRows, (tile + 1) * kComputeRows);
            for (uint32_t row = tile * kComputeRows; row < end; ++row) {
                assert(payload[row] == expected[row]);
                assert(weights[row] == (expected[row] ^ 0x5a5aULL));
            }
            return true;
        };
        std::shuffle(fragments.begin(), fragments.end(), rng);
        for (const auto& fragment : fragments) {
            for (uint32_t row = 0; row < fragment.rows; ++row) {
                assert(payload[fragment.begin + row] == UINT64_MAX);
                payload[fragment.begin + row] =
                    (static_cast<uint64_t>(fragment.source) << 32) | (fragment.sourceRow + row);
            }
            // Payload arrival alone cannot release GMM1/SwiGLU while this
            // slice's routed TopK weights have not arrived.
            assert(!canCompute(fragment.begin / kComputeRows));
            for (uint32_t row = 0; row < fragment.rows; ++row) {
                assert(weights[fragment.begin + row] == UINT64_MAX);
                weights[fragment.begin + row] = expected[fragment.begin + row] ^ 0x5a5aULL;
            }
            if constexpr (!WaitForFullExpert) {
                const uint32_t tile = fragment.begin / kReadyRows;
                ready[tile] += fragment.rows;
                assert(ready[tile] <= std::min(kReadyRows, validRows - tile * kReadyRows));
            }
            assert(remaining[fragment.owner] > 0);
            if (--remaining[fragment.owner] == 0) {
                ++expertDone;
            }
            assert(expertDone <= owners);
            if (expertDone == owners) {
                assert(payload == expected);
            }
            if constexpr (WaitForFullExpert) {
                assert(canCompute(fragment.begin / kComputeRows) == (expertDone == owners));
            } else {
                const uint32_t tile = fragment.begin / kReadyRows;
                const uint32_t target = std::min(kReadyRows, validRows - tile * kReadyRows);
                // Both kComputeRows-row compute tiles wait for the same row counter.
                for (uint32_t computeTile = tile * 2;
                     computeTile < std::min(tiles, tile * 2 + 2); ++computeTile) {
                    assert(canCompute(computeTile) == (ready[tile] == target));
                }
                if (ready[tile] == target) {
                    for (uint32_t row = tile * kReadyRows; row < tile * kReadyRows + target; ++row) {
                        assert(payload[row] == expected[row]);
                    }
                }
            }
            // Probe a different, possibly not-yet-arrived input tile as well.
            (void)canCompute(rng() % tiles);
        }
        expertDone += deferredEmpty;
        assert(expertDone == owners);
        assert(payload == expected);
        for (uint32_t tile = 0; tile < readyBlocks; ++tile) {
            assert(ready[tile] == (WaitForFullExpert ? 999U : std::min(kReadyRows, validRows - tile * kReadyRows)));
        }
        if (tiles > 0) {
            assert(canCompute(0));
            assert(expertLatched);
            const uint32_t readsAtCompletion = flagReads;
            for (uint32_t task = 0; task < 64; ++task) {
                assert(canCompute(rng() % tiles));
            }
            assert(flagReads == readsAtCompletion);
        }
    }
}

void CheckLayouts()
{
    constexpr ForwardDispatchUbLayout ep64x32(64, 32, 7168, 2);
    static_assert(ep64x32.syncBytes == 8 * 1024);
    static_assert(ep64x32.prevSumBytes == 128);
    static_assert(ep64x32.totalBytes <= 248 * 1024);
    static_assert(MegaMoeBF16Training::kMaxEpRanks == 1024);
    static_assert(MegaMoeBF16Training::kMaxLocalExperts == 32);
    static_assert(MegaMoeBF16Training::kMaxTotalExperts == 2048);
    static_assert(MegaMoeBF16Training::PeerSyncCountOffset(2) == 64 * 1024);
    static_assert(MegaMoeBF16Training::PeerDataOffset(2) == 72 * 1024);
    static_assert(MegaMoeBF16Training::PeerSyncCountOffset(768) == 64 * 1024);
    static_assert(MegaMoeBF16Training::PeerDataOffset(768) == 72 * 1024);
    static_assert(MegaMoeBF16Training::PeerSyncCountOffset(769) == 64 * 1024);
    static_assert(MegaMoeBF16Training::PeerDataOffset(769) == 72 * 1024);
    static_assert(MegaMoeBF16Training::PeerSyncCountOffset(1024) == 64 * 1024);
    static_assert(MegaMoeBF16Training::PeerDataOffset(1024) == 72 * 1024);
    static_assert(1024 * MegaMoeBF16Training::kPeerRankSyncSlotBytes <=
                  MegaMoeBF16Training::PeerSyncCountOffset(1024));
    constexpr CompactTokenCountExchangeLayout ep8Auto(
        8, 4,
        MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(8, 4));
    constexpr CompactTokenCountExchangeLayout ep8Forced(8, 4, true);
    constexpr CompactTokenCountExchangeLayout ep511Auto(
        511, 8,
        MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(511, 8));
    constexpr CompactTokenCountExchangeLayout ep512Auto(
        512, 4,
        MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(512, 4));
    constexpr CompactTokenCountExchangeLayout ep512Forced(512, 4, true);
    constexpr CompactTokenCountExchangeLayout ep512EightExperts(
        512, 8,
        MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(512, 8));
    constexpr CompactTokenCountExchangeLayout ep1024(
        1024, 2,
        MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(1024, 2));
    constexpr CompactTokenCountExchangeLayout experimental(1024, 2, false);
    static_assert(!ep8Auto.enabled && ep8Auto.tokenCountsOffset == 72 * 1024);
    static_assert(ep8Forced.enabled && ep8Forced.packetBytes == 512);
    static_assert(!ep511Auto.enabled);
    static_assert(ep512Auto.enabled && ep512Auto.packetBytes == 32 * 1024);
    static_assert(ep512Forced.enabled && ep512Forced.packetBytes == 32 * 1024);
    static_assert(ep512Forced.blockSumsBytes == 512);
    static_assert(ep512Forced.tokenCountsOffset == 72 * 1024 + 32 * 1024 + 512);
    static_assert(ep512Forced.tokenCountsBytes == 4 * 1024 * 1024);
    static_assert(ep512EightExperts.enabled);
    static_assert(ep512EightExperts.packetBytes == 32 * 1024);
    static_assert(ep512EightExperts.tokenCountsBytes == 8 * 1024 * 1024);
    static_assert(MegaMoeBF16Training::CanUseCompactTokenCountExchange(1024, 8));
    static_assert(!MegaMoeBF16Training::CanUseCompactTokenCountExchange(512, 9));
    static_assert(ep1024.enabled && ep1024.packetBytes == 64 * 1024);
    static_assert(ep1024.tokenCountsBytes == 8 * 1024 * 1024);
    static_assert(!experimental.enabled &&
                  experimental.tokenCountsOffset == 72 * 1024);
    for (uint64_t experts = 1; experts <= 32; ++experts) {
        const uint64_t worldSize = std::min<uint64_t>(1024, 2048 / experts);
        const ForwardDispatchUbLayout layout(worldSize, experts, 7168, 2);
        const uint64_t usedPrevSumBytes = experts * sizeof(int32_t);
        assert(layout.prevSumBytes >= usedPrevSumBytes);
        assert(layout.prevSumBytes - usedPrevSumBytes < 32);
        assert(layout.prevSumBytes % 32 == 0);
        assert(layout.totalBytes <= 248 * 1024);
    }

    constexpr ForwardTransientWorkspaceLayout legacy(8192, 4, 65536);
    constexpr ForwardDispatchTileReadyLayout ready(8192, 4, 65536);
    static_assert(kReadyRows == 2 * kComputeRows);
    static_assert(ready.flagSlotsPerExpert == 256);
    static_assert(ready.flagCount == 256);
    static_assert(ready.bytes == 16 * 1024);
    static_assert(ready.offset == legacy.totalBytes);
    static_assert(ready.totalBytes == legacy.totalBytes + 16 * 1024);
    static_assert(legacy.swigluToGmm2FlagOffset == 512ULL * 1024 * 1024);
    static_assert(ready.totalBytes - legacy.swigluToGmm2FlagOffset ==
                  16ULL * 1024 + 512);
    static_assert(ForwardDispatchTileReadyLayout::FragmentRows(kReadyRows - 176, 200) == 176);
    static_assert(ForwardDispatchTileReadyLayout::FragmentRows(kReadyRows - 1, 200) == 1);
    static_assert(ForwardDispatchTileReadyLayout::FragmentRows(kReadyRows, 300) == (kReadyRows < 300 ? kReadyRows : 300));
    static_assert(ForwardDispatchTileReadyLayout::RowsInReadyBlock(0, kReadyRows + 1) == kReadyRows);
    static_assert(ForwardDispatchTileReadyLayout::RowsInReadyBlock(1, kReadyRows + 1) == 1);
    static_assert(ForwardDispatchTileReadyLayout::RowsInReadyBlock(1, kReadyRows + kComputeRows) == kComputeRows);
    static_assert(ForwardDispatchTileReadyLayout::UseTileReadyForShape(4096, 16, 4));
    static_assert(!ForwardDispatchTileReadyLayout::UseTileReadyForShape(4096, 16, 8));
    static_assert(!ForwardDispatchTileReadyLayout::UseTileReadyForShape(4096, 16, 16));
    static_assert(!ForwardDispatchTileReadyLayout::UseTileReadyForShape(64, 16, 4));
    for (uint32_t experts : {5U, 6U, 7U}) {
        assert(ForwardDispatchTileReadyLayout::UseTileReadyForShape(4096, 16, experts));
    }
    constexpr uint64_t minRows = 2 * kReadyRows;
    static_assert(ForwardDispatchTileReadyLayout::UseTileReadyForStartupExpert(true, false, minRows));
    static_assert(!ForwardDispatchTileReadyLayout::UseTileReadyForStartupExpert(true, false, minRows - 1));
    static_assert(!ForwardDispatchTileReadyLayout::UseTileReadyForStartupExpert(true, true, 4096));
    static_assert(!ForwardDispatchTileReadyLayout::UseTileReadyForStartupExpert(false, false, 4096));
    static_assert(ForwardDispatchTileReadyLayout::ReadyBlocksPerProducerPartition(
        kReadyRows * 16, 16, 4, 8, 4) == 2);
    static_assert(ForwardDispatchTileReadyLayout::ReadyBlocksPerProducerPartition(
        4096, 16, 4, 512, 1) == 1);
    static_assert(ForwardDispatchTileReadyLayout::ReadyBlocksPerProducerPartition(
        4096, 16, 2, 1024, 1) == 1);
    constexpr TrainingSavedStateLayout saved(4096, 7168, 16, 8, 4, 65536, 8192, false);
    static_assert(saved.ptrAOffset == 0);
    static_assert(saved.cumsumOffset == 896ULL * 1024 * 1024);
    static_assert(saved.legacyBytes - saved.cumsumOffset == 661504);
    constexpr TrainingSavedStateLayout compact(
        4096, 7168, 16, 8, 4, 65536, 8192, false, true, false, false);
    constexpr DispatchOutputLayout dispatch(7168, 65536);
    static_assert(compact.ptrABytes == 0);
    static_assert(compact.dispatchedTopkWeightsBytes == 0);
    static_assert(compact.cumsumOffset == 0);
    static_assert(compact.legacyBytes == 530432);
    static_assert(dispatch.tokensBytes == 896ULL * 1024 * 1024);
    static_assert(dispatch.topkWeightsOffset == dispatch.tokensBytes);
    static_assert(dispatch.topkWeightsBytes == 128ULL * 1024);
    static_assert(dispatch.totalBytes == 896ULL * 1024 * 1024 + 128ULL * 1024);

    for (uint32_t capacity : {1U, 127U, 128U, 129U, 65536U, 524288U}) {
        for (uint32_t worldSize : {2U, 8U, 16U, 512U}) {
            for (uint32_t experts : {1U, 2U, 3U, 4U, 16U}) {
                for (bool saveMxFp8 : {false, true}) {
                    for (bool saveGateUp : {false, true}) {
                        const TrainingSavedStateLayout layout(
                            4096, 7168, 16, worldSize, experts, capacity,
                            8192, saveMxFp8, saveGateUp, saveMxFp8);
                        const uint64_t packedXBytes =
                            MegaMoeBF16Training::AlignUp(
                                static_cast<uint64_t>(capacity) * 7168 * 2,
                                512);
                        assert(layout.ptrAOffset == 0);
                        assert(layout.cumsumOffset == packedXBytes);
                        assert(layout.cumsumOffset <= layout.expandedRowIdxOffset);
                        assert(layout.expandedRowIdxOffset <=
                               layout.expandedRowIdxGatherOffset);
                        assert(layout.expandedRowIdxGatherOffset <=
                               layout.sumBeforeRankOffset);
                        assert(layout.sumBeforeRankOffset <=
                               layout.savedTokenCountsOffset);
                        assert(layout.savedTokenCountsOffset <=
                               layout.dispatchedTopkWeightsOffset);
                        assert(layout.dispatchedTopkWeightsOffset <=
                               layout.backwardProgressOffset);
                        assert(layout.backwardProgressOffset < layout.legacyBytes);
                        assert(layout.legacyBytes <= layout.totalBytes);
                    }
                }
            }
        }
    }

    for (uint32_t capacity : {1U, 127U, 128U, 129U, 191U, 192U, 193U, 255U, 256U, 257U, 383U, 384U, 385U, 767U, 768U, 769U, 65536U, 131072U}) {
        for (uint32_t experts : {1U, 4U, 16U}) {
            const ForwardTransientWorkspaceLayout transient(8192, experts, capacity);
            const ForwardDispatchTileReadyLayout layout(8192, experts, capacity);
            const uint64_t stride = layout.flagSlotsPerExpert;
            const uint64_t ptrA2Bytes = MegaMoeBF16Training::AlignUp(
                static_cast<uint64_t>(capacity) * 4096 * 2, 512);
            const uint64_t legacyFlagBytes = static_cast<uint64_t>(experts) *
                MegaMoeBF16Training::kIntCacheline *
                MegaMoeBF16Training::kInt32Bytes;
            assert(transient.swigluToGmm2FlagOffset == ptrA2Bytes);
            // Host allocates the complete range with torch::empty. The kernel
            // resets both legacy flag arrays at their actual aligned addresses
            // and every appended row-ready counter on each invocation.
            assert(transient.swigluToGmm2FlagOffset + legacyFlagBytes <=
                   transient.dispatchToGmm1FlagOffset);
            assert(transient.dispatchToGmm1FlagOffset + legacyFlagBytes <=
                   transient.totalBytes);
            assert(layout.offset == transient.totalBytes);
            assert(layout.totalBytes == transient.totalBytes + layout.bytes);
            assert(layout.flagCount == stride);
            // Legacy flags are 128-byte aligned; for one expert the tail
            // need not be 512-byte aligned, but every counter cacheline is.
            assert(layout.offset % 128 == 0 && layout.bytes % 512 == 0);
            // tile readiness is enabled for only one startup expert. Whichever local
            // expert is selected reuses this single counter set, and even the
            // worst case (all received rows in that expert) fits.
            const uint64_t last = ((capacity - 1) / kReadyRows) * 64;
            assert(last + 64 <= layout.bytes);
        }
    }
}

void CheckCompactTokenCountExchange(
    uint32_t worldSize, uint32_t expertsPerRank, uint32_t aivCores)
{
    constexpr int32_t arrivalTag = 0x800000;
    constexpr uint32_t packetElements = 16;
    constexpr uint32_t prefixOffset = 8;
    const uint32_t totalElements = worldSize * expertsPerRank;
    std::vector<int32_t> counts(totalElements);
    for (uint32_t index = 0; index < totalElements; ++index) {
        counts[index] = static_cast<int32_t>((index * 17U + index / 5U) % 11U);
    }
    std::vector<std::vector<int32_t>> packets(
        worldSize, std::vector<int32_t>(packetElements, 0));
    const uint32_t destinationsPerCore =
        (worldSize + aivCores - 1U) / aivCores;
    std::vector<int32_t> blockSums(aivCores, 0);
    for (uint32_t core = 0; core < aivCores; ++core) {
        const uint32_t begin = core * destinationsPerCore;
        const uint32_t end = std::min(worldSize, begin + destinationsPerCore);
        for (uint32_t destination = begin; destination < end; ++destination) {
            for (uint32_t expert = 0; expert < expertsPerRank; ++expert) {
                blockSums[core] += counts[destination * expertsPerRank + expert];
            }
        }
    }
    int32_t blockPrefix = 0;
    for (uint32_t core = 0; core < aivCores; ++core) {
        const uint32_t begin = core * destinationsPerCore;
        const uint32_t end = std::min(worldSize, begin + destinationsPerCore);
        int32_t localPrefix = 0;
        for (uint32_t destination = begin; destination < end; ++destination) {
            auto& packet = packets[destination];
            for (uint32_t expert = 0; expert < expertsPerRank; ++expert) {
                const int32_t count = counts[destination * expertsPerRank + expert];
                packet[expert] = count + (expert == 0 ? arrivalTag : 0);
                packet[prefixOffset + expert] = blockPrefix + localPrefix;
                localPrefix += count;
            }
        }
        assert(localPrefix == blockSums[core]);
        blockPrefix += blockSums[core];
    }
    int32_t expectedPrefix = 0;
    for (uint32_t destination = 0; destination < worldSize; ++destination) {
        auto packet = packets[destination];
        assert(packet[0] != 0);
        packet[0] -= arrivalTag;
        for (uint32_t expert = 0; expert < expertsPerRank; ++expert) {
            const int32_t count = counts[destination * expertsPerRank + expert];
            assert(packet[expert] == count);
            assert(packet[prefixOffset + expert] == expectedPrefix);
            expectedPrefix += count;
        }
    }
}

// DMA chunk size is independent of GM layout and Ready publication units.
void CheckDispatchCopyRows()
{
#if defined(DEEPEP_BF16_M192_FORWARD)
    static_assert(ForwardDispatchUbLayout::CopyRowsForForward(2) == 1);
#else
    static_assert(ForwardDispatchUbLayout::CopyRowsForForward(2) == 2);
    static_assert(ForwardDispatchUbLayout::CopyRowsForForward(3) == 3);
#endif
    for (uint64_t ep : {1U, 4U, 8U, 16U, 64U}) {
        for (uint64_t experts : {1U, 3U, 4U, 7U, 16U, 64U}) {
            for (uint64_t k : {1U, 31U, 32U, 33U, 256U, 7168U, 16384U}) {
                const ForwardDispatchUbLayout one(ep, experts, k, 1);
                const ForwardDispatchUbLayout two(ep, experts, k, 2);
                assert(one.syncBytes == two.syncBytes);
                assert(one.prevSumBytes == two.prevSumBytes);
                assert(one.cumsumBytes == two.cumsumBytes);
                assert(one.resetFlagBytes == two.resetFlagBytes);
                assert(one.copyElementsPerPing >= k);
                assert(two.copyElementsPerPing >= 2 * k);
                assert(one.copyBytes <= two.copyBytes);
                assert(two.totalBytes - one.totalBytes ==
                       two.copyBytes - one.copyBytes);
                // Check odd/partial fragments and repeated ping-pong reuse.
                for (uint64_t rows : {1U, 2U, 3U, 63U, 64U, 191U, 192U, 383U, 384U, 769U}) {
                    for (uint64_t chunk : {1U, 2U}) {
                        const ForwardDispatchUbLayout ub(ep, experts, k, chunk);
                        uint64_t next = 0;
                        for (uint64_t i = 0; i < (rows + chunk - 1) / chunk; ++i) {
                            const uint64_t begin = i * chunk;
                            const uint64_t count = std::min(chunk, rows - begin);
                            assert(begin == next && count > 0);
                            assert(count * k <= ub.copyElementsPerPing);
                            next += count;
                        }
                        assert(next == rows);
                    }
                }
            }
        }
    }
    constexpr ForwardDispatchUbLayout one(8, 4, 7168, 1);
    constexpr ForwardDispatchUbLayout two(8, 4, 7168, 2);
    static_assert(one.copyBytes == 28 * 1024);
    static_assert(two.copyBytes == 56 * 1024);
}

int main()
{
    CheckLayouts();
    CheckDispatchCopyRows();
    for (uint32_t aivCores : {56U, 64U, 72U}) {
        CheckCompactTokenCountExchange(512, 4, aivCores);
        CheckCompactTokenCountExchange(1024, 2, aivCores);
        CheckCompactTokenCountExchange(600, 3, aivCores);
        CheckCompactTokenCountExchange(512, 1, aivCores);
        CheckCompactTokenCountExchange(512, 8, aivCores);
    }
    std::mt19937 rng(831);
    // Cross-rank tiles, zero-token ranks/experts, split tails and a hot expert.
    for (const auto& counts : std::vector<std::vector<uint32_t>>{
            {80, 48, 129, 0}, {128, 128, 1, 0}, {255, 1, 128, 0},
            {0, 0, 0, 0}, {0, 1, 0, 0},
            {1, 127, 0, 129}, {0, 16384, 0, 0}, std::vector<uint32_t>(8, 2048)}) {
        for (uint32_t blocks : {1U, 3U, 4U, 7U}) {
            for (uint32_t capacity : {0U, 1U, 127U, 128U, 129U, 191U, 192U, 193U, 255U, 256U, 257U, 383U, 384U, 385U, 767U, 768U, 769U, 1000U, 65536U}) {
                CheckExpert<false>(counts, blocks, capacity, rng);
                CheckExpert<true>(counts, blocks, capacity, rng);
            }
        }
    }
    for (uint32_t ep : {2U, 4U, 8U, 16U, 128U, 256U}) {
        for (uint32_t aic : {28U, 32U, 36U}) {
            for (uint32_t trial = 0; trial < 12; ++trial) {
                std::vector<uint32_t> counts(ep);
                for (auto& count : counts) {
                    count = rng() % 387;
                }
                const uint32_t total = std::accumulate(counts.begin(), counts.end(), 0U);
                const uint32_t capacity = trial % 2 == 0 ? total : rng() % (total + 1);
                CheckExpert<false>(counts, std::max(1U, aic / ep), capacity, rng);
                CheckExpert<true>(counts, std::max(1U, aic / ep), capacity, rng);
            }
        }
    }
    std::cout << "PASS: BF16 " << kReadyRows << "-row readiness / " << kComputeRows << "-row compute, compact token-count pack/unpack"
                 ", 1/2-row Dispatch UB, layout, coverage, arrivals, tails, resets, streaming bypass"
                 ", expert-only publication and payload/weight completion\n";
}
