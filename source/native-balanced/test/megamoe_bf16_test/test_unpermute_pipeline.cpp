// Compile with -Itest/megamoe_bf16_test/unpermute_host_stub, WITHOUT CANN.
// Executes the production helper against shuffled MTE2/V/MTE3 queues. It is
// not a device simulator and does not establish device precision/performance.
#include "../../kernels/mega_moe_bf16/include/mega_moe_unpermute.h"

#include <algorithm>
#include <iostream>
#include <numeric>

using MegaMoeBF16Routing::UnpermutePlan;
constexpr uint32_t kUbBytes = 248 * 1024;

template<class T> void Register(std::vector<T> &values)
{
    AscendC::globalRegions.push_back({reinterpret_cast<uintptr_t>(values.data()), values.size() * sizeof(T)});
}

void Check(uint32_t tokens, uint32_t hidden, uint32_t topK, uint32_t cores, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<int32_t> indices(tokens * topK);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<bfloat16_t> inputs(uint64_t{tokens} * topK * hidden);
    // Finite mixed-sign BF16 values spanning exponents. This catches a changed
    // route order or a switch to one FP32 accumulation and a final BF16 cast.
    for (auto &input : inputs)
        input.bits = static_cast<uint16_t>((rng() & 0x807fU) | ((115 + rng() % 24) << 7));
    const bfloat16_t sentinel{0x5a5a};
    std::vector<bfloat16_t> output(uint64_t{tokens} * hidden + 32, sentinel);
    std::vector<bfloat16_t> expected = output;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t col = 0; col < hidden; ++col) {
            bfloat16_t sum = ToBf16(ToFloat(inputs[uint64_t{static_cast<uint32_t>(indices[token * topK])} * hidden + col]) + 0.0F);
            for (uint32_t expert = 1; expert < topK; ++expert)
                sum = ToBf16(ToFloat(sum) + ToFloat(inputs[uint64_t{static_cast<uint32_t>(indices[token * topK + expert])} * hidden + col]));
            expected[16 + uint64_t{token} * hidden + col] = sum;
        }
    }
    AscendC::globalRegions.clear();
    Register(indices);
    Register(inputs);
    Register(output);
    const uint32_t perCore = (tokens + cores - 1) / cores;
    const UnpermutePlan plan(hidden, topK, kUbBytes);
    assert(plan.enabled);
    for (uint32_t core = 0; core < cores; ++core) {
        const uint32_t begin = std::min(tokens, core * perCore);
        const uint32_t count = std::min(perCore, tokens - begin);
        AscendC::scheduler = {};
        AscendC::scheduler.rng.seed(seed + core);
        assert(MegaMoeBF16Routing::TryUnpermute(
            reinterpret_cast<GM_ADDR>(indices.data()), reinterpret_cast<GM_ADDR>(inputs.data()),
            reinterpret_cast<GM_ADDR>(output.data() + 16), hidden, topK, begin, count, kUbBytes));
        assert(!AscendC::scheduler.Pending());
        assert(AscendC::scheduler.indexCopies == (count + plan.batchTokens - 1) / plan.batchTokens);
        assert(AscendC::scheduler.getValues == count * topK);
        assert(AscendC::scheduler.allocatedBytes == (count ? 4 * plan.rowBytes + plan.kIndexBytes : 0));
        // Inactive cores and later core slices must not be touched.
        for (uint64_t i = 0; i < uint64_t{begin + count} * hidden; ++i)
            assert(output[16 + i].bits == expected[16 + i].bits);
        for (uint64_t i = 16 + uint64_t{begin + count} * hidden; i < output.size(); ++i)
            assert(output[i].bits == sentinel.bits);
    }
    for (size_t i = 0; i < output.size(); ++i) assert(output[i].bits == expected[i].bits);
}

int main()
{
    constexpr UnpermutePlan normal(7168, 16, kUbBytes);
    static_assert(normal.enabled && normal.batchTokens == 64);
    static_assert(4 * normal.rowBytes + normal.kIndexBytes == 60 * 1024);
    static_assert(UnpermutePlan(31232, 16, kUbBytes).enabled);
    static_assert(!UnpermutePlan(31248, 16, kUbBytes).enabled);
    static_assert(!UnpermutePlan(0, 16, kUbBytes).enabled);
    static_assert(!UnpermutePlan(17, 16, kUbBytes).enabled);
    static_assert(!UnpermutePlan(7168, 0, kUbBytes).enabled);
    static_assert(!UnpermutePlan(7168, 1025, kUbBytes).enabled);
    static_assert(!UnpermutePlan(UINT32_MAX, 16, kUbBytes).enabled);
    // Rejected plans must not allocate UB or touch any supplied address.
    assert(!MegaMoeBF16Routing::TryUnpermute(nullptr, nullptr, nullptr, 31248, 16, 0, 1, kUbBytes));
    assert(AscendC::scheduler.allocatedBytes == 0);
    Check(64, 7168, 16, 1, 1);   // Exactly the production per-AIV shape.
    Check(1, 31232, 3, 1, 2);    // Exact UB capacity boundary.
    Check(1, 16, 1024, 1, 3);    // Maximum supported TopK.
    uint32_t cases = 3;
    for (uint32_t topK : {1U, 2U, 3U, 7U, 16U, 17U, 127U}) {
        for (uint32_t cores : {1U, 3U, 64U}) {
            for (uint32_t seed = 1; seed <= 4; ++seed) {
                Check(2 * (1024 / topK) + 1, 16, topK, cores, seed);
                ++cases;
            }
        }
        Check(1, 16, topK, 64, topK); // One active core and 63 idle cores.
        ++cases;
    }
    std::cout << "PASS: " << cases << " Unpermute cases; production helper, randomized pipe queues, BF16 bits, tails, guards and capacity\n";
}
