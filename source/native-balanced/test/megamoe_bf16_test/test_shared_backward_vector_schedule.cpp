#include "../../kernels/mega_moe_bf16/include/shared_backward_vector_schedule.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using namespace MegaMoeSharedExpert::VectorSchedule;

static void Check(uint32_t rows, uint32_t width, uint32_t workers)
{
    const auto count = Count(rows, width);
    std::vector<uint8_t> ownership(count, 0);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> coverage(rows);
    for (uint32_t worker = 0; worker < workers; ++worker) {
        for (uint64_t index = worker; index < count; index += workers) {
            const auto tile = At(rows, width, index);
            assert(++ownership[index] == 1);
            assert(tile.rows > 0 && tile.columns > 0 && tile.Elements() <= kMaxElements);
            assert(tile.row + tile.rows <= rows && tile.column + tile.columns <= width);
            assert(tile.columns % 256 == 0);
            // Both separate V2 planes and interleaved ABI1 G/U are read with
            // a real GM row stride. No packed UB row may cross into Up/Gate.
            for (const uint32_t stride : {width, 2 * width}) {
                for (uint32_t r = 0; r < tile.rows; ++r) {
                    const uint64_t begin = uint64_t{tile.row + r} * stride + tile.column;
                    assert(begin + tile.columns <= uint64_t{tile.row + r} * stride + width);
                }
            }
            for (uint32_t r = tile.row; r < tile.row + tile.rows; ++r)
                coverage[r].emplace_back(tile.column, tile.column + tile.columns);
        }
    }
    assert(std::all_of(ownership.begin(), ownership.end(), [](uint8_t n) { return n == 1; }));
    for (auto& spans : coverage) {
        std::sort(spans.begin(), spans.end());
        uint32_t end = 0;
        for (const auto& span : spans) { assert(span.first == end); end = span.second; }
        assert(end == width);
    }
    assert(At(rows, width, count).Elements() == 0);
}

int main()
{
    static_assert(Count(8192, 3072) == 4096);
    static_assert(At(8192, 3072, 0).Elements() == 6144);
    static_assert(Count(8192, 7168) == 8192);
    static_assert(At(8192, 7168, 0).Elements() == 7168);
    static_assert(3 * 2 * kMaxElements * 2 <= 128 * 1024, "VJP double-buffer UB budget");
    assert(Count(0, 3072) == 0 && Count(257, 0) == 0);
    for (uint32_t rows : {1U, 2U, 63U, 64U, 65U, 127U, 128U, 129U, 257U, 8192U})
        for (uint32_t width : {256U, 512U, 3072U, 4096U, 4352U, 7168U, 8192U, 8448U, 16384U})
            for (uint32_t workers : {1U, 2U, 63U, 64U}) Check(rows, width, workers);
    std::cout << "SHARED_VECTOR_SCHEDULE_PASS\n";
}
