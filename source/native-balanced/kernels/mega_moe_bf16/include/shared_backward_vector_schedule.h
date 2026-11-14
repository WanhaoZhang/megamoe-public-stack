/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#pragma once
#include <cstdint>

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
#define MEGA_MOE_SHARED_VECTOR_HD __host_aicore__
#elif defined(__CCE__)
#define MEGA_MOE_SHARED_VECTOR_HD __host__ __aicore__
#else
#define MEGA_MOE_SHARED_VECTOR_HD
#endif

namespace MegaMoeSharedExpert {
namespace VectorSchedule {
constexpr uint32_t kMaxElements = 8192;
struct Tile {
    uint32_t row = 0, column = 0, rows = 0, columns = 0;
    MEGA_MOE_SHARED_VECTOR_HD constexpr uint32_t Elements() const { return rows * columns; }
};
MEGA_MOE_SHARED_VECTOR_HD constexpr uint32_t RowsPerTile(uint32_t width)
{
    return width != 0 && width <= kMaxElements / 2 ? 2 : 1;
}
MEGA_MOE_SHARED_VECTOR_HD constexpr uint32_t ColumnTiles(uint32_t width)
{
    return (width + kMaxElements - 1) / kMaxElements;
}
MEGA_MOE_SHARED_VECTOR_HD constexpr uint64_t Count(uint32_t rows, uint32_t width)
{
    if (!rows || !width) return 0;
    return ((uint64_t{rows} + RowsPerTile(width) - 1) / RowsPerTile(width)) * ColumnTiles(width);
}
// Pack at most two complete rows for normal F; split one row when width>UB.
// Callers supply their actual GM row stride, including ABI1 interleaved G/U.
MEGA_MOE_SHARED_VECTOR_HD constexpr Tile At(uint32_t rows, uint32_t width, uint64_t index)
{
    if (index >= Count(rows, width)) return {};
    const uint32_t per = RowsPerTile(width), cols = ColumnTiles(width);
    const uint32_t row = static_cast<uint32_t>(index / cols) * per;
    const uint32_t col = static_cast<uint32_t>(index % cols) * kMaxElements;
    return {row, col, rows - row < per ? rows - row : per,
        width - col < kMaxElements ? width - col : kMaxElements};
}
}  // namespace VectorSchedule
}  // namespace MegaMoeSharedExpert
#undef MEGA_MOE_SHARED_VECTOR_HD
