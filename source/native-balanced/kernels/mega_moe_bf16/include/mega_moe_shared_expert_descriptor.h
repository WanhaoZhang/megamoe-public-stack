#pragma once

#include <stdint.h>

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
#include <kernel_operator.h>
#define MEGA_MOE_SHARED_HOST_DEVICE __host_aicore__
#elif defined(__CCE__)
#include <kernel_operator.h>
#define MEGA_MOE_SHARED_HOST_DEVICE __host__ __aicore__
#else
#define MEGA_MOE_SHARED_HOST_DEVICE
#endif

// Optional forward ABI. The ordinary MegaMoeFullDescriptor and entry point do
// not change. Addresses are canonical contiguous BF16 tensors; weights remain
// three independent framework parameters rather than a concatenated copy.
struct MegaMoeSharedExpertDescriptor {
    uint64_t abi = 0;
    uint64_t weight_gate = 0;
    uint64_t weight_up = 0;
    uint64_t weight_down = 0;
    uint64_t raw_gate = 0;
    uint64_t raw_up = 0;
    uint64_t activation = 0;
    uint64_t ready = 0;
    uint64_t tokens = 0;
    uint64_t hidden = 0;
    uint64_t ffn = 0;
    uint64_t timer = 0;
    uint64_t flags = 0;
    uint64_t reserved = 0;
};

static_assert(sizeof(MegaMoeSharedExpertDescriptor) == 14 * sizeof(uint64_t),
    "shared expert forward ABI must contain fourteen 64-bit words");

namespace MegaMoeSharedExpertLayout {
constexpr uint32_t kAbi = 1;
constexpr uint32_t kMaxCores = 128;
constexpr uint32_t kReadyPhases = 3;
constexpr uint32_t kReadyLineInts = 32;
constexpr uint32_t kReadyElements = kReadyPhases * kMaxCores * kReadyLineInts;
constexpr uint32_t kTimerPhases = 4;
constexpr uint32_t kTimerElements = kTimerPhases * kMaxCores * 2;

// flags=0 retains the original whole-projection pipeline and its layouts.
// flags=1 adds per-GU-tile readiness and per-M-row activation completion after
// the unchanged ready prefix. Raw BF16 G/U/A remain full contiguous matrices.
constexpr uint64_t kFlagTilePipeline = 1;
constexpr uint64_t kFlagDetailTimer = 2;
constexpr uint32_t kTileRows = 128;
constexpr uint32_t kTileColumns = 256;
constexpr uint32_t kDetailTimerOffset = kTimerElements;
constexpr uint32_t kDetailTimerElements = 2 * kMaxCores * 2;
constexpr uint32_t kTileTimerOffset = kDetailTimerOffset + kDetailTimerElements;

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t RowTiles(uint64_t tokens)
{
    return (tokens + kTileRows - 1) / kTileRows;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t ColumnTiles(uint64_t width)
{
    return (width + kTileColumns - 1) / kTileColumns;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t GateUpTasks(uint64_t tokens, uint64_t ffn)
{
    return RowTiles(tokens) * ColumnTiles(ffn);
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t DownTasks(uint64_t tokens, uint64_t hidden)
{
    return RowTiles(tokens) * ColumnTiles(hidden);
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t GateUpReadyOffset(uint64_t task)
{
    return kReadyElements + task * kReadyLineInts;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t ActivationRowReadyOffset(
    uint64_t tokens, uint64_t ffn, uint64_t row)
{
    return kReadyElements + (GateUpTasks(tokens, ffn) + row) * kReadyLineInts;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t ReadyElements(uint64_t tokens, uint64_t ffn)
{
    return kReadyElements + (GateUpTasks(tokens, ffn) + RowTiles(tokens)) * kReadyLineInts;
}

// Each task owns exactly one raw-cycle begin/end pair. GU records enclose a
// consecutive Gate tile + Up tile, never a Down tile or readiness polling.
MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t GateUpTimerOffset(uint64_t task)
{
    return kTileTimerOffset + task * 2;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t SwigluTimerOffset(
    uint64_t tokens, uint64_t ffn, uint64_t task)
{
    return kTileTimerOffset + GateUpTasks(tokens, ffn) * 2 + task * 2;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t DownTimerOffset(
    uint64_t tokens, uint64_t ffn, uint64_t task)
{
    return kTileTimerOffset + GateUpTasks(tokens, ffn) * 4 + task * 2;
}

MEGA_MOE_SHARED_HOST_DEVICE constexpr uint64_t TimerElements(
    uint64_t tokens, uint64_t hidden, uint64_t ffn)
{
    return kTileTimerOffset + GateUpTasks(tokens, ffn) * 4 + DownTasks(tokens, hidden) * 2;
}
}  // namespace MegaMoeSharedExpertLayout

#undef MEGA_MOE_SHARED_HOST_DEVICE
