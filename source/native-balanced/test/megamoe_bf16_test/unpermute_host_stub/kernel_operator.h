#pragma once

// Host-only shim for test_unpermute_pipeline.cpp. Each device pipe has a FIFO;
// a randomized scheduler interleaves them and honors explicit event credits.
// This checks software scheduling, not CANN instruction or hardware semantics.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
constexpr int EVENT_ID0 = 0;
constexpr int EVENT_ID1 = 1;
constexpr int PIPE_V = 0;
constexpr int PIPE_ALL = 3;

struct bfloat16_t { uint16_t bits; };
inline float ToFloat(bfloat16_t value)
{
    uint32_t bits = uint32_t{value.bits} << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
inline bfloat16_t ToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return {static_cast<uint16_t>(bits >> 16)};
}

namespace AscendC {
enum class HardEvent { V_MTE2, MTE2_V, MTE3_V, V_MTE3, S_MTE2, MTE2_S };
constexpr int kSource[] = {0, 1, 2, 0, 3, 1};
constexpr int kTarget[] = {1, 0, 0, 2, 1, 3};
struct Scheduler {
    std::array<std::deque<std::function<bool()>>, 3> queues;
    bool credits[6][2] = {};
    std::mt19937 rng{1};
    uint32_t indexCopies = 0;
    uint32_t allocatedBytes = 0;
    uint32_t getValues = 0;
    void Step()
    {
        const unsigned first = rng() % 3;
        for (unsigned n = 0; n < 3; ++n) {
            auto &q = queues[(first + n) % 3];
            if (!q.empty() && q.front()()) {
                q.pop_front();
                return;
            }
        }
        throw std::runtime_error("event deadlock");
    }
    bool Pending() const
    {
        return !queues[0].empty() || !queues[1].empty() || !queues[2].empty();
    }
    void Drain() { while (Pending()) Step(); }
};
inline Scheduler scheduler;

template<HardEvent event> void SetFlag(int slot)
{
    const int e = static_cast<int>(event);
    auto set = [=]() {
        if (scheduler.credits[e][slot]) throw std::runtime_error("duplicate event credit");
        scheduler.credits[e][slot] = true;
        return true;
    };
    if (kSource[e] == 3) set();
    else scheduler.queues[kSource[e]].push_back(set);
}
template<HardEvent event> void WaitFlag(int slot)
{
    const int e = static_cast<int>(event);
    auto wait = [=]() {
        if (!scheduler.credits[e][slot]) return false;
        scheduler.credits[e][slot] = false;
        return true;
    };
    if (kTarget[e] == 3) {
        while (!wait()) scheduler.Step();
    } else {
        scheduler.queues[kTarget[e]].push_back(wait);
    }
}
template<HardEvent event> void SyncFunc() { SetFlag<event>(0); WaitFlag<event>(0); }
template<int pipe> void PipeBarrier()
{
    if constexpr (pipe == PIPE_ALL) scheduler.Drain();
}

struct Region { uintptr_t begin; size_t bytes; };
inline std::vector<Region> globalRegions;
inline void CheckGlobal(const void *ptr, size_t bytes)
{
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (const auto &region : globalRegions) {
        if (address >= region.begin && address - region.begin <= region.bytes &&
            bytes <= region.bytes - (address - region.begin)) return;
    }
    throw std::runtime_error("GM access out of bounds");
}
template<class T> struct GlobalTensor {
    T *ptr = nullptr;
    void SetGlobalBuffer(T *value) { ptr = value; }
    GlobalTensor operator[](uint64_t offset) const { return {ptr + offset}; }
};
template<class T> struct LocalTensor {
    std::shared_ptr<std::vector<uint8_t>> storage;
    size_t offset = 0;
    LocalTensor operator[](uint64_t index) const { return {storage, offset + index * sizeof(T)}; }
    T *Data(size_t count) const
    {
        if (offset > storage->size() || count * sizeof(T) > storage->size() - offset)
            throw std::runtime_error("UB access out of bounds");
        return reinterpret_cast<T *>(storage->data() + offset);
    }
    T GetValue(uint32_t index) const
    {
        ++scheduler.getValues;
        return Data(index + 1)[index];
    }
};
template<int = 0> struct TBuf {
    std::shared_ptr<std::vector<uint8_t>> storage;
    template<class T> LocalTensor<T> Get() { return {storage, 0}; }
};
struct TPipe {
    template<int position> void InitBuffer(TBuf<position> &buf, uint32_t bytes)
    {
        assert(bytes % 32 == 0);
        scheduler.allocatedBytes += bytes;
        buf.storage = std::make_shared<std::vector<uint8_t>>(bytes, 0xa5);
    }
    void Destroy()
    {
        if (scheduler.Pending()) throw std::runtime_error("UB released with pending operations");
        for (const auto &event : scheduler.credits)
            for (bool credit : event)
                if (credit) throw std::runtime_error("unconsumed event credit");
    }
};
struct DataCopyExtParams { uint32_t blockCount, blockLen, srcStride, dstStride, rsv; };
template<class T> struct DataCopyPadExtParams { bool pad; uint32_t left, right; T value; };
template<class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
    DataCopyExtParams params, DataCopyPadExtParams<T>)
{
    assert(params.blockCount == 1 && params.srcStride == 0 && params.dstStride == 0);
    ++scheduler.indexCopies;
    scheduler.queues[1].push_back([=]() {
        CheckGlobal(src.ptr, params.blockLen);
        std::memcpy(dst.Data(params.blockLen / sizeof(T)), src.ptr, params.blockLen);
        return true;
    });
}
template<class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count)
{
    scheduler.queues[1].push_back([=]() {
        CheckGlobal(src.ptr, count * sizeof(T));
        std::memcpy(dst.Data(count), src.ptr, count * sizeof(T));
        return true;
    });
}
template<class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count)
{
    scheduler.queues[2].push_back([=]() {
        CheckGlobal(dst.ptr, count * sizeof(T));
        std::memcpy(dst.ptr, src.Data(count), count * sizeof(T));
        return true;
    });
}
inline void Adds(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src, float scalar, uint32_t count)
{
    scheduler.queues[0].push_back([=]() {
        for (uint32_t i = 0; i < count; ++i) dst.Data(count)[i] = ToBf16(ToFloat(src.Data(count)[i]) + scalar);
        return true;
    });
}
inline void Add(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> a,
    LocalTensor<bfloat16_t> b, uint32_t count)
{
    scheduler.queues[0].push_back([=]() {
        for (uint32_t i = 0; i < count; ++i)
            dst.Data(count)[i] = ToBf16(ToFloat(a.Data(count)[i]) + ToFloat(b.Data(count)[i]));
        return true;
    });
}
}  // namespace AscendC
