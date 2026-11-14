/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#ifndef DEEPEP_TRANSPORT_BENCHMARK_H
#define DEEPEP_TRANSPORT_BENCHMARK_H

#include <cstdint>

enum class TransportKind : uint32_t {
    REMOTE_UDMA = 0,
    LOCAL_SDMA = 1,
    REMOTE_MTE = 2,
    LOCAL_MTE = 3,
};

enum class TransportOperation : uint32_t {
    READ = 0,
    WRITE = 1,
};

enum class BenchmarkMode : uint32_t {
    RTT = 0,
    BANDWIDTH = 1,
};

// Each AIV writes one independent result slot. Keep slots on separate 64-byte
// cache lines so concurrent GM stores from adjacent AIVs cannot overwrite a
// neighbour's fields.
struct alignas(64) TransportBenchmarkResult {
    uint64_t totalCycles;
    uint64_t minCycles;
    uint64_t maxCycles;
    uint64_t operations;
    uint64_t startCycle;
    uint64_t endCycle;
};

static_assert(sizeof(TransportBenchmarkResult) == 64,
              "TransportBenchmarkResult must occupy one cache line");

#endif
