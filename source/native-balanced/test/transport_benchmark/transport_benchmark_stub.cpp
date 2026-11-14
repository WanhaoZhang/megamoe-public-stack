/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

#include "transport_benchmark.h"

extern "C" void transport_benchmark_kernel_do(
    uint32_t blockDim, void *stream, uint8_t *data,
    uint8_t *results,
    uint32_t transport, uint32_t operation, uint32_t mode,
    uint32_t messageBytes, uint32_t strideBytes, uint64_t destinationBase,
    uint32_t warmup, uint32_t loops, uint32_t batchSize,
    uint32_t activeRanks, uint32_t useMte3);

extern "C" void transport_benchmark_udma_qp_kernel_do(
    uint32_t blockDim, void *stream, uint8_t *data, uint8_t *results,
    uint32_t operation, uint32_t rowBytes, uint32_t rowCount,
    uint64_t destinationBase, uint32_t warmupOperations,
    uint32_t measuredGroups, uint32_t bidirectional);

extern "C" void transport_benchmark_remote_mte_bi_rows_kernel_do(
    uint32_t blockDim, void *stream, uint8_t *data, uint8_t *results,
    uint32_t operation, uint32_t rowBytes, uint32_t totalRows,
    uint64_t destinationBase, uint32_t warmupOperations,
    uint32_t measuredGroups);

namespace {

constexpr uint64_t kLocalMemBytes = 3ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t kCycleNs = 1ULL;
constexpr uint32_t kValidationBytes = 32U;
constexpr uint32_t kRawRttMaxBytes = 64U * 1024U;

aclshmemx_uniqueid_t gDefaultUid{};

struct CaseDefinition {
    TransportKind transport;
    TransportOperation operation;
    const char *transportName;
    const char *operationName;
    bool bidirectional;
};

constexpr CaseDefinition kCases[] = {
    {TransportKind::REMOTE_UDMA, TransportOperation::READ, "remote_udma", "read", false},
    {TransportKind::REMOTE_UDMA, TransportOperation::WRITE, "remote_udma", "write", false},
    {TransportKind::REMOTE_UDMA, TransportOperation::READ, "remote_udma_bi", "read", true},
    {TransportKind::REMOTE_UDMA, TransportOperation::WRITE, "remote_udma_bi", "write", true},
    {TransportKind::LOCAL_SDMA, TransportOperation::READ, "local_sdma", "read", false},
    {TransportKind::LOCAL_SDMA, TransportOperation::WRITE, "local_sdma", "write", false},
    {TransportKind::REMOTE_MTE, TransportOperation::READ, "remote_mte", "read", false},
    {TransportKind::REMOTE_MTE, TransportOperation::WRITE, "remote_mte", "write", false},
    {TransportKind::REMOTE_MTE, TransportOperation::READ, "remote_mte_bi", "read", true},
    {TransportKind::REMOTE_MTE, TransportOperation::WRITE, "remote_mte_bi", "write", true},
    {TransportKind::LOCAL_MTE, TransportOperation::READ, "local_mte", "read", false},
    {TransportKind::LOCAL_MTE, TransportOperation::WRITE, "local_mte", "write", false},
};

const char *ModeName(BenchmarkMode mode)
{
    if (mode == BenchmarkMode::RTT) {
        return "rtt";
    }
    return "bandwidth";
}

bool CheckAcl(aclError status, const char *operation)
{
    if (status == ACL_ERROR_NONE) {
        return true;
    }
    std::cerr << "[ERROR] " << operation << " failed, acl error " << status << '\n';
    return false;
}

uint32_t ParseUint32(const char *value, const char *name, bool allowZero = false)
{
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value[0] == '\0' || end == nullptr || *end != '\0' ||
        (!allowZero && parsed == 0UL) ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "[ERROR] Invalid " << name << ": " << value << '\n';
        std::exit(2);
    }
    return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> ParseSizes(const char *value, const char *name)
{
    std::vector<uint32_t> sizes;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            std::cerr << "[ERROR] Empty entry in " << name << '\n';
            std::exit(2);
        }
        const uint32_t bytes = ParseUint32(token.c_str(), name);
        if (bytes % 32U != 0U) {
            std::cerr << "[ERROR] " << name
                      << " entries must be multiples of 32 bytes: " << bytes << '\n';
            std::exit(2);
        }
        sizes.push_back(bytes);
    }
    if (sizes.empty()) {
        std::cerr << "[ERROR] " << name << " must not be empty\n";
        std::exit(2);
    }
    return sizes;
}

std::vector<uint32_t> ParseCounts(const char *value, const char *name)
{
    std::vector<uint32_t> counts;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            std::cerr << "[ERROR] Empty entry in " << name << '\n';
            std::exit(2);
        }
        counts.push_back(ParseUint32(token.c_str(), name));
    }
    if (counts.empty()) {
        std::cerr << "[ERROR] " << name << " must not be empty\n";
        std::exit(2);
    }
    return counts;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1ULL) / alignment * alignment;
}

uint8_t RankPattern(uint32_t rank)
{
    return static_cast<uint8_t>(0x21U + (rank * 37U) % 173U);
}

class Resources {
public:
    ~Resources()
    {
        // Once an AI Core task faults, subsequent ACL/SHMEM teardown calls can
        // enter the already-faulted device context and crash the host process.
        // The process is exiting, so let the runtime reclaim these resources.
        if (deviceFaulted) {
            return;
        }
        if (results != nullptr) {
            aclrtFree(results);
        }
        if (shmemInitialized) {
            if (data != nullptr) {
                aclshmem_free(data);
            }
            aclshmem_finalize();
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(static_cast<int32_t>(rank));
        }
        if (aclInitialized) {
            aclFinalize();
        }
    }

    uint32_t rank = 0U;
    bool aclInitialized = false;
    bool deviceSet = false;
    bool shmemInitialized = false;
    bool deviceFaulted = false;
    aclrtStream stream = nullptr;
    uint8_t *data = nullptr;
    TransportBenchmarkResult *results = nullptr;
};

bool ShouldValidateCase(
    uint32_t rank, const CaseDefinition &definition,
    uint32_t activeRanks)
{
    if (definition.transport == TransportKind::REMOTE_UDMA ||
        definition.transport == TransportKind::REMOTE_MTE) {
        if (definition.operation == TransportOperation::READ) {
            return rank < activeRanks;
        }
        return activeRanks == 1U ? rank == 1U : rank < activeRanks;
    }
    return rank < activeRanks;
}

bool ValidateCase(
    uint32_t rank, uint32_t worldSize, const CaseDefinition &definition,
    uint32_t activeRanks, uint32_t blocks,
    uint32_t messageBytes, uint32_t strideBytes, uint64_t destinationBase,
    uint8_t *data)
{
    const bool shouldValidate =
        ShouldValidateCase(rank, definition, activeRanks);
    uint32_t expectedRank = rank;
    if (definition.transport == TransportKind::REMOTE_UDMA ||
        definition.transport == TransportKind::REMOTE_MTE) {
        if (definition.operation == TransportOperation::READ) {
            expectedRank = (rank + 1U) % worldSize;
        } else if (activeRanks == 1U) {
            expectedRank = 0U;
        } else {
            expectedRank = (rank + worldSize - 1U) % worldSize;
        }
    }

    if (!shouldValidate) {
        return true;
    }

    const size_t sampleBytes = std::min<size_t>(messageBytes, kValidationBytes);
    std::vector<uint8_t> sample(sampleBytes);
    const uint8_t expected = RankPattern(expectedRank);
    for (uint32_t block = 0; block < blocks; ++block) {
        uint8_t *destination = data + destinationBase +
            static_cast<uint64_t>(block) * strideBytes;
        if (!CheckAcl(aclrtMemcpy(
                sample.data(), sample.size(), destination, sample.size(),
                ACL_MEMCPY_DEVICE_TO_HOST), "validation memcpy head")) {
            return false;
        }
        if (!std::all_of(sample.begin(), sample.end(),
                [expected](uint8_t value) { return value == expected; })) {
            return false;
        }
        const bool headOnly =
            definition.transport == TransportKind::LOCAL_MTE &&
            definition.operation == TransportOperation::READ;
        if (messageBytes > sampleBytes && !headOnly) {
            if (!CheckAcl(aclrtMemcpy(
                    sample.data(), sample.size(),
                    destination + messageBytes - sampleBytes, sample.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST), "validation memcpy tail")) {
                return false;
            }
            if (!std::all_of(sample.begin(), sample.end(),
                    [expected](uint8_t value) { return value == expected; })) {
                return false;
            }
        }
    }
    return true;
}

bool ValidateUdmaQpCase(
    uint32_t rank, TransportOperation operation, bool bidirectional,
    uint32_t rowBytes,
    uint32_t rowCount, uint64_t destinationBase, uint8_t *data)
{
    const bool shouldValidate = bidirectional ||
        (operation == TransportOperation::READ && rank == 0U) ||
        (operation == TransportOperation::WRITE && rank == 1U);
    if (!shouldValidate) {
        return true;
    }

    const uint64_t transferBytes64 =
        static_cast<uint64_t>(rowBytes) * rowCount;
    if (transferBytes64 > std::numeric_limits<size_t>::max()) {
        return false;
    }
    const size_t transferBytes = static_cast<size_t>(transferBytes64);
    std::vector<uint8_t> actual(transferBytes);
    if (!CheckAcl(aclrtMemcpy(
            actual.data(), actual.size(), data + destinationBase,
            actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
            "UDMA QP validation memcpy")) {
        return false;
    }

    // Every validating destination contains the other rank's source pattern:
    // rank0 for unidirectional GET, rank1 for unidirectional PUT, and both
    // ranks for bidirectional GET/PUT.
    const uint32_t expectedRank = 1U - rank;
    const uint8_t expected = RankPattern(expectedRank);
    return std::all_of(
        actual.begin(), actual.end(),
        [expected](uint8_t value) { return value == expected; });
}

int RunUdmaQpBenchmark(int argc, char **argv)
{
    if (argc != 10) {
        std::cerr
            << "Usage: transport_benchmark udma_qp <rank> <world_size> <ip_port> "
            << "<row_sizes_csv> <row_counts_csv> <qp_count> "
            << "<warmup_operations> <measured_groups>\n";
        return 2;
    }

    const uint32_t rank = ParseUint32(argv[2], "rank", true);
    const uint32_t worldSize = ParseUint32(argv[3], "world_size");
    const char *ipPort = argv[4];
    const std::vector<uint32_t> rowSizes =
        ParseSizes(argv[5], "row_sizes");
    const std::vector<uint32_t> rowCounts =
        ParseCounts(argv[6], "row_counts");
    const uint32_t qpCount = ParseUint32(argv[7], "qp_count");
    const uint32_t warmupOperations =
        ParseUint32(argv[8], "warmup_operations", true);
    const uint32_t measuredGroups =
        ParseUint32(argv[9], "measured_groups");
    if (rank >= worldSize || worldSize != 2U) {
        std::cerr << "[ERROR] UDMA QP uni/bi benchmark requires "
                     "world_size == 2 and 0 <= rank < world_size\n";
        return 2;
    }
    if (qpCount > ACLSHMEM_MAX_QP_NUM) {
        std::cerr << "[ERROR] qp_count exceeds ACLSHMEM_MAX_QP_NUM: "
                  << qpCount << '\n';
        return 2;
    }
    uint64_t maxTransferBytes = 0ULL;
    for (uint32_t rowBytes : rowSizes) {
        for (uint32_t rowCount : rowCounts) {
            const uint64_t transferBytes =
                static_cast<uint64_t>(rowBytes) * rowCount;
            maxTransferBytes = std::max(maxTransferBytes, transferBytes);
        }
    }
    const uint64_t destinationBase = AlignUp(maxTransferBytes, 4096ULL);
    const uint64_t dataBytes64 = destinationBase * 2ULL;
    if (dataBytes64 > kLocalMemBytes / 2ULL ||
        dataBytes64 > std::numeric_limits<size_t>::max()) {
        std::cerr << "[ERROR] Requested UDMA QP buffers are too large: "
                  << dataBytes64 << " bytes\n";
        return 2;
    }
    const size_t dataBytes = static_cast<size_t>(dataBytes64);
    const size_t resultBytes = static_cast<size_t>(qpCount) *
        sizeof(TransportBenchmarkResult);

    Resources resources;
    resources.rank = rank;
    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.aclInitialized = true;
    if (!CheckAcl(aclrtSetDevice(static_cast<int32_t>(rank)), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    aclshmemx_init_attr_t attributes{};
    if (test_set_attr(
            static_cast<int32_t>(rank), static_cast<int32_t>(worldSize),
            kLocalMemBytes, ipPort, gDefaultUid, &attributes) != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] test_set_attr failed\n";
        return 1;
    }
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    const int qpStatus =
        aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qpCount);
    if (qpStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_set_qp_num failed, status "
                  << qpStatus << '\n';
        return 1;
    }
    const int initStatus =
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed, status "
                  << initStatus << '\n';
        return 1;
    }
    resources.shmemInitialized = true;

    resources.data = static_cast<uint8_t *>(aclshmem_malloc(dataBytes));
    if (resources.data == nullptr) {
        std::cerr << "[ERROR] aclshmem_malloc failed for "
                  << dataBytes << " bytes\n";
        return 1;
    }
    if (!CheckAcl(aclrtMalloc(
            reinterpret_cast<void **>(&resources.results), resultBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc UDMA QP results")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemset(
            resources.data, static_cast<size_t>(destinationBase),
            RankPattern(rank), static_cast<size_t>(destinationBase)),
            "initialize UDMA QP source")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemset(
            resources.data + destinationBase,
            static_cast<size_t>(destinationBase), 0,
            static_cast<size_t>(destinationBase)),
            "initialize UDMA QP destination")) {
        return 1;
    }
    aclshmem_barrier_all();

    bool allValid = true;
    for (bool bidirectional : {false, true}) {
        const char *directionName = bidirectional ? "bi" : "uni";
        for (TransportOperation operation : {
                 TransportOperation::READ, TransportOperation::WRITE}) {
            const char *operationName = operation == TransportOperation::READ ?
                (bidirectional ? "read_bi" : "read") :
                (bidirectional ? "write_bi" : "write");
            for (uint32_t rowBytes : rowSizes) {
                for (uint32_t rowCount : rowCounts) {
                    // SHMEM keeps the requested QP count, but a row sweep does
                    // not need to launch AIVs that have no row to transfer.
                    // Example: 16 rows with qpCount=28 uses QP/AIV 0..15.
                    const uint32_t activeQpCount =
                        std::min(qpCount, rowCount);
                    const uint64_t transferBytes =
                        static_cast<uint64_t>(rowBytes) * rowCount;
                    if (!CheckAcl(aclrtMemset(
                            resources.data + destinationBase,
                            static_cast<size_t>(transferBytes), 0,
                            static_cast<size_t>(transferBytes)),
                            "clear UDMA QP destination")) {
                        return 1;
                    }
                    if (!CheckAcl(aclrtMemset(
                            resources.results, resultBytes, 0, resultBytes),
                            "clear UDMA QP results")) {
                        return 1;
                    }
                    aclshmem_barrier_all();

                    transport_benchmark_udma_qp_kernel_do(
                        activeQpCount, resources.stream, resources.data,
                        reinterpret_cast<uint8_t *>(resources.results),
                        static_cast<uint32_t>(operation), rowBytes, rowCount,
                        destinationBase, warmupOperations, measuredGroups,
                        bidirectional ? 1U : 0U);
                    if (!CheckAcl(
                            aclrtSynchronizeStream(resources.stream),
                            "UDMA QP benchmark kernel")) {
                        resources.deviceFaulted = true;
                        return 1;
                    }
                    aclshmem_barrier_all();

                const bool valid = ValidateUdmaQpCase(
                    rank, operation, bidirectional, rowBytes, rowCount,
                    destinationBase, resources.data);
                allValid = allValid && valid;
                const bool shouldValidate = bidirectional ||
                    (operation == TransportOperation::READ && rank == 0U) ||
                    (operation == TransportOperation::WRITE && rank == 1U);
                if (shouldValidate) {
                    std::cout << "[VERIFY] rank=" << rank
                              << " transport=remote_udma_qp"
                              << " op=" << operationName
                              << " row_bytes=" << rowBytes
                              << " rows=" << rowCount
                              << " qp_count=" << qpCount
                              << " direction=" << directionName
                              << " status=" << (valid ? "PASS" : "FAIL")
                              << '\n';
                }

                if (!bidirectional && rank != 0U) {
                    continue;
                }

                std::vector<TransportBenchmarkResult> hostResults(activeQpCount);
                if (!CheckAcl(aclrtMemcpy(
                        hostResults.data(),
                        hostResults.size() * sizeof(hostResults[0]),
                        resources.results,
                        hostResults.size() * sizeof(hostResults[0]),
                        ACL_MEMCPY_DEVICE_TO_HOST),
                        "copy UDMA QP results")) {
                    return 1;
                }
                uint64_t maxCoreCycles = 0ULL;
                uint64_t earliestStartCycle =
                    std::numeric_limits<uint64_t>::max();
                uint64_t latestEndCycle = 0ULL;
                uint64_t submittedOperations = 0ULL;
                for (const TransportBenchmarkResult &result : hostResults) {
                    maxCoreCycles = std::max(
                        maxCoreCycles, result.totalCycles);
                    earliestStartCycle = std::min(
                        earliestStartCycle, result.startCycle);
                    latestEndCycle = std::max(
                        latestEndCycle, result.endCycle);
                    submittedOperations += result.operations;
                }
                const uint64_t expectedOperations =
                    static_cast<uint64_t>(rowCount) * measuredGroups;
                if (submittedOperations != expectedOperations) {
                    std::cerr << "[ERROR] UDMA QP operation accounting mismatch: got "
                              << submittedOperations << ", expected "
                              << expectedOperations << '\n';
                    return 1;
                }
                if (latestEndCycle < earliestStartCycle) {
                    std::cerr << "[ERROR] Invalid UDMA QP timing window: start="
                              << earliestStartCycle << ", end="
                              << latestEndCycle << '\n';
                    return 1;
                }
                // AIV blocks do not necessarily enter the measured region at
                // the same cycle.  The rank makespan must cover the earliest
                // participating AIV start through the latest AIV completion;
                // max(end - start) alone undercounts launch skew and inflates
                // short-message, many-QP bandwidth.
                const uint64_t makespanCycles =
                    latestEndCycle - earliestStartCycle;
                const double seconds =
                    static_cast<double>(makespanCycles) * kCycleNs * 1.0e-9;
                const double totalMeasuredBytes =
                    static_cast<double>(transferBytes) * measuredGroups;
                const double bandwidthGbps = seconds > 0.0 ?
                    totalMeasuredBytes / seconds / 1.0e9 : 0.0;
                std::cout << std::fixed << std::setprecision(6)
                          << "[RESULT] rank=" << rank
                          << " transport=remote_udma_qp"
                          << " op=" << operationName
                          << " mode=bandwidth"
                          << " direction=" << directionName
                          << " row_bytes=" << rowBytes
                          << " rows=" << rowCount
                          << " total_bytes=" << transferBytes
                          << " qp_count=" << qpCount
                          << " active_qp_count=" << activeQpCount
                          << " blocks=" << activeQpCount
                          << " warmup_operations=" << warmupOperations
                          << " measured_groups=" << measuredGroups
                          << " pipe=pipe_s"
                          << " total_cycles=" << makespanCycles
                          << " max_core_cycles=" << maxCoreCycles
                          << " global_start_cycle=" << earliestStartCycle
                          << " global_end_cycle=" << latestEndCycle
                          << " submitted_operations=" << submittedOperations
                          << " gbps=" << bandwidthGbps
                          << " verify=" << (valid ? "PASS" : "FAIL")
                          << '\n';
                }
            }
        }
    }

    aclshmem_barrier_all();
    if (!allValid) {
        std::cerr << "[ERROR] UDMA QP benchmark validation failed on rank "
                  << rank << '\n';
        return 1;
    }
    std::cout << "[SUCCESS] UDMA QP benchmark completed on rank "
              << rank << '\n';
    return 0;
}

bool ValidateRemoteMteBiRowsCase(
    uint32_t rank, uint32_t worldSize, TransportOperation operation,
    uint32_t rowBytes, uint32_t totalRows,
    uint64_t destinationBase, uint8_t *data)
{
    const uint64_t transferBytes64 =
        static_cast<uint64_t>(rowBytes) * totalRows;
    if (transferBytes64 > std::numeric_limits<size_t>::max()) {
        return false;
    }
    const size_t transferBytes = static_cast<size_t>(transferBytes64);
    std::vector<uint8_t> actual(transferBytes);
    if (!CheckAcl(aclrtMemcpy(
            actual.data(), actual.size(), data + destinationBase,
            actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
            "Remote MTE BI row validation memcpy")) {
        return false;
    }

    const uint32_t expectedRank = operation == TransportOperation::READ ?
        (rank + 1U) % worldSize : (rank + worldSize - 1U) % worldSize;
    const uint8_t expected = RankPattern(expectedRank);
    return std::all_of(
        actual.begin(), actual.end(),
        [expected](uint8_t value) { return value == expected; });
}

int RunRemoteMteBiRowsBenchmark(int argc, char **argv)
{
    if (argc != 10) {
        std::cerr
            << "Usage: transport_benchmark remote_mte_bi_rows "
            << "<rank> <world_size> <ip_port> <row_sizes_csv> "
            << "<total_rows_csv> <blocks_csv> <warmup_operations> "
            << "<measured_groups>\n";
        return 2;
    }

    const uint32_t rank = ParseUint32(argv[2], "rank", true);
    const uint32_t worldSize = ParseUint32(argv[3], "world_size");
    const char *ipPort = argv[4];
    const std::vector<uint32_t> rowSizes =
        ParseSizes(argv[5], "row_sizes");
    const std::vector<uint32_t> totalRowCounts =
        ParseCounts(argv[6], "total_rows");
    const std::vector<uint32_t> blockCounts =
        ParseCounts(argv[7], "blocks");
    const uint32_t warmupOperations =
        ParseUint32(argv[8], "warmup_operations", true);
    const uint32_t measuredGroups =
        ParseUint32(argv[9], "measured_groups");
    if (rank >= worldSize || worldSize < 2U) {
        std::cerr << "[ERROR] Remote MTE BI row benchmark requires "
                  << "world_size >= 2 and 0 <= rank < world_size\n";
        return 2;
    }

    for (uint32_t totalRows : totalRowCounts) {
        for (uint32_t blocks : blockCounts) {
            if (blocks > totalRows || totalRows % blocks != 0U) {
                std::cerr << "[ERROR] total_rows must be divisible by blocks: "
                          << "total_rows=" << totalRows
                          << " blocks=" << blocks << '\n';
                return 2;
            }
        }
    }

    uint64_t maxTransferBytes = 0ULL;
    for (uint32_t rowBytes : rowSizes) {
        for (uint32_t totalRows : totalRowCounts) {
            maxTransferBytes = std::max(
                maxTransferBytes,
                static_cast<uint64_t>(rowBytes) * totalRows);
        }
    }
    const uint64_t destinationBase = AlignUp(maxTransferBytes, 4096ULL);
    const uint64_t dataBytes64 = destinationBase * 2ULL;
    const uint32_t maxBlocks =
        *std::max_element(blockCounts.begin(), blockCounts.end());
    if (dataBytes64 > kLocalMemBytes / 2ULL ||
        dataBytes64 > std::numeric_limits<size_t>::max()) {
        std::cerr << "[ERROR] Requested Remote MTE BI row buffers are too large: "
                  << dataBytes64 << " bytes\n";
        return 2;
    }
    const size_t dataBytes = static_cast<size_t>(dataBytes64);
    const size_t resultBytes = static_cast<size_t>(maxBlocks) *
        sizeof(TransportBenchmarkResult);

    Resources resources;
    resources.rank = rank;
    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.aclInitialized = true;
    if (!CheckAcl(aclrtSetDevice(static_cast<int32_t>(rank)), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    aclshmemx_init_attr_t attributes{};
    if (test_set_attr(
            static_cast<int32_t>(rank), static_cast<int32_t>(worldSize),
            kLocalMemBytes, ipPort, gDefaultUid, &attributes) != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] test_set_attr failed\n";
        return 1;
    }
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
    const int initStatus =
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed, status "
                  << initStatus << '\n';
        return 1;
    }
    resources.shmemInitialized = true;

    resources.data = static_cast<uint8_t *>(aclshmem_malloc(dataBytes));
    if (resources.data == nullptr) {
        std::cerr << "[ERROR] aclshmem_malloc failed for "
                  << dataBytes << " bytes\n";
        return 1;
    }
    if (!CheckAcl(aclrtMalloc(
            reinterpret_cast<void **>(&resources.results), resultBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc Remote MTE BI row results")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemset(
            resources.data, static_cast<size_t>(destinationBase),
            RankPattern(rank), static_cast<size_t>(destinationBase)),
            "initialize Remote MTE BI row source")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemset(
            resources.data + destinationBase,
            static_cast<size_t>(destinationBase), 0,
            static_cast<size_t>(destinationBase)),
            "initialize Remote MTE BI row destination")) {
        return 1;
    }
    aclshmem_barrier_all();

    bool allValid = true;
    for (TransportOperation operation : {
             TransportOperation::READ, TransportOperation::WRITE}) {
        const char *operationName =
            operation == TransportOperation::READ ? "read" : "write";
        for (uint32_t rowBytes : rowSizes) {
            for (uint32_t totalRows : totalRowCounts) {
                const uint64_t transferBytes =
                    static_cast<uint64_t>(rowBytes) * totalRows;
                for (uint32_t blocks : blockCounts) {
                    const uint32_t rowsPerBlock = totalRows / blocks;
                    if (!CheckAcl(aclrtMemset(
                            resources.data + destinationBase,
                            static_cast<size_t>(transferBytes), 0,
                            static_cast<size_t>(transferBytes)),
                            "clear Remote MTE BI row destination")) {
                        return 1;
                    }
                    if (!CheckAcl(aclrtMemset(
                            resources.results, resultBytes, 0, resultBytes),
                            "clear Remote MTE BI row results")) {
                        return 1;
                    }
                    aclshmem_barrier_all();

                    transport_benchmark_remote_mte_bi_rows_kernel_do(
                        blocks, resources.stream, resources.data,
                        reinterpret_cast<uint8_t *>(resources.results),
                        static_cast<uint32_t>(operation), rowBytes, totalRows,
                        destinationBase, warmupOperations, measuredGroups);
                    if (!CheckAcl(
                            aclrtSynchronizeStream(resources.stream),
                            "Remote MTE BI row benchmark kernel")) {
                        resources.deviceFaulted = true;
                        return 1;
                    }
                    aclshmem_barrier_all();

                    const bool valid = ValidateRemoteMteBiRowsCase(
                        rank, worldSize, operation, rowBytes, totalRows,
                        destinationBase, resources.data);
                    allValid = allValid && valid;
                    std::cout << "[VERIFY] rank=" << rank
                              << " transport=remote_mte_bi_rows"
                              << " op=" << operationName
                              << " row_bytes=" << rowBytes
                              << " total_rows=" << totalRows
                              << " rows_per_block=" << rowsPerBlock
                              << " blocks=" << blocks
                              << " status=" << (valid ? "PASS" : "FAIL")
                              << '\n';

                    std::vector<TransportBenchmarkResult> hostResults(blocks);
                    if (!CheckAcl(aclrtMemcpy(
                            hostResults.data(),
                            hostResults.size() * sizeof(hostResults[0]),
                            resources.results,
                            hostResults.size() * sizeof(hostResults[0]),
                            ACL_MEMCPY_DEVICE_TO_HOST),
                            "copy Remote MTE BI row results")) {
                        return 1;
                    }
                    uint64_t maxCoreCycles = 0ULL;
                    uint64_t earliestStart = std::numeric_limits<uint64_t>::max();
                    uint64_t latestEnd = 0ULL;
                    uint64_t submittedOperations = 0ULL;
                    for (const TransportBenchmarkResult &result : hostResults) {
                        maxCoreCycles = std::max(
                            maxCoreCycles, result.totalCycles);
                        earliestStart = std::min(
                            earliestStart, result.startCycle);
                        latestEnd = std::max(latestEnd, result.endCycle);
                        submittedOperations += result.operations;
                    }
                    const uint64_t expectedOperations =
                        static_cast<uint64_t>(totalRows) * measuredGroups;
                    if (submittedOperations != expectedOperations) {
                        std::cerr << "[ERROR] Remote MTE BI row operation "
                                  << "accounting mismatch: got "
                                  << submittedOperations << ", expected "
                                  << expectedOperations << '\n';
                        return 1;
                    }
                    if (earliestStart == std::numeric_limits<uint64_t>::max() ||
                        latestEnd < earliestStart) {
                        std::cerr << "[ERROR] Invalid Remote MTE BI row timing "
                                  << "window\n";
                        return 1;
                    }
                    const uint64_t makespanCycles = latestEnd - earliestStart;
                    const double seconds =
                        static_cast<double>(makespanCycles) * kCycleNs * 1.0e-9;
                    const double totalMeasuredBytes =
                        static_cast<double>(transferBytes) * measuredGroups;
                    const double bandwidthGbps = seconds > 0.0 ?
                        totalMeasuredBytes / seconds / 1.0e9 : 0.0;
                    std::cout << std::fixed << std::setprecision(6)
                              << "[RESULT] rank=" << rank
                              << " transport=remote_mte_bi_rows"
                              << " op=" << operationName
                              << " mode=row_sweep"
                              << " row_bytes=" << rowBytes
                              << " total_rows=" << totalRows
                              << " rows_per_block=" << rowsPerBlock
                              << " total_bytes=" << transferBytes
                              << " blocks=" << blocks
                              << " warmup_operations=" << warmupOperations
                              << " measured_groups=" << measuredGroups
                              << " loops=" << rowsPerBlock
                              << " batch=0"
                              << " total_cycles=" << makespanCycles
                              << " max_core_cycles=" << maxCoreCycles
                              << " global_start_cycle=" << earliestStart
                              << " global_end_cycle=" << latestEnd
                              << " submitted_operations=" << submittedOperations
                              << " gbps=" << bandwidthGbps
                              << " verify=" << (valid ? "PASS" : "FAIL")
                              << '\n';
                }
            }
        }
    }

    aclshmem_barrier_all();
    if (!allValid) {
        std::cerr << "[ERROR] Remote MTE BI row benchmark validation failed on rank "
                  << rank << '\n';
        return 1;
    }
    std::cout << "[SUCCESS] Remote MTE BI row benchmark completed on rank "
              << rank << '\n';
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    std::cout << std::unitbuf;
    if (argc > 1 && std::string(argv[1]) == "udma_qp") {
        return RunUdmaQpBenchmark(argc, argv);
    }
    if (argc > 1 && std::string(argv[1]) == "remote_mte_bi_rows") {
        return RunRemoteMteBiRowsBenchmark(argc, argv);
    }
    if (argc != 14 && argc != 15) {
        std::cerr
            << "Usage: transport_benchmark <rank> <world_size> <ip_port> "
            << "<remote_mte_bandwidth_sizes_csv> "
            << "<local_mte_bandwidth_sizes_csv> "
            << "<udma_bandwidth_sizes_csv> "
            << "<warmup> <loops> "
            << "<remote_mte_blocks> <local_mte_blocks> "
            << "<local_sdma_blocks> "
            << "<batch_size> <pipe_s|pipe_mte3> [all|udma_loop_sweep]\n";
        return 2;
    }

    const uint32_t rank = ParseUint32(argv[1], "rank", true);
    const uint32_t worldSize = ParseUint32(argv[2], "world_size");
    const char *ipPort = argv[3];
    const std::vector<uint32_t> remoteMteBandwidthSizes =
        ParseSizes(argv[4], "remote_mte_bandwidth_sizes");
    const std::vector<uint32_t> localMteBandwidthSizes =
        ParseSizes(argv[5], "local_mte_bandwidth_sizes");
    const std::vector<uint32_t> udmaBandwidthSizes =
        ParseSizes(argv[6], "udma_bandwidth_sizes");
    const std::vector<uint32_t> rttSizes{64U, 512U};
    const uint32_t benchmarkWarmup = ParseUint32(argv[7], "warmup", true);
    const uint32_t benchmarkLoops = ParseUint32(argv[8], "loops");
    const std::vector<uint32_t> remoteMteBlockCounts =
        ParseCounts(argv[9], "remote_mte_blocks");
    const std::vector<uint32_t> localMteBlockCounts =
        ParseCounts(argv[10], "local_mte_blocks");
    const std::vector<uint32_t> localSdmaBlockCounts =
        ParseCounts(argv[11], "local_sdma_blocks");
    const uint32_t batchSize = ParseUint32(argv[12], "batch_size", true);
    const std::string pipeMode = argv[13];
    const std::string benchmarkSelection = argc == 15 ? argv[14] : "all";
    if (rank >= worldSize || worldSize < 2U) {
        std::cerr << "[ERROR] Require 0 <= rank < world_size and world_size >= 2\n";
        return 2;
    }
    if (pipeMode != "pipe_s" && pipeMode != "pipe_mte3") {
        std::cerr << "[ERROR] pipe must be pipe_s or pipe_mte3\n";
        return 2;
    }
    if (benchmarkSelection != "all" &&
        benchmarkSelection != "udma_loop_sweep") {
        std::cerr << "[ERROR] benchmark selection must be all or udma_loop_sweep\n";
        return 2;
    }
    if (*std::max_element(rttSizes.begin(), rttSizes.end()) >
        kRawRttMaxBytes) {
        std::cerr << "[ERROR] RTT probe exceeds the 64 KiB UB buffer\n";
        return 2;
    }

    const uint32_t commonMaxBytes = std::max({
        *std::max_element(rttSizes.begin(), rttSizes.end()),
        *std::max_element(
            remoteMteBandwidthSizes.begin(), remoteMteBandwidthSizes.end()),
        *std::max_element(udmaBandwidthSizes.begin(), udmaBandwidthSizes.end())});
    const uint32_t localMteMaxBytes = *std::max_element(
        localMteBandwidthSizes.begin(), localMteBandwidthSizes.end());
    const uint32_t maxBlocks = std::max({
        *std::max_element(remoteMteBlockCounts.begin(), remoteMteBlockCounts.end()),
        *std::max_element(localMteBlockCounts.begin(), localMteBlockCounts.end()),
        *std::max_element(localSdmaBlockCounts.begin(), localSdmaBlockCounts.end())});
    const uint64_t commonStride64 = AlignUp(commonMaxBytes, 4096ULL);
    const uint64_t localMteStride64 = AlignUp(localMteMaxBytes, 4096ULL);
    const uint64_t maxStride64 = std::max(commonStride64, localMteStride64);
    const uint64_t destinationBase = maxStride64 * maxBlocks;
    const uint64_t dataBytes64 = destinationBase * 2ULL;
    if (commonStride64 > std::numeric_limits<uint32_t>::max() ||
        localMteStride64 > std::numeric_limits<uint32_t>::max() ||
        dataBytes64 > kLocalMemBytes / 2ULL) {
        std::cerr << "[ERROR] Requested benchmark buffers are too large: "
                  << dataBytes64 << " bytes\n";
        return 2;
    }
    const uint32_t commonStrideBytes = static_cast<uint32_t>(commonStride64);
    const uint32_t localMteStrideBytes =
        static_cast<uint32_t>(localMteStride64);
    const size_t dataBytes = static_cast<size_t>(dataBytes64);
    const size_t resultBytes = static_cast<size_t>(maxBlocks) *
        sizeof(TransportBenchmarkResult);

    Resources resources;
    resources.rank = rank;
    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.aclInitialized = true;
    if (!CheckAcl(aclrtSetDevice(static_cast<int32_t>(rank)), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    aclshmemx_init_attr_t attributes{};
    if (test_set_attr(
            static_cast<int32_t>(rank), static_cast<int32_t>(worldSize),
            kLocalMemBytes, ipPort, gDefaultUid, &attributes) != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] test_set_attr failed\n";
        return 1;
    }
    attributes.option_attr.data_op_engine_type =
        static_cast<data_op_engine_type_t>(
            ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA | ACLSHMEM_DATA_OP_UDMA);
    const int initStatus =
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initStatus != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] aclshmemx_init_attr failed, status "
                  << initStatus << '\n';
        return 1;
    }
    resources.shmemInitialized = true;

    resources.data = static_cast<uint8_t *>(aclshmem_malloc(dataBytes));
    if (resources.data == nullptr) {
        std::cerr << "[ERROR] aclshmem_malloc failed for " << dataBytes << " bytes\n";
        return 1;
    }
    if (!CheckAcl(aclrtMalloc(
            reinterpret_cast<void **>(&resources.results), resultBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc results")) {
        return 1;
    }

    if (!CheckAcl(aclrtMemset(
            resources.data, static_cast<size_t>(destinationBase),
            RankPattern(rank), static_cast<size_t>(destinationBase)),
            "initialize source")) {
        return 1;
    }
    if (!CheckAcl(aclrtMemset(
            resources.data + destinationBase, static_cast<size_t>(destinationBase),
            0, static_cast<size_t>(destinationBase)), "initialize destination")) {
        return 1;
    }
    aclshmem_barrier_all();

    bool allValid = true;
    const uint32_t useMte3 = pipeMode == "pipe_mte3" ? 1U : 0U;
    for (BenchmarkMode mode : {
             BenchmarkMode::RTT,
             BenchmarkMode::BANDWIDTH}) {
        for (const CaseDefinition &definition : kCases) {
            if (benchmarkSelection == "udma_loop_sweep" &&
                (mode != BenchmarkMode::BANDWIDTH ||
                 definition.transport != TransportKind::REMOTE_UDMA ||
                 !definition.bidirectional)) {
                continue;
            }
            if (definition.bidirectional && mode != BenchmarkMode::BANDWIDTH) {
                continue;
            }
            const std::vector<uint32_t> *sizes = &rttSizes;
            if (mode == BenchmarkMode::BANDWIDTH) {
                if (definition.transport == TransportKind::REMOTE_MTE) {
                    sizes = &remoteMteBandwidthSizes;
                } else if (definition.transport == TransportKind::LOCAL_MTE) {
                    sizes = &localMteBandwidthSizes;
                } else {
                    sizes = &udmaBandwidthSizes;
                }
            }
            if (definition.transport == TransportKind::LOCAL_SDMA &&
                definition.operation == TransportOperation::WRITE) {
                if (rank == 0U) {
                    std::cout
                        << "[SKIP] transport=local_sdma op=write "
                        << "reason=sdma_write_not_supported_on_ascend950\n";
                }
                continue;
            }
            for (uint32_t messageBytes : *sizes) {
                std::vector<uint32_t> blockCounts{1U};
                if (mode == BenchmarkMode::BANDWIDTH &&
                    definition.transport == TransportKind::REMOTE_MTE) {
                    blockCounts = remoteMteBlockCounts;
                } else if (mode == BenchmarkMode::BANDWIDTH &&
                    definition.transport == TransportKind::LOCAL_MTE) {
                    blockCounts = localMteBlockCounts;
                } else if (mode == BenchmarkMode::BANDWIDTH &&
                    definition.transport == TransportKind::LOCAL_SDMA) {
                    blockCounts = localSdmaBlockCounts;
                }
                for (uint32_t blocks : blockCounts) {
                    const uint32_t strideBytes =
                        definition.transport == TransportKind::LOCAL_MTE ?
                        localMteStrideBytes : commonStrideBytes;
                const size_t destinationBytes =
                    static_cast<size_t>(blocks) * strideBytes;
                const bool remoteTransport =
                    definition.transport == TransportKind::REMOTE_UDMA ||
                    definition.transport == TransportKind::REMOTE_MTE;
                const uint32_t activeRanks = mode == BenchmarkMode::BANDWIDTH ?
                    (remoteTransport ?
                        (definition.bidirectional ? worldSize : 1U) : worldSize) : 1U;
                const uint32_t warmup = benchmarkWarmup;
                const uint64_t loops64 = benchmarkLoops;
                if (loops64 > std::numeric_limits<uint32_t>::max()) {
                    std::cerr << "[ERROR] Loop count exceeds uint32 range\n";
                    return 2;
                }
                const uint32_t loops = static_cast<uint32_t>(loops64);

                if (!CheckAcl(aclrtMemset(
                        resources.data + destinationBase,
                        destinationBytes, 0, destinationBytes),
                        "clear destination")) {
                    return 1;
                }
                if (!CheckAcl(aclrtMemset(
                        resources.results, resultBytes, 0, resultBytes),
                        "clear results")) {
                    return 1;
                }
                aclshmem_barrier_all();

                transport_benchmark_kernel_do(
                    blocks, resources.stream, resources.data,
                    reinterpret_cast<uint8_t *>(resources.results),
                    static_cast<uint32_t>(definition.transport),
                    static_cast<uint32_t>(definition.operation),
                    static_cast<uint32_t>(mode), messageBytes, strideBytes,
                    destinationBase, warmup, loops, batchSize,
                    activeRanks, useMte3);
                if (!CheckAcl(
                        aclrtSynchronizeStream(resources.stream),
                        "transport benchmark kernel")) {
                    resources.deviceFaulted = true;
                    return 1;
                }
                aclshmem_barrier_all();

                const bool valid = ValidateCase(
                    rank, worldSize, definition, activeRanks, blocks,
                    messageBytes, strideBytes, destinationBase, resources.data);
                allValid = allValid && valid;
                if (ShouldValidateCase(
                        rank, definition, activeRanks)) {
                    std::cout << "[VERIFY] rank=" << rank
                              << " transport=" << definition.transportName
                              << " op=" << definition.operationName
                              << " mode=" << ModeName(mode)
                              << " bytes=" << messageBytes
                              << " status=" << (valid ? "PASS" : "FAIL") << '\n';
                }

                if (rank >= activeRanks) {
                    continue;
                }
                std::vector<TransportBenchmarkResult> hostResults(blocks);
                if (!CheckAcl(aclrtMemcpy(
                        hostResults.data(), hostResults.size() * sizeof(hostResults[0]),
                        resources.results, hostResults.size() * sizeof(hostResults[0]),
                        ACL_MEMCPY_DEVICE_TO_HOST), "copy benchmark results")) {
                    return 1;
                }

                uint64_t totalCycles = 0ULL;
                uint64_t minCycles = std::numeric_limits<uint64_t>::max();
                uint64_t maxCycles = 0ULL;
                uint64_t earliestStartCycle = std::numeric_limits<uint64_t>::max();
                uint64_t latestEndCycle = 0ULL;
                for (const TransportBenchmarkResult &result : hostResults) {
                    totalCycles = std::max(totalCycles, result.totalCycles);
                    minCycles = std::min(minCycles, result.minCycles);
                    maxCycles = std::max(maxCycles, result.maxCycles);
                    if (mode == BenchmarkMode::BANDWIDTH) {
                        earliestStartCycle = std::min(
                            earliestStartCycle, result.startCycle);
                        latestEndCycle = std::max(
                            latestEndCycle, result.endCycle);
                    }
                }
                // GetSystemCycle() values from different AIV cores are not a
                // guaranteed common wall-clock timeline. Use the slowest
                // participating block's elapsed cycles for aggregate bandwidth
                // instead of subtracting absolute timestamps across cores.
                const double averageNs = mode != BenchmarkMode::BANDWIDTH ?
                    static_cast<double>(hostResults[0].totalCycles) * kCycleNs /
                        hostResults[0].operations : 0.0;
                const double seconds =
                    static_cast<double>(totalCycles) * kCycleNs * 1.0e-9;
                const double transferredBytes = static_cast<double>(messageBytes) *
                    loops * blocks;
                const double bandwidthGbps = mode == BenchmarkMode::BANDWIDTH &&
                    seconds > 0.0 ? transferredBytes / seconds / 1.0e9 : 0.0;
                std::cout << std::fixed << std::setprecision(6)
                          << "[RESULT] rank=" << rank
                          << " transport=" << definition.transportName
                          << " op=" << definition.operationName
                          << " mode=" << ModeName(mode)
                          << " bytes=" << messageBytes
                          << " blocks=" << blocks
                          << " warmup=" << warmup
                          << " loops=" << loops
                          << " batch=" << batchSize
                          << " pipe=" << pipeMode
                          << " total_cycles=" << totalCycles
                          << " start_cycle="
                          << (mode == BenchmarkMode::BANDWIDTH ?
                              earliestStartCycle : 0ULL)
                          << " end_cycle="
                          << (mode == BenchmarkMode::BANDWIDTH ?
                              latestEndCycle : 0ULL)
                          << " min_cycles=" << minCycles
                          << " max_cycles=" << maxCycles
                          << " avg_ns=" << averageNs
                          << " min_ns=" << static_cast<double>(minCycles) * kCycleNs
                          << " max_ns=" << static_cast<double>(maxCycles) * kCycleNs
                          << " gbps=" << bandwidthGbps
                          << " verify=" << (valid ? "PASS" : "FAIL") << '\n';
                }
            }
        }
    }

    aclshmem_barrier_all();
    if (!allValid) {
        std::cerr << "[ERROR] Transport benchmark validation failed on rank "
                  << rank << '\n';
        return 1;
    }
    std::cout << "[SUCCESS] Transport benchmark completed on rank " << rank << '\n';
    return 0;
}
