/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "shmem.h"

#include "transport_benchmark.h"

using namespace AscendC;

namespace {

constexpr uint32_t kScratchBytes = 64U * 1024U;
constexpr uint32_t kSyncId = 0U;

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void LocalMteRead(
    __gm__ uint8_t *src, LocalTensor<uint8_t> scratch, uint32_t bytes)
{
    scratch.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    DataCopyPadExtParams<uint8_t> padParams{};
    uint32_t copied = 0U;
    while (copied < bytes) {
        const uint32_t chunk = MinU32(kScratchBytes, bytes - copied);
        DataCopyExtParams copyParams(1, chunk, 0, 0, 0);
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(src + copied);
        DataCopyPad(scratch, source, copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        copied += chunk;
    }
}

__aicore__ inline void LocalMteWrite(
    __gm__ uint8_t *dst, LocalTensor<uint8_t> scratch, uint32_t bytes)
{
    scratch.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
    uint32_t copied = 0U;
    while (copied < bytes) {
        const uint32_t chunk = MinU32(kScratchBytes, bytes - copied);
        DataCopyExtParams copyParams(1, chunk, 0, 0, 0);
        GlobalTensor<uint8_t> destination;
        destination.SetGlobalBuffer(dst + copied);
        DataCopyPad(destination, scratch, copyParams);
        PipeBarrier<PIPE_ALL>();
        copied += chunk;
    }
}

__aicore__ inline void RawRemoteMteRead(
    GlobalTensor<uint8_t> &remoteSource, LocalTensor<uint8_t> scratch,
    uint64_t offset, uint32_t bytes)
{
    scratch.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    DataCopy(scratch, remoteSource[offset], bytes);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RawRemoteMteWrite(
    GlobalTensor<uint8_t> &remoteDestination, LocalTensor<uint8_t> scratch,
    uint64_t offset, uint32_t bytes)
{
    scratch.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
    DataCopy(remoteDestination[offset], scratch, bytes);
    PipeBarrier<PIPE_ALL>();
}

template <pipe_t UDMA_PIPE>
__aicore__ inline void IssueTransfer(
    TransportKind transport, TransportOperation operation,
    __gm__ uint8_t *dst, __gm__ uint8_t *src,
    LocalTensor<uint8_t> scratch, uint32_t bytes, int peer)
{
    __ubuf__ uint8_t *scratchPtr =
        reinterpret_cast<__ubuf__ uint8_t *>(scratch.GetPhyAddr());
    if (transport == TransportKind::REMOTE_UDMA) {
        if constexpr (UDMA_PIPE == PIPE_S) {
            if (operation == TransportOperation::READ) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_S>(
                    dst, src, static_cast<__ubuf__ uint8_t *>(nullptr),
                    bytes, peer, kSyncId);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
                    dst, src, static_cast<__ubuf__ uint8_t *>(nullptr),
                    bytes, peer, kSyncId);
            }
        } else {
            if (operation == TransportOperation::READ) {
                aclshmemx_udma_get_nbi<uint8_t, PIPE_MTE3>(
                    dst, src, scratchPtr, bytes, peer, kSyncId);
            } else {
                aclshmemx_udma_put_nbi<uint8_t, PIPE_MTE3>(
                    dst, src, scratchPtr, bytes, peer, kSyncId);
            }
        }
        return;
    }

    if (transport == TransportKind::LOCAL_SDMA) {
        if (operation == TransportOperation::READ) {
            aclshmemx_sdma_get_nbi(
                dst, src, scratchPtr, kScratchBytes, bytes, peer, kSyncId);
        } else {
            aclshmemx_sdma_put_nbi(
                dst, src, scratchPtr, kScratchBytes, bytes, peer, kSyncId);
        }
        return;
    }

    if (transport == TransportKind::REMOTE_MTE) {
        if (operation == TransportOperation::READ) {
            aclshmemx_mte_get_nbi(
                dst, src, scratchPtr, kScratchBytes, bytes, peer, kSyncId);
        } else {
            aclshmemx_mte_put_nbi(
                dst, src, scratchPtr, kScratchBytes, bytes, peer, kSyncId);
        }
        return;
    }

    if (operation == TransportOperation::READ) {
        LocalMteRead(src, scratch, bytes);
    } else {
        LocalMteWrite(dst, scratch, bytes);
    }
}

__aicore__ inline void CompleteTransfer(
    TransportKind transport, LocalTensor<uint8_t> scratch, int peer);

template <pipe_t UDMA_PIPE>
__aicore__ inline void RunRttTransfer(
    TransportKind transport, TransportOperation operation,
    __gm__ uint8_t *dst, __gm__ uint8_t *src,
    GlobalTensor<uint8_t> &remoteDestination,
    GlobalTensor<uint8_t> &remoteSource,
    LocalTensor<uint8_t> scratch, uint64_t offset,
    uint32_t bytes, int peer)
{
    if (transport == TransportKind::REMOTE_MTE) {
        if (operation == TransportOperation::READ) {
            RawRemoteMteRead(remoteSource, scratch, offset, bytes);
        } else {
            RawRemoteMteWrite(remoteDestination, scratch, offset, bytes);
        }
        return;
    }
    IssueTransfer<UDMA_PIPE>(
        transport, operation, dst + offset, src + offset,
        scratch, bytes, peer);
    CompleteTransfer(transport, scratch, peer);
}

__aicore__ inline void CompleteTransfer(
    TransportKind transport, LocalTensor<uint8_t> scratch, int peer)
{
    if (transport == TransportKind::REMOTE_UDMA) {
        aclshmemx_udma_quiet(peer);
    } else if (transport == TransportKind::LOCAL_SDMA) {
        aclshmemx_sdma_quiet(
            reinterpret_cast<__ubuf__ uint8_t *>(scratch.GetPhyAddr()),
            kScratchBytes, kSyncId);
    } else if (transport == TransportKind::REMOTE_MTE) {
        aclshmemx_mte_quiet();
    }
}

template <pipe_t UDMA_PIPE>
__aicore__ inline void RunBenchmark(
    GM_ADDR data, GM_ADDR resultAddress,
    uint32_t transportValue, uint32_t operationValue,
    uint32_t modeValue, uint32_t messageBytes, uint32_t strideBytes,
    uint64_t destinationBase,
    uint32_t warmup, uint32_t loops, uint32_t batchSize,
    uint32_t activeRanks)
{
    const uint32_t rank = static_cast<uint32_t>(aclshmem_my_pe());
    const uint32_t rankCount = static_cast<uint32_t>(aclshmem_n_pes());
    const uint32_t block = GetBlockIdx();
    auto *results = reinterpret_cast<__gm__ TransportBenchmarkResult *>(resultAddress);
    if (rank >= activeRanks) {
        return;
    }

    TPipe pipe;
    TBuf<TPosition::VECOUT> scratchBuffer;
    pipe.InitBuffer(scratchBuffer, kScratchBytes);
    LocalTensor<uint8_t> scratch =
        scratchBuffer.GetWithOffset<uint8_t>(kScratchBytes, 0);

    const TransportKind transport = static_cast<TransportKind>(transportValue);
    const TransportOperation operation =
        static_cast<TransportOperation>(operationValue);
    const BenchmarkMode mode = static_cast<BenchmarkMode>(modeValue);
    const bool remote = transport == TransportKind::REMOTE_UDMA ||
        transport == TransportKind::REMOTE_MTE;
    const int peer = remote ? static_cast<int>((rank + 1U) % rankCount) :
        static_cast<int>(rank);

    const uint64_t blockOffset = static_cast<uint64_t>(block) * strideBytes;
    __gm__ uint8_t *source =
        reinterpret_cast<__gm__ uint8_t *>(data) + blockOffset;
    __gm__ uint8_t *destination =
        reinterpret_cast<__gm__ uint8_t *>(data) + destinationBase + blockOffset;
    GlobalTensor<uint8_t> rttRemoteSource;
    GlobalTensor<uint8_t> rttRemoteDestination;
    rttRemoteSource.SetGlobalBuffer(source);
    rttRemoteDestination.SetGlobalBuffer(destination);
    if (transport == TransportKind::REMOTE_MTE && mode == BenchmarkMode::RTT) {
        rttRemoteSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
            aclshmem_ptr(source, peer)));
        rttRemoteDestination.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(
                aclshmem_ptr(destination, peer)));
    }

    // Direct UB-to-GM RTT probes need initialized UB contents. This
    // preparation is outside the measured interval.
    if ((transport == TransportKind::LOCAL_MTE ||
         (transport == TransportKind::REMOTE_MTE &&
          mode == BenchmarkMode::RTT)) &&
        operation == TransportOperation::WRITE) {
        const uint32_t initialBytes = MinU32(kScratchBytes, messageBytes);
        DataCopyExtParams copyParams(1, initialBytes, 0, 0, 0);
        DataCopyPadExtParams<uint8_t> padParams{};
        GlobalTensor<uint8_t> sourceTensor;
        sourceTensor.SetGlobalBuffer(source);
        scratch.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        DataCopyPad(scratch, sourceTensor, copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    const bool nbiBandwidth = mode == BenchmarkMode::BANDWIDTH &&
        (transport == TransportKind::REMOTE_UDMA ||
         transport == TransportKind::REMOTE_MTE ||
         transport == TransportKind::LOCAL_SDMA);
    for (uint32_t iteration = 0; iteration < warmup; ++iteration) {
        if (mode == BenchmarkMode::RTT) {
            const uint32_t slots = strideBytes / messageBytes;
            const uint64_t offset =
                static_cast<uint64_t>(iteration % slots) * messageBytes;
            RunRttTransfer<UDMA_PIPE>(
                transport, operation, destination, source,
                rttRemoteDestination, rttRemoteSource,
                scratch, offset, messageBytes, peer);
        } else {
            IssueTransfer<UDMA_PIPE>(
                transport, operation, destination, source, scratch,
                messageBytes, peer);
            if (!nbiBandwidth) {
                CompleteTransfer(transport, scratch, peer);
            }
        }
    }
    if (nbiBandwidth && warmup != 0U) {
        CompleteTransfer(transport, scratch, peer);
    }

    uint64_t totalCycles = 0U;
    uint64_t minCycles = ~static_cast<uint64_t>(0U);
    uint64_t maxCycles = 0U;
    uint64_t startCycle = 0U;
    uint64_t endCycle = 0U;
    if (mode == BenchmarkMode::RTT) {
        const uint64_t start = GetSystemCycle();
        for (uint32_t iteration = 0; iteration < loops; ++iteration) {
            const uint32_t slots = strideBytes / messageBytes;
            const uint64_t offset =
                static_cast<uint64_t>(iteration % slots) * messageBytes;
            RunRttTransfer<UDMA_PIPE>(
                transport, operation, destination, source,
                rttRemoteDestination, rttRemoteSource,
                scratch, offset, messageBytes, peer);
        }
        totalCycles = GetSystemCycle() - start;
        minCycles = totalCycles / loops;
        maxCycles = minCycles;
    } else {
        // SHMEM udma_perftest defines batch=0 (or batch>loops) as full
        // async. Other NBI transports use the same explicit benchmark
        // convention instead of an arbitrary fixed batch size.
        const uint32_t effectiveBatch =
            batchSize == 0U || batchSize > loops ? loops : batchSize;
        startCycle = GetSystemCycle();
        const uint32_t batches = loops / effectiveBatch;
        for (uint32_t batch = 0; batch < batches; ++batch) {
            for (uint32_t iteration = 0; iteration < effectiveBatch; ++iteration) {
                IssueTransfer<UDMA_PIPE>(
                    transport, operation, destination, source, scratch,
                    messageBytes, peer);
            }
            CompleteTransfer(transport, scratch, peer);
        }
        const uint32_t remaining = loops % effectiveBatch;
        for (uint32_t iteration = 0; iteration < remaining; ++iteration) {
            IssueTransfer<UDMA_PIPE>(
                transport, operation, destination, source, scratch,
                messageBytes, peer);
        }
        if (remaining != 0U) {
            CompleteTransfer(transport, scratch, peer);
        }
        endCycle = GetSystemCycle();
        totalCycles = endCycle - startCycle;
        minCycles = totalCycles;
        maxCycles = totalCycles;
    }

    // Make a local MTE read observable from the host without including this
    // diagnostic store in the measured interval.
    if (transport == TransportKind::LOCAL_MTE &&
        operation == TransportOperation::READ) {
        DataCopyExtParams copyParams(1, MinU32(messageBytes, 32U), 0, 0, 0);
        GlobalTensor<uint8_t> destinationTensor;
        destinationTensor.SetGlobalBuffer(destination);
        DataCopyPad(destinationTensor, scratch, copyParams);
        PipeBarrier<PIPE_ALL>();
    }
    if (mode == BenchmarkMode::RTT &&
        transport == TransportKind::REMOTE_MTE) {
        if (operation == TransportOperation::READ) {
            scratch.address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
            GlobalTensor<uint8_t> destinationTensor;
            destinationTensor.SetGlobalBuffer(destination);
            DataCopy(destinationTensor, scratch, messageBytes);
            PipeBarrier<PIPE_ALL>();
        } else {
            RawRemoteMteWrite(
                rttRemoteDestination, scratch, 0U, messageBytes);
        }
    }

    results[block].totalCycles = totalCycles;
    results[block].minCycles = minCycles;
    results[block].maxCycles = maxCycles;
    results[block].operations = loops;
    results[block].startCycle = startCycle;
    results[block].endCycle = endCycle;
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
transport_benchmark_pipe_s_kernel(
    GM_ADDR data, GM_ADDR results,
    uint32_t transport, uint32_t operation, uint32_t mode,
    uint32_t messageBytes, uint32_t strideBytes,
    uint64_t destinationBase,
    uint32_t warmup, uint32_t loops, uint32_t batchSize,
    uint32_t activeRanks)
{
    RunBenchmark<PIPE_S>(
        data, results, transport, operation, mode, messageBytes,
        strideBytes, destinationBase, warmup, loops, batchSize, activeRanks);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
transport_benchmark_pipe_mte3_kernel(
    GM_ADDR data, GM_ADDR results,
    uint32_t transport, uint32_t operation, uint32_t mode,
    uint32_t messageBytes, uint32_t strideBytes,
    uint64_t destinationBase,
    uint32_t warmup, uint32_t loops, uint32_t batchSize,
    uint32_t activeRanks)
{
    RunBenchmark<PIPE_MTE3>(
        data, results, transport, operation, mode, messageBytes,
        strideBytes, destinationBase, warmup, loops, batchSize, activeRanks);
}

extern "C" void transport_benchmark_kernel_do(
    uint32_t blockDim, void *stream, uint8_t *data,
    uint8_t *results,
    uint32_t transport, uint32_t operation, uint32_t mode,
    uint32_t messageBytes, uint32_t strideBytes,
    uint64_t destinationBase,
    uint32_t warmup, uint32_t loops, uint32_t batchSize,
    uint32_t activeRanks, uint32_t useMte3)
{
    if (useMte3 != 0U) {
        transport_benchmark_pipe_mte3_kernel<<<blockDim, nullptr, stream>>>(
            data, results, transport, operation, mode, messageBytes,
            strideBytes, destinationBase, warmup, loops, batchSize, activeRanks);
        return;
    }
    transport_benchmark_pipe_s_kernel<<<blockDim, nullptr, stream>>>(
        data, results, transport, operation, mode, messageBytes,
        strideBytes, destinationBase, warmup, loops, batchSize, activeRanks);
}

namespace {

__aicore__ inline void IssueUdmaQpRows(
    TransportOperation operation, __gm__ uint8_t *destination,
    __gm__ uint8_t *source, uint32_t rowBytes, uint32_t rowCount,
    int peer, uint32_t qpIdx, uint32_t qpCount)
{
    for (uint32_t row = qpIdx; row < rowCount; row += qpCount) {
        const uint64_t offset = static_cast<uint64_t>(row) * rowBytes;
        if (operation == TransportOperation::READ) {
            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_S>(
                destination + offset, source + offset,
                static_cast<__ubuf__ uint8_t *>(nullptr), rowBytes,
                peer, qpIdx, kSyncId);
        } else {
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S>(
                destination + offset, source + offset,
                static_cast<__ubuf__ uint8_t *>(nullptr), rowBytes,
                peer, qpIdx, kSyncId);
        }
    }
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
transport_benchmark_udma_qp_kernel(
    GM_ADDR data, GM_ADDR results, uint32_t operationValue,
    uint32_t rowBytes, uint32_t rowCount, uint64_t destinationBase,
    uint32_t warmupOperations, uint32_t measuredGroups,
    uint32_t bidirectionalValue)
{
    const uint32_t rank = static_cast<uint32_t>(aclshmem_my_pe());
    const bool bidirectional = bidirectionalValue != 0U;
    if (!bidirectional && rank != 0U) {
        return;
    }
    const uint32_t qpIdx = GetBlockIdx();
    const uint32_t qpCount = GetBlockNum();
    // The host path requires exactly two ranks. Unidirectional cases keep the
    // original rank0 initiator; bidirectional cases run both ranks together.
    const int peer = static_cast<int>(1U - rank);
    auto *benchmarkResults =
        reinterpret_cast<__gm__ TransportBenchmarkResult *>(results);
    __gm__ uint8_t *source = reinterpret_cast<__gm__ uint8_t *>(data);
    __gm__ uint8_t *destination = source + destinationBase;
    const TransportOperation operation =
        static_cast<TransportOperation>(operationValue);
    const uint64_t rowsForQp =
        1ULL + (static_cast<uint64_t>(rowCount) - 1ULL - qpIdx) / qpCount;

    // Keep warmup semantics identical to the Remote MTE row sweep: each AIV
    // issues warmupOperations individual row transfers, followed by one quiet.
    for (uint32_t iteration = 0U; iteration < warmupOperations; ++iteration) {
        const uint32_t row = qpIdx +
            static_cast<uint32_t>(iteration % rowsForQp) * qpCount;
        const uint64_t offset = static_cast<uint64_t>(row) * rowBytes;
        if (operation == TransportOperation::READ) {
            aclshmemx_udma_qp_get_nbi<uint8_t, PIPE_S>(
                destination + offset, source + offset,
                static_cast<__ubuf__ uint8_t *>(nullptr), rowBytes,
                peer, qpIdx, kSyncId);
        } else {
            aclshmemx_udma_qp_put_nbi<uint8_t, PIPE_S>(
                destination + offset, source + offset,
                static_cast<__ubuf__ uint8_t *>(nullptr), rowBytes,
                peer, qpIdx, kSyncId);
        }
    }
    if (warmupOperations != 0U) {
        aclshmemx_udma_qp_quiet(peer, qpIdx);
    }

    // SyncAll requires the platform-wide AIV set.  This benchmark may launch
    // fewer blocks when qpCount < the physical AIV count, so using SyncAll
    // would wait for blocks that do not exist. Time every independent QP on
    // its owning AIV and let the host form one rank-wide makespan.
    const uint64_t startCycle = GetSystemCycle();
    for (uint32_t group = 0U; group < measuredGroups; ++group) {
        IssueUdmaQpRows(
            operation, destination, source, rowBytes, rowCount,
            peer, qpIdx, qpCount);
        aclshmemx_udma_qp_quiet(peer, qpIdx);
    }
    const uint64_t endCycle = GetSystemCycle();
    const uint64_t totalCycles = endCycle - startCycle;

    // Result slots are 64-byte aligned and isolated, so concurrent AIV stores
    // cannot overwrite a neighbouring QP's timing record.
    benchmarkResults[qpIdx].totalCycles = totalCycles;
    benchmarkResults[qpIdx].minCycles = totalCycles;
    benchmarkResults[qpIdx].maxCycles = totalCycles;
    benchmarkResults[qpIdx].operations = rowsForQp * measuredGroups;
    benchmarkResults[qpIdx].startCycle = startCycle;
    benchmarkResults[qpIdx].endCycle = endCycle;
}

extern "C" void transport_benchmark_udma_qp_kernel_do(
    uint32_t blockDim, void *stream, uint8_t *data, uint8_t *results,
    uint32_t operation, uint32_t rowBytes, uint32_t rowCount,
    uint64_t destinationBase, uint32_t warmupOperations,
    uint32_t measuredGroups, uint32_t bidirectional)
{
    transport_benchmark_udma_qp_kernel<<<blockDim, nullptr, stream>>>(
        data, results, operation, rowBytes, rowCount, destinationBase,
        warmupOperations, measuredGroups, bidirectional);
}

namespace {

__aicore__ inline void IssueRemoteMteRow(
    TransportOperation operation, __gm__ uint8_t *destination,
    __gm__ uint8_t *source, __ubuf__ uint8_t *scratch,
    uint32_t rowBytes, uint32_t row, int peer)
{
    const uint64_t offset = static_cast<uint64_t>(row) * rowBytes;
    if (operation == TransportOperation::READ) {
        aclshmemx_mte_get_nbi(
            destination + offset, source + offset, scratch,
            kScratchBytes, rowBytes, peer, kSyncId);
    } else {
        aclshmemx_mte_put_nbi(
            destination + offset, source + offset, scratch,
            kScratchBytes, rowBytes, peer, kSyncId);
    }
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void
transport_benchmark_remote_mte_bi_rows_kernel(
    GM_ADDR data, GM_ADDR results, uint32_t operationValue,
    uint32_t rowBytes, uint32_t totalRows, uint64_t destinationBase,
    uint32_t warmupOperations, uint32_t measuredGroups)
{
    const uint32_t rank = static_cast<uint32_t>(aclshmem_my_pe());
    const uint32_t rankCount = static_cast<uint32_t>(aclshmem_n_pes());
    const uint32_t block = GetBlockIdx();
    const uint32_t blocks = GetBlockNum();
    const uint32_t rowsPerBlock = totalRows / blocks;
    const uint32_t firstRow = block * rowsPerBlock;
    const int peer = static_cast<int>((rank + 1U) % rankCount);
    const TransportOperation operation =
        static_cast<TransportOperation>(operationValue);

    TPipe pipe;
    TBuf<TPosition::VECOUT> scratchBuffer;
    pipe.InitBuffer(scratchBuffer, kScratchBytes);
    LocalTensor<uint8_t> scratchTensor =
        scratchBuffer.GetWithOffset<uint8_t>(kScratchBytes, 0);
    __ubuf__ uint8_t *scratch =
        reinterpret_cast<__ubuf__ uint8_t *>(scratchTensor.GetPhyAddr());
    __gm__ uint8_t *source = reinterpret_cast<__gm__ uint8_t *>(data);
    __gm__ uint8_t *destination = source + destinationBase;

    // warmup counts individual row transfers, not complete row groups.
    for (uint32_t iteration = 0U; iteration < warmupOperations; ++iteration) {
        const uint32_t row = firstRow + iteration % rowsPerBlock;
        IssueRemoteMteRow(
            operation, destination, source, scratch,
            rowBytes, row, peer);
    }
    if (warmupOperations != 0U) {
        aclshmemx_mte_quiet();
    }

    const uint64_t startCycle = GetSystemCycle();
    for (uint32_t group = 0U; group < measuredGroups; ++group) {
        for (uint32_t localRow = 0U; localRow < rowsPerBlock; ++localRow) {
            IssueRemoteMteRow(
                operation, destination, source, scratch,
                rowBytes, firstRow + localRow, peer);
        }
        // Complete every configured row group independently. Otherwise the
        // effective queue depth would become totalRows * measuredGroups.
        aclshmemx_mte_quiet();
    }
    const uint64_t endCycle = GetSystemCycle();

    auto *benchmarkResults =
        reinterpret_cast<__gm__ TransportBenchmarkResult *>(results);
    const uint64_t totalCycles = endCycle - startCycle;
    benchmarkResults[block].totalCycles = totalCycles;
    benchmarkResults[block].minCycles = totalCycles;
    benchmarkResults[block].maxCycles = totalCycles;
    benchmarkResults[block].operations =
        static_cast<uint64_t>(rowsPerBlock) * measuredGroups;
    benchmarkResults[block].startCycle = startCycle;
    benchmarkResults[block].endCycle = endCycle;
}

extern "C" void transport_benchmark_remote_mte_bi_rows_kernel_do(
    uint32_t blockDim, void *stream, uint8_t *data, uint8_t *results,
    uint32_t operation, uint32_t rowBytes, uint32_t totalRows,
    uint64_t destinationBase, uint32_t warmupOperations,
    uint32_t measuredGroups)
{
    transport_benchmark_remote_mte_bi_rows_kernel<<<blockDim, nullptr, stream>>>(
        data, results, operation, rowBytes, totalRows,
        destinationBase, warmupOperations, measuredGroups);
}
