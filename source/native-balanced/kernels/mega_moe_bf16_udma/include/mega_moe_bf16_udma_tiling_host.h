/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */

#ifndef MEGA_MOE_BF16_UDMA_TILING_HOST_H
#define MEGA_MOE_BF16_UDMA_TILING_HOST_H

#include <cstdint>

namespace MegaMoeBF16UDMAHost {

constexpr uint64_t ALIGN_BYTES = 512;
constexpr uint64_t BF16_BYTES = 2;
constexpr uint64_t SIGNAL_BYTES = 8;
constexpr uint64_t READY_COUNTER_BYTES = 4;
constexpr uint64_t COMBINE_ROW_GROUP = 128;

inline uint64_t Align(uint64_t value)
{
    return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

inline uint64_t CalcSharedActivationSize(
    uint64_t maxRecvTokens,
    uint64_t hiddenSize)
{
    return Align(maxRecvTokens * hiddenSize * BF16_BYTES);
}

struct ExtraBufferSize {
    uint64_t workspaceBytes;
    uint64_t peerBytes;
};

inline ExtraBufferSize CalcExtraBufferSize(
    uint64_t m,
    uint64_t topk,
    uint64_t worldSize,
    uint64_t expertPerRank,
    uint64_t maxRecvTokens,
    uint64_t hiddenSize)
{
    const uint64_t dispatchSignals = Align(
        worldSize * expertPerRank * SIGNAL_BYTES);
    // The same symmetric buffer is used sequentially as InitRouting scratch,
    // Dispatch receive/GMM1 input, and remote Combine staging.
    const uint64_t sharedActivation = CalcSharedActivationSize(
        maxRecvTokens, hiddenSize);
    const uint64_t routedTopkWeights = Align(m * topk * BF16_BYTES);
    const uint64_t combineReady = Align(
        expertPerRank *
        ((maxRecvTokens + COMBINE_ROW_GROUP - 1) / COMBINE_ROW_GROUP) *
        READY_COUNTER_BYTES);
    return {
        combineReady,
        routedTopkWeights + dispatchSignals + ALIGN_BYTES +
            sharedActivation};
}

}  // namespace MegaMoeBF16UDMAHost

#endif  // MEGA_MOE_BF16_UDMA_TILING_HOST_H
