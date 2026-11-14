/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * MegaMoe BF16 M-group scheduling prototype.  The public operator and kernel
 * entry are intentionally independent from MegaMoeBF16 so that Wave
 * scheduling can be validated without changing the production path.
 */
#ifndef DEEPEP_MEGA_MOE_BF16_MGROUP_H
#define DEEPEP_MEGA_MOE_BF16_MGROUP_H

#include "mega_moe_bf16.h"

namespace MegaMoeImpl {

struct ExpertRowCursor {
    uint32_t expertIdx{0};
    uint32_t rowOffsetInExpert{0};
};

struct ExpertRowSlice {
    uint32_t expertIdx{0};
    uint32_t rowBegin{0};
    uint32_t rowEnd{0};
    uint32_t readySlot{0};

    __aicore__ inline uint32_t Rows() const
    {
        return rowEnd - rowBegin;
    }
};

class MegaMoeBF16MGroup : public MegaMoeBF16 {
public:
    static constexpr uint32_t M_GROUP_ROWS = 256;
    static constexpr uint32_t GMM1_LOGICAL_TILES_PER_AIC_TARGET = 4;

    __aicore__ inline MegaMoeBF16MGroup() = default;
    __aicore__ inline void Process();
    template<uint32_t SaveMode> __aicore__ inline void ProcessWithSaveMode();

private:
    __aicore__ inline uint32_t GetExpertRows(uint32_t expertIdx);
    __aicore__ inline uint64_t GetExpertRowBase(uint32_t expertIdx);
    __aicore__ inline bool NextSlice(
        ExpertRowCursor &cursor,
        uint32_t maxGroups,
        ExpertRowSlice &slice);
    __aicore__ inline uint32_t GroupsPerWave();
    __aicore__ inline uint32_t MaxGroupsPerExpert();
    __aicore__ inline uint32_t ReadySlotCapacity();
    __aicore__ inline GM_ADDR ReadyFlagBase();
    __aicore__ inline void ResetMGroupReadyFlags();
    __aicore__ inline void SetSliceAddresses(
        GMMAddrInfo &gmmAddrInfo,
        const ExpertRowSlice &slice,
        uint32_t gmmIndex);
};

__aicore__ inline uint32_t MegaMoeBF16MGroup::GetExpertRows(
    uint32_t expertIdx)
{
    int64_t rows = 0;
    if (groupListType_ == 0) {
        const int64_t end = groupList_.GetValue(expertIdx);
        const int64_t begin = expertIdx == 0
            ? 0
            : groupList_.GetValue(expertIdx - 1);
        rows = end - begin;
    } else {
        rows = groupList_.GetValue(expertIdx);
    }
    const uint64_t base = GetExpertRowBase(expertIdx);
    if (rows <= 0 || base >= maxOutputSize_) {
        return 0;
    }
    const uint64_t remaining = maxOutputSize_ - base;
    return static_cast<uint32_t>(
        static_cast<uint64_t>(rows) < remaining ? rows : remaining);
}

__aicore__ inline uint64_t MegaMoeBF16MGroup::GetExpertRowBase(
    uint32_t expertIdx)
{
    if (expertIdx == 0) {
        return 0;
    }
    if (groupListType_ == 0) {
        const int64_t base = groupList_.GetValue(expertIdx - 1);
        return base > 0 ? static_cast<uint64_t>(base) : 0;
    }
    uint64_t base = 0;
    for (uint32_t idx = 0; idx < expertIdx; ++idx) {
        const int32_t rows = groupList_.GetValue(idx);
        if (rows > 0) {
            base += static_cast<uint32_t>(rows);
        }
    }
    return base;
}

__aicore__ inline bool MegaMoeBF16MGroup::NextSlice(
    ExpertRowCursor &cursor,
    uint32_t maxGroups,
    ExpertRowSlice &slice)
{
    while (cursor.expertIdx < expertPerRank_) {
        const uint32_t rows = GetExpertRows(cursor.expertIdx);
        if (cursor.rowOffsetInExpert >= rows) {
            ++cursor.expertIdx;
            cursor.rowOffsetInExpert = 0;
            continue;
        }
        const uint64_t maxRows = static_cast<uint64_t>(maxGroups) * M_GROUP_ROWS;
        const uint64_t requestedEnd =
            static_cast<uint64_t>(cursor.rowOffsetInExpert) + maxRows;
        const uint32_t end = requestedEnd < rows
            ? static_cast<uint32_t>(requestedEnd)
            : rows;
        slice = {
            cursor.expertIdx,
            cursor.rowOffsetInExpert,
            end,
            cursor.expertIdx * MaxGroupsPerExpert() +
                cursor.rowOffsetInExpert / M_GROUP_ROWS};
        cursor.rowOffsetInExpert = end;
        if (cursor.rowOffsetInExpert == rows) {
            ++cursor.expertIdx;
            cursor.rowOffsetInExpert = 0;
        }
        return true;
    }
    return false;
}

__aicore__ inline uint32_t MegaMoeBF16MGroup::GroupsPerWave()
{
    const uint32_t outputN = Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF;
    const uint32_t mTiles = Ops::Base::CeilDiv(M_GROUP_ROWS,
        MegaMoeBF16CatlassImpl::L1_TILE_M);
    const uint32_t nTiles = Ops::Base::CeilDiv(outputN,
        MegaMoeBF16CatlassImpl::L1_TILE_N);
    // One BF16 scheduler task computes both gate and up weight blocks, so it
    // represents two logical N tiles in the official Wave formula.
    const uint32_t gmm1LogicalTilesPerGroup =
        mTiles * nTiles * SWIGLU_N_HALF;
    const uint32_t gmm1TargetTiles =
        GMM1_LOGICAL_TILES_PER_AIC_TARGET * blockNum_;
    const uint32_t gmm1Groups = Ops::Base::CeilDiv(
        gmm1TargetTiles, gmm1LogicalTilesPerGroup);
    const uint32_t gmm2NTiles = Ops::Base::CeilDiv(
        static_cast<uint32_t>(Get<K_VALUE>(problemShape_)),
        MegaMoeBF16CatlassImpl::L1_TILE_N);
    const uint32_t gmm2TilesPerGroup = mTiles * gmm2NTiles;
    const uint32_t gmm2Groups = Ops::Base::CeilDiv(
        blockNum_, gmm2TilesPerGroup);
    const uint32_t groups = gmm1Groups > gmm2Groups
        ? gmm1Groups
        : gmm2Groups;
    return groups == 0 ? 1 : groups;
}

__aicore__ inline uint32_t MegaMoeBF16MGroup::MaxGroupsPerExpert()
{
    return Ops::Base::CeilDiv(
        static_cast<uint64_t>(maxOutputSize_),
        static_cast<uint64_t>(M_GROUP_ROWS));
}

__aicore__ inline uint32_t MegaMoeBF16MGroup::ReadySlotCapacity()
{
    return expertPerRank_ * MaxGroupsPerExpert();
}

__aicore__ inline GM_ADDR MegaMoeBF16MGroup::ReadyFlagBase()
{
    if (useSplitTrainingState_) {
        const MegaMoeBF16Training::ForwardTransientWorkspaceLayout layout(
            params_.tilingData->n,
            params_.tilingData->expertPerRank,
            params_.tilingData->maxOutputSize);
        return params_.workspaceInfo.ptrA2 + layout.totalBytes;
    }
    return params_.workspaceInfo.ptrA + params_.workspaceInfo.workspaceSize;
}

__aicore__ inline void MegaMoeBF16MGroup::ResetMGroupReadyFlags()
{
    if constexpr (g_coreType == AscendC::AIV) {
        const uint32_t slots = ReadySlotCapacity();
        __gm__ int32_t *flags =
            reinterpret_cast<__gm__ int32_t *>(ReadyFlagBase());
        for (uint32_t slot = aivCoreIdx_; slot < slots;
             slot += blockAivNum_) {
            WriteGmByPassDCache(flags + slot * INT_CACHELINE, 0);
        }
    }
}

__aicore__ inline void MegaMoeBF16MGroup::SetSliceAddresses(
    GMMAddrInfo &gmmAddrInfo,
    const ExpertRowSlice &slice,
    uint32_t gmmIndex)
{
    const uint64_t expertRowBase = GetExpertRowBase(slice.expertIdx);
    const uint64_t globalRow = expertRowBase + slice.rowBegin;
    const uint64_t n = Get<N_VALUE>(problemShape_);
    const uint64_t outputN = n / SWIGLU_N_HALF;
    const uint64_t k = Get<K_VALUE>(problemShape_);

    gmmAddrInfo.groupFlagList =
        reinterpret_cast<__gm__ int32_t *>(ReadyFlagBase()) +
        slice.readySlot * INT_CACHELINE;
    gmmAddrInfo.groupFlagList2 =
        reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.ptrFlagDispatchToGmm1) +
        slice.expertIdx * INT_CACHELINE;

    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA +
            globalRow * k * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale;
        gmmAddrInfo.bGlobal = params_.bGmAddr +
            static_cast<uint64_t>(slice.expertIdx) * n * k *
                sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr;
        if constexpr (g_coreType == AscendC::AIV) {
            AscendC::Coord<
                int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
                epilogueOffset{
                    static_cast<int64_t>(globalRow * outputN),
                    0,
                    static_cast<int64_t>(slice.readySlot),
                    static_cast<int64_t>(globalRow),
                    0,
                    0};
            epilogueOp_.UpdateGlobalAddr(epilogueOffset);
        }
    } else {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 +
            globalRow * outputN * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale;
        gmmAddrInfo.bGlobal = params_.b2GmAddr +
            static_cast<uint64_t>(slice.expertIdx) * k * outputN *
                sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr;
    }
}

__aicore__ inline void MegaMoeBF16MGroup::Process()
{
    using namespace MegaMoeBF16Training;
    switch (ResolveSavedGateUpMode(params_.tilingData->savedGateUpMode, params_.tilingData->swigluLimit)) {
        case kSaveNone: ProcessWithSaveMode<kSaveNone>(); break;
        case kSaveBf16: ProcessWithSaveMode<kSaveBf16>(); break;
        default: ProcessWithSaveMode<kSaveMxFp8>(); break;
    }
}

template<uint32_t SaveMode>
__aicore__ inline void MegaMoeBF16MGroup::ProcessWithSaveMode()
{
#if defined(ENABLE_TIMER)
    if constexpr (g_coreType == AscendC::AIV) {
        timer_.template Tok<Overwrite>(AscendTimer::AIV_INIT_PARAM);
        timer_.Tik();
    }
#endif
    const int64_t oriOverflowMode =
        GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();

    moe_init_routing_v3<bfloat16_t, QuantOutType>(
        params_.aGmAddr,
        params_.expertIdxGmAddr,
        nullptr,
        nullptr,
        params_.peermemInfo.ptrA0,
        params_.workspaceInfo.expandedRowIdx,
        tokenPerExpertCurRankAddr_,
        nullptr,
        params_.workspaceInfo.expandedRowIdxGather,
        params_.workspaceInfo.ptrA,
        &params_.tilingData->moeInitRoutingTilingData,
        params_.tilingData->moeInitRoutingTilingData.tilingKey);
    // Use the shared short-lived gather phase before MGroup/Dispatch UB setup.
    bool routedWeightsReady = false;
    if constexpr (g_coreType == AscendC::AIV) {
        routedWeightsReady = MegaMoeBF16Routing::TryGatherTopkWeights(
            params_.probsGmAddr, params_.workspaceInfo.expandedRowIdx,
            params_.peermemInfo.ptrRoutedTopkWeights,
            static_cast<uint64_t>(m_) * topK_, aivCoreIdx_, blockAivNum_);
    }
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);
    if (!routedWeightsReady) {
        GatherRoutedTopkWeights();
    }
    SyncAll<true>();
    ResetWorkSpaceFlagList();
    ResetMGroupReadyFlags();
    SyncAll<true>();
    CrossRankSyncCntPerExpert();

    if (subBlockIdx_ == 0) {
        PipeBarrier<PIPE_ALL>();
        firstPipe.Destroy();
    }
    SyncAll<false>();

    // Phase A: preserve the production Dispatch protocol, but finish all
    // Dispatch work before the first compute-only Wave.  This keeps the
    // AIV1 Dispatch UB lifetime disjoint from GMM2 fixpipe UB.
    GMMAddrInfo dispatchAddrInfo{};
    preOffset_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(dispatchAddrInfo, 0);
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 1) {
                GlobalTensor<QuantOutType> dispatchedX;
                dispatchedX.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(dispatchAddrInfo.aGlobal));
                DispatchTokens(dispatchAddrInfo, expertIdx, dispatchedX);
            }
        }
    }

    if constexpr (g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
            NotifyCube(GMM2_UB_READY_FLAG);
        }
    }
    if constexpr (g_coreType == AscendC::AIC) {
        WaitForVector(GMM2_UB_READY_FLAG + FLAG_ID_MAX);
    }

    const uint32_t modelN = params_.tilingData->n;
    const uint32_t modelK = k_;
    Get<N_VALUE>(problemShape_) = modelN;
    Get<K_VALUE>(problemShape_) = modelK;
    epilogueOp_.Init({
        params_.workspaceInfo.ptrA2,
        params_.workspaceInfo.ptrA2Scale,
        ReadyFlagBase(),
        params_.workspaceInfo.ptrDispatchedTopkWeights,
        nullptr,
        nullptr,
        nullptr,
        ALIGN_256,
        ALIGN_256,
        params_.tilingData->swigluLimit,
        params_.workspaceInfo.ptrSavedGate,
        params_.workspaceInfo.ptrSavedUp,
        params_.workspaceInfo.ptrSavedGateScale, params_.workspaceInfo.ptrSavedUpScale,
        params_.workspaceInfo.ptrSavedGateHigh, params_.workspaceInfo.ptrSavedUpLow,
        params_.workspaceInfo.ptrSavedUpHigh});
    epilogueOp_.UpdateNextProblem(
        {0, modelN / SWIGLU_N_HALF, modelK, 0});

    uint32_t gmm1StartBlock = 0;
    uint32_t gmm2StartBlock = 0;
    gmm1VecSetSyncCom_ = 0;
    gmm2VecSetSyncCom_ = 0;
    gmm2PingPongIdx_ = 0;

    ExpertRowCursor cursor{};
    const uint32_t groupsPerWave = GroupsPerWave();
    // Construct GMM2 first so its CATLASS event state can be moved from the
    // default IDs 0-3 to IDs 4-7 before GMM1 claims IDs 0-3.  The resources
    // are static handles to the same physical L1/L0/UB storage, while the two
    // software pipelines keep independent event state across all waves.
    MegaMoeBF16CatlassImpl::GroupMatmulParam<1> gmm2Params(
        modelK, modelN / SWIGLU_N_HALF);
    MegaMoeBF16CatlassImpl::GroupMatmulParam<0> gmm1Params(
        modelN, modelK, params_.tilingData->blockNumPerEP);
    while (cursor.expertIdx < expertPerRank_) {
        const ExpertRowCursor waveStartCursor = cursor;
        uint32_t waveGroupCount = 0;
        GMMAddrInfo gmmAddrInfo{};

        // Like the official Wave path, merge adjacent M-groups from the same
        // expert into one larger problem.  Crossing an expert boundary still
        // starts a separate problem because the weight address changes.
        while (waveGroupCount < groupsPerWave) {
            ExpertRowSlice slice{};
            const uint32_t remainingGroups =
                groupsPerWave - waveGroupCount;
            if (!NextSlice(cursor, remainingGroups, slice)) {
                break;
            }
            waveGroupCount += Ops::Base::CeilDiv(
                slice.Rows(), M_GROUP_ROWS);
            Get<M_VALUE>(problemShape_) = slice.Rows();
            SetSliceAddresses(gmmAddrInfo, slice, 0);
            MegaMoeBF16CatlassImpl::GroupMatmulSwiglu<false, false, SaveMode>(
                gmm1Params,
                epilogueOp_,
                params_,
                problemShape_,
                gmmAddrInfo,
                gmm1StartBlock,
                gmm1VecSetSyncCom_,
                slice.expertIdx,
                false,
                1  // Whole-expert MGroup path does not reorder Ready blocks.
                MEGA_MOE_BF16_TIMER_ARG(timer_));
        }
        gmm1Params.blockMmad.HandoffSharedLocalMemory();

        ExpertRowCursor gmm2Cursor = waveStartCursor;
        uint32_t replayedGroupCount = 0;
        while (replayedGroupCount < waveGroupCount) {
            ExpertRowSlice slice{};
            const uint32_t remainingGroups =
                waveGroupCount - replayedGroupCount;
            if (!NextSlice(gmm2Cursor, remainingGroups, slice)) {
                break;
            }
            replayedGroupCount += Ops::Base::CeilDiv(
                slice.Rows(), M_GROUP_ROWS);
            Get<M_VALUE>(problemShape_) = slice.Rows();
            SetSliceAddresses(gmmAddrInfo, slice, 1);
            MegaMoeBF16CatlassImpl::GroupMatmul2(
                gmm2Params,
                params_,
                problemShape_,
                gmmAddrInfo,
                gmm2StartBlock,
                gmm2VecSetSyncCom_,
                slice.expertIdx,
                slice.rowBegin,
                gmm2PingPongIdx_,
                rankId_
                MEGA_MOE_BF16_TIMER_ARG(timer_));
        }
        gmm2Params.blockMmad.HandoffSharedLocalMemory();
    }

    EndSync();
    EndGMM2Sync();
    SyncAll<true>();
    // ptrA, packed TopK weights and routing metadata are persistent. Snapshot
    // only the mutable SHMEM token-count table before counters are reset.
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1) {
            TPipe savePipe;
            savePipe.InitBuffer(
                SaveStateBuf_, MegaMoeForwardSave::kCopyBufferBytes);
            SaveForwardState();
            PipeBarrier<PIPE_ALL>();
            savePipe.Destroy();
        }
    }
    SyncAll<true>();
    if constexpr (g_coreType == AscendC::AIV) {
        ResetTokenPerExpert();
        SyncAll<true>();
        CrossRankSyncInWorldSize();
        UnpermuteV2();
        PipeBarrier<PIPE_ALL>();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
#if defined(ENABLE_TIMER)
    timerAll_.template Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
#endif
}

}  // namespace MegaMoeImpl

#endif  // DEEPEP_MEGA_MOE_BF16_MGROUP_H
