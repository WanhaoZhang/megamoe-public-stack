#pragma once

#include "mega_moe_shared_expert_descriptor.h"
#include "mega_moe_bf16_catlass_impl.h"

namespace MegaMoeSharedExpert {
using namespace AscendC;
namespace TilePipeline {
using namespace AscendC;
using namespace MegaMoeSharedExpertLayout;

__aicore__ inline uint32_t Core()
{
    return GetBlockIdx() / GetTaskRation();
}

__aicore__ inline __gm__ int32_t *GateReady(
    const MegaMoeSharedExpertDescriptor &d, uint64_t task)
{
    return reinterpret_cast<__gm__ int32_t *>(d.ready) + GateUpReadyOffset(task);
}

__aicore__ inline __gm__ int32_t *ActivationReady(
    const MegaMoeSharedExpertDescriptor &d, uint64_t row)
{
    return reinterpret_cast<__gm__ int32_t *>(d.ready) +
        ActivationRowReadyOffset(d.tokens, d.ffn, row);
}

__aicore__ inline void Stamp(
    const MegaMoeSharedExpertDescriptor &d, uint64_t offset, uint32_t edge)
{
#if defined(ENABLE_TIMER) && !defined(MEGAFLUX_SUMMARY_TIMER)
    if (d.timer) {
        WriteGmByPassDCache<int64_t>(
            reinterpret_cast<__gm__ int64_t *>(d.timer) + offset + edge,
            static_cast<int64_t>(GetSystemCycle()));
    }
#endif
}

// One scope per participating core, including dependencies/interleaving.
__aicore__ inline void PhaseStamp(const MegaMoeSharedExpertDescriptor &d,int phase,int edge) {
#if defined(ENABLE_TIMER) && defined(MEGAFLUX_SUMMARY_TIMER)
    if(d.timer)WriteGmByPassDCache<int64_t>((__gm__ int64_t*)d.timer+(phase*128+Core())*2+edge,GetSystemCycle());
#endif
}

// One object owns the AIC's existing CATLASS local resources. Every task is a
// real 128x256 output tile with complete K. GU and Down share these resources
// sequentially; there is no second MMAD owner or split-K output accumulation.
class GemmTiles {
    using Impl = MegaMoeBF16CatlassImpl::ForwardGmTileCopy<Catlass::layout::ColumnMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        MegaMoeBF16CatlassImpl::DispatchPolicy,
        MegaMoeBF16CatlassImpl::L1TileShape,
        MegaMoeBF16CatlassImpl::L0TileShape,
        bfloat16_t, bfloat16_t, bfloat16_t, void, Impl>;

public:
    __aicore__ inline GemmTiles() : mm_(resource_) {}

    __aicore__ inline void Run(
        uint64_t input, uint64_t weight, uint64_t output,
        uint32_t m, uint32_t n, uint32_t k, uint32_t row, uint32_t column)
    {
        GlobalTensor<bfloat16_t> ag, bg, cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weight));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        const auto al = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
        const auto bl = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(k, n);
        const auto cl = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, n);
        auto at = tla::MakeTensor(ag, al, Catlass::Arch::PositionGM{});
        auto bt = tla::MakeTensor(bg, bl, Catlass::Arch::PositionGM{});
        auto ct = tla::MakeTensor(cg, cl, Catlass::Arch::PositionGM{});
        const uint32_t rows = m - row < kTileRows ? m - row : kTileRows;
        const uint32_t columns = n - column < kTileColumns ? n - column : kTileColumns;
        auto a = tla::GetTile(at, tla::MakeCoord(row, 0U), tla::MakeShape(rows, k));
        auto b = tla::GetTile(bt, tla::MakeCoord(0U, column), tla::MakeShape(k, columns));
        auto c = tla::GetTile(ct, tla::MakeCoord(row, column), tla::MakeShape(rows, columns));
        mm_(a, b, c, Catlass::GemmCoord{rows, columns, k});
    }

    __aicore__ inline void Drain()
    {
        // UNIT_FLAG MMAD may return with FIX->GM still in flight. Publish no
        // tile readiness, reuse no output, and destroy no owner before this.
        SetFlag<HardEvent::FIX_S>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::ALL>();
    }

private:
    Catlass::Arch::Resource<MegaMoeBF16CatlassImpl::ArchTag> resource_;
    BlockMmad mm_;
};

__aicore__ inline void GateUpTile(
    const MegaMoeSharedExpertDescriptor &d, GM_ADDR input, GemmTiles &mm, uint64_t task)
{
    const uint64_t columns = ColumnTiles(d.ffn);
    const uint32_t row = static_cast<uint32_t>(task / columns) * kTileRows;
    const uint32_t column = static_cast<uint32_t>(task % columns) * kTileColumns;
    const uint64_t record = GateUpTimerOffset(task);
    Stamp(d, record, 0);
    mm.Run(reinterpret_cast<uint64_t>(input), d.weight_gate, d.raw_gate,
        static_cast<uint32_t>(d.tokens), static_cast<uint32_t>(d.ffn),
        static_cast<uint32_t>(d.hidden), row, column);
    mm.Run(reinterpret_cast<uint64_t>(input), d.weight_up, d.raw_up,
        static_cast<uint32_t>(d.tokens), static_cast<uint32_t>(d.ffn),
        static_cast<uint32_t>(d.hidden), row, column);
    mm.Drain();
    WriteGmByPassDCache(GateReady(d, task), int32_t{1});
    Stamp(d, record, 1);
}

__aicore__ inline bool IsActivationReady(
    const MegaMoeSharedExpertDescriptor &d, uint64_t row)
{
    return ReadGmByPassDCache(ActivationReady(d, row)) >=
        static_cast<int32_t>(ColumnTiles(d.ffn));
}

__aicore__ inline void DownTile(
    const MegaMoeSharedExpertDescriptor &d, GM_ADDR output, GemmTiles &mm, uint64_t task)
{
    const uint64_t columns = ColumnTiles(d.hidden);
    const uint32_t row = static_cast<uint32_t>(task / columns) * kTileRows;
    const uint32_t column = static_cast<uint32_t>(task % columns) * kTileColumns;
    const uint64_t record = DownTimerOffset(d.tokens, d.ffn, task);
    DataSyncBarrier<MemDsbT::ALL>();
    Stamp(d, record, 0);
    mm.Run(d.activation, d.weight_down, reinterpret_cast<uint64_t>(output),
        static_cast<uint32_t>(d.tokens), static_cast<uint32_t>(d.hidden),
        static_cast<uint32_t>(d.ffn), row, column);
    mm.Drain();
    Stamp(d, record, 1);
}

// Keep VF captures limited to UB pointers and scalar values. Clipping dispatch
// stays outside VF, matching the compiler-compatible backward recompute path.
template<bool Clip>
__aicore__ inline void ActivationVector(
    LocalTensor<bfloat16_t> gate, LocalTensor<bfloat16_t> up,
    uint32_t count, float limit)
{
    auto *gptr = reinterpret_cast<__ubuf__ bfloat16_t *>(gate.GetPhyAddr());
    auto *uptr = reinterpret_cast<__ubuf__ bfloat16_t *>(up.GetPhyAddr());
    constexpr uint32_t lanes = 256 / sizeof(float);
    const uint16_t repeats = static_cast<uint16_t>((count + lanes - 1) / lanes);
    __VEC_SCOPE__ {
        Reg::RegTensor<float> hi, lo;
        if constexpr (Clip) {
            Reg::Duplicate(hi, limit);
            Reg::Duplicate(lo, -limit);
        }
        uint32_t remaining = count;
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t offset = static_cast<uint32_t>(repeat) * lanes;
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::RegTensor<bfloat16_t> gb, ub, result;
            Reg::RegTensor<float> gf, uf, neg, ex, denominator, silu, value;
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(gb, gptr + offset);
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(ub, uptr + offset);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kCastBF16ToFP32>(gf, gb, mask);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kCastBF16ToFP32>(uf, ub, mask);
            if constexpr (Clip) {
                Reg::Min(gf, gf, hi, mask);
                Reg::Max(uf, uf, lo, mask);
                Reg::Min(uf, uf, hi, mask);
            }
            Reg::Muls(neg, gf, -1.0F, mask);
            Reg::Exp(ex, neg, mask);
            Reg::Adds(denominator, ex, 1.0F, mask);
            Reg::Div<float, &MegaMoeBF16Impl::kDivMode>(silu, gf, denominator, mask);
            Reg::Mul(value, silu, uf, mask);
            Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kCastFP32ToBF16>(result, value, mask);
            Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(gptr + offset, result, mask);
        }
    }
}

__aicore__ inline void SwigluTile(
    const MegaMoeSharedExpertDescriptor &d, uint64_t task, float limit)
{
    constexpr uint32_t elements = kTileRows * kTileColumns;
    LocalTensor<bfloat16_t> gate{TPosition::VECIN, 0, elements};
    LocalTensor<bfloat16_t> up{TPosition::VECIN, elements * uint32_t{sizeof(bfloat16_t)}, elements};
    const uint64_t columns = ColumnTiles(d.ffn);
    const uint64_t rowIndex = task / columns;
    const uint64_t row = rowIndex * kTileRows;
    const uint64_t column = (task % columns) * kTileColumns;
    const uint32_t rows = static_cast<uint32_t>(
        d.tokens - row < kTileRows ? d.tokens - row : kTileRows);
    const uint32_t width = static_cast<uint32_t>(
        d.ffn - column < kTileColumns ? d.ffn - column : kTileColumns);
    GlobalTensor<bfloat16_t> gateGm, upGm, activationGm;
    gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(d.raw_gate));
    upGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(d.raw_up));
    activationGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(d.activation));
    gateGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    upGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    activationGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    const uint64_t offset = row * d.ffn + column;
    const uint64_t record = SwigluTimerOffset(d.tokens, d.ffn, task);
    Stamp(d, record, 0);
    SyncFunc<HardEvent::V_MTE2>();
    SyncFunc<HardEvent::MTE3_MTE2>();
    DataCopyExtParams load{static_cast<uint16_t>(rows), width * uint32_t{sizeof(bfloat16_t)},
        static_cast<uint32_t>((d.ffn - width) * sizeof(bfloat16_t)), 0, 0};
    DataCopyPad(gate, gateGm[offset], load, {});
    DataCopyPad(up, upGm[offset], load, {});
    SyncFunc<HardEvent::MTE2_V>();
    // Host contract requires F divisible by 256, so each real column tile has
    // width=256. M may end in a partial 128-row tile without extra elements.
    const uint32_t count = rows * width;
    if (limit == kMegaMoeSwigluLimitDisabled) {
        ActivationVector<false>(gate, up, count, limit);
    } else {
        ActivationVector<true>(gate, up, count, limit);
    }
    SyncFunc<HardEvent::V_MTE3>();
    DataCopyExtParams store{static_cast<uint16_t>(rows), width * uint32_t{sizeof(bfloat16_t)},
        0, static_cast<uint32_t>((d.ffn - width) * sizeof(bfloat16_t)), 0};
    DataCopyPad(activationGm[offset], gate, store);
    SyncFunc<HardEvent::MTE3_S>();
    DataSyncBarrier<MemDsbT::ALL>();
    // Independent N tiles atomically retire into one padded M-row counter.
    // A Down tile is enabled only when every F tile for these rows retired.
    AtomicAdd(ActivationReady(d, rowIndex), int32_t{1});
    Stamp(d, record, 1);
}
}  // namespace TilePipeline

// This startup never reads activation readiness. Every AIC must reach the
// original mixed SyncAll so both AIV lanes can finish InitRouting first.
__aicore__ inline void GateUpStartup(
    const MegaMoeSharedExpertDescriptor &d, GM_ADDR input)
{
    if constexpr (g_coreType == AscendC::AIC) {
        const uint64_t task = TilePipeline::Core();
        if (task < MegaMoeSharedExpertLayout::GateUpTasks(d.tokens, d.ffn)) {
            TilePipeline::PhaseStamp(d,0,0);
            TilePipeline::GemmTiles mm;
            TilePipeline::GateUpTile(d, input, mm, task);
        }
    }
}

// Called only after the existing mixed SyncAll. Static task ownership makes
// every AIC/AIV0 pair useful; ready Down work is interleaved fairly with GU.
// Full persistent GM planes remove producer backpressure and cross-row reuse.
__aicore__ inline void ProcessTiled(
    const MegaMoeSharedExpertDescriptor &d, GM_ADDR input, GM_ADDR output, float limit)
{
    using namespace MegaMoeSharedExpertLayout;
    const uint64_t workers = GetBlockNum();
    const uint64_t core = TilePipeline::Core();
    const uint64_t guTasks = GateUpTasks(d.tokens, d.ffn);
    if constexpr (g_coreType == AscendC::AIC) {
        TilePipeline::GemmTiles mm;
        uint64_t gu = core + workers;  // startup already produced task=core
        uint64_t down = core;
        const uint64_t downTasks = DownTasks(d.tokens, d.hidden);
        const uint64_t downColumns = ColumnTiles(d.hidden);
        if(core>=guTasks&&core<downTasks)TilePipeline::PhaseStamp(d,0,0);
        bool allowDown = true;
        while (gu < guTasks || down < downTasks) {
            const bool haveGu = gu < guTasks;
            if (down < downTasks && (allowDown || !haveGu) &&
                TilePipeline::IsActivationReady(d, down / downColumns)) {
                TilePipeline::DownTile(d, output, mm, down);
                down += workers;
                // At most one Down tile between GU pairs while this core
                // still owns GU tasks. Neither queue can starve the other.
                allowDown = false;
            } else if (haveGu) {
                TilePipeline::GateUpTile(d, input, mm, gu);
                gu += workers;
                allowDown = true;
            } else {
                // Only local Down tasks remain. Other cores/AIV0 can still
                // finish the required GU/S tiles; no producer waits on us.
                while (!TilePipeline::IsActivationReady(d, down / downColumns)) {}
            }
        }
        mm.Drain();
        if(core<guTasks||core<downTasks)TilePipeline::PhaseStamp(d,0,1);
        WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(d.ready) +
            (2 * kMaxCores + core) * kReadyLineInts, int32_t{1});
    } else if constexpr (g_coreType == AscendC::AIV) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
        if(core<guTasks)TilePipeline::PhaseStamp(d,1,0);
        for (uint64_t task = core; task < guTasks; task += workers) {
            while (ReadGmByPassDCache(TilePipeline::GateReady(d, task)) != 1) {}
            DataSyncBarrier<MemDsbT::ALL>();
            TilePipeline::SwigluTile(d, task, limit);
        }
        PipeBarrier<PIPE_ALL>();
        if(core<guTasks)TilePipeline::PhaseStamp(d,1,1);
    }
}
}  // namespace MegaMoeSharedExpert
