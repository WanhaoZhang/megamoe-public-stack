/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef MEGA_MOE_SHARED_EXPERT_BACKWARD_H
#define MEGA_MOE_SHARED_EXPERT_BACKWARD_H

#include "mega_moe_shared_expert_types.h"
#include "mega_moe_bf16_backward_catlass_impl.h"
#include "shared_backward_vector_schedule.h"

namespace MegaMoeSharedExpert {
using namespace AscendC;
namespace Bwd = MegaMoeBF16BackwardCatlassImpl;

enum ControlPhase : uint32_t {
    DA_READY = 0,
    DZ_READY = 1,
    DX_READY = 2,
    VECTOR_RETIRED = 3,
    RECOMPUTE_GU_READY = 4,
    RECOMPUTE_A_READY = 5
};
constexpr uint32_t kVectorElements = VectorSchedule::kMaxElements;
enum class SharedVectorOp { Vjp, Recompute, Merge };
template<SharedVectorOp Op>
__aicore__ inline void RunSharedVectorTiles(const SharedExpertBackwardArgs& a,
    uint64_t rawGate, uint64_t rawUp, uint64_t rowStride, bool clip);

// Separate V2 raw-cycle tensor; never writes into the routed Timer schema.
// Phases 0/2/4..8 use AIC lane 0..31; phases 1/3/9 use AIV lane 0..63.
__aicore__ inline void BackwardStampV2(uint64_t timer, uint32_t phase, uint32_t edge)
{
#if defined(ENABLE_TIMER)
    if (timer) {
        const uint32_t lane = g_coreType == AIC
            ? GetBlockIdx() / GetTaskRation() : GetBlockIdx();
        auto p = reinterpret_cast<__gm__ int64_t*>(timer) +
            (phase * kBackwardTimerLanesV2 + lane) * 2 + edge;
        AscendC::WriteGmByPassDCache<int64_t>(p, static_cast<int64_t>(GetSystemCycle()));
    }
#endif
}

__aicore__ inline __gm__ int32_t* Flag(
    const SharedExpertBackwardArgs& a, uint32_t phase, uint32_t worker)
{
    return reinterpret_cast<__gm__ int32_t*>(a.control) +
        (phase * kMaxParticipants + worker) * kControlLineWords;
}

// One producer per cache line and exact epochs: no counter reset or reuse can
// turn a previous invocation into readiness for this invocation.
__aicore__ inline void Publish(
    const SharedExpertBackwardArgs& a, uint32_t phase, uint32_t worker)
{
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::ALL>();
    auto p = Flag(a, phase, worker);
    *p = static_cast<int32_t>(a.epoch);
    GlobalTensor<int32_t> flag;
    flag.SetGlobalBuffer(p);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(flag);
    DataSyncBarrier<MemDsbT::ALL>();
}

__aicore__ inline void AcquireAll(
    const SharedExpertBackwardArgs& a, uint32_t phase, uint32_t workers)
{
    for (uint32_t worker = 0; worker < workers; ++worker) {
        auto p = Flag(a, phase, worker);
        while (AscendC::ReadGmByPassDCache(p) != static_cast<int32_t>(a.epoch)) {}
    }
    DataSyncBarrier<MemDsbT::ALL>();
}

// Independent, complete-K CATLASS GEMM. Its resource is destroyed before the
// caller publishes readiness or constructs the next GEMM. No routed tensor,
// shape, cursor, tile-ready flag or operand-ready context is modified.
template<class LayoutA, class LayoutB>
__aicore__ inline void GemmTiles(Bwd::BackwardGroupMatmulParam<LayoutA, LayoutB, 0, true>& mm,
    uint64_t aAddr,uint64_t bAddr,uint64_t cAddr,uint32_t m,uint32_t n,uint32_t k) {
            GlobalTensor<bfloat16_t> a, b, c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(aAddr));
            b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bAddr));
            c.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cAddr));
            auto ta = tla::MakeTensor(a, tla::MakeLayout<bfloat16_t, LayoutA>(m, k),
                Catlass::Arch::PositionGM{});
            auto tb = tla::MakeTensor(b, tla::MakeLayout<bfloat16_t, LayoutB>(k, n),
                Catlass::Arch::PositionGM{});
            auto tc = tla::MakeTensor(c, tla::MakeLayout<bfloat16_t, Bwd::RowMajor>(m, n),
                Catlass::Arch::PositionGM{});
            Bwd::BlockScheduler scheduler({m, n, k},
                Catlass::MakeCoord(Bwd::L1_TILE_M, Bwd::L1_TILE_N));
            const uint32_t loops = scheduler.GetCoreLoops();
            for (uint32_t i = mm.blockIdx; i < loops; i += mm.blockNum) {
                auto coord = scheduler.GetBlockCoord(i);
                auto shape = scheduler.GetActualBlockShape(coord);
                const uint32_t row = coord.m() * Bwd::L1_TILE_M;
                const uint32_t col = coord.n() * Bwd::L1_TILE_N;
                const uint32_t depth = coord.k() * Bwd::L1_TILE_K;
                auto ba = tla::GetTile(ta, tla::MakeCoord(row, depth),
                    tla::MakeShape(shape.m(), shape.k()));
                auto bb = tla::GetTile(tb, tla::MakeCoord(depth, col),
                    tla::MakeShape(shape.k(), shape.n()));
                auto bc = tla::GetTile(tc, tla::MakeCoord(row, col),
                    tla::MakeShape(shape.m(), shape.n()));
                mm.blockMmad(ba, bb, bc, shape);
            }
    PipeBarrier<PIPE_ALL>();
}
template<class LayoutA, class LayoutB = Bwd::RowMajor>
__aicore__ inline void Gemm(uint64_t aAddr,uint64_t bAddr,uint64_t cAddr,uint32_t m,uint32_t n,uint32_t k) {
    if constexpr (g_coreType == AIC) {
        { Bwd::BackwardGroupMatmulParam<LayoutA,LayoutB,0,true> mm;
          GemmTiles<LayoutA,LayoutB>(mm,aAddr,bAddr,cAddr,m,n,k); }
        PipeBarrier<PIPE_ALL>();
    }
}
template<class LayoutA, class LayoutB = Bwd::RowMajor>
__aicore__ inline void GemmPair(uint64_t a0,uint64_t b0,uint64_t c0,
    uint64_t a1,uint64_t b1,uint64_t c1,uint32_t m,uint32_t n,uint32_t k,
    uint64_t timer=0,int phase0=-1,int phase1=-1) {
    if constexpr (g_coreType == AIC) {
      {
        Bwd::BackwardGroupMatmulParam<LayoutA,LayoutB,0,true> mm;
        if(phase0>=0)BackwardStampV2(timer,phase0,0);
        GemmTiles<LayoutA,LayoutB>(mm,a0,b0,c0,m,n,k);
        if(phase0>=0)BackwardStampV2(timer,phase0,1);
        if(phase1>=0)BackwardStampV2(timer,phase1,0);
        GemmTiles<LayoutA,LayoutB>(mm,a1,b1,c1,m,n,k);
        if(phase1>=0)BackwardStampV2(timer,phase1,1);
      }
      PipeBarrier<PIPE_ALL>();
    }
}

// The raw GU supplies the clamp derivative, including equality at both clamp
// boundaries. dA, raw G/U, sigmoid and intermediates are FP32 inside the VF;
// only the two dG/dU outputs cross a BF16 tensor boundary.
__aicore__ inline void SwiGLUVjp(LocalTensor<bfloat16_t> da,
    LocalTensor<bfloat16_t> gate, LocalTensor<bfloat16_t> up, float limit,
    uint32_t elements, bool clip = true)
{
    auto dp = reinterpret_cast<__ubuf__ bfloat16_t*>(da.GetPhyAddr());
    auto gp = reinterpret_cast<__ubuf__ bfloat16_t*>(gate.GetPhyAddr());
    auto uptr = reinterpret_cast<__ubuf__ bfloat16_t*>(up.GetPhyAddr());
    __VEC_SCOPE__ {
        uint32_t remaining = elements;
        for (uint16_t repeat = 0; repeat < (elements + 63) / 64; ++repeat) {
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::RegTensor<bfloat16_t> db, gb, ub, dgb, dub;
            Reg::RegTensor<float> d, rawG, rawU, g, u, one, zero, hi, lo;
            Reg::RegTensor<float> neg, ex, denom, sig, silu, inner, tmp, dg, du;
            Reg::MaskReg gateInside, upAboveLow, upBelowHigh;
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(db, dp + repeat * 64);
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(gb, gp + repeat * 64);
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(ub, uptr + repeat * 64);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(d, db, mask);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(rawG, gb, mask);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(rawU, ub, mask);
            Reg::Duplicate(one, 1.0F, mask);
            Reg::Duplicate(zero, 0.0F, mask);
            Reg::Duplicate(hi, limit, mask);
            Reg::Duplicate(lo, -limit, mask);
            if (clip) {
                Reg::Compare<float, CMPMODE::LE>(gateInside, rawG, hi, mask);
                Reg::Compare<float, CMPMODE::GE>(upAboveLow, rawU, lo, mask);
                Reg::Compare<float, CMPMODE::LE>(upBelowHigh, rawU, hi, mask);
                Reg::Min(g, rawG, hi, mask);
                Reg::Max(u, rawU, lo, mask);
                Reg::Min(u, u, hi, mask);
            } else {
                Reg::Muls(g, rawG, 1.0F, mask);
                Reg::Muls(u, rawU, 1.0F, mask);
            }
            Reg::Muls(neg, g, -1.0F, mask);
            Reg::Exp(ex, neg, mask);
            Reg::Adds(denom, ex, 1.0F, mask);
            Reg::Div<float, &MegaMoeBF16Impl::kBackwardDivMode>(sig, one, denom, mask);
            Reg::Mul(silu, g, sig, mask);
            Reg::Mul(du, d, silu, mask);
            if (clip) {
                Reg::Select<float>(du, du, zero, upAboveLow);
                Reg::Select<float>(du, du, zero, upBelowHigh);
            }
            Reg::Sub(tmp, one, sig, mask);
            Reg::Mul(inner, g, tmp, mask);
            Reg::Adds(inner, inner, 1.0F, mask);
            // Match Triton's FP32 grouping: dsilu = sig * inner, then
            // dG = (dA * U) * dsilu. Reassociation changes BF16 boundaries.
            Reg::Mul(inner, sig, inner, mask);
            Reg::Mul(tmp, d, u, mask);
            Reg::Mul(dg, tmp, inner, mask);
            if (clip) {
                Reg::Select<float>(dg, dg, zero, gateInside);
            }
            Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kBackwardCastFP32ToBF16>(dgb, dg, mask);
            Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kBackwardCastFP32ToBF16>(dub, du, mask);
            Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(dp + repeat * 64, dgb, mask);
            Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(gp + repeat * 64, dub, mask);
        }
    }
}

__aicore__ inline void ProduceDZPlanes(const SharedExpertBackwardArgs& a,
    uint64_t rawGate, uint64_t rawUp, uint64_t rowStride, bool clip = true)
{
    RunSharedVectorTiles<SharedVectorOp::Vjp>(a, rawGate, rawUp, rowStride, clip);
}

__aicore__ inline void AddBF16(LocalTensor<bfloat16_t> dst,
    LocalTensor<bfloat16_t> src, uint32_t elements)
{
    auto d = reinterpret_cast<__ubuf__ bfloat16_t*>(dst.GetPhyAddr());
    auto s = reinterpret_cast<__ubuf__ bfloat16_t*>(src.GetPhyAddr());
    __VEC_SCOPE__ {
        uint32_t remaining = elements;
        for (uint16_t repeat = 0; repeat < (elements + 63) / 64; ++repeat) {
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::RegTensor<bfloat16_t> db, sb, result;
            Reg::RegTensor<float> df, sf;
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(db, d + repeat * 64);
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(sb, s + repeat * 64);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(df, db, mask);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(sf, sb, mask);
            Reg::Add(df, df, sf, mask);
            Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kBackwardCastFP32ToBF16>(result, df, mask);
            Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(d + repeat * 64, result, mask);
        }
    }
}

__aicore__ inline void MergeDX(const SharedExpertBackwardArgs& a)
{
    RunSharedVectorTiles<SharedVectorOp::Merge>(a, a.dxg, a.dxu, a.hidden, false);
}

// S0 is deliberately serial across shared and routed branches, but all shared
// GEMMs and VJP execute inside the same mixed-core MegaMoE device launch.
// Complete the prefix and retire its CATLASS/TPipe resources before routed P0.
__aicore__ inline void RunBackwardS0Planes(const SharedExpertBackwardArgs& a,
    GM_ADDR dy, uint64_t rawGate, uint64_t rawUp, uint64_t rowStride,
    uint64_t timer = 0, bool clip = true, bool recompute = false)
{
    const uint32_t aics = GetBlockNum();
    const uint32_t aivs = aics * 2;
    if constexpr (g_coreType == AIC) {
        const uint32_t worker = GetBlockIdx() / GetTaskRation();
        BackwardStampV2(timer, 2, 0);
        Gemm<Bwd::RowMajor>(reinterpret_cast<uint64_t>(dy), a.wd, a.da, a.m, a.ffn, a.hidden);
        BackwardStampV2(timer, 2, 1);
        Publish(a, DA_READY, worker);
        if (recompute) AcquireAll(a, RECOMPUTE_A_READY, aivs);
        // dW_down reads dY/A only: hide vector VJP before waiting for dG/dU.
        BackwardStampV2(timer, 4, 0);
        Gemm<Bwd::ColumnMajor>(reinterpret_cast<uint64_t>(dy), a.a, a.dwd, a.hidden, a.ffn, a.m);
        BackwardStampV2(timer, 4, 1);
        AcquireAll(a, DZ_READY, aivs);
        GemmPair<Bwd::ColumnMajor>(a.dg,a.x,a.dwg,a.du,a.x,a.dwu,a.ffn, a.hidden, a.m,timer,5,6);
        GemmPair<Bwd::RowMajor>(a.dg,a.wg,a.dxg,a.du,a.wu,a.dxu,a.m, a.hidden, a.ffn,timer,7,8);
        Publish(a, DX_READY, worker);
    } else {
        AcquireAll(a, DA_READY, aics);
        BackwardStampV2(timer, 3, 0);
        ProduceDZPlanes(a, rawGate, rawUp, rowStride, clip);
        BackwardStampV2(timer, 3, 1);
        Publish(a, DZ_READY, GetBlockIdx());
        AcquireAll(a, DX_READY, aics);
        BackwardStampV2(timer, 9, 0);
        MergeDX(a);
        BackwardStampV2(timer, 9, 1);
        Publish(a, VECTOR_RETIRED, GetBlockIdx());
    }
    AcquireAll(a, VECTOR_RETIRED, aivs);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunBackwardS0(const SharedExpertBackwardArgs& a, GM_ADDR dy)
{
    RunBackwardS0Planes(a, dy, a.gu, a.gu + uint64_t{a.ffn} * sizeof(bfloat16_t),
        uint64_t{2} * a.ffn);
}

// Restore only the shared activation, preserving the forward's FP32
// g/(1+exp(-g))*u grouping and BF16 output. No planner, token transport or
// routed forward is invoked by the V2 recompute mode.
// Keep the VF capture small: only two UB addresses and one FP32 limit.
// Keep descriptor loads and runtime clip dispatch outside the VF: B237's
// instruction selector crashed on the original in-loop recompute VF.
template<bool Clip>
__aicore__ inline void RecomputeActivationVector(
    LocalTensor<bfloat16_t> gate, LocalTensor<bfloat16_t> up, float limit, uint32_t elements)
{
    auto gp = reinterpret_cast<__ubuf__ bfloat16_t*>(gate.GetPhyAddr());
    auto uptr = reinterpret_cast<__ubuf__ bfloat16_t*>(up.GetPhyAddr());
    __VEC_SCOPE__ {
        uint32_t remaining = elements;
        for (uint16_t repeat = 0; repeat < (elements + 63) / 64; ++repeat) {
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::RegTensor<bfloat16_t> gb, ub, result;
            Reg::RegTensor<float> g, u, hi, lo, neg, ex, denom, silu, value;
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(gb, gp + repeat * 64);
            Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(ub, uptr + repeat * 64);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(g, gb, mask);
            Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kBackwardCastBF16ToFP32>(u, ub, mask);
            Reg::Duplicate(hi, limit, mask);
            Reg::Duplicate(lo, -limit, mask);
            if constexpr (Clip) {
                Reg::Min(g, g, hi, mask);
                Reg::Max(u, u, lo, mask);
                Reg::Min(u, u, hi, mask);
            }
            Reg::Muls(neg, g, -1.0F, mask);
            Reg::Exp(ex, neg, mask);
            Reg::Adds(denom, ex, 1.0F, mask);
            Reg::Div<float, &MegaMoeBF16Impl::kBackwardDivMode>(silu, g, denom, mask);
            Reg::Mul(value, silu, u, mask);
            Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kBackwardCastFP32ToBF16>(result, value, mask);
            Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(gp + repeat * 64, result, mask);
        }
    }
}

__aicore__ inline void LoadSharedVectorPlane(LocalTensor<bfloat16_t> dst,
    GlobalTensor<bfloat16_t> src, const VectorSchedule::Tile& tile, uint64_t stride)
{
    DataCopyPad(dst, src[uint64_t{tile.row} * stride + tile.column],
        DataCopyExtParams{static_cast<uint16_t>(tile.rows), tile.columns * uint32_t{2},
            static_cast<uint32_t>((stride - tile.columns) * 2), 0, 0},
        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
}

__aicore__ inline void StoreSharedVectorPlane(GlobalTensor<bfloat16_t> dst,
    LocalTensor<bfloat16_t> src, const VectorSchedule::Tile& tile, uint32_t stride)
{
    DataCopyPad(dst[uint64_t{tile.row} * stride + tile.column], src,
        DataCopyExtParams{static_cast<uint16_t>(tile.rows), tile.columns * uint32_t{2},
            0, (stride - tile.columns) * uint32_t{2}, 0});
}

template<SharedVectorOp Op>
__aicore__ inline void LoadSharedVectorSlot(LocalTensor<bfloat16_t> first,
    LocalTensor<bfloat16_t> second, LocalTensor<bfloat16_t> third,
    GlobalTensor<bfloat16_t> input0, GlobalTensor<bfloat16_t> input1,
    GlobalTensor<bfloat16_t> input2, const VectorSchedule::Tile& tile,
    uint32_t width, uint64_t rowStride, uint32_t slot)
{
    const auto event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
    WaitFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    LoadSharedVectorPlane(first[slot * kVectorElements], input0, tile,
        Op == SharedVectorOp::Vjp ? width : rowStride);
    LoadSharedVectorPlane(second[slot * kVectorElements], input1, tile, rowStride);
    if constexpr (Op == SharedVectorOp::Vjp) {
        LoadSharedVectorPlane(third[slot * kVectorElements], input2, tile, rowStride);
    }
    SetFlag<HardEvent::MTE2_V>(event);
}

// Two private slots: next load overlaps the current VF/store. Both vector
// reads and MTE3 stores retire before a slot is reused. Drain every event
// before destroying TPipe; ABI3's transport later reuses static staging UB.
template<SharedVectorOp Op>
__aicore__ inline void RunSharedVectorTiles(const SharedExpertBackwardArgs& a,
    uint64_t rawGate, uint64_t rawUp, uint64_t rowStride, bool clip)
{
    if constexpr (g_coreType == AIV) {
        TPipe pipe;
        TBuf<> firstBuf, secondBuf, thirdBuf;
        pipe.InitBuffer(firstBuf, 2 * kVectorElements * sizeof(bfloat16_t));
        pipe.InitBuffer(secondBuf, 2 * kVectorElements * sizeof(bfloat16_t));
        if constexpr (Op == SharedVectorOp::Vjp) {
            pipe.InitBuffer(thirdBuf, 2 * kVectorElements * sizeof(bfloat16_t));
        }
        auto first = firstBuf.Get<bfloat16_t>();
        auto second = secondBuf.Get<bfloat16_t>();
        auto third = first;
        if constexpr (Op == SharedVectorOp::Vjp) third = thirdBuf.Get<bfloat16_t>();
        GlobalTensor<bfloat16_t> input0, input1, input2, output0, output1;
        input0.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
            Op == SharedVectorOp::Vjp ? a.da : rawGate));
        input1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
            Op == SharedVectorOp::Vjp ? rawGate : rawUp));
        input2.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rawUp));
        output0.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
            Op == SharedVectorOp::Vjp ? a.dg : (Op == SharedVectorOp::Merge ? a.dxg : a.a)));
        output1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.du));
        const uint32_t width = Op == SharedVectorOp::Merge ? a.hidden : a.ffn;
        const uint64_t tiles = VectorSchedule::Count(a.m, width);
        const uint32_t workers = GetBlockNum() * 2;
        const uint64_t begin = GetBlockIdx();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        if (begin < tiles) {
            LoadSharedVectorSlot<Op>(first, second, third, input0, input1, input2,
                VectorSchedule::At(a.m, width, begin), width, rowStride, 0);
        }
        uint32_t iteration = 0;
        for (uint64_t index = begin; index < tiles; index += workers, ++iteration) {
            const uint32_t slot = iteration & 1U;
            const auto event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::MTE2_V>(event);
            if (index + workers < tiles) {
                LoadSharedVectorSlot<Op>(first, second, third, input0, input1, input2,
                    VectorSchedule::At(a.m, width, index + workers), width, rowStride, slot ^ 1U);
            }
            const auto tile = VectorSchedule::At(a.m, width, index);
            auto v0 = first[slot * kVectorElements];
            auto v1 = second[slot * kVectorElements];
            if constexpr (Op == SharedVectorOp::Vjp) {
                SwiGLUVjp(v0, v1, third[slot * kVectorElements], a.limit, tile.Elements(), clip);
            } else if constexpr (Op == SharedVectorOp::Merge) {
                AddBF16(v0, v1, tile.Elements());
            } else {
                if (clip) RecomputeActivationVector<true>(v0, v1, a.limit, tile.Elements());
                else RecomputeActivationVector<false>(v0, v1, a.limit, tile.Elements());
            }
            SetFlag<HardEvent::V_MTE2>(event);
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            StoreSharedVectorPlane(output0, v0, tile, width);
            if constexpr (Op == SharedVectorOp::Vjp) {
                StoreSharedVectorPlane(output1, v1, tile, width);
            }
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void RecomputeActivationV2(const SharedExpertBackwardArgsV2& v)
{
    RunSharedVectorTiles<SharedVectorOp::Recompute>(v.base, v.raw_gate, v.raw_up,
        v.base.ffn, v.base.limit != kMegaMoeSwigluLimitDisabled);
}

__aicore__ inline void RunBackwardS0V2(const SharedExpertBackwardArgsV2& v, GM_ADDR dy)
{
    const auto& a = v.base;
    if (v.recompute) {
        if constexpr (g_coreType == AIC) {
            BackwardStampV2(v.timer, 0, 0);
            GemmPair<Bwd::RowMajor,Bwd::ColumnMajor>(a.x,a.wg,v.raw_gate,
                a.x,a.wu,v.raw_up,a.m,a.ffn,a.hidden);
            BackwardStampV2(v.timer, 0, 1);
            Publish(a, RECOMPUTE_GU_READY, GetBlockIdx() / GetTaskRation());
        } else {
            AcquireAll(a, RECOMPUTE_GU_READY, GetBlockNum());
            BackwardStampV2(v.timer, 1, 0);
            RecomputeActivationV2(v);
            BackwardStampV2(v.timer, 1, 1);
            Publish(a, RECOMPUTE_A_READY, GetBlockIdx());
        }
        // AIC continues with dA while AIV reconstructs A. Only dW_down
        // consumes A; RunBackwardS0Planes waits immediately before that GEMM.
    }
    RunBackwardS0Planes(a, dy, v.raw_gate, v.raw_up, a.ffn, v.timer,
        a.limit != kMegaMoeSwigluLimitDisabled, v.recompute != 0);
}

// ABI3 alone lends the vector-idle portions of S0 to routed transport. Each
// hook must retire its UB/MTE resources before it returns; all 64 AIVs take
// both collective hooks, including producers with no local payload.
template<class Hooks>
__aicore__ inline void RunBackwardS0V3(
    const SharedExpertBackwardArgsV3& v3, GM_ADDR dy, Hooks& hooks)
{
    const auto& v = v3.shared;
    const auto& a = v.base;
    const uint32_t aics = GetBlockNum(), aivs = aics * 2;
    if constexpr (g_coreType == AIV) {
        hooks.PrepareSources();
    }
    if (v.recompute) {
        if constexpr (g_coreType == AIC) {
            BackwardStampV2(v.timer, 0, 0);
            GemmPair<Bwd::RowMajor,Bwd::ColumnMajor>(a.x,a.wg,v.raw_gate,
                a.x,a.wu,v.raw_up,a.m,a.ffn,a.hidden);
            BackwardStampV2(v.timer, 0, 1);
            Publish(a, RECOMPUTE_GU_READY, GetBlockIdx() / GetTaskRation());
        } else {
            AcquireAll(a, RECOMPUTE_GU_READY, aics);
            BackwardStampV2(v.timer, 1, 0);
            RecomputeActivationV2(v);
            BackwardStampV2(v.timer, 1, 1);
            Publish(a, RECOMPUTE_A_READY, GetBlockIdx());
        }
        // dA is independent of reconstructed A; defer its wait to dW_down.
    }
    if constexpr (g_coreType == AIC) {
        const uint32_t worker = GetBlockIdx() / GetTaskRation();
        BackwardStampV2(v.timer, 2, 0);
        Gemm<Bwd::RowMajor>(reinterpret_cast<uint64_t>(dy), a.wd, a.da, a.m, a.ffn, a.hidden);
        BackwardStampV2(v.timer, 2, 1);
        Publish(a, DA_READY, worker);
        if (v.recompute) AcquireAll(a, RECOMPUTE_A_READY, aivs);
        BackwardStampV2(v.timer, 4, 0);
        Gemm<Bwd::ColumnMajor>(reinterpret_cast<uint64_t>(dy), a.a, a.dwd, a.hidden, a.ffn, a.m);
        BackwardStampV2(v.timer, 4, 1);
        AcquireAll(a, DZ_READY, aivs);
        BackwardStampV2(v.timer, 5, 0);
        Gemm<Bwd::ColumnMajor>(a.dg, a.x, a.dwg, a.ffn, a.hidden, a.m);
        BackwardStampV2(v.timer, 5, 1);
        BackwardStampV2(v.timer, 6, 0);
        Gemm<Bwd::ColumnMajor>(a.du, a.x, a.dwu, a.ffn, a.hidden, a.m);
        BackwardStampV2(v.timer, 6, 1);
        BackwardStampV2(v.timer, 7, 0);
        Gemm<Bwd::RowMajor>(a.dg, a.wg, a.dxg, a.m, a.hidden, a.ffn);
        BackwardStampV2(v.timer, 7, 1);
        BackwardStampV2(v.timer, 8, 0);
        Gemm<Bwd::RowMajor>(a.du, a.wu, a.dxu, a.m, a.hidden, a.ffn);
        BackwardStampV2(v.timer, 8, 1);
        Publish(a, DX_READY, worker);
    } else {
        AcquireAll(a, DA_READY, aics);
        BackwardStampV2(v.timer, 3, 0);
        ProduceDZPlanes(a, v.raw_gate, v.raw_up, a.ffn,
            a.limit != kMegaMoeSwigluLimitDisabled);
        BackwardStampV2(v.timer, 3, 1);
        Publish(a, DZ_READY, GetBlockIdx());
        // ProduceDZPlanes's local TPipe is destroyed before the hook binds
        // static staging UB. AIC concurrently performs its five grad GEMMs.
        hooks.Transport();
        hooks.RetireSourceReads();
        AcquireAll(a, DX_READY, aics);
        BackwardStampV2(v.timer, 9, 0);
        MergeDX(a);
        BackwardStampV2(v.timer, 9, 1);
        Publish(a, VECTOR_RETIRED, GetBlockIdx());
    }
    // In ABI3 this is also the X/P global read-lease guard. Routed P1 cannot
    // consume P or overwrite ptrD merely because an early DO flag is ready.
    AcquireAll(a, VECTOR_RETIRED, aivs);
    PipeBarrier<PIPE_ALL>();
}
}  // namespace MegaMoeSharedExpert
#endif
