#pragma once

#include "mega_moe_shared_expert_descriptor.h"
#include "mega_moe_bf16_catlass_impl.h"
#include "mega_moe_shared_expert_tile.h"

namespace MegaMoeSharedExpert {
using namespace AscendC;
using namespace MegaMoeSharedExpertLayout;

enum ReadyPhase : uint32_t { kGateUpReady = 0, kActivationReady = 1, kOutputReady = 2 };
enum TimerPhase : uint32_t { kGateUp = 0, kSwiglu = 1, kDown = 2, kOutputWait = 3 };

__aicore__ inline uint32_t Core() { return GetBlockIdx() / GetTaskRation(); }

__aicore__ inline bool IsTiled(const MegaMoeSharedExpertDescriptor &d)
{
    return (d.flags & kFlagTilePipeline) != 0;
}

__aicore__ inline void StampDetail(
    const MegaMoeSharedExpertDescriptor &d, uint32_t phase, uint32_t lane, uint32_t edge)
{
#if defined(ENABLE_TIMER)
    if (d.timer && (d.flags & kFlagDetailTimer)) {
        WriteGmByPassDCache<int64_t>(reinterpret_cast<__gm__ int64_t *>(d.timer) +
            kDetailTimerOffset + (phase * kMaxCores + lane) * 2 + edge,
            static_cast<int64_t>(GetSystemCycle()));
    }
#endif
}

__aicore__ inline void Stamp(
    const MegaMoeSharedExpertDescriptor &d, uint32_t phase, uint32_t lane, uint32_t edge)
{
#if defined(ENABLE_TIMER)
    if (d.timer != 0) {
        auto *cycles = reinterpret_cast<__gm__ int64_t *>(d.timer);
        // Packed 16-byte lane records share scalar DCache lines. A cached GM
        // assignment may write back a stale neighbouring lane; emit one
        // uncached 64-bit store without changing the public Timer layout.
        AscendC::WriteGmByPassDCache<int64_t>(
            cycles + (phase * kMaxCores + lane) * 2 + edge,
            static_cast<int64_t>(GetSystemCycle()));
    }
#endif
}

__aicore__ inline __gm__ int32_t *ReadyAddress(
    const MegaMoeSharedExpertDescriptor &d, uint32_t phase, uint32_t core)
{
    return reinterpret_cast<__gm__ int32_t *>(d.ready) +
        (phase * kMaxCores + core) * kReadyLineInts;
}

__aicore__ inline void Publish(
    const MegaMoeSharedExpertDescriptor &d, uint32_t phase)
{
    // One producer owns a whole cacheline. The host supplies a fresh zeroed
    // ready tensor for every invocation; no route epoch or flags are reused.
    WriteGmByPassDCache(ReadyAddress(d, phase, Core()), int32_t{1});
}

__aicore__ inline void WaitAll(
    const MegaMoeSharedExpertDescriptor &d, uint32_t phase)
{
    for (uint32_t core = 0; core < GetBlockNum(); ++core) {
        auto *ready = ReadyAddress(d, phase, core);
        while (ReadGmByPassDCache(ready) != 1) {}
    }
}

__aicore__ inline void GateUp(
    const MegaMoeSharedExpertDescriptor &d, GM_ADDR input)
{
    if constexpr (g_coreType == AIC) {
        Stamp(d, kGateUp, Core(), 0);
        MegaMoeBF16CatlassImpl::SharedExpertGemmGm(input,
            reinterpret_cast<GM_ADDR>(d.weight_gate), reinterpret_cast<GM_ADDR>(d.raw_gate),
            d.tokens, d.ffn, d.hidden);
        MegaMoeBF16CatlassImpl::SharedExpertGemmGm(input,
            reinterpret_cast<GM_ADDR>(d.weight_up), reinterpret_cast<GM_ADDR>(d.raw_up),
            d.tokens, d.ffn, d.hidden);
        Publish(d, kGateUpReady);
        Stamp(d, kGateUp, Core(), 1);
    }
}

__aicore__ inline void Swiglu(const MegaMoeSharedExpertDescriptor &d, float limit)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
        WaitAll(d, kGateUpReady);
        Stamp(d, kSwiglu, Core(), 0);
        constexpr uint32_t chunk = 4096;
        constexpr uint32_t lanes = 256 / sizeof(float);
        LocalTensor<bfloat16_t> gate{TPosition::VECIN, 0, chunk};
        LocalTensor<bfloat16_t> up{TPosition::VECIN, chunk * sizeof(bfloat16_t), chunk};
        GlobalTensor<bfloat16_t> gateGm, upGm, activationGm;
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(d.raw_gate));
        upGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(d.raw_up));
        activationGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(d.activation));
        gateGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        upGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        activationGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        const uint64_t elements = d.tokens * d.ffn;
        const bool clip = limit != kMegaMoeSwigluLimitDisabled;
        auto *gateUb = reinterpret_cast<__ubuf__ bfloat16_t *>(gate.GetPhyAddr());
        auto *upUb = reinterpret_cast<__ubuf__ bfloat16_t *>(up.GetPhyAddr());
        for (uint64_t begin = static_cast<uint64_t>(Core()) * chunk; begin < elements;
             begin += static_cast<uint64_t>(GetBlockNum()) * chunk) {
            const uint32_t count = static_cast<uint32_t>(
                elements - begin < chunk ? elements - begin : chunk);
            const uint16_t repeats = static_cast<uint16_t>((count + lanes - 1) / lanes);
            SyncFunc<HardEvent::V_MTE2>();
            DataCopyExtParams copy{1, count * uint32_t{sizeof(bfloat16_t)}, 0, 0, 0};
            DataCopyPad(gate, gateGm[begin], copy, {});
            DataCopyPad(up, upGm[begin], copy, {});
            SyncFunc<HardEvent::MTE2_V>();
            __VEC_SCOPE__ {
                Reg::RegTensor<float> lim, negLim;
                if (clip) {
                    Reg::Duplicate(lim, limit);
                    Reg::Duplicate(negLim, -limit);
                }
                uint32_t remaining = count;
                for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                    const uint32_t offset = static_cast<uint32_t>(repeat) * lanes;
                    auto mask = Reg::UpdateMask<float>(remaining);
                    Reg::RegTensor<bfloat16_t> g, u, result;
                    Reg::RegTensor<float> gf, uf, neg, exp, denom, silu, value;
                    Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(g, gateUb + offset);
                    Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(u, upUb + offset);
                    Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kCastBF16ToFP32>(gf, g, mask);
                    Reg::Cast<float, bfloat16_t, MegaMoeBF16Impl::kCastBF16ToFP32>(uf, u, mask);
                    if (clip) {
                        Reg::Min(gf, gf, lim, mask);
                        Reg::Max(uf, uf, negLim, mask);
                        Reg::Min(uf, uf, lim, mask);
                    }
                    Reg::Muls(neg, gf, -1.0F, mask);
                    Reg::Exp(exp, neg, mask);
                    Reg::Adds(denom, exp, 1.0F, mask);
                    Reg::Div<float, &MegaMoeBF16Impl::kDivMode>(silu, gf, denom, mask);
                    Reg::Mul(value, silu, uf, mask);
                    Reg::Cast<bfloat16_t, float, MegaMoeBF16Impl::kCastFP32ToBF16>(result, value, mask);
                    Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(gateUb + offset, result, mask);
                }
            }
            SyncFunc<HardEvent::V_MTE3>();
            DataCopyPad(activationGm[begin], gate, copy);
            SyncFunc<HardEvent::MTE3_V>();
        }
        // Finish the last activation write before scalar publication or reuse
        // of AIV0 UB by the ordinary routed SwiGLU epilogue.
        SyncFunc<HardEvent::MTE3_S>();
        PipeBarrier<PIPE_ALL>();
        Publish(d, kActivationReady);
        Stamp(d, kSwiglu, Core(), 1);
    }
}

__aicore__ inline void Down(const MegaMoeSharedExpertDescriptor &d, GM_ADDR output)
{
    if constexpr (g_coreType == AIC) {
        WaitAll(d, kActivationReady);
        Stamp(d, kDown, Core(), 0);
        MegaMoeBF16CatlassImpl::SharedExpertGemmGm(
            reinterpret_cast<GM_ADDR>(d.activation), reinterpret_cast<GM_ADDR>(d.weight_down),
            output, d.tokens, d.hidden, d.ffn);
        Publish(d, kOutputReady);
        Stamp(d, kDown, Core(), 1);
    }
}

__aicore__ inline void WaitOutput(const MegaMoeSharedExpertDescriptor &d)
{
    if constexpr (g_coreType == AIV) {
        const uint32_t lane = GetBlockIdx();
        Stamp(d, kOutputWait, lane, 0);
        WaitAll(d, kOutputReady);
        Stamp(d, kOutputWait, lane, 1);
    }
}
}  // namespace MegaMoeSharedExpert
