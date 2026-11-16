#pragma once
#include "mega_moe_worker_queue.h"
#include "mega_moe_stage_timer.h"
#include "mega_moe_training_workspace.h"

// Integration-only backward restore. Source homes have been refreshed from
// this layer's live FSDP Parameter before the existing entry rendezvous.
// Each paired AIV1 owns disjoint chunks; the other AIV and all AICs can
// dispatch/compute while these producers use their own upper UB.
namespace MegaFluxBackwardRestore {
using namespace AscendC;
constexpr int WORKERS=128, LINE=16, CHUNK=4096;
constexpr int PANEL_ROWS=256;

__aicore__ inline __gm__ int32_t* Flag(uint64_t ready,int slot,int projection,int worker) {
    return (__gm__ int32_t*)ready+((slot*2+projection)*WORKERS+worker)*LINE;
}

__aicore__ inline void CopyExperts(const MegaMoeReplicaDescriptor& d,uint64_t ready,int epoch,
        bool w1First=false) {
    if constexpr(g_coreType==AIV) {
        if(GetSubBlockIdx()!=1)return;
        const int worker=GetBlockIdx()/GetTaskRation(),workers=GetBlockNum();
        int order[32],used=0;
        for(int slot=0;slot<d.slots;++slot) {
            const int logical=MegaFluxWorker::Read((__gm__ int32_t*)d.plan+d.rank*d.slots+slot);
            if(logical<0)continue;
            int pos=used;
            while(pos>0 && MegaFluxWorker::Read((__gm__ int32_t*)d.plan+d.rank*d.slots+order[pos-1])>logical) {
                order[pos]=order[pos-1];--pos;
            }
            order[pos]=slot;++used;
        }
        // P0 gather and table copies retired at the preceding all-AIV barrier.
        // AIV1 has no dispatch ring; its [224,240) KiB copy buffers are lent
        // until Drain before it enters the original SwiGLU backward loop.
        MegaFluxWorker::Copy copy;
        for(int pos=0;pos<used;++pos) {
            const int slot=order[pos];
            const int logical=MegaFluxWorker::Read((__gm__ int32_t*)d.plan+d.rank*d.slots+slot);
            const uint64_t peer=(uint64_t)aclshmem_ptr((GM_ADDR)d.base,logical/d.home);
            // Saved modes consume W2 first; opt-in none must recompute GU
            // with W1 before P1. Keep the exact same publication protocol.
            for(int stage=0;stage<2;++stage) {
                const int p=MegaMoeBF16Training::BackwardRestoreProjection(stage,w1First);
                const int64_t plane=p?d.hidden*d.ffn:2*d.ffn*d.hidden;
                const uint64_t home=p?d.home2:d.home1, replica=p?d.replica2:d.replica1;
                const int64_t chunks=(plane+CHUNK-1)/CHUNK;
                for(int64_t chunk=worker;chunk<chunks;chunk+=workers) {
                    const int64_t begin=chunk*CHUNK;
                    const int n=plane-begin<CHUNK?plane-begin:CHUNK;
                    copy.Push(peer+home-d.base+(logical%d.home*plane+begin)*2,
                              replica+(slot*plane+begin)*2,n);
                }
                copy.Drain();
                SetFlag<HardEvent::MTE3_S>(EVENT_ID4);WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
                PipeBarrier<PIPE_ALL>();
                // Includes zero-task producers; a ready epoch means payload
                // writes completed, not merely that their DMA was issued.
                MegaFluxWorker::Release(Flag(ready,slot,p,worker),epoch);
            }
        }
    }
}

__aicore__ inline void WaitExpert(uint64_t ready,int slot,int projection,int epoch) {
    if constexpr(g_coreType==AIC) {
        for(int worker=0;worker<GetBlockNum();++worker)
            while(MegaFluxWorker::Read(Flag(ready,slot,projection,worker))!=epoch){}
        DataSyncBarrier<MemDsbT::ALL>();
    }
}

// Canonical BF16 layouts are W1=[2F,H] and W2=[H,F].  Every worker publishes
// one flag after its last MTE3 for a panel; an AIC waits for all workers before
// constructing the corresponding GEMM operand.  The common ready buffer is
// indexed as slot -> W1 panels -> W2 panels -> worker lines.
__aicore__ inline __gm__ int32_t* PanelFlag(uint64_t ready,int slot,int projection,
        int panel,int w1Panels,int w2Panels,int worker) {
    const int index=slot*(w1Panels+w2Panels) + (projection ? w1Panels : 0) + panel;
    return (__gm__ int32_t*)ready+(index*WORKERS+worker)*LINE;
}

__aicore__ inline void WaitPanel(uint64_t ready,int slot,int projection,int panel,
        int w1Panels,int w2Panels,int epoch) {
    if constexpr(g_coreType==AIC) {
        const int panels=projection?w2Panels:w1Panels;
        if(!ready || slot<0 || panel<0 || panel>=panels)return;
        for(int worker=0;worker<GetBlockNum();++worker)
            while(MegaFluxWorker::Read(PanelFlag(ready,slot,projection,panel,
                w1Panels,w2Panels,worker))!=epoch){}
        DataSyncBarrier<MemDsbT::ALL>();
    }
}

template<int OnlyProjection = -1>
__aicore__ inline void CopyExpertsPanel(const MegaMoeReplicaDescriptor& d,
        uint64_t ready,int epoch,uint64_t trace=0,bool w1First=false) {
    if constexpr(g_coreType==AIV) {
        if(GetSubBlockIdx()!=1)return;
        const int worker=GetBlockIdx()/GetTaskRation(),workers=GetBlockNum();
        const int w1Panels=(2*d.ffn+PANEL_ROWS-1)/PANEL_ROWS;
        const int w2Panels=(d.hidden+PANEL_ROWS-1)/PANEL_ROWS;
        MegaFluxWorker::Copy copy;
        // W2 is consumed first by dH in saved modes. Opt-in none restores
        // W1 first for GU recompute. Empty
        // slots still publish each panel so stale epochs cannot be mistaken
        // for a completed payload on a later invocation.
        for(int stage=0;stage<2;++stage) {
            const int projection=MegaMoeBF16Training::BackwardRestoreProjection(stage,w1First);
            if constexpr (OnlyProjection >= 0) {
                if (projection != OnlyProjection) continue;
            }
            const auto phaseTimer=MegaFluxStageTimer::BeginAsync(trace,true,MegaFluxStageTimer::BWD_RESTORE_PHASE,-1,projection);
            const int panels=projection?w2Panels:w1Panels;
            const int64_t width=projection?d.hidden:2*d.ffn;
            const int64_t rowWidth=projection?d.ffn:d.hidden;
            const int64_t plane=width*rowWidth;
            const uint64_t home=projection?d.home2:d.home1;
            const uint64_t replica=projection?d.replica2:d.replica1;
            for(int slot=0;slot<d.slots;++slot) {
                const int logical=MegaFluxWorker::Read((__gm__ int32_t*)d.plan+
                    d.rank*d.slots+slot);
                const bool active=logical>=0;
                const uint64_t peer=active?(uint64_t)aclshmem_ptr(
                    (GM_ADDR)d.base,logical/d.home):0;
                for(int panel=0;panel<panels;++panel) {
#if MEGAFLUX_RESTORE_DELAY_CYCLES > 0
                    // Diagnostic producer stress only. All workers delay at
                    // the same later-panel boundary; default builds have none.
                    if(active && panel==1) {
                        const int64_t begin=GetSystemCycle();
                        while(GetSystemCycle()-begin<MEGAFLUX_RESTORE_DELAY_CYCLES){}
                    }
#endif
                    const auto copyTimer=active?MegaFluxStageTimer::Begin(trace,true,MegaFluxStageTimer::RESTORE_COPY,-1,projection,panel,worker,-1,slot):0;
                    const int64_t first=(int64_t)panel*PANEL_ROWS;
                    const int64_t rows=width-first<PANEL_ROWS?width-first:PANEL_ROWS;
                    const int64_t elements=rows*rowWidth;
                    const int64_t chunks=(elements+CHUNK-1)/CHUNK;
                    for(int64_t chunk=worker;chunk<chunks;chunk+=workers) {
                        if(active) {
                            const int64_t begin=chunk*CHUNK;
                            const int n=elements-begin<CHUNK?elements-begin:CHUNK;
                            const int64_t offset=first*rowWidth+begin;
                            const int64_t srcOffset=(logical%d.home)*plane+offset;
                            const int64_t dstOffset=slot*plane+offset;
                            copy.Push(peer+home-d.base+srcOffset*2,
                                replica+dstOffset*2,n);
                        }
                    }
                    copy.Drain();
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID4);
                    WaitFlag<HardEvent::MTE3_S>(EVENT_ID4);
                    PipeBarrier<PIPE_ALL>();
                    MegaFluxStageTimer::End(copyTimer);
                    MegaFluxWorker::Release(PanelFlag(ready,slot,projection,panel,
                        w1Panels,w2Panels,worker),epoch);
                    MegaFluxStageTimer::Mark(trace,true,MegaFluxStageTimer::RESTORE_READY,-1,projection,panel,worker,-1,slot);
                }
            }
            MegaFluxStageTimer::EndAsync(phaseTimer);
        }
    }
}
}
