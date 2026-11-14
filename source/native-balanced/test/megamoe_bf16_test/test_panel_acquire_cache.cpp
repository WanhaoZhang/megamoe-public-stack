#define __aicore__
#include "mega_moe_panel_acquire_cache.h"
#include <cassert>
#include <cstdint>
#include <iostream>
using MegaFluxBackwardRestore::PanelAcquireCache;

int main() {
    PanelAcquireCache c;
    int waits=0;
    auto consume=[&](uint64_t flags,int slot,int p,int w1,int w2,int epoch,int panel) {
        if(!c.AlreadyAcquired(flags,slot,p,w1,w2,epoch,panel)) {
            ++waits; // Stand-in callback returns only after all producers and DSB.
            c.RecordAcquire(panel);
        }
    };
    for(int outputTile=0;outputTile<67;++outputTile)
        for(int panel=0;panel<24;++panel)consume(0x1000,0,0,24,28,7,panel);
    assert(waits==24);
    consume(0x1000,0,0,24,28,8,0);assert(waits==25); // new epoch
    consume(0x1000,1,0,24,28,8,0);assert(waits==26); // new slot
    consume(0x1000,1,1,24,28,8,0);assert(waits==27); // new projection
    consume(0x2000,1,1,24,28,8,0);assert(waits==28); // different ready buffer
    consume(0x2000,1,1,25,28,8,0);assert(waits==29); // changed W1 layout
    consume(0x2000,1,1,25,29,8,0);assert(waits==30); // changed W2 layout
    consume(0x2000,1,1,25,29,8,63);consume(0x2000,1,1,25,29,8,63);assert(waits==31);
    for(int i=0;i<3;++i)consume(0x2000,1,1,25,29,8,64);assert(waits==34);
    for(int i=0;i<3;++i)consume(0x2000,1,1,25,29,8,-1);assert(waits==37);
    PanelAcquireCache nextInvocation;
    assert(!nextInvocation.AlreadyAcquired(0x2000,1,1,25,29,8,0));
    // An attempted lookup without a completed wait must never populate a bit.
    assert(!nextInvocation.AlreadyAcquired(0x2000,1,1,25,29,8,0));
    std::cout << "PANEL_CACHE_CPP_CONTRACT_PASS waits=" << waits << '\n';
}
