#pragma once
#include <cstdint>

namespace MegaFluxBackwardRestore {
// Private to one AIC and one expert's projection GEMM call. Payload is immutable after
// its exact-epoch acquire; this object never survives a kernel invocation.
struct PanelAcquireCache {
    uint64_t ready = 0, acquired = 0;
    int slot = -1, projection = -1, w1Panels = -1, w2Panels = -1, epoch = -1;

    __aicore__ inline bool AlreadyAcquired(uint64_t flags, int s, int p,
            int w1, int w2, int e, int panel) {
        if (ready != flags || slot != s || projection != p ||
                w1Panels != w1 || w2Panels != w2 || epoch != e) {
            ready = flags; slot = s; projection = p;
            w1Panels = w1; w2Panels = w2; epoch = e; acquired = 0;
        }
        return panel >= 0 && panel < 64 && (acquired & (uint64_t(1) << panel));
    }

    // Called only after the original all-producer WaitPanel + DSB returns.
    __aicore__ inline void RecordAcquire(int panel) {
        if (panel >= 0 && panel < 64) acquired |= uint64_t(1) << panel;
    }
};
}
