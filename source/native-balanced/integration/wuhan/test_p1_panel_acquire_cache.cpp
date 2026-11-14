// Host-check the same cache class instantiated by P1; no mock device math.
#include <cassert>
#include <cstdint>
#include <initializer_list>
#define __aicore__
#include "../../kernels/mega_moe_bf16/include/mega_moe_panel_acquire_cache.h"

int main() {
    using MegaFluxBackwardRestore::PanelAcquireCache;
    for (int panels : {1, 28, 64}) {
        PanelAcquireCache cache;
        for (int p = 0; p < panels; ++p) {
            assert(!cache.AlreadyAcquired(4096, 2, 1, 24, panels, 7, p));
            cache.RecordAcquire(p);
            assert(cache.AlreadyAcquired(4096, 2, 1, 24, panels, 7, p));
        }
        assert(!cache.AlreadyAcquired(4096, 2, 1, 24, panels, 8, 0));
        cache.RecordAcquire(0);
        assert(!cache.AlreadyAcquired(8192, 2, 1, 24, panels, 8, 0));
        cache.RecordAcquire(0);
        assert(!cache.AlreadyAcquired(8192, 3, 1, 24, panels, 8, 0));
        cache.RecordAcquire(0);
        assert(!cache.AlreadyAcquired(8192, 3, 0, 24, panels, 8, 0));
        cache.RecordAcquire(0);
        assert(!cache.AlreadyAcquired(8192, 3, 0, 25, panels, 8, 0));
        cache.RecordAcquire(0);
        assert(!cache.AlreadyAcquired(8192, 3, 0, 25, panels + 1, 8, 0));
        for (int p : {-1, 64, 65}) {
            cache.RecordAcquire(p);
            assert(!cache.AlreadyAcquired(8192, 3, 0, 25, panels + 1, 8, p));
        }
        PanelAcquireCache nextInvocation;
        assert(!nextInvocation.AlreadyAcquired(8192, 3, 0, 25, panels + 1, 8, 0));
    }
}
