#pragma once
#include <cstdint>
#ifndef MEGAFLUX_DIRECT_INLINE
#define MEGAFLUX_DIRECT_INLINE inline
#endif

namespace MegaFluxMoonEPDirect {
// n is the agreed M*TopK per rank, hence the mean owner load. Use exact
// integer arithmetic after global histogram validation; no host decision.
MEGAFLUX_DIRECT_INLINE bool KeepHome(int64_t maximum, int64_t n,
                                    int ranks, int permille) {
    return ranks == 8 && permille >= 1000 && permille <= 8000 && n > 0 &&
           maximum * 1000 <= n * static_cast<int64_t>(permille);
}
}
