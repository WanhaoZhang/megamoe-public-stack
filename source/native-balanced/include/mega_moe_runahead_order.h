#pragma once
#ifndef MEGAFLUX_RUNAHEAD_INLINE
#define MEGAFLUX_RUNAHEAD_INLINE inline
#endif
namespace MegaFluxRunahead {
MEGAFLUX_RUNAHEAD_INLINE bool Enabled(int mode,bool bf16,bool gradientPipe,
        int restore,int extent,int home) {
    if(mode<=0 || mode>5 || !bf16)return false;
    const bool replicated=gradientPipe && restore==2;
    const bool homeOnly=!gradientPipe && extent==home;
    return replicated || (mode>=5 && homeOnly);
}
MEGAFLUX_RUNAHEAD_INLINE bool HomeFirst(int mode,bool enabled,bool producer) {
    return enabled && mode>=2 && (producer || mode<4);
}
// Each row and ping is32B-aligned; retain256B slack and the tiling DMA limit.
// Returning0 preserves the complete legacy layout for unsupported rows.
MEGAFLUX_RUNAHEAD_INLINE int TransportRows(int bytes,int hidden,int tilingRows) {
    if(hidden<=0 || hidden%16 || tilingRows<=0 || bytes<=256)return 0;
    const int fit=(bytes-256)/2/(hidden*2);
    return fit<tilingRows?fit:tilingRows;
}
}
