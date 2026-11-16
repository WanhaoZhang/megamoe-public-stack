#pragma once
#ifndef MEGAFLUX_ORDER_INLINE
#define MEGAFLUX_ORDER_INLINE inline
#endif
#ifndef MEGAFLUX_PRIORITY_INLINE
#define MEGAFLUX_PRIORITY_INLINE MEGAFLUX_ORDER_INLINE
#endif
#include "mega_moe_gradient_priority.h"

namespace MegaFluxGradientOrder {
// Single metadata pass for EP2/4/8. Reader supplies exact-epoch plan values;
// no cached key survives this call or the current immutable plan lease.
template<class Reader>
MEGAFLUX_ORDER_INLINE bool Build(Reader& reader,int ranks,int rank,int home,
        int slots,int extent,int* order) {
    if ((ranks!=2 && ranks!=4 && ranks!=8) || rank<0 || rank>=ranks ||
        home<=0 || home>32 || slots<0 || slots>32 || extent!=home+slots || extent>64)
        return false;
    int contributors[256],local[32],keys[64];
    for(int i=0;i<256;++i)contributors[i]=0;
    for(int i=0;i<slots;++i)local[i]=-1;
    for(int i=0;i<ranks*slots;++i) {
        const int logical=reader.Read(i);
        if(logical>=ranks*home || logical < -1)return false;
        if(logical>=0)++contributors[logical];
        if(i/slots==rank)local[i%slots]=logical;
    }
    for(int e=0;e<extent;++e) {
        const int logical=e<home?rank*home+e:local[e-home];
        keys[e]=logical<0?100000+e:MegaFluxPriority::Key(logical,contributors[logical],ranks);
        order[e]=e;
    }
    for(int i=1;i<extent;++i) {
        const int value=order[i];int j=i-1;
        while(j>=0 && keys[order[j]]>keys[value]) {
            order[j+1]=order[j];--j;
        }
        order[j+1]=value;
    }
    return true;
}
}
