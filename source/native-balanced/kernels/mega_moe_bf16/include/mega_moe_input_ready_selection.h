#pragma once
#ifndef MEGAFLUX_INPUT_READY_INLINE
#define MEGAFLUX_INPUT_READY_INLINE inline
#endif
// Counts are the existing per-local-expert placement counts; never change
// routes or work order. Equal sizes prefer the first expert. One invocation
// owns one ready-counter set, retired/reset by the existing protocol.
namespace MegaFluxInputReady {
MEGAFLUX_INPUT_READY_INLINE int Largest(const int* counts,int count,bool cumulative,
        long long capacity,int minRows) {
    long long previous=0,consumed=0,bestRows=0;
    int selected=-1;
    for(int e=0;e<count;++e) {
        const long long value=counts[e];
        long long rows=cumulative?value-previous:value;
        previous=value;
        if(rows<0) return -1;
        const long long remaining=capacity>consumed?capacity-consumed:0;
        if(rows>remaining)rows=remaining;
        consumed+=rows;
        if(rows>=minRows && rows>bestRows){bestRows=rows;selected=e;}
    }
    return selected;
}
}
