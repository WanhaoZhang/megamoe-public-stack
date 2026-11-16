#pragma once
#include <cstdint>
#include <type_traits>
#ifndef MEGAFLUX_EARLY_DW1_WIDE
#define MEGAFLUX_EARLY_DW1_WIDE 0
#endif
#ifndef MEGAFLUX_GRADIENT_PRIORITY
#define MEGAFLUX_GRADIENT_PRIORITY 0
#endif
#ifndef MEGAFLUX_PRIORITY_INLINE
#define MEGAFLUX_PRIORITY_INLINE inline
#endif
namespace MegaFluxPriority {
using CompletionMask = std::conditional_t<MEGAFLUX_EARLY_DW1_WIDE,uint64_t,uint32_t>;
constexpr int CompletionExperts = sizeof(CompletionMask)*8;
MEGAFLUX_PRIORITY_INLINE int Key(int logical,int remoteContributors,int ranks) {
    if(!remoteContributors)return 10000+logical;
#if MEGAFLUX_GRADIENT_PRIORITY
    return (ranks-remoteContributors)*256+logical;
#else
    return logical;
#endif
}
}
