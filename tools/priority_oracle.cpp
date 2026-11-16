#define MEGAFLUX_GRADIENT_PRIORITY 1
#define MEGAFLUX_EARLY_DW1_WIDE 1
#include "mega_moe_gradient_priority.h"
extern "C" __declspec(dllexport) int priority_key(int logical,int count,int ranks) {
    return MegaFluxPriority::Key(logical,count,ranks);
}
extern "C" __declspec(dllexport) uint64_t completion_mask(const int* experts,int count) {
    MegaFluxPriority::CompletionMask mask=0;
    for(int i=0;i<count;++i)mask|=MegaFluxPriority::CompletionMask(1)<<experts[i];
    return mask;
}
