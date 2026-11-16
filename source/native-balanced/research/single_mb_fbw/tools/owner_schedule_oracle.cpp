#define MEGAFLUX_SCHEDULE_INLINE inline
#include "mega_moe_owner_schedule.h"
extern "C" __declspec(dllexport) long long owner_task(long long ordinal,int worker,int workers,int whole) {
    return whole ? MegaFluxOwnerSchedule::Task<true>(ordinal,worker,workers,1)
                 : MegaFluxOwnerSchedule::Task<false>(ordinal,worker,workers,1);
}
