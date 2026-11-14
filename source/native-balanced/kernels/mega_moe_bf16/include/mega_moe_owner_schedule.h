#pragma once
#ifndef MEGAFLUX_SCHEDULE_INLINE
#define MEGAFLUX_SCHEDULE_INLINE __aicore__ inline
#endif
// Pure integer partition, shared by the device and the host exhaustive test.
// Odd workers are the early AIV1 consumers. Each group owns R+1 residues;
// its even worker owns residue 0, its odd worker owns residues 1..R.
namespace MegaFluxOwnerSchedule {
template<bool WholeGroup = false>
MEGAFLUX_SCHEDULE_INLINE long long Task(long long ordinal,int worker,int workers,int ratio) {
    const int share=WholeGroup?ratio+1:((worker&1)?ratio:1);
    const int residue=WholeGroup?ordinal%share:((worker&1)?1+ordinal%share:0);
    return (ordinal/share)*(workers/2)*(ratio+1)
        +(worker/2)*(ratio+1)+residue;
}
}
