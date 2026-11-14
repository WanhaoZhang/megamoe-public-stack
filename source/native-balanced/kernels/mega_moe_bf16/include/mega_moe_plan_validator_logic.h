#pragma once
#ifndef MEGAFLUX_VALIDATOR_INLINE
#define MEGAFLUX_VALIDATOR_INLINE inline
#endif
namespace MegaFluxPlanValidation {
struct Result {int active,required,status;};
// Per-source and per-destination work is independent. The device distributes
// it over 32 workers; host tests also exercise these exact summary rules.
struct Summary {int error=0,mismatch=0,active=0,required=0;};
template<class Reader>
MEGAFLUX_VALIDATOR_INLINE void CheckSource(Reader read,int src,int ep,int home,
        int slots,int epoch,int capacity,Summary& out) {
    const int placement=ep*home+ep*(home+slots),meta=placement+ep*slots,stride=meta+3;
    if(read(src*stride+meta))out.error=1;
    if(read(src*stride+meta+1)!=epoch||read(src*stride+meta+2)!=capacity)out.mismatch=1;
    for(int i=0;i<ep*slots;++i)
        if(read(src*stride+placement+i)!=read(placement+i))out.mismatch=1;
}
template<class Reader,class Counts>
MEGAFLUX_VALIDATOR_INLINE void CheckDestination(Reader read,Counts counts,int r,
        int ep,int home,int slots,Summary& out) {
    const int width=home+slots,placement=ep*home+ep*width,meta=placement+ep*slots,stride=meta+3;
    int rows=0;
    for(int s=0;s<width;++s) {
        const int count=counts(r*width+s);rows+=count;
        if(s>=home) {
            const int index=r*slots+s-home;
            if(read(placement+index)>=0)++out.active;
            if(count==0)for(int src=0;src<ep;++src)
                if(read(src*stride+placement+index)>=0)out.error=1;
        }
    }
    if(rows>out.required)out.required=rows;
}
MEGAFLUX_VALIDATOR_INLINE Result Finish(const Summary& s,int native,int capacity) {
    const int required=(s.required+127)/128*128;
    int status=s.mismatch?2:(s.error?1:(required>capacity?3:0));
    if(native>status)status=native;
    return {s.active,required,status};
}
}
