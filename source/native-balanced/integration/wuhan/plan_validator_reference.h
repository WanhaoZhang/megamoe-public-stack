#pragma once
// Test-only independent serial reference; never included in a production kernel.
namespace MegaFluxPlanValidationReference {
// Portable scalar metadata logic. The device reader bypasses GM cache;
// the host differential test uses the same code with an ordinary array reader.
template<class Reader>
inline MegaFluxPlanValidation::Result Finalize(Reader read,int native,int ep,int home,
        int slots,int epoch,int capacity) {
    const int experts=ep*home,width=home+slots,pc=ep*width;
    const int placement=experts+pc,meta=placement+ep*slots,stride=meta+3;
    int error=0,mismatch=0,active=0,required=0;
    for(int r=0;r<ep;++r) {
        if(read(r*stride+meta)!=0)error=1;
        if(read(r*stride+meta+1)!=epoch||read(r*stride+meta+2)!=capacity)mismatch=1;
        for(int i=0;i<ep*slots;++i)
            if(read(r*stride+placement+i)!=read(placement+i))mismatch=1;
    }
    for(int r=0;r<ep;++r) {
        int rows=0;
        for(int s=0;s<width;++s) {
            int count=0;for(int src=0;src<ep;++src)count+=read(src*stride+experts+r*width+s);
            rows+=count;
            if(s>=home) {
                const int index=r*slots+s-home;
                if(read(placement+index)>=0)++active;
                if(count==0)for(int src=0;src<ep;++src)
                    if(read(src*stride+placement+index)>=0)error=1;
            }
        }
        if(rows>required)required=rows;
    }
    required=(required+127)/128*128;
    int status=mismatch?2:(error?1:(required>capacity?3:0));
    if(native>status)status=native;
    return {active,required,status};
}
}
