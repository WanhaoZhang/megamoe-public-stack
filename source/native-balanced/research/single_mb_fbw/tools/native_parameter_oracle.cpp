#include "mega_moe_unified_policy.h"
#if defined(_WIN32)
#define PARAM_EXPORT __declspec(dllexport)
#else
#define PARAM_EXPORT
#endif
extern "C" PARAM_EXPORT int native_parameters(int p,int home,int slots,int beta,int penalty,int gain,const int* source,int* cp,int* quota) {
    if(p<1||p>8||home<1||home>32||slots<0||p*slots>256||beta<=0||penalty<0||gain<0)return 1;
    UnifiedPlanner::State a{};a.p=p;a.h=home;a.s=slots;a.e=p*home;
    for(int e=0;e<a.e;++e)for(int s=0;s<p;++s)a.c[e]+=source[s*a.e+e];
    if(!UnifiedPlanner::native(a,beta,penalty,gain))return 2;
    for(int i=0;i<p*slots;++i)cp[i]=a.cp[i];
    for(int e=0;e<a.e;++e) {
        int row[8],out[64];for(int s=0;s<p;++s)row[s]=source[s*a.e+e];
        if(!UnifiedPlanner::source_quota(row,a.q+e*p,p,e/home,0,out))return 3;
        for(int s=0;s<p;++s)for(int d=0;d<p;++d)quota[(s*a.e+e)*p+d]=out[s*p+d];
    }
    return 0;
}
