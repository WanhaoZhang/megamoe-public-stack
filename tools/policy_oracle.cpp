#include "mega_moe_unified_policy.h"
extern "C" int policy_reference(int p,int home,int slots,int policy,const int* source,int* cp,int* quota) {
    UnifiedPlanner::State a{};a.p=p;a.h=home;a.s=slots;a.e=p*home;
    int total=0;for(int x=0;x<a.e;++x){a.c[x]=0;for(int s=0;s<p;++s)a.c[x]+=source[s*a.e+x];total+=a.c[x];}
    if(!UnifiedPlanner::run(a,policy,total/p))return 1;
    for(int i=0;i<p*slots;++i)cp[i]=a.cp[i];
    for(int x=0;x<a.e;++x){int row[8],out[64];for(int s=0;s<p;++s)row[s]=source[s*a.e+x];
        int mode=policy==2?1:policy==3?2:policy==4?3:0;
        if(!UnifiedPlanner::source_quota(row,(policy==4?a.instances:a.q)+x*p,p,x/home,mode,out))return 2;
        for(int s=0;s<p;++s)for(int d=0;d<p;++d)quota[(s*a.e+x)*p+d]=out[s*p+d];
    }return 0;
}
