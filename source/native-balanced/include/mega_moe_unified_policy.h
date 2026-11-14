#pragma once
// Fixed-owner policy ports. Provenance and numerical/transport adaptations:
// planner-executor-ep8-lab/references/MANIFEST.json and POLICY-PORTS.md.
// No parameters, streams, SHMEM writes or optimizer state are visible here.
#include <cstdint>
#ifndef UP_INLINE
#define UP_INLINE inline
#endif
namespace UnifiedPlanner {
constexpr int EMAX=256, PMAX=8, CMAX=256;
UP_INLINE int mn(int a,int b){return a<b?a:b;}
UP_INLINE int mx(int a,int b){return a>b?a:b;}
UP_INLINE void sort(int* idx,int n,const int* value,bool descending) {
    for(int i=0;i<n;++i)idx[i]=i;
    for(int i=1;i<n;++i){int v=idx[i],j=i-1;
        while(j>=0 && (descending?value[idx[j]]<value[v]:value[idx[j]]>value[v])){idx[j+1]=idx[j];--j;}idx[j+1]=v;}
}
struct State {
    int p,h,s,e,source_policy;
    int evaluations,rounds,adopted,model_flags;
    float predicted_before,predicted_after;
    int c[EMAX],q[EMAX*PMAX],cp[CMAX],instances[EMAX*PMAX];
    UP_INLINE void home() {
        for(int x=0;x<e;++x)for(int d=0;d<p;++d)q[x*p+d]=(x/h==d?c[x]:0);
        for(int i=0;i<p*s;++i)cp[i]=-1;
    }
    UP_INLINE bool canonical() {
        for(int d=0;d<p;++d){int used=0;
            for(int x=0;x<e;++x)if(x/h!=d && q[x*p+d]>0){if(used>=s)return false;cp[d*s+used++]=x;}
            while(used<s)cp[d*s+used++]=-1;
        }return true;
    }
};
UP_INLINE bool moon(State& a,int target) {
    if(a.s!=a.h)return false;
    a.home();int balance[PMAX],transfer[PMAX*PMAX],remaining[EMAX];
    for(int i=0;i<a.p*a.p;++i)transfer[i]=0;
    for(int r=0;r<a.p;++r){balance[r]=-target;for(int x=r*a.h;x<(r+1)*a.h;++x)balance[r]+=a.c[x];}
    for(int step=0;step<a.p*a.p;++step){int owner=0,d=0;
        for(int r=1;r<a.p;++r){if(balance[r]>balance[owner])owner=r;if(balance[r]<balance[d])d=r;}
        if(balance[owner]<=0||balance[d]>=0)break;
        int take=-balance[d];transfer[owner*a.p+d]=take;balance[owner]-=take;balance[d]=0;
    }
    for(int owner=0;owner<a.p;++owner){
        for(int j=0;j<a.h;++j)remaining[j]=a.c[owner*a.h+j];
        for(int step=0;step<a.p+a.h;++step){int d=0,j=0;
            for(int r=1;r<a.p;++r)if(transfer[owner*a.p+r]>transfer[owner*a.p+d])d=r;
            int wanted=transfer[owner*a.p+d];if(wanted<=0)break;
            for(int k=1;k<a.h;++k)if(remaining[k]>remaining[j])j=k;
            int take=mn(remaining[j],wanted);if(take<=0)return false;
            int x=owner*a.h+j;a.q[x*a.p+owner]-=take;a.q[x*a.p+d]+=take;
            remaining[j]-=take;transfer[owner*a.p+d]-=take;
        }
        for(int r=0;r<a.p;++r)if(transfer[owner*a.p+r])return false;
    }
    for(int d=0;d<a.p;++d){int used=0,rows=0;
        for(int x=0;x<a.e;++x){int count=a.q[x*a.p+d];rows+=count;if(x/a.h==d||count<=0)continue;
            if(used>=a.s)return false;int j=used;
            while(j>0){int prev=a.cp[d*a.s+j-1],value=a.q[prev*a.p+d];
                if(value>count||(value==count&&prev>x))break;
                a.cp[d*a.s+j]=prev;--j;}a.cp[d*a.s+j]=x;++used;
        }if(rows!=target)return false;
    }return true;
}
UP_INLINE bool native(State& a,int beta=768,int penalty=2,int gain=4) {
    a.home();int load[PMAX],original[PMAX],blocks[EMAX],ds[PMAX],rs[PMAX],xs[EMAX],values[EMAX];
    for(int r=0;r<a.p;++r)load[r]=0;
    int total=0;
    for(int x=0;x<a.e;++x){blocks[x]=(a.c[x]+beta-1)/beta;total+=blocks[x];
        for(int d=0;d<a.p;++d)a.q[x*a.p+d]=d==x/a.h?blocks[x]:0;load[x/a.h]+=blocks[x];}
    for(int r=0;r<a.p;++r)original[r]=load[r];int target=(total+a.p-1)/a.p;
    for(int round=0;round<=total;++round){bool moved=false;sort(ds,a.p,load,true);
        for(int di=0;di<a.p;++di){int d=ds[di];if(load[d]<=target)continue;
            for(int x=0;x<a.e;++x)values[x]=a.q[x*a.p+d];sort(xs,a.e,values,true);
            for(int xi=0;xi<a.e;++xi){int x=xs[xi];if(!a.q[x*a.p+d])continue;sort(rs,a.p,load,false);
                for(int ri=0;ri<a.p;++ri){int r=rs[ri];if(r==d||load[r]>=target)continue;int existing=-1,free=-1;
                    for(int j=0;j<a.s;++j){int v=a.cp[r*a.s+j];if(v==x&&existing<0)existing=j;if(v<0&&free<0)free=j;}
                    if(r!=x/a.h&&existing<0&&free<0)continue;
                    int take=mn(a.q[x*a.p+d],mn(target-load[r],load[d]-target));if(take<=0)continue;
                    if(r!=x/a.h&&existing<0)a.cp[r*a.s+free]=x;
                    a.q[x*a.p+d]-=take;a.q[x*a.p+r]+=take;load[d]-=take;load[r]+=take;moved=true;
                    if(load[d]<=target)break;
                }if(load[d]<=target)break;
            }
        }if(!moved)break;
    }
    if(penalty||gain){
        for(int it=0;it<a.p*a.s;++it){int aug[PMAX];for(int r=0;r<a.p;++r)aug[r]=load[r];
            for(int r=0;r<a.p;++r)for(int j=0;j<a.s;++j){int x=a.cp[r*a.s+j];if(x>=0&&a.q[x*a.p+r]>0)aug[x/a.h]+=penalty;}
            int d=0;for(int r=1;r<a.p;++r)if(aug[r]>aug[d])d=r;
            int best=0x7fffffff,bx=-1,br=-1,bs=-1;
            for(int r=0;r<a.p;++r)for(int j=0;j<a.s;++j){int x=a.cp[r*a.s+j];if(x<0||x/a.h!=d)continue;int v=a.q[x*a.p+r];
                if(v>0&&(v<best||(v==best&&x<bx))){best=v;bx=x;br=r;bs=j;}}
            if(bx<0||best>=penalty)break;a.q[bx*a.p+br]=0;a.q[bx*a.p+d]+=best;load[br]-=best;load[d]+=best;a.cp[br*a.s+bs]=-1;
        }
        int aug[PMAX];for(int r=0;r<a.p;++r)aug[r]=load[r];
        for(int r=0;r<a.p;++r)for(int j=0;j<a.s;++j){int x=a.cp[r*a.s+j];if(x>=0&&a.q[x*a.p+r]>0)aug[x/a.h]+=penalty;}
        int before=0,after=0;for(int r=0;r<a.p;++r){before=mx(before,original[r]);after=mx(after,aug[r]);}
        if(after+gain>=before){for(int x=0;x<a.e;++x)for(int r=0;r<a.p;++r)a.q[x*a.p+r]=r==x/a.h?blocks[x]:0;for(int i=0;i<a.p*a.s;++i)a.cp[i]=-1;}
    }
    for(int x=0;x<a.e;++x){int left=a.c[x];for(int d=0;d<a.p;++d){int take=mn(left,a.q[x*a.p+d]*beta);a.q[x*a.p+d]=take;left-=take;}}
    for(int d=0;d<a.p;++d)for(int j=0;j<a.s;++j){int x=a.cp[d*a.s+j];if(x>=0&&!a.q[x*a.p+d])a.cp[d*a.s+j]=-1;}
    return true;
}
UP_INLINE bool echo(State& a) {
    a.home();int spill[EMAX],capacity[PMAX],xs[EMAX],ds[PMAX],local[EMAX],value[EMAX];
    int total=0;for(int x=0;x<a.e;++x)total+=a.c[x];int avg=total/a.p;
    for(int r=0;r<a.p;++r){int load=0;for(int j=0;j<a.h;++j){load+=a.c[r*a.h+j];value[j]=a.c[r*a.h+j];}
        capacity[r]=mx(0,avg-load);sort(local,a.h,value,false);int keep=avg;
        for(int j=0;j<a.h;++j){int x=r*a.h+local[j],v=mn(keep,a.c[x]);keep-=v;spill[x]=a.c[x]-v;}}
    sort(xs,a.e,spill,true);sort(ds,a.p,capacity,true);
    int dstart=0;
    for(int ri=0;ri<a.p;++ri){int d=ds[ri],dend=dstart+capacity[d],xstart=0;
        int overlap[EMAX],order[EMAX];
        for(int i=0;i<a.e;++i){int x=xs[i],xend=xstart+spill[x];overlap[i]=mx(0,mn(xend,dend)-mx(xstart,dstart));xstart=xend;}
        sort(order,a.e,overlap,true);
        for(int j=0;j<mn(a.s,a.e);++j){int i=order[j],x=xs[i],v=overlap[i];if(v){a.q[x*a.p+d]+=v;a.q[x*a.p+x/a.h]-=v;}}
        dstart=dend;
    }return a.canonical();
}
UP_INLINE bool eplb(State& a) {
    a.home();int mult[EMAX],replica[CMAX],order[CMAX],used[PMAX];float instance[EMAX],load[PMAX];
    for(int x=0;x<a.e;++x)mult[x]=1;
    for(int j=0;j<a.p*a.s;++j){int best=0;float value=(float)a.c[0]/mult[0];
        for(int x=1;x<a.e;++x){float v=(float)a.c[x]/mult[x];if(v>value){best=x;value=v;}}
        replica[j]=best;++mult[best];order[j]=j;
    }
    for(int x=0;x<a.e;++x)instance[x]=(float)a.c[x]/mult[x];
    for(int r=0;r<a.p;++r){load[r]=0;used[r]=0;for(int j=0;j<a.h;++j)load[r]+=instance[r*a.h+j];}
    for(int i=1;i<a.p*a.s;++i){int v=order[i],j=i-1;
        while(j>=0&&instance[replica[order[j]]]<instance[replica[v]]){order[j+1]=order[j];--j;}order[j+1]=v;}
    for(int x=0;x<a.e;++x)for(int d=0;d<a.p;++d)a.instances[x*a.p+d]=d==x/a.h?1:0;
    for(int i=0;i<a.p*a.s;++i){int x=replica[order[i]],d=-1;
        for(int r=0;r<a.p;++r)if(used[r]<a.s&&(d<0||load[r]<load[d]))d=r;
        if(d<0)return false;++a.instances[x*a.p+d];
        ++used[d];load[d]+=instance[x];
    }
    for(int x=0;x<a.e;++x){int begin=0;
        for(int d=0;d<a.p;++d){int end=begin+a.instances[x*a.p+d],rem=a.c[x]%mult[x];
            a.q[x*a.p+d]=(a.c[x]/mult[x])*a.instances[x*a.p+d]+mn(rem,end)-mn(rem,begin);begin=end;}}
    return a.canonical();
}
UP_INLINE bool ultra_oracle(State& a,int threshold,bool dynamic,int minTokens,bool allowZero) {
    a.home();int load[PMAX],slack[PMAX],used[PMAX],sources[PMAX],xs[EMAX],values[EMAX];
    for(int r=0;r<a.p;++r){load[r]=0;used[r]=0;for(int j=0;j<a.h;++j)load[r]+=a.c[r*a.h+j];slack[r]=mx(threshold-load[r],0);}
    sort(sources,a.p,load,true);
    for(int i=0;i<a.p;++i){int owner=sources[i],need=mx(load[owner]-threshold,0);
        for(int j=0;j<a.h;++j)values[j]=a.c[owner*a.h+j];sort(xs,a.h,values,true);
        for(int j=0;j<a.h&&need;++j){int x=owner*a.h+xs[j],available=a.c[x]-(!allowZero&&a.c[x]>0?1:0);
            while(need>0&&available>0){int feasible=0;
                for(int r=0;r<a.p;++r)if(r!=owner&&slack[r]>0&&used[r]<a.s&&a.q[x*a.p+r]==0)++feasible;
                if(!feasible)break;int floor=dynamic?mx(1,(need+feasible-1)/feasible):minTokens,d=-1;
                for(int r=0;r<a.p;++r)if(r!=owner&&slack[r]>0&&used[r]<a.s&&a.q[x*a.p+r]==0&&mn(need,mn(slack[r],available))>=floor)
                    if(d<0||slack[r]>slack[d])d=r;
                if(d<0)break;int take=mn(need,mn(slack[d],available));
                a.q[x*a.p+owner]-=take;a.q[x*a.p+d]+=take;slack[d]-=take;++used[d];need-=take;available-=take;
            }
        }if(need)return false;
    }return true;
}
UP_INLINE bool ultra(State& a,int minTokens=1024,bool allowZero=false,float eps=0.01f) {
    int total=0,high=0;for(int r=0;r<a.p;++r){int load=0;for(int j=0;j<a.h;++j)load+=a.c[r*a.h+j];total+=load;high=mx(high,load);}
    int low=(total+a.p-1)/a.p;high=mx(high,low);float f=low*(1.0f+eps);int fast=(int)f;if(fast<f)++fast;fast=mx(fast,low);
    for(int retry=0;retry<3;++retry){if(ultra_oracle(a,fast,true,minTokens,allowZero))return a.canonical();fast=mn(high,fast+mx(1,(high+99)/100));}
    if(ultra_oracle(a,low,false,minTokens,allowZero))return a.canonical();low=mn(low+1,high);
    while(low<high){int span=high-low,probes[4],first=-1;
        for(int w=0;w<4;++w){int t=span<=4?low+w:low+(int)((int64_t)span*(w+1)/5);probes[w]=t;
            if(t<high&&ultra_oracle(a,t,false,minTokens,allowZero)){first=w;break;}}
        if(span<=4){if(first>=0)low=high=probes[first];else low=high;}
        else if(first>=0){high=probes[first];if(first>0)low=probes[first-1]+1;}
        else low=probes[3]+1;
    }
    if(!ultra_oracle(a,low,false,minTokens,allowZero))return false;return a.canonical();
}
// The shared compiler allocates one expert's source transportation matrix.
// mode0: source-major interval; mode1: local-fill + balanced remainders;
// mode2: ECHO's breadth-first floor + depth-first residual.
UP_INLINE bool source_quota(const int* source,const int* dest,int p,int owner,int mode,int* out) {
    int n=0;for(int s=0;s<p;++s)n+=source[s];for(int i=0;i<p*p;++i)out[i]=0;
    if(!n)return true;
    if(mode==3){int copies=0;for(int d=0;d<p;++d)copies+=dest[d];if(copies<=0)return false;
        int lo=0;for(int s=0;s<p;++s){int hi=lo+source[s],begin=0;
            for(int d=0;d<p;++d){int end=begin+dest[d];
                int upper=hi/copies*dest[d]+mn(hi%copies,end)-mn(hi%copies,begin);
                int lower=lo/copies*dest[d]+mn(lo%copies,end)-mn(lo%copies,begin);
                out[s*p+d]=upper-lower;begin=end;}lo=hi;}return true;}
    if(mode==0){int lo=0;for(int s=0;s<p;++s){int hi=lo+source[s],a=0;
        for(int d=0;d<p;++d){int b=a+dest[d];out[s*p+d]=mx(0,mn(hi,b)-mx(lo,a));a=b;}lo=hi;}return true;}
    int left[PMAX],room[PMAX];int64_t frac[PMAX*PMAX];
    for(int s=0;s<p;++s)left[s]=source[s];for(int d=0;d<p;++d)room[d]=dest[d];
    if(mode==2){for(int s=0;s<p;++s)for(int d=0;d<p;++d)if(d!=owner){int v=(int)((int64_t)source[s]*dest[d]/n);out[s*p+d]=v;left[s]-=v;room[d]-=v;}
        for(int d=0;d<p;++d)if(d!=owner)for(int s=0;s<p;++s){int v=mn(left[s],room[d]);out[s*p+d]+=v;left[s]-=v;room[d]-=v;}
        for(int s=0;s<p;++s){out[s*p+owner]+=left[s];room[owner]-=left[s];left[s]=0;}
    }else{
        for(int d=0;d<p;++d){int v=mn(left[d],room[d]);out[d*p+d]=v;left[d]-=v;room[d]-=v;}
        int denom=0;for(int d=0;d<p;++d)denom+=room[d];
        if(denom){int original[PMAX],need[PMAX];for(int s=0;s<p;++s)original[s]=left[s];for(int d=0;d<p;++d)need[d]=room[d];
            for(int s=0;s<p;++s)for(int d=0;d<p;++d){int64_t prod=(int64_t)original[s]*need[d];int v=(int)(prod/denom);
                out[s*p+d]+=v;left[s]-=v;room[d]-=v;frac[s*p+d]=prod%denom;}
            for(int it=0;it<p*p;++it){int bs=-1,bd=-1;int64_t best=-2;
                for(int d=0;d<p;++d)if(room[d])for(int s=0;s<p;++s)if(left[s]&&frac[s*p+d]>best){bs=s;bd=d;best=frac[s*p+d];}
                if(bs<0)break;out[bs*p+bd]++;left[bs]--;room[bd]--;frac[bs*p+bd]=-1;
            }
        }
    }
    for(int s=0;s<p;++s){int sum=0;for(int d=0;d<p;++d){if(out[s*p+d]<0)return false;sum+=out[s*p+d];}if(sum!=source[s])return false;}
    for(int d=0;d<p;++d){int sum=0;for(int s=0;s<p;++s)sum+=out[s*p+d];if(sum!=dest[d])return false;}return true;
}
UP_INLINE bool run(State& a,int policy,int target) {
    if(a.p<1||a.p>PMAX||a.e>EMAX||a.p*a.s>CMAX||a.h<1||a.s<0)return false;
    if(policy==0)return moon(a,target);
    if(policy==1)return native(a);
    if(policy==2)return ultra(a);
    if(policy==3)return echo(a);
    if(policy==4)return eplb(a);
    return false; // step_time is refused until a calibrated model is supplied.
}
} // namespace UnifiedPlanner
