#pragma once
// Same scalar implementation is compiled on host for reference differential
// checks and on the NPU. No allocation, STL, floats or host synchronization.
#ifndef MEGAFLUX_SEARCH_INLINE
#define MEGAFLUX_SEARCH_INLINE __aicore__ inline
#endif
namespace MegaFluxSearch {
constexpr int EMAX=256,PMAX=8;
MEGAFLUX_SEARCH_INLINE int Min(int a,int b){return a<b?a:b;}
MEGAFLUX_SEARCH_INLINE int Max(int a,int b){return a>b?a:b;}
MEGAFLUX_SEARCH_INLINE void Order(int* out,const int* values,int n,bool desc){
    for(int i=0;i<n;++i)out[i]=i;
    for(int i=1;i<n;++i){int v=out[i],j=i-1;
        while(j>=0&&(desc?values[out[j]]<values[v]:values[out[j]]>values[v])){out[j+1]=out[j];--j;}
        out[j+1]=v;}
}
MEGAFLUX_SEARCH_INLINE void Home(const int* blocks,int E,int P,int* q,int* load){
    for(int r=0;r<P;++r)load[r]=0;
    for(int e=0;e<E;++e)for(int r=0;r<P;++r){int v=r==e/(E/P)?blocks[e]:0;q[e*P+r]=v;load[r]+=v;}
}
MEGAFLUX_SEARCH_INLINE int Pairs(const int* q,int E,int P,int* ex,int* rx){
    int n=0;for(int e=0;e<E;++e)for(int r=0;r<P;++r)
        if(r!=e/(E/P)&&q[e*P+r]>0){ex[n]=e;rx[n++]=r;}
    return n;
}
MEGAFLUX_SEARCH_INLINE void Owners(const int* q,int E,int P,int* owners){
    for(int r=0;r<P;++r)owners[r]=0;
    for(int e=0;e<E;++e)for(int r=0;r<P;++r)if(r!=e/(E/P)&&q[e*P+r]>0)++owners[e/(E/P)];
}
MEGAFLUX_SEARCH_INLINE void Candidate(const int* blocks,int E,int P,int S,int largest,int multiple,int* q,int* load){
    Home(blocks,E,P,q,load);int total=0;for(int e=0;e<E;++e)total+=blocks[e];int target=(total+P-1)/P,home=E/P;
    int donors[PMAX],values[EMAX],expertorder[EMAX];
    int ex[EMAX],rx[EMAX],order[EMAX],keys[EMAX];
    for(int round=0;round<S+2;++round){
        bool moved=false;
        for(int r=0;r<P;++r){values[r]=load[r];if(largest){values[r]=0;for(int e=r*home;e<(r+1)*home;++e)values[r]=Max(values[r],q[e*P+r]);}}
        Order(donors,values,P,true);
        for(int di=0;di<P;++di){int d=donors[di];if(load[d]<=target)continue;
            int n=Pairs(q,E,P,ex,rx),count=0;
            for(int i=0;i<n;++i)if(ex[i]/home==d){ex[count]=ex[i];rx[count]=rx[i];keys[count]=load[rx[i]];++count;}
            // Initial enumeration is (expert,rank); stable load sort preserves ties.
            Order(order,keys,count,false);
            for(int i=0;i<count;++i){int j=order[i],e=ex[j],r=rx[j];int take=Min(q[e*P+d],Max(0,target-load[r]));
                if(take){q[e*P+d]-=take;q[e*P+r]+=take;load[d]-=take;load[r]+=take;moved=true;}
                if(load[d]<=target)break;}
            if(load[d]<=target)continue;
            for(int i=0;i<home;++i)values[i]=q[(d*home+i)*P+d];Order(expertorder,values,home,true);
            for(int ei=0;ei<home;++ei){int e=d*home+expertorder[ei],opened=0;
                while(q[e*P+d]>0&&load[d]>target){
                    int receiver=-1;
                    for(int r=0;r<P;++r){int used=0;for(int x=0;x<E;++x)if(x/home!=r&&q[x*P+r]>0)++used;
                        if(r!=d&&load[r]<target&&q[e*P+r]==0&&used<S&&(receiver<0||load[r]<load[receiver]))receiver=r;}
                    if(receiver<0||(!multiple&&opened))break;
                    int take=Min(q[e*P+d],target-load[receiver]);q[e*P+d]-=take;q[e*P+receiver]+=take;load[d]-=take;load[receiver]+=take;
                    moved=true;++opened;}
                if(load[d]<=target)break;}
        }
        if(!moved)break;
    }
}
MEGAFLUX_SEARCH_INLINE void Run(const int* blocks,int E,int P,int S,int penalty,int gain,bool training,int* q,int* copies){
    int load[PMAX],original[PMAX];Home(blocks,E,P,q,original);
    int total=0,before=0;for(int r=0;r<P;++r){total+=original[r];before=Max(before,original[r]);}
    // Strict training/min_gain fallback can be proved before any candidate work.
    bool strict=training||gain>0;
    if(!S || (strict&&static_cast<long long>((total+P-1)/P)+gain>=before)){
        for(int i=0;i<P*S;++i)copies[i]=-1;return;}
    int temp[EMAX*PMAX],bestMax=0x7fffffff,bestRep=0x7fffffff;
    int ex[EMAX],rx[EMAX],order[EMAX],keys[EMAX];
    for(int c=0;c<4;++c){Candidate(blocks,E,P,S,c/2,c%2,temp,load);
        int maxload=0;for(int r=0;r<P;++r)maxload=Max(maxload,load[r]);int replicas=Pairs(temp,E,P,ex,rx);
        if(maxload<bestMax||(maxload==bestMax&&replicas<bestRep)){
            bestMax=maxload;bestRep=replicas;for(int i=0;i<E*P;++i)q[i]=temp[i];}}
    for(int r=0;r<P;++r){load[r]=0;for(int e=0;e<E;++e)load[r]+=q[e*P+r];}
    int n=Pairs(q,E,P,ex,rx);for(int i=0;i<n;++i)keys[i]=q[ex[i]*P+rx[i]];Order(order,keys,n,false);
    for(int i=0;i<n;++i){int j=order[i],e=ex[j],r=rx[j],v=q[e*P+r],owner=e/(E/P),target=-1;if(!v)continue;
        for(int t=0;t<P;++t)if(t!=r&&(t==owner||q[e*P+t]>0)&&load[t]+v<=bestMax){
            if(target<0||(t==owner&&target!=owner)||(t!=owner&&target!=owner&&load[t]<load[target]))target=t;}
        if(target>=0){q[e*P+r]=0;q[e*P+target]+=v;load[r]-=v;load[target]+=v;}}
    for(int i=0;i<E*P;++i){int donor=0,receiver=0;
        for(int r=1;r<P;++r){if(load[r]>load[donor])donor=r;if(load[r]<load[receiver])receiver=r;}
        if(load[donor]-load[receiver]<=1)break;
        int e=0;for(;e<E;++e)if(q[e*P+donor]>0&&(receiver==e/(E/P)||q[e*P+receiver]>0))break;
        if(e==E)break;--q[e*P+donor];++q[e*P+receiver];--load[donor];++load[receiver];}
    int owners[PMAX];
    if(training)for(int i=0;i<P*S;++i){Owners(q,E,P,owners);int owner=0;
        for(int r=1;r<P;++r)if(static_cast<long long>(load[r])+static_cast<long long>(penalty)*owners[r]>static_cast<long long>(load[owner])+static_cast<long long>(penalty)*owners[owner])owner=r;
        int eBest=-1,rBest=-1,vBest=0x7fffffff;
        for(int e=0;e<E;++e)if(e/(E/P)==owner)for(int r=0;r<P;++r)if(r!=owner&&q[e*P+r]>0&&q[e*P+r]<vBest){eBest=e;rBest=r;vBest=q[e*P+r];}
        if(eBest<0||vBest>=penalty)break;
        q[eBest*P+rBest]=0;q[eBest*P+owner]+=vBest;load[rBest]-=vBest;load[owner]+=vBest;}
    Owners(q,E,P,owners);long long proxy=0;
    for(int r=0;r<P;++r){long long v=static_cast<long long>(load[r])+(training?static_cast<long long>(penalty)*owners[r]:0);if(v>proxy)proxy=v;}
    if(strict&&proxy+gain>=before)Home(blocks,E,P,q,load);
    for(int r=0;r<P;++r){int s=0;for(int e=0;e<E;++e)if(e/(E/P)!=r&&q[e*P+r]>0)copies[r*S+s++]=e;
        for(;s<S;++s)copies[r*S+s]=-1;}
}
} // namespace MegaFluxSearch
