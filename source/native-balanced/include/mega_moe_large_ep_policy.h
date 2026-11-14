#include "mega_moe_capacity.h"
#pragma once
#include <cstdint>

// Shared by the AIV GM planner and host-only policy differential tests.
// Q/CP/Quota retain their pointer address space, including __gm__ on device.
namespace MegaMoeLargePolicy {
constexpr int MaxE = 2048, MaxP = 256;
__aicore__ inline int Min(int a, int b) { return a < b ? a : b; }
__aicore__ inline int Max(int a, int b) { return a > b ? a : b; }
__aicore__ inline void Sort(int* order, int n, const int* values, bool descending) {
    for (int i=0;i<n;++i) order[i]=i;
    for (int i=1;i<n;++i) {
        int v=order[i],j=i-1;
        while (j>=0 && (descending ? values[order[j]]<values[v] : values[order[j]]>values[v])) {
            order[j+1]=order[j];--j;
        }
        order[j+1]=v;
    }
}
template<class Q, class CP>
__aicore__ inline bool Native(const int* counts, int experts, int ranks, int home, int slots,
                              int beta, int penalty, int gain, Q q, CP cp, int& before) {
    if (experts<=0 || experts>MaxE || ranks<2 || ranks>MaxP || home<=0 ||
        experts!=home*ranks || slots<0 || beta<=0 || penalty<0 || gain<0) return false;
    int load[MaxP],original[MaxP];
    int donors[MaxP],receivers[MaxP],order[MaxE];
    for (int r=0;r<ranks;++r) load[r]=original[r]=0;
    for (int i=0;i<ranks*slots;++i) cp[i]=-1;
    int64_t total=0,totalRows=0;
    for (int e=0;e<experts;++e) {
        if (counts[e]<0) return false;
        totalRows+=counts[e];if(totalRows>=2147483392LL)return false;
        int blocks=(static_cast<int64_t>(counts[e])+beta-1)/beta;
        total+=blocks;
        if (total>2147483520LL) return false;
        for (int r=0;r<ranks;++r) q[e*ranks+r]=r==e/home ? blocks : 0;
        load[e/home]+=blocks;original[e/home]+=blocks;
    }
    const int target=(total+ranks-1)/ranks;
    // Each move clears an original donor/expert edge or fills a receiver or
    // retires a donor. Receivers never exceed target, so cannot become donors.
    for (int round=0;round<=experts+2*ranks;++round) {
        bool moved=false;Sort(donors,ranks,load,true);
        for (int di=0;di<ranks;++di) {
            int d=donors[di];if (load[d]<=target) continue;
            // Sort expert indices directly against GM Q. Avoid another E-word
            // local cache: NPU runtime enforces a 32 KiB default stack.
            for (int e=0;e<experts;++e) order[e]=e;
            for (int i=1;i<experts;++i) {
                int v=order[i],j=i-1,value=q[v*ranks+d];
                while (j>=0 && q[order[j]*ranks+d]<value) {order[j+1]=order[j];--j;}
                order[j+1]=v;
            }
            for (int ei=0;ei<experts;++ei) {
                int e=order[ei];if (!q[e*ranks+d]) continue;
                Sort(receivers,ranks,load,false);
                for (int ri=0;ri<ranks;++ri) {
                    int r=receivers[ri];if (r==d || load[r]>=target) continue;
                    int existing=-1,free=-1;
                    for (int s=0;s<slots;++s) {
                        int v=cp[r*slots+s];
                        if (v==e && existing<0) existing=s;
                        if (v<0 && free<0) free=s;
                    }
                    if (r!=e/home && existing<0 && free<0) continue;
                    int take=Min(q[e*ranks+d],Min(target-load[r],load[d]-target));
                    if (take<=0) continue;
                    if (r!=e/home && existing<0) cp[r*slots+free]=e;
                    q[e*ranks+d]-=take;q[e*ranks+r]+=take;
                    load[d]-=take;load[r]+=take;moved=true;
                    if (load[d]<=target) break;
                }
                if (load[d]<=target) break;
            }
        }
        if (!moved) break;
        if (round==experts+2*ranks) return false;
    }
    if (penalty || gain) {
        for (int iteration=0;iteration<ranks*slots;++iteration) {
            int64_t aug[MaxP];for (int r=0;r<ranks;++r) aug[r]=load[r];
            for (int r=0;r<ranks;++r) for (int s=0;s<slots;++s) {
                int e=cp[r*slots+s];if (e>=0 && q[e*ranks+r]>0) aug[e/home]+=penalty;
            }
            int d=0;for (int r=1;r<ranks;++r) if (aug[r]>aug[d]) d=r;
            int bestv=0x7fffffff,beste=-1,bestr=-1,bests=-1;
            for (int r=0;r<ranks;++r) for (int s=0;s<slots;++s) {
                int e=cp[r*slots+s];if (e<0 || e/home!=d) continue;
                int v=q[e*ranks+r];if (v<=0) continue;
                if (v<bestv || (v==bestv && e<beste)) {bestv=v;beste=e;bestr=r;bests=s;}
            }
            if (beste<0 || bestv>=penalty) break;
            q[beste*ranks+bestr]=0;q[beste*ranks+d]+=bestv;
            load[bestr]-=bestv;load[d]+=bestv;cp[bestr*slots+bests]=-1;
        }
        int64_t aug[MaxP];for (int r=0;r<ranks;++r) aug[r]=load[r];
        for (int r=0;r<ranks;++r) for (int s=0;s<slots;++s) {
            int e=cp[r*slots+s];if (e>=0 && q[e*ranks+r]>0) aug[e/home]+=penalty;
        }
        int64_t previous=0,after=0;
        for (int r=0;r<ranks;++r) {
            if (original[r]>previous) previous=original[r];
            if (aug[r]>after) after=aug[r];
        }
        if (after+gain>=previous) {
            for (int e=0;e<experts;++e) for (int r=0;r<ranks;++r)
                q[e*ranks+r]=r==e/home?(static_cast<int64_t>(counts[e])+beta-1)/beta:0;
            for (int i=0;i<ranks*slots;++i) cp[i]=-1;
        }
    }
    before=0;for (int r=0;r<ranks;++r) before=Max(before,original[r]);
    for (int r=0;r<ranks;++r) for (int s=0;s<slots;++s) {
        int e=cp[r*slots+s];if (e>=0 && !q[e*ranks+r]) cp[r*slots+s]=-1;
    }
    return true;
}
template<class Q, class CP, class Quota>
__aicore__ inline bool Moon(const int* counts,int experts,int ranks,int home,int target,
                            Q q,CP copies,Quota quota) {
    if (ranks<2 || ranks>MaxP || experts!=ranks*home || experts>MaxE || home<1 || home>MEGAMOE_MAX_LOCAL_EXPERTS/2) return false;
    int64_t totalRows=0;
    for(int e=0;e<experts;++e) {if(counts[e]<0)return false;totalRows+=counts[e];}
    if(totalRows>=2147483392LL || totalRows!=static_cast<int64_t>(ranks)*target)return false;
    int balance[MaxP],remaining[MEGAMOE_MAX_LOCAL_EXPERTS / 2];
    for (int i=0;i<ranks*ranks;++i) quota[i]=0;
    for (int r=0;r<ranks;++r) {
        balance[r]=-target;
        for (int e=r*home;e<(r+1)*home;++e) {
            if (counts[e]<0) return false;
            balance[r]+=counts[e];
            for (int d=0;d<ranks;++d) q[e*ranks+d]=d==r?counts[e]:0;
        }
    }
    for (int i=0;i<ranks*home;++i) copies[i]=-1;
    for (int step=0;step<ranks*ranks;++step) {
        int owner=0,dest=0;
        for (int r=1;r<ranks;++r) {
            if (balance[r]>balance[owner]) owner=r;
            if (balance[r]<balance[dest]) dest=r;
        }
        if (balance[owner]<=0 || balance[dest]>=0) break;
        int take=-balance[dest];quota[owner*ranks+dest]=take;
        balance[owner]-=take;balance[dest]=0;
    }
    for (int owner=0;owner<ranks;++owner) {
        for (int j=0;j<home;++j) remaining[j]=counts[owner*home+j];
        for (int step=0;step<ranks+home;++step) {
            int dest=0,local=0;
            for (int r=1;r<ranks;++r) if (quota[owner*ranks+r]>quota[owner*ranks+dest]) dest=r;
            int wanted=quota[owner*ranks+dest];if (wanted<=0) break;
            for (int j=1;j<home;++j) if (remaining[j]>remaining[local]) local=j;
            int take=Min(remaining[local],wanted);if (take<=0) return false;
            int e=owner*home+local;
            q[e*ranks+owner]-=take;q[e*ranks+dest]+=take;
            remaining[local]-=take;quota[owner*ranks+dest]-=take;
        }
        for (int r=0;r<ranks;++r) if (quota[owner*ranks+r]) return false;
    }
    for (int r=0;r<ranks;++r) {
        int used=0,rows=0;
        for (int e=0;e<experts;++e) {
            int count=q[e*ranks+r];rows+=count;
            if (e/home==r || count<=0) continue;
            if (used>=home) return false;
            int pos=used;
            while (pos>0) {
                int prev=copies[r*home+pos-1],value=q[prev*ranks+r];
                if (value>count || (value==count && prev>e)) break;
                copies[r*home+pos]=prev;--pos;
            }
            copies[r*home+pos]=e;++used;
        }
        if (rows!=target) return false;
    }
    return true;
}
}
