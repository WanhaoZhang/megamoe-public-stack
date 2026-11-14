#include "mega_moe_capacity.h"
#pragma once
// Policy port only, pinned to Ascend MoonEP dev_server_dedup@36be681.
// q is [expert, destination]. No token-block rounding or cost pruning.
namespace MegaFluxMoonEP {
__aicore__ inline bool Run(const int* counts,int experts,int ranks,int home,
                          int target,int* q,int* copies) {
    int balance[8],quota[64],remaining[MEGAMOE_MAX_LOCAL_EXPERTS / 2];
    for(int i=0;i<64;++i)quota[i]=0;
    for(int r=0;r<ranks;++r) {
        balance[r]=-target;
        for(int e=r*home;e<(r+1)*home;++e)balance[r]+=counts[e];
    }
    for(int step=0;step<ranks*ranks;++step) {
        int owner=0,dest=0;
        for(int r=1;r<ranks;++r) {
            if(balance[r]>balance[owner])owner=r;
            if(balance[r]<balance[dest])dest=r;
        }
        if(balance[owner]<=0 || balance[dest]>=0)break;
        int take=-balance[dest];quota[owner*ranks+dest]=take;
        balance[owner]-=take;balance[dest]=0;
    }
    for(int owner=0;owner<ranks;++owner) {
        for(int j=0;j<home;++j)remaining[j]=counts[owner*home+j];
        for(int step=0;step<ranks+home;++step) {
            int dest=0,local=0;
            for(int r=1;r<ranks;++r)if(quota[owner*ranks+r]>quota[owner*ranks+dest])dest=r;
            int wanted=quota[owner*ranks+dest];if(wanted<=0)break;
            for(int j=1;j<home;++j)if(remaining[j]>remaining[local])local=j;
            int take=remaining[local]<wanted?remaining[local]:wanted;
            if(take<=0)return false;
            int e=owner*home+local;
            q[e*ranks+owner]-=take;q[e*ranks+dest]+=take;
            remaining[local]-=take;quota[owner*ranks+dest]-=take;
        }
        for(int r=0;r<ranks;++r)if(quota[owner*ranks+r])return false;
    }
    for(int r=0;r<ranks;++r) {
        int used=0,rows=0;
        for(int e=0;e<experts;++e) {
            int count=q[e*ranks+r];rows+=count;
            if(e/home==r || count<=0)continue;
            if(used>=home)return false;
            int pos=used;
            while(pos>0) {
                int prev=copies[r*home+pos-1],value=q[prev*ranks+r];
                if(value>count || (value==count && prev>e))break;
                copies[r*home+pos]=prev;--pos;
            }
            copies[r*home+pos]=e;++used;
        }
        if(rows!=target)return false;
    }
    return true;
}
}
