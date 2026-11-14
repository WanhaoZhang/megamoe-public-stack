#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
using namespace AscendC;
namespace {
constexpr int MAX_E=256, MAX_P=8, WORKERS=32;
__aicore__ inline int Read(__gm__ int32_t* p) {return ReadGmBypassDCache(p);}
__aicore__ inline int Min(int a,int b) {return a<b?a:b;}
__aicore__ inline int Max(int a,int b) {return a>b?a:b;}
__aicore__ inline void Publish() {
    GlobalTensor<uint8_t> gm;
    DataCacheCleanAndInvalid<uint8_t,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(gm);
    DataSyncBarrier<MemDsbT::ALL>();
}
__aicore__ inline void Sort(int* order,int n,int* values,bool descending) {
    for(int i=0;i<n;++i)order[i]=i;
    for(int i=1;i<n;++i) {
        int v=order[i],j=i-1;
        while(j>=0 && (descending?values[order[j]]<values[v]:values[order[j]]>values[v])) {
            order[j+1]=order[j];--j;
        }
        order[j+1]=v;
    }
}
}

// All routing stays on device. Each rank computes the same small deterministic
// policy from published histograms; no full-route gather or host greedy search.
extern "C" [[bisheng::core_ratio(0,1)]] __global__ __aicore__
void mega_moe_replica_planner_kernel(uint64_t ffts,GM_ADDR base,GM_ADDR ids,
    GM_ADDR copies,GM_ADDR physical,GM_ADDR logical,GM_ADDR assignment,GM_ADDR header,
    uint64_t histoff,uint64_t metaoff,uint64_t erroroff,int n,int topk,
    int home,int slots,int rank,int ranks,int beta,int penalty,int gain,int epoch) {
    util_set_ffts_config(ffts);
    if ASCEND_IS_AIC {return;}
    const int worker=GetBlockIdx()/(GetSubBlockNum()>0?GetSubBlockNum():1);
    const int experts=home*ranks,es=home+slots;
    const int chunk=((n+WORKERS-1)/WORKERS+31)/32*32;
    const int begin=Min(n,worker*chunk),end=Min(n,begin+chunk);
    auto hist=(__gm__ int32_t*)(base+histoff);
    auto meta=(__gm__ int32_t*)(base+metaoff);
    auto errors=(__gm__ int32_t*)(base+erroroff);
    auto prefixes=(__gm__ int32_t*)(base+erroroff+WORKERS*128);
    auto totals=prefixes+MAX_E*WORKERS;
    auto input=(__gm__ int32_t*)ids;
    auto cpg=(__gm__ int32_t*)copies;auto qout=(__gm__ int32_t*)assignment;
    auto out=(__gm__ int32_t*)physical;auto lc=(__gm__ int32_t*)logical;
    auto hdr=(__gm__ int32_t*)header;
    int local[MAX_E];for(int e=0;e<experts;++e)local[e]=0;
    int bad=0;
    for(int i=begin;i<end;++i) {int e=Read(input+i);if(e<0||e>=experts)bad=1;else ++local[e];}
    for(int e=0;e<experts;++e)hist[worker*MAX_E+e]=local[e];
    errors[worker*32]=bad;
    if(worker==0) {
        meta[0]=epoch;meta[1]=n;meta[2]=topk;meta[3]=experts;
        meta[4]=slots;meta[5]=beta;meta[6]=penalty;meta[7]=gain;
    }
    Publish();aclshmemx_barrier_all_vec();
    // Parallelize global counting and prefix construction by expert. Each
    // writer owns full 128-byte rows, avoiding scalar DCache line races.
    int globalError=0;
    for(int r=0;r<ranks;++r) {
        auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);
        auto pe=(__gm__ int32_t*)(peer+erroroff);
        if(Read(pe+worker*32))globalError=1;
        if(worker<8) {
            auto pm=(__gm__ int32_t*)(peer+metaoff);
            if(Read(pm+worker)!=Read(meta+worker))globalError=2;
        }
    }
    errors[worker*32+1]=globalError;
    for(int e=worker;e<experts;e+=WORKERS) {
        int total=0,prefix=0;
        for(int r=0;r<ranks;++r) {
            auto peer=(__gm__ uint8_t*)aclshmem_ptr(base,r);
            auto ph=(__gm__ int32_t*)(peer+histoff);
            for(int w=0;w<WORKERS;++w) {
                int v=Read(ph+w*MAX_E+e);total+=v;if(r<rank)prefix+=v;
            }
        }
        totals[e*32]=total;
        for(int w=0;w<WORKERS;++w) {
            prefixes[e*32+w]=prefix;prefix+=Read(hist+w*MAX_E+e);
        }
    }
    Publish();aclshmemx_barrier_all_vec();
    if(worker==0) {
        int counts[MAX_E],blocks[MAX_E],q[MAX_E*MAX_P],load[MAX_P],original[MAX_P],cp[MAX_E];
        int donororder[MAX_P],receiverorder[MAX_P],expertorder[MAX_E],values[MAX_E];
        for(int e=0;e<experts;++e)counts[e]=Read(totals+e*32);
        int error=0;
        for(int w=0;w<WORKERS;++w)error=Max(error,Read(errors+w*32+1));
        for(int r=0;r<ranks;++r)load[r]=original[r]=0;
        for(int i=0;i<ranks*slots;++i)cp[i]=-1;
        int total=0;
        for(int e=0;e<experts;++e) {
            blocks[e]=(counts[e]+beta-1)/beta;lc[e]=counts[e];total+=blocks[e];
            for(int r=0;r<ranks;++r)q[e*ranks+r]=r==e/home?blocks[e]:0;
            load[e/home]+=blocks[e];original[e/home]+=blocks[e];
        }
        const int target=(total+ranks-1)/ranks;
        if(!error)for(int round=0;round<=total;++round) {
            bool moved=false;Sort(donororder,ranks,load,true);
            for(int di=0;di<ranks;++di) {
                int d=donororder[di];if(load[d]<=target)continue;
                for(int e=0;e<experts;++e)values[e]=q[e*ranks+d];
                Sort(expertorder,experts,values,true);
                for(int ei=0;ei<experts;++ei) {
                    int e=expertorder[ei];if(!q[e*ranks+d])continue;
                    Sort(receiverorder,ranks,load,false);
                    for(int ri=0;ri<ranks;++ri) {
                        int r=receiverorder[ri];if(r==d||load[r]>=target)continue;
                        int existing=-1,free=-1;
                        for(int s=0;s<slots;++s) {
                            int v=cp[r*slots+s];
                            if(v==e && existing<0)existing=s;
                            if(v<0 && free<0)free=s;
                        }
                        if(r!=e/home && existing<0 && free<0)continue;
                        int take=Min(q[e*ranks+d],Min(target-load[r],load[d]-target));
                        if(take<=0)continue;
                        if(r!=e/home && existing<0)cp[r*slots+free]=e;
                        q[e*ranks+d]-=take;q[e*ranks+r]+=take;load[d]-=take;load[r]+=take;moved=true;
                        if(load[d]<=target)break;
                    }
                    if(load[d]<=target)break;
                }
            }
            if(!moved)break;
        }
        if(penalty||gain) {
            for(int iteration=0;iteration<ranks*slots;++iteration) {
                int aug[MAX_P];for(int r=0;r<ranks;++r)aug[r]=load[r];
                for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                    int e=cp[r*slots+s];if(e>=0 && q[e*ranks+r]>0)aug[e/home]+=penalty;
                }
                int d=0;for(int r=1;r<ranks;++r)if(aug[r]>aug[d])d=r;
                int bestv=0x7fffffff,beste=-1,bestr=-1,bests=-1;
                for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                    int e=cp[r*slots+s];if(e<0||e/home!=d)continue;
                    int v=q[e*ranks+r];if(v<=0)continue;
                    if(v<bestv || (v==bestv && e<beste)) {bestv=v;beste=e;bestr=r;bests=s;}
                }
                if(beste<0||bestv>=penalty)break;
                q[beste*ranks+bestr]=0;q[beste*ranks+d]+=bestv;
                load[bestr]-=bestv;load[d]+=bestv;cp[bestr*slots+bests]=-1;
            }
            int aug[MAX_P];for(int r=0;r<ranks;++r)aug[r]=load[r];
            for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
                int e=cp[r*slots+s];if(e>=0 && q[e*ranks+r]>0)aug[e/home]+=penalty;
            }
            int before=0,after=0;
            for(int r=0;r<ranks;++r) {before=Max(before,original[r]);after=Max(after,aug[r]);}
            if(after+gain>=before) {
                for(int e=0;e<experts;++e)for(int r=0;r<ranks;++r)q[e*ranks+r]=r==e/home?blocks[e]:0;
                for(int i=0;i<ranks*slots;++i)cp[i]=-1;
            }
        }
        int active=0;for(int r=0;r<ranks;++r)for(int s=0;s<slots;++s) {
            int e=cp[r*slots+s];
            if(e>=0 && q[e*ranks+r]==0)cp[r*slots+s]=-1;
            else if(e>=0)++active;
        }
        for(int i=0;i<ranks*slots;++i)cpg[i]=cp[i];
        int rows[MAX_P],homeRows[MAX_P];for(int r=0;r<ranks;++r)rows[r]=homeRows[r]=0;
        for(int e=0;e<experts;++e) {
            homeRows[e/home]+=counts[e];int offset=0;
            for(int r=0;r<ranks;++r) {
                qout[e*ranks+r]=q[e*ranks+r];
                int next=offset+q[e*ranks+r]*beta;
                rows[r]+=Min(counts[e],next)-Min(counts[e],offset);offset=next;
            }
        }
        int maxrecv=1,before=0;
        for(int r=0;r<ranks;++r){maxrecv=Max(maxrecv,Max(rows[r],homeRows[r]));before=Max(before,original[r]);}
        hdr[0]=active;hdr[1]=(maxrecv+127)/128*128;hdr[2]=epoch;hdr[3]=error;
        hdr[4]=before;hdr[5]=0;hdr[6]=active;hdr[7]=0;
        Publish();
    }
    aclshmemx_barrier_all_vec();
    if(!Read(hdr+3)) {
        int cachedQ[MAX_E*MAX_P],cachedCopies[MAX_E];
        for(int i=0;i<experts*ranks;++i)cachedQ[i]=Read(qout+i);
        for(int i=0;i<ranks*slots;++i)cachedCopies[i]=Read(cpg+i);
        for(int e=0;e<experts;++e)local[e]=Read(prefixes+e*32+worker);
        for(int i=begin;i<end;++i) {
            int e=Read(input+i),block=local[e]++/beta,r=0;
            for(;r<ranks-1;++r) {int v=cachedQ[e*ranks+r];if(block<v)break;block-=v;}
            int slot=e%home;
            if(r!=e/home)for(int s=0;s<slots;++s)if(cachedCopies[r*slots+s]==e){slot=home+s;break;}
            out[i]=r*es+slot;
        }
        Publish();
    }
    // No rank may overwrite its histogram for the next layer invocation before
    // all peers finish consuming source prefixes for this epoch.
    aclshmemx_barrier_all_vec();
}
extern "C" __attribute__((visibility("default"))) void mega_moe_replica_planner_do(
    void* stream,uint64_t ffts,uint8_t* base,uint8_t* ids,uint8_t* copies,uint8_t* physical,
    uint8_t* logical,uint8_t* assignment,uint8_t* header,uint64_t histogram,uint64_t meta,uint64_t errors,
    int n,int topk,int home,int slots,int rank,int ranks,int beta,int penalty,int gain,int epoch) {
    mega_moe_replica_planner_kernel<<<WORKERS,nullptr,stream>>>(ffts,base,ids,copies,physical,logical,assignment,header,
        histogram,meta,errors,n,topk,home,slots,rank,ranks,beta,penalty,gain,epoch);
}
