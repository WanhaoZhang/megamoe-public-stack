// Portable test adapter; the independent DAG reference is step_time_reference.py.
#include <vector>
#include <cstdint>
#include "mega_moe_step_time_policy.h"
struct Memory {
    const int* costs;const int* counts;int* flow;std::vector<int> scratch;
    Memory(const int* m,const int* s,int* q):costs(m),counts(s),flow(q),scratch(32768,0){}
    int cost(int i){return costs[i];}int get(int i){return scratch.at(i);}
    void put(int i,int v){scratch.at(i)=v;}int source(int s,int e){return counts[s*256+e];}
    int quota(int s,int e,int d){return flow[(s*256+e)*8+d];}
    void set_quota(int s,int e,int d,int v){flow[(s*256+e)*8+d]=v;}
};
extern "C" __declspec(dllexport) int search_reference(const int* source,const int* model,int* copies,int* flow,float* stats){
    UnifiedPlanner::State a{};a.p=8;a.e=256;a.h=a.s=32;
    int total=0;for(int e=0;e<256;++e)for(int s=0;s<8;++s){a.c[e]+=source[s*256+e];total+=source[s*256+e];}
    a.home();Memory m(model,source,flow);StepTime::Engine<Memory> engine{a,m,model[9]};
    if(!engine.run(model[10],total/8))return 1;
    for(int i=0;i<256;++i)copies[i]=a.cp[i];
    stats[0]=float(a.evaluations);stats[1]=float(a.rounds);stats[2]=float(a.adopted);stats[3]=float(a.source_policy);
    stats[4]=a.predicted_before;stats[5]=a.predicted_after;stats[6]=float(a.model_flags);stats[7]=float(model[9]);
    return 0;
}
extern "C" __declspec(dllexport) int score_reference(const int* source,const int* model,const int* copies,const int* quota,int64_t* answer){
    UnifiedPlanner::State a{};a.p=8;a.e=256;a.h=a.s=32;
    for(int i=0;i<256;++i)a.cp[i]=copies[i];
    std::vector<int> flow(quota,quota+16384);Memory m(model,source,flow.data());
    for(int s=0;s<8;++s)for(int e=0;e<256;++e)for(int d=0;d<8;++d)a.q[e*8+d]+=flow[(s*256+e)*8+d];
    StepTime::Engine<Memory> engine{a,m,model[9]};for(int d=0;d<8;++d)for(int k=0;k<64;++k)engine.cache_slot(d,k);
    *answer=engine.score();return 0;
}
extern "C" __declspec(dllexport) int phase_reference(const int* source,const int* model,const int* copies,const int* quota,int64_t* answer){
    UnifiedPlanner::State a{};a.p=8;a.e=256;a.h=a.s=32;
    for(int i=0;i<256;++i)a.cp[i]=copies[i];
    std::vector<int> flow(quota,quota+16384);Memory m(model,source,flow.data());
    for(int s=0;s<8;++s)for(int e=0;e<256;++e)for(int d=0;d<8;++d)a.q[e*8+d]+=flow[(s*256+e)*8+d];
    StepTime::Engine<Memory> engine{a,m,model[9]};for(int d=0;d<8;++d)for(int k=0;k<64;++k)engine.cache_slot(d,k);
    engine.score();int64_t fw=0,bw=0,owner=0;
    for(int d=0;d<8;++d){fw=StepTime::maxv(fw,engine.fw[d]);bw=StepTime::maxv(bw,engine.tail[d]);owner=StepTime::maxv(owner,engine.owners[d]);}
    answer[0]=fw;
    answer[1]=StepTime::maxv(StepTime::maxv(bw,owner),engine.network(6*model[7]+16,6*model[7]*model[8],6*model[7]*model[8])*model[24]/1000000)+model[23];
    answer[2]=owner;return 0;
}
