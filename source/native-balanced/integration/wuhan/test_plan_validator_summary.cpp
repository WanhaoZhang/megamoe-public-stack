#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "../../kernels/mega_moe_bf16/include/mega_moe_plan_validator_logic.h"
#include "plan_validator_reference.h"

struct Read {
    const int* p;
    int operator()(int i) const { return p[i]; }
};

int main() {
    std::mt19937 rng(20261029);
    int cases=0;
    for(int ep : {2,4,8,16,32,64,128,256})
    for(int home : {1,2,4,8,16})
    for(int slots : {0,1,2,4,8,16}) {
        const int experts=ep*home,pc=ep*(home+slots),copies=ep*slots;
        if(home+slots>32 || pc>2048 || copies>2048)continue;
        const int placement=experts+pc,meta=placement+copies,stride=meta+3;
        std::vector<int> original(ep*stride,0);
        for(int src=0;src<ep;++src) {
            for(int i=0;i<experts+pc;++i)original[src*stride+i]=rng()%5;
            std::fill_n(original.data()+src*stride+placement,copies,-1);
            original[src*stride+meta+1]=17;
            original[src*stride+meta+2]=1048576;
        }
        for(int fault=0;fault<8;++fault) {
            auto data=original;
            if(fault==1)data[(ep-1)*stride+meta]=1;
            if(fault==2)data[(ep-1)*stride+meta+1]=18;
            if(fault==3)data[(ep-1)*stride+meta+2]=128;
            if(fault==4 && slots)data[(ep-1)*stride+placement]=1;
            if(fault==5 && slots) {
                for(int src=0;src<ep;++src) {
                    data[src*stride+placement]=home;
                    data[src*stride+experts+home]=0;
                }
            }
            if(fault==6)for(int src=0;src<ep;++src)data[src*stride+meta+2]=128;
            std::vector<int> counts(pc,0);
            for(int i=0;i<pc;++i)for(int src=0;src<ep;++src)
                counts[i]+=data[src*stride+experts+i];
            const int capacity=fault==6?128:1048576;
            for(int native=0;native<=3;++native) {
                MegaFluxPlanValidation::Summary all;
                for(int worker=0;worker<32;++worker) {
                    MegaFluxPlanValidation::Summary partial;
                    for(int rank=worker;rank<ep;rank+=32) {
                        MegaFluxPlanValidation::CheckSource(Read{data.data()},rank,ep,home,slots,17,capacity,partial);
                        MegaFluxPlanValidation::CheckDestination(Read{data.data()},Read{counts.data()},rank,ep,home,slots,partial);
                    }
                    all.error|=partial.error;all.mismatch|=partial.mismatch;
                    all.active+=partial.active;all.required=std::max(all.required,partial.required);
                }
                auto actual=MegaFluxPlanValidation::Finish(all,native,capacity);
                auto expected=MegaFluxPlanValidationReference::Finalize(Read{data.data()},native,ep,home,slots,17,capacity);
                assert(actual.active==expected.active && actual.required==expected.required && actual.status==expected.status);
                ++cases;
            }
        }
    }
    std::cout << "PLAN_VALIDATOR_SUMMARY_PASS cases=" << cases << '\n';
}
