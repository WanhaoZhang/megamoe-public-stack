#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#define __aicore__
#include "mega_moe_moonep_algorithm.h"

int main() {
    const int ranks = 8;
    const int home = MEGAMOE_MAX_LOCAL_EXPERTS / 2;
    const int experts = ranks * home;
    const int target = 1024;
    std::mt19937 random(42);
    for (int test = 0; test < 1000; ++test) {
        std::vector<int> counts(experts), quota(experts*ranks), copies(ranks*home,-1);
        for(int item=0;item<ranks*target;++item) {
            int expert = random()%experts;
            if (test%3==1 && item%2==0) expert=0;
            if (test%3==2 && item%2==0) expert=experts-1;
            ++counts[expert];
        }
        for(int expert=0;expert<experts;++expert) quota[expert*ranks+expert/home]=counts[expert];
        assert(MegaFluxMoonEP::Run(counts.data(),experts,ranks,home,target,quota.data(),copies.data()));
        for(int expert=0;expert<experts;++expert) {
            int sum=0;
            for(int rank=0;rank<ranks;++rank) { assert(quota[expert*ranks+rank]>=0);sum+=quota[expert*ranks+rank]; }
            assert(sum==counts[expert]);
        }
        for(int rank=0;rank<ranks;++rank) {
            int sum=0;
            for(int expert=0;expert<experts;++expert) {
                int count=quota[expert*ranks+rank];sum+=count;
                if(count && expert/home!=rank) {
                    assert(std::find(copies.begin()+rank*home,copies.begin()+(rank+1)*home,expert)!=copies.begin()+(rank+1)*home);
                }
            }
            assert(sum==target);
        }
    }
    std::cout << "POLICY_PASS capacity=" << MEGAMOE_MAX_LOCAL_EXPERTS << " cases=1000" << std::endl;
}
