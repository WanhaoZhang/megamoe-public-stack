#include "mega_moe_gradient_order.h"
struct Reader {
    const int* plan;int reads=0;
    int Read(int index){++reads;return plan[index];}
};
extern "C" __declspec(dllexport) int build_gradient_order(const int* plan,int ranks,
        int rank,int home,int slots,int extent,int* order,int* reads) {
    Reader reader{plan};bool ok=MegaFluxGradientOrder::Build(reader,ranks,rank,home,slots,extent,order);
    *reads=reader.reads;return ok?0:1;
}
