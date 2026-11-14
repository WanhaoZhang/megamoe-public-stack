"""Compile the real operand wrapper with CPU tensor/copy stubs.

Exercises ADL through the same unqualified GetTile used in CATLASS's K loop.
It does not emulate DMA visibility or claim NPU acceptance.
"""
from pathlib import Path
import os
import subprocess
import tempfile


def main():
    env_root=os.environ.get('MEGAFLUX_SOURCE_ROOT')
    root=Path(env_root) if env_root else Path(__file__).resolve().parents[2]
    header=root/'kernels/mega_moe_bf16/include/mega_moe_restore_operand.h'
    # Production device includes are replaced by small host tensor stubs only;
    # the wrapper implementation below is compiled without changing its body.
    implementation='\n'.join(line for line in header.read_text().splitlines()
        if not line.startswith(('#include', '#pragma')))
    stub=r'''
#include <cassert>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#define __aicore__
std::vector<std::pair<int,int>> waits;
int copies=0;
namespace tla {
template<class T> struct Tensor {
  using Coord=std::tuple<int,int>;
  Coord c,s;
  Coord coord()const{return c;} Coord originShape()const{return s;}
};
template<int I,class T> auto get(T const& t){return std::get<I>(t);}
template<class T,class C,class S> auto GetTile(T const& t,C const& c,S const& s){
 return Tensor<int>{{get<0>(t.coord())+get<0>(c),get<1>(t.coord())+get<1>(c)},s};
}
}
namespace Catlass {
namespace Arch {struct Ascend950{};}
namespace layout {struct ColumnMajor{}; struct RowMajor{};}
namespace Gemm::Tile {
template<class Arch,class Src,class Dst> struct TileCopyTla {
 template<class D,class S> void operator()(D const&,S const&){++copies;}
};
}}
namespace MegaFluxBackwardRestore {
constexpr int PANEL_ROWS=256;
inline void WaitPanel(uint64_t,int,int projection,int panel,int,int,int){
 waits.emplace_back(projection,panel);
}
}
struct Address {uint64_t restoreReady=1;int restoreSlot=0,restoreMode=2,
 restoreW1Panels=32,restoreW2Panels=28,restoreEpoch=7;};
'''
    tests=r'''
int main(){
 using namespace MegaFluxBackwardRestore;
 Address a;
 auto outer=WithOperandReady<Catlass::layout::RowMajor>(tla::Tensor<int>{{0,256},{7168,256}},a,1);
 ReadyOperandCopy<decltype(outer),int> copy;
 // Delayed producer: only each consumed K panel may be waited. The first
 // load cannot falsely certify panels1..27; the prefetch waits the next one.
 for(int k=0;k<28;++k){
   auto inner=GetTile(outer,std::make_tuple(k*256,0),std::make_tuple(256,256));
   static_assert(std::is_same_v<decltype(inner.ready),OperandReady>);
   copy(0,inner);
 }
 assert(waits.size()==28 && copies==28);
 for(int k=0;k<28;++k)assert(waits[k]==std::make_pair(1,k));
 waits.clear();
 auto w1=WithOperandReady<Catlass::layout::ColumnMajor>(tla::Tensor<int>{{0,768},{512,384}},a,0);
 ReadyOperandCopy<decltype(w1),int> recompute;
 for(int k=0;k<2;++k)recompute(0,GetTile(w1,std::make_tuple(k*256,0),std::make_tuple(256,384)));
 assert((waits==std::vector<std::pair<int,int>>{{0,3},{0,4}}));
 waits.clear();
 a.restoreSlot=-1;
 auto home=WithOperandReady<Catlass::layout::RowMajor>(tla::Tensor<int>{{0,0},{512,256}},a,1);
 ReadyOperandCopy<decltype(home),int> homeCopy;
 homeCopy(0,GetTile(home,std::make_tuple(0,0),std::make_tuple(256,256)));
 assert(waits.empty());
 a.restoreSlot=0;a.restoreMode=1;
 auto expert=WithOperandReady<Catlass::layout::RowMajor>(tla::Tensor<int>{{0,0},{512,256}},a,1);
 ReadyOperandCopy<decltype(expert),int> expertCopy;
 expertCopy(0,GetTile(expert,std::make_tuple(0,0),std::make_tuple(256,256)));
 assert(waits.empty());
}
'''
    with tempfile.TemporaryDirectory(prefix='restore-operand-host-') as directory:
        source=Path(directory)/'test.cpp'
        source.write_text(stub+'\n'+implementation+'\n'+tests)
        binary=Path(directory)/'test'
        subprocess.run(['g++','-std=c++17','-Wall','-Werror',str(source),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)
    print('RESTORE_OPERAND_ADL_FULL_K_ROW_COLUMN_TAIL_HOME_CPU_PASS')


if __name__=='__main__':
    main()
