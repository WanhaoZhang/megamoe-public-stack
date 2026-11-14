#pragma once

#include "mega_moe_backward_restore.h"
#include "tla/tensor.hpp"

#ifndef MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE
#define MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE 0
#endif
#if MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE
#include "mega_moe_panel_acquire_cache.h"
#endif

// CATLASS calls GetTile again inside its complete-K MMAD. Keep readiness on
// that operand through ADL, so its real GM->L1 B copy waits for the rows it
// will read, including the next-K prefetch. No K loop or accumulation changes.
namespace MegaFluxBackwardRestore {
struct OperandReady {
    uint64_t flags=0;
    int slot=-1,projection=0,w1Panels=0,w2Panels=0,epoch=0,rowAxis=0;
#if MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE
    PanelAcquireCache* cache=nullptr;
#endif
};

template<class Tensor>
struct ReadyOperand : Tensor {
    OperandReady ready;
    __aicore__ inline ReadyOperand(const Tensor& tensor,const OperandReady& context)
        : Tensor(tensor),ready(context) {}
};

template<class Tensor,class Coord,class Shape>
__aicore__ inline auto GetTile(const ReadyOperand<Tensor>& tensor,
        const Coord& coord,const Shape& shape) {
    auto tile=tla::GetTile(static_cast<const Tensor&>(tensor),coord,shape);
    return ReadyOperand<decltype(tile)>(tile,tensor.ready);
}

template<class Tensor>
struct OperandBase {using Type=Tensor;};
template<class Tensor>
struct OperandBase<ReadyOperand<Tensor>> {using Type=Tensor;};

template<class TensorSrc,class TensorDst>
struct ReadyOperandCopy {
    using Plain=typename OperandBase<TensorSrc>::Type;
    using Copy=Catlass::Gemm::Tile::TileCopyTla<Catlass::Arch::Ascend950,Plain,TensorDst>;
    Copy copy;
    int previousFirst=-1,previousLast=-1;

    template<class Dst,class Src>
    __aicore__ inline void operator()(const Dst& dst,const Src& src) {copy(dst,src);}

    template<class Dst,class Src>
    __aicore__ inline void operator()(const Dst& dst,const ReadyOperand<Src>& src) {
        const auto& r=src.ready;
        if(r.flags && r.slot>=0) {
            const int begin=r.rowAxis?static_cast<int>(tla::get<1>(src.coord())):
                                      static_cast<int>(tla::get<0>(src.coord()));
            const int rows=r.rowAxis?static_cast<int>(tla::get<1>(src.originShape())):
                                     static_cast<int>(tla::get<0>(src.originShape()));
            const int first=begin/PANEL_ROWS,last=(begin+rows-1)/PANEL_ROWS;
            // Recompute B is column-major: every K slice of one output tile
            // reads the same physical W1 rows. Their epoch remains immutable.
            if(rows>0 && (first!=previousFirst || last!=previousLast)) {
#if MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE
                for(int panel=first;panel<=last;++panel) {
                    if(r.cache && r.cache->AlreadyAcquired(r.flags,r.slot,r.projection,
                            r.w1Panels,r.w2Panels,r.epoch,panel))continue;
                    WaitPanel(r.flags,r.slot,r.projection,panel,r.w1Panels,r.w2Panels,r.epoch);
                    if(r.cache)r.cache->RecordAcquire(panel);
                }
#else
                for(int panel=first;panel<=last;++panel)
                    WaitPanel(r.flags,r.slot,r.projection,panel,r.w1Panels,r.w2Panels,r.epoch);
#endif
                previousFirst=first;previousLast=last;
            }
        }
        copy(dst,static_cast<const Src&>(src));
    }
};

template<class Layout,class Tensor,class Address>
__aicore__ inline auto WithOperandReady(const Tensor& tensor,const Address& addr,int projection
#if MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE
        ,PanelAcquireCache* cache=nullptr
#endif
        ) {
    OperandReady r;
    if(addr.restoreMode==2) {
        r.flags=addr.restoreReady;r.slot=addr.restoreSlot;r.projection=projection;
        r.w1Panels=addr.restoreW1Panels;r.w2Panels=addr.restoreW2Panels;
        r.epoch=addr.restoreEpoch;
        r.rowAxis=std::is_same_v<Layout,Catlass::layout::ColumnMajor>?1:0;
#if MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE
        r.cache=cache;
#endif
    }
    return ReadyOperand<Tensor>(tensor,r);
}
}
