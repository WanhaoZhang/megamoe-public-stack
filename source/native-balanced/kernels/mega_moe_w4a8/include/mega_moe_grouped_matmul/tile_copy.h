//
// Created by z00937986 on 26-5-20.
//

#ifndef ASCEND_DEEPEP_MEGA_MOE_W4A8_TILE_COPY_H
#define ASCEND_DEEPEP_MEGA_MOE_W4A8_TILE_COPY_H

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "tla/tensor.hpp"
#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "tla/tensor.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"

namespace Catlass::Gemm {
    // Block Mmad Policies add
    template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L1_SCALE_FACTOR_K_ = 16, uint32_t L0C_STAGES_ = 1,
        bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
        uint32_t L0B_STAGES_ = 2, uint32_t UB_STAGES_ = 2>
    struct MmadMxW4A8 : public MmadBase<ArchTag_, false> {
        static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
        static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
        static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
        static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
        static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
        static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
        static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
        /// GM→L1 MX scale 一次驻留的 L1 K 条带个数。设为 1 时每条 L1 K 条带各搬一次 scale。
        static constexpr uint32_t L1_SCALE_FACTOR_K = L1_SCALE_FACTOR_K_;
        static constexpr uint32_t UB_STAGES = UB_STAGES_;
    };
}

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    uint32_t TargetUBIdx = 0,
    bool ReluEnable = false,
    class Enable = void
>
struct CopyL0CToTargetUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to aiv one ub, can not find the specialization.");
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_, uint32_t TargetUBIdx_>
struct CopyL0CToTargetUBTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity::NO_QUANT,
    TargetUBIdx_,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr auto targetUBIdx = TargetUBIdx_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.subBlockId = targetUBIdx;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
    struct TileCopyTla<
        Arch::Ascend950,
        tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
        tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::VECCALC>,
        std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>
        > {
        //w4A8 特殊的nZ， 每个ELE_NUM_PER_C0为32和fp8对齐；
        static constexpr uint32_t ELE_NUM_PER_C0 = 32;

        // Methods
        CATLASS_DEVICE
        TileCopyTla(){};

        template <class TensorDst, class TensorSrc>
        CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
        {
            static_assert(
                tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                    && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                    && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::VECCALC,
                "The input parameters do not match. TensorSrc must be GM and nZ, "
                "while TensorDst must be L1 and nZ"
            );

            uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(tla::get<0>(srcTensor.originShape()));
            uint32_t blockLen = CeilDiv<2>(tla::get<1>(srcTensor.originShape()));

            AscendC::DataCopyParams repeatParams;

            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 / 4  - blockLen;
            repeatParams.dstStride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 / 4  - blockLen;

            auto dstOffset = dstTensor.layout()(dstTensor.coord());
            auto srcOffset = RoundDown<ELE_NUM_PER_C0>(tla::get<0>(srcTensor.coord())) * tla::get<0, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.shape()) +
            tla::get<1>(srcTensor.coord()) * ELE_NUM_PER_C0 + tla::get<0>(srcTensor.coord()) % ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
        }
    };

template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA_,
    class ElementB_,
    class LayoutTagB_,
    class ElementPrologueB_,
    class PrologueLayoutBTag_,
    class ElementMxScaleA_,
    class LayoutMxScaleATag_,
    class ElementMxScaleB_,
    class LayoutMxScaleBTag_,
    class ElementC_,
    class LayoutTagC_,
    class ElementBias = void,
    uint32_t TargetUBIdx = 0,
    CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,
    bool ReluEnable = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool COPY_L0C_TO_GM = false
>
struct PackedMx4A8TileCopyTla : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA_, ElementB_, LayoutTagB_,
    ElementC_, LayoutTagC_, ElementBias> {
    using ElementMxScaleA = ElementMxScaleA_;
    using ElementMxScaleB = ElementMxScaleB_;

    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;

    using LayoutTagL1MxScaleA = layout::zZ;
    using LayoutTagL1MxScaleB = layout::nN;
    using LayoutL1MxScaleA = detail::TagToLayout_t<ElementMxScaleA, LayoutTagL1MxScaleA>;
    using LayoutL1MxScaleB = detail::TagToLayout_t<ElementMxScaleB, LayoutTagL1MxScaleB>;

    using TensorL1MxScaleA = tla::Tensor<AscendC::LocalTensor<ElementMxScaleA>, LayoutL1MxScaleA,
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL1MxScaleB = tla::Tensor<AscendC::LocalTensor<ElementMxScaleB>, LayoutL1MxScaleB,
        tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    template <class TensorMxScaleA>
    using CopyGmToL1MxScaleA = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleA, TensorL1MxScaleA>;

    template <class TensorMxScaleB>
    using CopyGmToL1MxScaleB = Gemm::Tile::TileCopyTla<ArchTag, TensorMxScaleB, TensorL1MxScaleB>;

    //MXW4A8 的GM -> UB 搬运
    using ElementPrologueB = ElementPrologueB_;
    using PrologueLayoutB = detail::TagToLayout_t<ElementPrologueB, PrologueLayoutBTag_>;
    using LayoutMxScaleA = decltype(tla::MakeMxScaleLayout<ElementMxScaleA, LayoutMxScaleATag_, false>(uint32_t(0), uint32_t(0)));
    using LayoutMxScaleB = decltype(tla::MakeMxScaleLayout<ElementMxScaleB, LayoutMxScaleBTag_, true>(uint32_t(0), uint32_t(0)));

    using LayoutTagBub = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB_, LayoutTagB>>::L1BType::Layout;
    using LayoutUB = detail::TagToLayout_t<ElementPrologueB, LayoutTagBub>;
    using TensorUB =
            tla::Tensor<AscendC::LocalTensor<ElementPrologueB>, LayoutUB, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::VECCALC>;
    template <class TensorB>
    using CopyGmToUB = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorUB>;

    // 重写 CopyL0CToDst
    static constexpr CopyL0CToUBMode CopyMode = CopyMode_;
    using TensorL0C = typename PackedTileCopyTla<
        ArchTag,
        ElementA_,
        LayoutTagA,
        ElementB_,
        LayoutTagB,
        ElementC_,
        LayoutTagC,
        ElementBias>::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        std::conditional_t<COPY_L0C_TO_GM,
            Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>,
            Gemm::Tile::CopyL0CToTargetUBTla<ArchTag, TensorL0C, TensorC, CopyMode,
                                             DEQUANT_GRANULARITY, TargetUBIdx, ReluEnable>>;
};

} // namespace Catlass::Gemm::Tile

#endif //OPS_TRANSFORMER_MAIN_TILE_COPY_H
