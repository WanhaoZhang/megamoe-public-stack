/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * MGroup-only CATLASS resource overlay.
 *
 * The regular CATLASS Resource allocates local tensors through TPipe/TBuf.
 * MGroup creates GMM1 and GMM2 parameters for every wave, so repeatedly
 * constructing that Resource adds exposed TPipe setup cost.  This target-local
 * overlay binds the same CATLASS resource members to their fixed physical
 * local-memory positions instead.  It is intentionally visible only while
 * building ascend_deepep_mega_moe_bf16_mgroup; the CATLASS submodule and the
 * ordinary BF16/UDMA kernels continue to use the upstream Resource.
 */
#ifndef CATLASS_ARCH_RESOURCE_HPP
#define CATLASS_ARCH_RESOURCE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Arch {

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    CATLASS_DEVICE AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }

protected:
    CATLASS_DEVICE LocalTensorBufferBase() = default;

    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer : LocalTensorBufferBase {
public:
    CATLASS_DEVICE LocalTensorBuffer()
    {
        if constexpr (
            Position == AscendC::TPosition::A1 || Position == AscendC::TPosition::B1 ||
            Position == AscendC::TPosition::C1) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L1_SIZE);
        } else if constexpr (Position == AscendC::TPosition::A2) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L0A_SIZE);
        } else if constexpr (Position == AscendC::TPosition::B2) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L0B_SIZE);
        } else if constexpr (Position == AscendC::TPosition::C2) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::BIAS_SIZE);
        } else if constexpr (Position == AscendC::TPosition::CO1) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::L0C_SIZE);
        } else if constexpr (Position == AscendC::TPosition::C2PIPE2GM) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::FIXBUF_SIZE);
        } else if constexpr (
            Position == AscendC::TPosition::VECIN || Position == AscendC::TPosition::VECOUT ||
            Position == AscendC::TPosition::VECCALC) {
            this->tensor = AscendC::LocalTensor<uint8_t>(Position, 0, ArchTag::UB_SIZE);
        } else {
            static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported MGroup static local tensor position.");
        }
    }
};

template <class ArchTag>
struct Resource {
public:
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> fpBuf;

    CATLASS_DEVICE Resource()
    {
        // mega_moe_bf16_mgroup() initializes the SoC state once at kernel
        // entry.  Repeating InitSocState for every wave would reintroduce the
        // fixed setup cost this overlay is intended to remove.
    }
};

} // namespace Catlass::Arch

#endif // CATLASS_ARCH_RESOURCE_HPP
