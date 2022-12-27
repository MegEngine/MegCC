/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/ElemwiseHelper/ElemwiseHelper.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <memory>
#include <sstream>
#include <string>
#include "Arm/ArmCommon/ArmSimdHelper.h"
#include "Arm/ArmCommon/ElemwiseHelper/ElemwiseHelper.h"
#include "Common/ElemwiseCommon.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace Arm64 {

class ElemwiseGenBase {
public:
    //! gen the code out side the compute kernel, just address offset, for loop
    virtual std::string GenCodeBody(std::vector<std::string>) const = 0;

    //! Gen elemwise kernel asm computing init code, init for the necessary simd
    //! variable, such as zero in Relu
    virtual std::string GenKernelAsmInit(std::vector<std::string>) const = 0;

    //! Gen the simd elemwise compute code, and the degree of unroll is specific
    //! by first param
    virtual std::string GenKernelSimdUnroll(std::vector<std::string>) const = 0;

    //! Gen the naive C elemwise compute code, and the degree of unroll is
    //! specific by first param
    virtual std::string GenKernelNaiveUnroll(
            std::vector<std::string>) const = 0;

    virtual ~ElemwiseGenBase() {}
};

//! The Unary elemwise kernel base
class ElemwiseGenUnary : public ElemwiseGenBase {
public:
    std::string m_src_dtype;
    std::string m_dst_dtype;
    bool m_inline_mode;
    std::unique_ptr<ArmCommon::ArmSimdHelper> m_src_simd;
    std::unique_ptr<ArmCommon::ArmSimdHelper> m_dst_simd;
    bool m_i32_to_qs8;
    std::unique_ptr<ArmCommon::ElemwiseGenUnarySigmoid> m_common_sigmoid_gen;
    ElemwiseGenUnary(std::string src_dtype = "f32",
                     std::string dst_dtype = "f32", bool inline_mode = false)
            : m_src_dtype(src_dtype),
              m_dst_dtype(dst_dtype),
              m_inline_mode(inline_mode) {
        m_src_simd = std::make_unique<ArmCommon::ArmSimdHelper>(src_dtype);
        m_dst_simd = std::make_unique<ArmCommon::ArmSimdHelper>(dst_dtype);
        m_common_sigmoid_gen =
                std::make_unique<ArmCommon::ElemwiseGenUnarySigmoid>(
                        src_dtype, dst_dtype, inline_mode);
        m_i32_to_qs8 = Utils::is_int_dtype(m_src_dtype, 32) &&
                       Utils::is_int_dtype(m_dst_dtype, 8);
    };
    std::string GenCodeBody(std::vector<std::string>) const override;
    virtual std::string GenInlineName() const = 0;
};

//! create the elemwise helper implement according to the mode and operand
struct ElemwiseHelperFunc {
    static std::shared_ptr<ElemwiseGenBase> CreateGenHelper(
            std::string mode, std::vector<CCOperand> operands);
    static std::string BcastType2String(BcastType bcast_type);
};

/************************************Unary***********************************/

#define DEFINE_NNARY_OP(_name)                                                 \
    class _name : public ElemwiseGenUnary {                                    \
    public:                                                                    \
        _name(std::string src_dtype = "f32", std::string dst_dtype = "f32",    \
              bool inline_mode = false)                                        \
                : ElemwiseGenUnary(SymbolHelper::gen_valid_dtype(src_dtype),   \
                                   SymbolHelper::gen_valid_dtype(dst_dtype),   \
                                   inline_mode) {}                             \
        std::string GenKernelAsmInit(std::vector<std::string>) const override; \
        std::string GenKernelSimdUnroll(                                       \
                std::vector<std::string>) const override;                      \
        std::string GenKernelNaiveUnroll(                                      \
                std::vector<std::string>) const override;                      \
        std::string GenInlineName() const override;                            \
    };

DEFINE_NNARY_OP(ElemwiseGenUnarySigmoid)
#undef DEFINE_NNARY_OP

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
