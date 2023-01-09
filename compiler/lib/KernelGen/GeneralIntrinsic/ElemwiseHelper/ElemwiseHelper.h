/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ElemwiseHelper/ElemwiseHelper.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <memory>
#include <sstream>
#include <string>
#include "Common/ElemwiseCommon.h"
#include "GeneralIntrinsic/GISimdHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class ElemwiseGenBase {
public:
    //! gen the code out side the compute kernel, just address offset, for loop
    virtual std::string GenCodeBody(std::vector<std::string>) const = 0;

    //! Gen elemwise kernel init code, init for the necessary simd variable,
    //! such as zero in Relu
    virtual std::string GenKernelSimdInit(std::vector<std::string>) const = 0;

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
    std::unique_ptr<GISimdHelper> m_src_simd;
    std::unique_ptr<GISimdHelper> m_dst_simd;
    bool m_i32_to_qs8;
    ElemwiseGenUnary(std::string src_dtype = "f32",
                     std::string dst_dtype = "f32", bool inline_mode = false)
            : m_src_dtype(src_dtype),
              m_dst_dtype(dst_dtype),
              m_inline_mode(inline_mode) {
        m_src_simd = std::make_unique<GISimdHelper>(src_dtype);
        m_dst_simd = std::make_unique<GISimdHelper>(dst_dtype);
        m_i32_to_qs8 = Utils::is_int_dtype(m_src_dtype, 32) &&
                       Utils::is_int_dtype(m_dst_dtype, 8);
    };
    std::string GenCodeBody(std::vector<std::string>) const override;
    virtual std::string GenInlineName() const = 0;
};

//! The Binary elemwise kernel base
class ElemwiseGenBinary : public ElemwiseGenBase {
public:
    BcastType m_bcast_type;
    bool m_should_reverse;

public:
    ElemwiseGenBinary(const CCOperand& operand0, const CCOperand& operand1) {
        m_bcast_type = GetBcastType(operand0, operand1);
        m_should_reverse = WhetherShouldReverse(operand0, operand1);
    }
    std::string GenCodeBody(std::vector<std::string>) const override;

    //! get the BcastType of the elemwise compute
    static BcastType GetBcastType(const CCOperand& operand0,
                                  const CCOperand& operand1);

    //! whether the inputs should reverse, such scalar_vec should reverse to
    //! vec_scalar
    static bool WhetherShouldReverse(const CCOperand& operand0,
                                     const CCOperand& operand1);
};

//! The Ternary elemwise kernel base
//! TODO: add ternary elemwise kernel here
class ElemwiseGenTernary : public ElemwiseGenBase {
    BcastType m_bcast_type;

public:
    ElemwiseGenTernary(const CCOperand& operand0, const CCOperand& operand1,
                       const CCOperand& operand2) {
        m_bcast_type = GetBcastType(operand0, operand1, operand2);
    }
    std::string GenCodeBody(std::vector<std::string>) const override;

    //! get the BcastType of the elemwise compute
    static BcastType GetBcastType(const CCOperand& operand0,
                                  const CCOperand& operand1,
                                  const CCOperand& operand2);
    static bool is_available(BcastType bcast_type);
};

//! create the elemwise helper implement according to the mode and operand
struct ElemwiseHelperFunc {
    static std::shared_ptr<ElemwiseGenBase> CreateGenHelper(
            std::string mode, std::vector<CCOperand> operands);
    static std::string BcastType2String(BcastType bcast_type);
};

/************************************Unary***********************************/

#define DEFINE_UNARY_OP(_name)                                               \
    class _name : public ElemwiseGenUnary {                                  \
    public:                                                                  \
        _name(std::string src_dtype = "f32", std::string dst_dtype = "f32",  \
              bool inline_mode = false)                                      \
                : ElemwiseGenUnary(SymbolHelper::gen_valid_dtype(src_dtype), \
                                   SymbolHelper::gen_valid_dtype(dst_dtype), \
                                   inline_mode) {}                           \
        std::string GenKernelSimdInit(                                       \
                std::vector<std::string>) const override;                    \
        std::string GenKernelSimdUnroll(                                     \
                std::vector<std::string>) const override;                    \
        std::string GenKernelNaiveUnroll(                                    \
                std::vector<std::string>) const override;                    \
        std::string GenInlineName() const override;                          \
    };

DEFINE_UNARY_OP(ElemwiseGenUnaryRelu)
DEFINE_UNARY_OP(ElemwiseGenUnaryExp)
DEFINE_UNARY_OP(ElemwiseGenUnarySigmoid)
DEFINE_UNARY_OP(ElemwiseGenUnaryHswish)
#undef DEFINE_UNARY_OP

/************************************Binary***********************************/
#define DEFINE_BINARY_OP(_name)                                     \
    class _name : public ElemwiseGenBinary {                        \
    public:                                                         \
        _name(const CCOperand& operand0, const CCOperand& operand1) \
                : ElemwiseGenBinary(operand0, operand1) {}          \
        std::string GenKernelSimdInit(                              \
                std::vector<std::string>) const override;           \
        std::string GenKernelSimdUnroll(                            \
                std::vector<std::string>) const override;           \
        std::string GenKernelNaiveUnroll(                           \
                std::vector<std::string>) const override;           \
    };

DEFINE_BINARY_OP(ElemwiseGenBinaryAdd)
DEFINE_BINARY_OP(ElemwiseGenBinarySub)
DEFINE_BINARY_OP(ElemwiseGenBinaryMul)
DEFINE_BINARY_OP(ElemwiseGenBinaryTrueDiv)
DEFINE_BINARY_OP(ElemwiseGenBinaryFuseAddRelu)
DEFINE_BINARY_OP(ElemwiseGenBinaryMax)
DEFINE_BINARY_OP(ElemwiseGenBinaryMin)
#undef DEFINE_BINARY_OP
//! TODO: add more binary elemwise here
/************************************Ternary***********************************/

#define DEFINE_TERNARY_OP(_name)                                      \
    class _name : public ElemwiseGenTernary {                         \
    public:                                                           \
        _name(const CCOperand& operand0, const CCOperand& operand1,   \
              const CCOperand& operand2)                              \
                : ElemwiseGenTernary(operand0, operand1, operand2) {} \
        std::string GenKernelSimdInit(                                \
                std::vector<std::string>) const override;             \
        std::string GenKernelSimdUnroll(                              \
                std::vector<std::string>) const override;             \
        std::string GenKernelNaiveUnroll(                             \
                std::vector<std::string>) const override;             \
    };

DEFINE_TERNARY_OP(ElemwiseGenTernaryFuseMulAdd3)

#undef DEFINE_TERNARY_OP
}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
