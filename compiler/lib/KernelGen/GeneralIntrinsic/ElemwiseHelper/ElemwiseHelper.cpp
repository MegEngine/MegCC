/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ElemwiseHelper/ElemwiseHelper.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "ElemwiseHelper.h"
#include "Utils/SymbolHelper.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

#define CASE_DISPATCH(_mode, _helper_name)       \
    if (mode == _mode) {                         \
        return std::make_shared<_helper_name>(); \
    }

#define CASE_DISPATCH_ARG(_mode, _helper_name, ...)         \
    if (mode == _mode) {                                    \
        return std::make_shared<_helper_name>(__VA_ARGS__); \
    }

std::shared_ptr<ElemwiseGenBase> ElemwiseHelperFunc::CreateGenHelper(
        std::string mode, std::vector<CCOperand> operands) {
    size_t nr_operands = operands.size();
    if (nr_operands == 2) {
        CASE_DISPATCH("RELU", ElemwiseGenUnaryRelu);
        CASE_DISPATCH("EXP", ElemwiseGenUnaryExp);
        CASE_DISPATCH("SIGMOID", ElemwiseGenUnarySigmoid);
        CASE_DISPATCH("H_SWISH", ElemwiseGenUnaryHswish);
    } else if (nr_operands == 3) {
        CASE_DISPATCH_ARG("ADD", ElemwiseGenBinaryAdd, operands[0],
                          operands[1]);
        CASE_DISPATCH_ARG("SUB", ElemwiseGenBinarySub, operands[0],
                          operands[1]);
        CASE_DISPATCH_ARG("MUL", ElemwiseGenBinaryMul, operands[0],
                          operands[1]);
        CASE_DISPATCH_ARG("TRUE_DIV", ElemwiseGenBinaryTrueDiv, operands[0],
                          operands[1]);
        CASE_DISPATCH_ARG("FUSE_ADD_RELU", ElemwiseGenBinaryFuseAddRelu,
                          operands[0], operands[1]);
        CC_ABORT << "Binary mode: " << mode << " not Implement now\n";
    } else if (nr_operands == 4) {
        CASE_DISPATCH_ARG("FUSE_MUL_ADD3", ElemwiseGenTernaryFuseMulAdd3,
                          operands[0], operands[1], operands[2]);
        CC_ABORT << "Ternary mode: " << mode << " not Implement now\n";
    } else {
        CC_ABORT << mode << " not Implement now\n";
    }
    return nullptr;
}

#undef CASE_DISPATCH
#undef CASE_DISPATCH_ARG

#define CASE_BCAST_TYPE(_name) \
    case _name:                \
        return #_name;
std::string ElemwiseHelperFunc::BcastType2String(BcastType bcast_type) {
    switch (bcast_type) {
        CASE_BCAST_TYPE(VEC)
        CASE_BCAST_TYPE(VEC_VEC)
        CASE_BCAST_TYPE(VEC_BCAST101)
        CASE_BCAST_TYPE(VEC_BCAST101x4)
        CASE_BCAST_TYPE(VEC_SCALAR)
        CASE_BCAST_TYPE(SCALAR_VEC)
        CASE_BCAST_TYPE(BCAST101_VEC)
        CASE_BCAST_TYPE(BCAST101x4_VEC)
        CASE_BCAST_TYPE(BV_VEC)
        CASE_BCAST_TYPE(VEC_BV)
        CASE_BCAST_TYPE(NAIVE)
        CASE_BCAST_TYPE(DYNAMIC_TYPE)
        CASE_BCAST_TYPE(VEC_VEC_VEC)
        CASE_BCAST_TYPE(VEC_VEC_SCALAR)
        CASE_BCAST_TYPE(BCAST101_VEC_BCAST101)
        CASE_BCAST_TYPE(BCAST101x4_VEC_BCAST101x4)
        CASE_BCAST_TYPE(VEC_BCAST101_VEC)
        CASE_BCAST_TYPE(VEC_BCAST101x4_VEC)
        CASE_BCAST_TYPE(VEC_SCALAR_VEC)
        CASE_BCAST_TYPE(VEC_SCALAR_SCALAR)
        CASE_BCAST_TYPE(UNKNOWN_BCAST_TYPE)
        default:
            CC_ABORT << "Unknow model " << bcast_type << ", "
                     << VEC_BCAST101_VEC << "\n";
            return "";
    }
}
#undef CASE_BCAST_TYPE

// vim: syntax=cpp.doxygen
