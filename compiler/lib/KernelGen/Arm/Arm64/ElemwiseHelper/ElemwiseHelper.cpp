/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/ElemwiseHelper/ElemwiseHelper.cpp
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
using namespace Arm64;

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
        CASE_DISPATCH("SIGMOID", ElemwiseGenUnarySigmoid);
    } else {
        CC_ABORT << mode << " not Implement now\n";
    }
    return nullptr;
}

#undef CASE_DISPATCH
#undef CASE_DISPATCH_ARG

std::string ElemwiseHelperFunc::BcastType2String(BcastType bcast_type) {
    return ArmCommon::ElemwiseHelperFunc::BcastType2String(bcast_type);
}

// vim: syntax=cpp.doxygen
