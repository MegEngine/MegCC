/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/MatMulKernel/MatmulCommon.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {
#define FIX_BODY_GUARD                                                         \
    std::string GetBodyGuardBegin(TContext* ctx) const override { return ""; } \
    std::string GetBodyGuardEnd(TContext* ctx) const override { return ""; }
class GIKernelFunc : public KernelFunc {
public:
    FIX_BODY_GUARD
};

#undef FIX_BODY_GUARD
}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
