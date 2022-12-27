/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/Elemwise/Elemwise.h
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
namespace Arm64 {

class ElemwiseKernel : public KernelFunc {
public:
    virtual ~ElemwiseKernel(){};
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;

    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
