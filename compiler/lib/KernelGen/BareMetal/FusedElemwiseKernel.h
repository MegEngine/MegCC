/**
 * \file
 * compiler/lib/KernelGen/BareMetal/ElemwiseKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class FusedElmwiseKernel : public KernelFunc {
public:
    virtual ~FusedElmwiseKernel() = default;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelSymbol(TContext* context) const override;

    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
