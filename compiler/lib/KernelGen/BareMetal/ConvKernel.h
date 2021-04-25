/**
 * \file
 * compiler/lib/KernelGen/BareMetal/ConvKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"
#include "Common/ConvKernel.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class ConvGeneral : public ConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
// vim: syntax=cpp.doxygen
