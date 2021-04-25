/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Pooling.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <sstream>
#include <string>
#include "Common/PoolingKernel.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class PoolingKernel : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
