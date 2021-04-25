/**
 * \file
 * compiler/lib/KernelGen/Common/PoolingKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
class PoolingImpl : public KernelFunc {
public:
    std::string GetKernelSymbol(TContext* context) const override;
};

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
