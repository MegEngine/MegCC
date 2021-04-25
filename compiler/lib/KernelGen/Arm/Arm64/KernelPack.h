/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/KernelPack.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#pragma once
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(
            KernelPack::KernType kernel_type);
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
