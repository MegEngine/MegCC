/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/KernelPack.h
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
namespace Armv7 {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(
            KernelPack::KernType kernel_type);
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
