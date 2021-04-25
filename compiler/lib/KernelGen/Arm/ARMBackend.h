/**
 * \file
 * compiler/lib/KernelGen/Arm/ARMBackend.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
namespace megcc {
namespace KernelGen {

struct ARMBackend {
    static constexpr int cacheline_byte = 64;
};

}  // namespace KernelGen
}  // namespace megcc