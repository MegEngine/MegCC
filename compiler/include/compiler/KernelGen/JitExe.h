/**
 * \file compiler/include/compiler/JitExe.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2021 Megvii Inc. All rights reserved.
 */
#pragma once

#include <sstream>
#include <vector>
#include "KernelGen.h"
namespace megcc {
namespace KernelGen {

class JitExec {
public:
    using KernelFn = megcc::KernelGen::KernelFunc;
    //! Jit compile get workspace C kernel and run, return the real workspace
    //! size
    static size_t jit_exec_and_get_workspace(const KernelFn* func, TContext* ctx);
};

}  // namespace KernelGen
}  // namespace megcc
// vim: syntax=cpp.doxygen
