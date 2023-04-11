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
