#pragma once
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace WebAssembly {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(
            KernelPack::KernType kernel_type);
};

}  // namespace WebAssembly
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
