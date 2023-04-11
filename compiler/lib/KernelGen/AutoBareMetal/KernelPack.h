#pragma once

#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace AutoBareMetal {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(KernelPack::KernType kernel_type);
};

}  // namespace AutoBareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
