#pragma once
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Armv7 {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(
            KernelPack::KernType kernel_type, const bool with_dot);
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
