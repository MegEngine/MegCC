#pragma once
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(KernelPack::KernType kernel_type);
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
