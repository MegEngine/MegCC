#pragma once
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

struct ArchKernelPack {
    static std::vector<const KernelFunc*> GetKernel(KernelPack::KernType kernel_type);
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
