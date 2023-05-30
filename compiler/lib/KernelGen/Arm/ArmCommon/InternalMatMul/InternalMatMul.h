#pragma once
#include <sstream>
#include <string>
#include "Arm/ArmCommon/InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class Int16M8N8K8MatMulKernel : public ArmCommon::MatmulInternal {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelSignature(TContext*) const override;
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen