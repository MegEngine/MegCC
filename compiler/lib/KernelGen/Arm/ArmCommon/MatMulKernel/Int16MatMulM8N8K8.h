#pragma once
#include <sstream>
#include <string>
#include "Arm/ArmCommon/InternalMatMul/InternalMatMul.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class Int16MatMulM8N8K8 : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    Int16M8N8K8MatMulKernel m_internal_kernel;
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen