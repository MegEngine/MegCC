#pragma once
#include <sstream>
#include <string>
#include "Arm/Arm64/InternalKernel/InternalKernel.h"
#include "Arm/Arm64/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace Arm64 {

class Fp32BatchedMatmul : public Arm64KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    MatmulM8N12Kernel gemm_kernel;
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
