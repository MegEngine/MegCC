#pragma once
#include <sstream>
#include <string>
#include "Arm/Armv7/InternalKernel/InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Armv7 {

class Int8x8x32MatMulMK4 : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override {
        return GetWorkspaceBodyCondition(context, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* context) const override {
        return GetWorkspaceBodyCondition(context, true);
    }

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    Int8x8x32MK4MatMulKernel m_internal_kernel;
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen