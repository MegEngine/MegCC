#pragma once
#include <string>
#include "Arm/ArmCommon/InternalKernel.h"
#include "Arm/Armv7/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace Armv7 {

class MatmulM4N12Kernel : public Armv7MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;

    std::string GetPackBWorkspaceBody(TContext*) const override;
};

class MatmulM4N12MK4Kernel : public Armv7MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;

    std::string GetPackBWorkspaceBody(TContext*) const override;
};

class MatmulM4N8MK4Kernel : public Armv7InternalKernelFunc {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelSignature(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;
};

class Int8x8x32MK4MatMulKernel : public Armv7MatmulInternal {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetPackAWorkspaceBody(TContext*) const override;
    std::string GetPackBWorkspaceBody(TContext*) const override;
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
