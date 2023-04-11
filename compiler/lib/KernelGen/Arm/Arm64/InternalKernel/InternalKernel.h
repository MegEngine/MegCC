#pragma once
#include <string>
#include "Arm/Arm64/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {

class MatmulM4N16MK4Kernel : public Arm64InternalKernelFunc {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelSignature(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;
};

class MatmulM8N12MK4Kernel : public Arm64MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;
    std::string GetPackBWorkspaceBody(TContext*) const override;
};

class MatmulM8N12Kernel : public Arm64MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;
    std::string GetPackBWorkspaceBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;
};

class MatmulInt8DotM8N12MK4Kernel : public Arm64MatmulInternal {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext*) const override;
    bool need_post_process(TContext*) const override;

    std::string GetPackAWorkspaceBody(TContext*) const override;
    std::string GetPackBWorkspaceBody(TContext*) const override;
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
