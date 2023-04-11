#pragma once
#include <memory>
#include <string>
#include "Arm/Armv7/ConvKernel/Winograd/WinogradF23Strategy4x8MK4.h"
#include "Arm/Armv7/InternalKernel/InternalKernel.h"
#include "Arm/Armv7/KernelCommon.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Armv7 {

class Conv1x1FloatMk4 : public Armv7ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    MatmulM4N12MK4Kernel m_inner_gemm;
};

class ConvIm2colFloat : public Armv7ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    ArmCommon::MatmulInternal* GetInnerCtxMatmul(TContext* ctx) const;
};

class WinogradFloatF23NCHW44 : public Armv7ConvImpl {
    mutable ArmCommon::WinogradFrameNchw44 m_framework;
    mutable WinogradF23Strategy4x8MK4 m_winograd_strategy;

public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;
};

class ConvFloatNCHWNCHW443x3s2 : public Armv7ConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
