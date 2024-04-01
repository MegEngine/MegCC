#pragma once
#include <memory>
#include <string>
#include "Arm/ArmCommon/ConvKernel/Fp32/Winograd/WinogradCommon.h"
#include "Common/ConvKernel.h"
#include "ConvKernel/Fp32/Winograd/WinogradF23Strategy4x16MK4.h"
#include "InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {
class Conv1x1FloatMk4 : public Arm64ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    MatmulM8N12MK4Kernel m_inner_gemm;
};

class Conv1x1DotMk4 : public Arm64ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    bool need_temp_dst(TContext* ctx) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    MatmulInt8DotM8N12MK4Kernel m_inner_gemm;
};

class Int8Conv1x1NCHW44 : public Arm64ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    bool need_temp_dst(TContext* ctx) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    MatmulInt8M4N4K16MK4Kernel m_inner_gemm;
};

class ConvBiasIm2colI8mmNCHW44 : public Arm64ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    MatmulInt8I8mmM8K8N12MK4Kernel m_inner_gemm;
};

class ConvIm2colFloat : public Arm64ConvImpl {
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

class ConvIm2colDot : public Arm64ConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    ArmCommon::MatmulInternal* GetInnerCtxMatmul(TContext* ctx) const;
};

class ConvDotNCHWNCHW44Common : public Arm64ConvImpl {
protected:
    bool IsAvailableCommon(TContext* context, const uint32_t valid_stride) const;

public:
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class ConvDotNCHWNCHW44Stride1 : public ConvDotNCHWNCHW44Common {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
};

class ConvDotNCHWNCHW44Stride2 : public ConvDotNCHWNCHW44Common {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
};

class WinogradFloatF23Nchw44 : public Arm64ConvImpl {
    mutable ArmCommon::WinogradFrameNchw44 m_framework;
    mutable WinogradF23Strategy4x16MK4 m_winograd_strategy;

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

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
