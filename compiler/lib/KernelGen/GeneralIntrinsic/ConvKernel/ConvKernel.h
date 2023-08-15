#pragma once
#include <memory>
#include <string>
#include "Common/ConvKernel.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Im2col/F32StrategyM4N12.h"
#include "Im2col/F32StrategyM4N8.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Winograd/WinogradF23Strategy4x8MK4.h"
#include "Winograd/WinogradF43Strategy4x16MK4.h"
#include "Winograd/WinogradF63Strategy4x16MK4.h"
#include "compiler/KernelGen/KernelGen.h"
#include "fp16/F16StrategyM8N8.h"
#include "fp16/WinogradF23Strategy8x8MK8.h"
#include "fp16/WinogradF43Strategy8x8MK8.h"
#include "fp16/WinogradF63Strategy8x8MK8.h"

namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {
class GIConvImpl : public ConvImpl {
public:
    virtual std::string GetKernelSymbol(TContext* context) const override {
        auto sub_str = GetKernelSubSymbol(context);
        if (sub_str.size() > 0) {
            return "GI_" + sub_str + "_" + ConvImpl::GetKernelSymbol(context);
        } else {
            return "GI_" + ConvImpl::GetKernelSymbol(context);
        }
    }
    virtual std::string GetKernelSubSymbol(TContext* context) const { return ""; }
};

class ConvFloatNCHWNCHW44 : public GIConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class ChannelWiseFloatMk4 : public GIConvImpl {
    //! gen channel wise k5s1 kernel
    std::string GenBodyMk4K5S1(TContext* contxt) const;

    //! gen channel wise k3s1 kernel
    std::string GenBodyMk4K3S1(TContext* contxt) const;

    //! gen channel wise k3s2 kernel
    std::string GenBodyMk4K3S2(TContext* contxt) const;

    //! gen channel wise k5s2 kernel
    std::string GenBodyMk4K5S2(TContext* contxt) const;

public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSubSymbol(TContext* context) const override {
        return "chanwise";
    };
};

class ConvIm2colFloat : public GIConvImpl {
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
    mutable Im2colFrameNchwxx m_framework;
    mutable F32StrategyM4N12 m_strategy;
};

class WinogradFloatF23NCHW44 : public GIConvImpl {
    mutable WinogradFrameNchw44 m_framework;
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

class Conv1x1FloatMk4 : public GIConvImpl {
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

class Float32NchwBackward : public GIConvImpl {
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
    std::shared_ptr<TContext> GetInnerGemmCtx(TContext* ctx) const;
    MatmulM4N12Kernel m_inner_gemm;
};

class WinogradFloatF43NCHW44 : public GIConvImpl {
    mutable WinogradFrameNchw44 m_framework;
    mutable WinogradF43Strategy4x16MK4 m_winograd_strategy;

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

class WinogradFloatF63NCHW44 : public GIConvImpl {
    mutable WinogradFrameNchw44 m_framework;
    mutable WinogradF63Strategy4x16MK4 m_winograd_strategy;

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

class WinogradFp16F23NCHW88 : public GIConvImpl {
    mutable WinogradFrameNchw88 m_framework;
    mutable WinogradF23Strategy8x8MK8 m_winograd_strategy;

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

class WinogradFp16F43NCHW88 : public GIConvImpl {
    mutable WinogradFrameNchw88 m_framework;
    mutable WinogradF43Strategy8x8MK8 m_winograd_strategy;

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

class WinogradFp16F63NCHW88 : public GIConvImpl {
    mutable WinogradFrameNchw88 m_framework;
    mutable WinogradF63Strategy8x8MK8 m_winograd_strategy;

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

class ConvIm2colFloat16M8N8 : public GIConvImpl {
public:
    std::string GetKernelSymbol(TContext* context) const override;
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;

    std::string GetWorkspaceBody(TContext* ctx) const override;

private:
    mutable Im2colFrameNchwxx m_framework;
    mutable F16StrategyM8N8 m_strategy;
};

class ConvIm2colFloatM4N8 : public GIConvImpl {
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
    mutable Im2colFrameNchwxx m_framework;
    mutable F32StrategyM4N8 m_strategy;
};
class Conv1x1Float16MK8 : public GIConvImpl {
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
    Fp16MatmulM8N8MK8Kernel m_inner_gemm;
};
class ConvFloat16NCHWNCHW88 : public GIConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class ChannelWiseFloat16Mk8 : public GIConvImpl {
    //! gen channel wise k5s1 kernel
    std::string GenBodyMk8K5S1(TContext* contxt) const;

    //! gen channel wise k3s1 kernel
    std::string GenBodyMk8K3S1(TContext* contxt) const;

    //! gen channel wise k3s2 kernel
    std::string GenBodyMk8K3S2(TContext* contxt) const;

    //! gen channel wise k5s2 kernel
    std::string GenBodyMk8K5S2(TContext* contxt) const;

public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSubSymbol(TContext* context) const override {
        return "chanwise_fp16";
    };
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
