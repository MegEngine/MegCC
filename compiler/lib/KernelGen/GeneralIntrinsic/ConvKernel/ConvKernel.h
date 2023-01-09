/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/ConvKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <memory>
#include <string>
#include "Common/ConvKernel.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Winograd/WinogradF23Strategy4x8MK4.h"
#include "Winograd/WinogradF43Strategy4x16MK4.h"
#include "Winograd/WinogradF63Strategy4x16MK4.h"
#include "compiler/KernelGen/KernelGen.h"

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
    virtual std::string GetKernelSubSymbol(TContext* context) const {
        return "";
    }
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
    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    GeneralIntrinsic::MatmulInternal* GetInnerCtxMatmul(TContext* ctx) const;
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

    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

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
    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

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
class WinogradFloatF43Nchw44 : public GIConvImpl {
    mutable WinogradFrameNchw44 m_framework;
    mutable WinogradF43Strategy4x16MK4 m_winograd_strategy;

public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;

    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;
};

class WinogradFloatF63Nchw44 : public GIConvImpl {
    mutable WinogradFrameNchw44 m_framework;
    mutable WinogradF63Strategy4x16MK4 m_winograd_strategy;

public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;

    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
