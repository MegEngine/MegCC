/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/ConvKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <memory>
#include <string>
#include "Common/ConvKernel.h"
#include "ConvKernel/Fp32/Winograd/WinogradF23Strategy4x16MK4.h"
#include "Arm/ArmCommon/ConvKernel/Fp32/Winograd/WinogradCommon.h"
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
    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

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
    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    bool need_temp_dst(TContext* ctx) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    MatmulInt8DotM8N12MK4Kernel m_inner_gemm;
};

class ConvIm2colFloat : public Arm64ConvImpl {
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
    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
    ArmCommon::MatmulInternal* GetInnerCtxMatmul(TContext* ctx) const;
};

class ConvDotNCHWNCHW44 : public Arm64ConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class ChannelWiseInt8Mk4K3 : public Arm64ConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;

    std::string GetWorkspaceBody(TContext* context) const override;
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

    std::vector<KernelObj> GetDependInternalSymbol(
            TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc 

// vim: syntax=cpp.doxygen
