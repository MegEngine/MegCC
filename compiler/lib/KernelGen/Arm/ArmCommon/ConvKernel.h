#pragma once
#include <memory>
#include <string>
#include "Arm/ArmCommon/ConvKernel/Int8/Winograd/WinogradCommon.h"
#include "Arm/ArmCommon/ConvKernel/Int8/Winograd/WinogradF23Strategy8x8Nchw44MK8Int8.h"
#include "Common/ConvKernel.h"
#include "InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "compiler/Common/TContext.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {
class ArmCommonConvImpl : public ConvImpl {
public:
    virtual std::string GetKernelSymbol(TContext* context) const override {
        auto sub_str = GetKernelSubSymbol(context);
        if (sub_str.size() > 0) {
            return "ArmCommon_" + sub_str + "_" + ConvImpl::GetKernelSymbol(context);
        } else {
            return "ArmCommon_" + ConvImpl::GetKernelSymbol(context);
        }
    }
    virtual std::string GetKernelSubSymbol(TContext* context) const { return ""; }
};

class ConvFloatNCHWNCHW44 : public ArmCommonConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class ChannelWiseFloatMk4 : public ArmCommonConvImpl {
    //! gen channel wise k5s1 kernel
    std::string GenBodyMk4K5S1(TContext* contxt) const;
    //! gen channel wise k5s2 kernel
    std::string GenBodyMk4K5S2(TContext* contxt) const;

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
    std::vector<KernelObj> GetDependInternalSymbol(TContext* ctx) const override;
};
class ChannelWiseInt8Nchw44 : public ArmCommonConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;

    std::string GetWorkspaceBody(TContext* context) const override;
};

class DirectInt8NCHW44 : public ArmCommonConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* ctx) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
    //! init gen
    std::string GetInitBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class Int8NchwNchw44ConvS1 : public ArmCommonConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetInitBody(TContext* context) const override;
    std::string GetKernelSubSymbol(TContext* context) const override {
        return "nchw_nchw44_s1";
    }
    std::string GetWorkspaceBody(TContext* context) const override;
};

class Int8NchwNchw44ConvS2 : public ArmCommonConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetInitBody(TContext* context) const override;
    std::string GetKernelSubSymbol(TContext* context) const override {
        return "nchw_nchw44_s2";
    }
    std::string GetWorkspaceBody(TContext* context) const override;
};

class WinogradFloatF23Nchw44MK8Int8 : public ArmCommonConvImpl {
    mutable ArmCommon::WinogradFrameNchw44Int8 m_framework;
    mutable WinogradF23Strategy8x8Nchw44MK8Int8 m_winograd_strategy;

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

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
