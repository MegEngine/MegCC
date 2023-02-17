
/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/CvCommon.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class CVKernelImpl : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override { return false; };
    std::string GetKernelBody(TContext* context) const override { return ""; };
    std::string GetKernelSymbol(TContext* context) const override { return ""; };
    virtual std::string GetCVKernelSubSymbol(TContext* context) const {
        CC_ABORT << "must impl cv subsymbol";
        return "";
    };
    std::string GetCVKernelSymbol(TContext* context) const override final {
        std::stringstream ss;
        Utils::cv_kern_sym_add_prefix(context, "armcommon", ss);
        ss << GetCVKernelSubSymbol(context);
        return ss.str();
    };
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc