/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Pooling.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <sstream>
#include <string>
#include "Common/PoolingKernel.h"
#include "Utils/SymbolHelper.h"
namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class PoolingNchw44Fp32 : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override {
        return "ArmCommon_" + PoolingImpl::GetKernelSymbol(context) + "_" +
               SymbolHelper::gen_io_str(context) + "_fallback";
    };
};

class PoolingNchw44QInt8 : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc