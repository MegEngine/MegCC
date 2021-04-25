/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/KernelCommon.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "Arm/ArmCommon/InternalKernel.h"
#include "Common/ConvKernel.h"
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Armv7 {

#define FIX_BODY_GUARD                                            \
    std::string GetBodyGuardBegin(TContext* ctx) const override { \
        return "\n#if defined(__arm__)\n";                        \
    }                                                             \
    std::string GetBodyGuardEnd(TContext* ctx) const override {   \
        return "\n#endif\n";                                      \
    }

class Armv7MatmulInternal : public ArmCommon::MatmulInternal {
public:
    FIX_BODY_GUARD
};

struct Armv7InternalKernelFunc : public InternalKernelFunc {
    FIX_BODY_GUARD
};

class Armv7KernelFunc : public KernelFunc {
public:
    FIX_BODY_GUARD
};

class Armv7ConvImpl : public ConvImpl {
public:
    virtual std::string GetKernelSymbol(TContext* context) const override {
        return "Armv7_" + ConvImpl::GetKernelSymbol(context);
    }
    FIX_BODY_GUARD
};

#undef FIX_BODY_GUARD

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc