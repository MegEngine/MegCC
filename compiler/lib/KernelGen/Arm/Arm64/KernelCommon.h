#pragma once
#include <string>
#include "Arm/ArmCommon/InternalKernel.h"
#include "Common/ConvKernel.h"
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {
#define FIX_BODY_GUARD                                            \
    std::string GetBodyGuardBegin(TContext* ctx) const override { \
        return "\n#if defined(__aarch64__)\n";                    \
    }                                                             \
    std::string GetBodyGuardEnd(TContext* ctx) const override { return "\n#endif\n"; }

class Arm64MatmulInternal : public ArmCommon::MatmulInternal {
public:
    FIX_BODY_GUARD
};

struct Arm64InternalKernelFunc : public InternalKernelFunc {
    FIX_BODY_GUARD
};

class Arm64KernelFunc : public KernelFunc {
public:
    FIX_BODY_GUARD
};

class Arm64ConvImpl : public ConvImpl {
public:
    virtual std::string GetKernelSymbol(TContext* context) const override {
        return "Arm64_" + ConvImpl::GetKernelSymbol(context);
    }
    FIX_BODY_GUARD
};

#undef FIX_BODY_GUARD

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc