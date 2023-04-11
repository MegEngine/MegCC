#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {
#define FIX_BODY_GUARD                                                         \
    std::string GetBodyGuardBegin(TContext* ctx) const override { return ""; } \
    std::string GetBodyGuardEnd(TContext* ctx) const override { return ""; }
class GIKernelFunc : public KernelFunc {
public:
    FIX_BODY_GUARD
};

#undef FIX_BODY_GUARD
}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
