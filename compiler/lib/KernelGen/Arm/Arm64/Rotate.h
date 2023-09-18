#pragma once
#include <string>
#include <sstream>
#include "compiler/KernelGen/KernelGen.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
namespace megcc {
namespace KernelGen {
namespace Arm64 {

class RotateKernel : public KernelFunc  {
public:
    bool IsAvailable(TContext* context) const override { return false; };
    std::string GetKernelBody(TContext* context) const override { return ""; };
    std::string GetKernelSymbol(TContext* context) const override { return ""; };
    bool IsCVAvailable(TContext* context) const override;
    std::string GetCVKernelBody(TContext* context) const override;
    std::string GetCVKernelSubSymbol(TContext* context) const;
    std::string GetCVKernelSignature(TContext* context) const override;

    std::string GetCVKernelSymbol(TContext* context) const override final {
        std::stringstream ss;
        Utils::cv_kern_sym_add_prefix(context, "arm64", ss);
        ss << GetCVKernelSubSymbol(context);
        return ss.str();
    };
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen