#pragma once
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace Arm64 {

class RotateKernel : public KernelFunc  {
public:
    bool IsCVAvailable(TContext* context) const override;
    std::string GetCVKernelBody(TContext* context) const override;
    std::string GetCVKernelSubSymbol(TContext* context) const;
    std::string GetCVKernelSignature(TContext* context) const override;
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc