#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class ReduceKernel : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc