#pragma once
#include <sstream>
#include <string>
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace Arm64 {

class ElemwiseMultiTypeKernel : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelSymbol(TContext* context) const override;

    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
