#pragma once
#include <string>
#include "Common/ConvKernel.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class ConvGeneral : public ConvImpl {
public:
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
// vim: syntax=cpp.doxygen
