#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {
class ConvBackDataGeneral : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;

    std::string GetKernelSymbol(TContext* context) const override;
    //! kernel gen
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
// vim: syntax=cpp.doxygen
