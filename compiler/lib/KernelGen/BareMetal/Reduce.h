#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class ReduceKernel : public KernelFunc {
public:
    virtual ~ReduceKernel(){};
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
