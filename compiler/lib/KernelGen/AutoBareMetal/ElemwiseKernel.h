#pragma once
#include <sstream>
#include <string>
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace AutoBareMetal {

class ElmwiseKernel : public KernelFunc {
public:
    virtual ~ElmwiseKernel(){};
    bool IsAvailable(TContext* context) const override;
    //! kernel gen
    std::string GetKernelSymbol(TContext* context) const override;

    std::string GetKernelBody(TContext* context) const override;

    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
};

}  // namespace AutoBareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
