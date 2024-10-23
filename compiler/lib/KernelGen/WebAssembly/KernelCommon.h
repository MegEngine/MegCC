#pragma once
#include <string>
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace WebAssembly {


class WebAssemblyKernelFunc : public KernelFunc {

};

class ExpNeonKernel : public InternalKernelFunc {
public:
    std::string GetKernelSymbol(TContext*) const override;

    std::string GetKernelSignature(TContext*) const override;

    std::string GetKernelBody(TContext*) const override;
};



}  // namespace WebAssembly
}  // namespace KernelGen
}  // namespace megcc