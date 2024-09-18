#pragma once
#include <sstream>
#include <string>
#include "WebAssembly/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace WebAssembly {

class Fp32MatMulM4N12 : public WebAssemblyKernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
private:
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
};

}  // namespace WebAssembly
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
