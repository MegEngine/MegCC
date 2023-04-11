#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class TopkKernel : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override;
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBody(ctx);
    }

private:
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc