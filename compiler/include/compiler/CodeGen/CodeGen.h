#pragma once

#include <sstream>
#include <vector>
#include "compiler/Common/Logger.h"
#include "compiler/Common/TContext.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace codegen {
struct AutoKernelFunc {
    virtual ~AutoKernelFunc(){};
    virtual bool IsAvailable(TContext* context) const = 0;
    virtual std::string GetKernelSymbol(TContext* context) const = 0;
    virtual KernelGen::KernelObj GetKernelObj(TContext* context) const = 0;
};

AutoKernelFunc* GenCode(KernelGen::KernelPack::KernType kern_type);

}  // namespace codegen
}  // namespace megcc
