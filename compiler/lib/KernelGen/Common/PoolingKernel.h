#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
class PoolingImpl : public KernelFunc {
public:
    std::string GetKernelSymbol(TContext* context) const override;
};

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
