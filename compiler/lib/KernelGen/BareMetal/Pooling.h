#pragma once
#include <sstream>
#include <string>
#include "Common/PoolingKernel.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class PoolingKernel : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
