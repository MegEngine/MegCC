#pragma once
#include <sstream>
#include <string>
#include "Common/PoolingKernel.h"
#include "Utils/SymbolHelper.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class PoolingNchw44Fp32 : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
};

class PoolingNchw44QInt8 : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
};

class PoolingNchw88Fp16 : public PoolingImpl {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
};
}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc