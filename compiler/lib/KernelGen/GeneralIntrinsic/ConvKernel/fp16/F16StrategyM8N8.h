#pragma once
#include <string>
#include "../Im2colCommon.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class F16StrategyM8N8 : public Im2colStrategyBase {
public:
    // matmul call
    virtual KernelGen::InternalKernelFunc* GetInnerCtxMatmul(TContext* ctx) override;
    virtual std::string GetInnerCtxMatmulSym(TContext* ctx) override;

private:
    Fp16MatmulM8N8MK8Kernel m_inner_gemm;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen