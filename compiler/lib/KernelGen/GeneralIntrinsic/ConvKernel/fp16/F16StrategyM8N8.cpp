#include "F16StrategyM8N8.h"
#include <string>
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
KernelGen::InternalKernelFunc* F16StrategyM8N8::GetInnerCtxMatmul(TContext*) {
    return &m_inner_gemm;
}

std::string F16StrategyM8N8::GetInnerCtxMatmulSym(TContext* ctx) {
    return m_inner_gemm.GetKernelSymbol(ctx) + "_naked";
}