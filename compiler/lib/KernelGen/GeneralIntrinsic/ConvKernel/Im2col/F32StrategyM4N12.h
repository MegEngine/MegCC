#pragma once
#include <string>
#include "../Im2colCommon.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class F32StrategyM4N12 : public Im2colStrategyBase {
public:
    // matmul call
    virtual KernelGen::InternalKernelFunc* GetInnerCtxMatmul(TContext* ctx) override {
        return GetInnerMatmul(ctx);
    };

    virtual std::string GetInnerCtxMatmulSym(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetNakedKernelSymbol(ctx);
    };

    virtual std::string PackBSym(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackBSymbol(ctx);
    };

    virtual std::string PackASym(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackASymbol(ctx);
    };

    virtual std::string GetPackAWorkspaceSym(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackAWorkspaceSymbol(ctx);
    };

    virtual std::string GetPackBWorkspaceSym(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackBWorkspaceSymbol(ctx);
    };

    virtual std::string GetPackASignature(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackASignature(ctx);
    };

    virtual std::string GetPackAWorkspaceSignature(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackAWorkspaceSignature(ctx);
    };

    virtual std::string GetPackBSignature(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackBSignature(ctx);
    };

    virtual std::string GetPackBWorkspaceSignature(TContext* ctx) override {
        return GetInnerMatmul(ctx)->GetPackBWorkspaceSignature(ctx);
    };

private:
    GIMatmulInternal* GetInnerMatmul(TContext* ctx);
    MatmulM4N12MK4Kernel m_inner_mk4_gemm;
    MatmulM4N12Kernel m_inner_gemm;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen