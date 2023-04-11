#pragma once
#include "MatMulCommon.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class Fp32MatMulM4N12 : public GIKernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
};

class Fp32MatMulM4N8K4 : public GIKernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
};

class Fp32GevmKernel : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
};

class Fp32GemvKernel : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class Fp32GemvMk4Kernel : public KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
};

class Fp32MatMulM4N12K4 : public GIKernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, false);
    }
    std::string GetWorkspaceBodyAndJitExec(TContext* ctx) const override {
        return GetWorkspaceBodyCondition(ctx, true);
    }

private:
    std::string GetWorkspaceBodyCondition(TContext* ctx, bool jit) const;
    std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) const;
};
}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
