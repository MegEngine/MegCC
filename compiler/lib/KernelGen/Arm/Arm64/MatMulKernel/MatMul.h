#pragma once
#include <sstream>
#include <string>
#include "Arm/Arm64/KernelCommon.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace Arm64 {

class Fp32MatMulM4N16K4 : public Arm64KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
};

class Fp32MatMulM8N12 : public Arm64KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::string GetWorkspaceBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
};

class Fp32MatMulM8N12K4 : public Arm64KernelFunc {
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

class Int8DotMatMulM8N12K4 : public Arm64KernelFunc {
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

class Fp16MatMulM8N8K8 : public Arm64KernelFunc {
public:
    bool IsAvailable(TContext* context) const override;
    std::string GetKernelSymbol(TContext* context) const override;
    std::string GetKernelBody(TContext* context) const override;
    std::vector<KernelObj> GetDependInternalSymbol(TContext* context) const override;
};
}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
