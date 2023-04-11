#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "compiler/CodeGen/CodeGen.h"

namespace megcc {
namespace codegen {
struct AutoKernelFuncInternal : AutoKernelFunc {
    virtual void CreateCompute(
            mlir::Block* entryBlock, mlir::OpBuilder& op_builder,
            mlir::MLIRContext* ctx, TContext* context) const = 0;
    virtual void CreatePass(
            mlir::PassManager& pm, mlir::MLIRContext* ctx, TContext* context) const = 0;

    virtual KernelGen::KernelObj GetKernelObj(TContext* context) const override;

    KernelGen::KernelObj CompileKernel(mlir::ModuleOp mod, std::string func_name) const;
};

}  // namespace codegen
}  // namespace megcc
