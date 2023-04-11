#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

void populateMemoryForwardingPatterns(RewritePatternSet& patterns);

std::unique_ptr<OperationPass<FuncOp>> createMemoryForwardingPass();

std::unique_ptr<OperationPass<FuncOp>> createStaticMemoryPlanningPass();

void populateKernelMaterializationPatterns(RewritePatternSet& patterns);

std::unique_ptr<OperationPass<ModuleOp>> createKernelMaterializationPass();

std::unique_ptr<OperationPass<FuncOp>> createKernelFinalCleanPass();

#define GEN_PASS_REGISTRATION
#include "compiler/Dialect/Kernel/Transforms/Passes.h.inc"

}  // namespace mlir

// vim: syntax=cpp.doxygen
