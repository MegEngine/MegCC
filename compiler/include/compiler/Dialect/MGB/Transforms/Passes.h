#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createMGBFuseKernelPass();

#define GEN_PASS_REGISTRATION
#include "compiler/Dialect/MGB/Transforms/Passes.h.inc"

}  // namespace mlir

// vim: syntax=cpp.doxygen
