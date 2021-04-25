/**
 * \file compiler/include/compiler/Dialect/Kernel/Transforms/Passes.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
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
