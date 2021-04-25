/**
 * \file compiler/include/compiler/Dialect/MGB/Transforms/Passes.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createMGBFuseKernelPass();

#define GEN_PASS_REGISTRATION
#include "compiler/Dialect/MGB/Transforms/Passes.h.inc"

}  // namespace mlir

// vim: syntax=cpp.doxygen
