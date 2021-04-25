/**
 * \file compiler/include/compiler/Dialect/MGB/Transforms/Passes.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_TRANSFORM
#define MGB_TRANSFORM

include "mlir/Pass/PassBase.td"

def MGBFuseKernelPass : Pass<"mgb-fuse-kernel", "FuncOp"> {
  let summary = "fuse mgb kernel to remove redundant opr";
  let dependentDialects = ["MGB::MGBDialect"];
  let constructor = "mlir::createMGBFuseKernelPass()";
}

#endif // MGB_TRANSFORM
