/**
 * \file compiler/include/compiler/Dialect/Kernel/Transforms/Passes.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef KERNEL_TRANSFORM
#define KERNEL_TRANSFORM

include "mlir/Pass/PassBase.td"

def MemoryForwardingPass : Pass<"memory-forwarding", "FuncOp"> {
  let summary = "perform readonly memory fowarding on reshape-like op, memref stride should be determined after this pass";
  let dependentDialects = ["Kernel::KernelDialect"];
  let constructor = "mlir::createMemoryForwardingPass()";
}

def StaticMemoryPlanningPass : Pass<"static-memory-planning", "FuncOp"> {
  let summary = "make static memory planning for memref with constant shape to avoid alloc/free on each execution";
  let dependentDialects = ["Kernel::KernelDialect"];
  let constructor = "mlir::createStaticMemoryPlanningPass()";
}

def KernelMaterializationPass : Pass<"kernel-materialization", "ModuleOp"> {
  let summary = "materialize abstract kernels to actual kernel definitions and kernel calls";
  let dependentDialects = ["Kernel::KernelDialect"];
  let constructor = "mlir::createKernelMaterializationPass()";
}

def KernelFinalCleanPass : Pass<"kernel-clean", "FuncOp"> {
  let summary = "clean unused or forward kernel opr";
  let dependentDialects = ["Kernel::KernelDialect"];
  let constructor = "mlir::createKernelFinalCleanPass()";
}

#endif // KERNEL_TRANSFORM
