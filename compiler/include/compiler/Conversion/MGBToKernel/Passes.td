/**
 * \file compiler/include/compiler/Conversion/MGBToKernel/Passes.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MLIR_MGB_TO_KERNEL
#define MLIR_MGB_TO_KERNEL

include "mlir/Pass/PassBase.td"

def MGBToKernelPass : Pass<"MGB-to-Kernel", "ModuleOp"> {
  let summary = "lowering mgb computing graph to the graph of abstract kernels";
  let dependentDialects = [
      "MGB::MGBDialect",
      "Kernel::KernelDialect",
      "memref::MemRefDialect"];
  let constructor = "mlir::createMGBToKernelPass()";
}

#endif // MLIR_MGB_TO_KERNEL
