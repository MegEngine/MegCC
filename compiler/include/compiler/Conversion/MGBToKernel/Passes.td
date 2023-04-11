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
