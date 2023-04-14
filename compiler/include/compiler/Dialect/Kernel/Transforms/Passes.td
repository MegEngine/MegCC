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

#endif // KERNEL_TRANSFORM
