#ifndef MGB_TRANSFORM
#define MGB_TRANSFORM

include "mlir/Pass/PassBase.td"

def MGBFuseKernelPass : Pass<"mgb-fuse-kernel", "FuncOp"> {
  let summary = "fuse mgb kernel to remove redundant opr";
  let dependentDialects = ["MGB::MGBDialect"];
  let constructor = "mlir::createMGBFuseKernelPass()";
}

#endif // MGB_TRANSFORM
