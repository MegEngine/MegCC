#pragma once
#include "mlir/IR/BuiltinOps.h"
namespace megcc {
namespace codegen {

mlir::MLIRContext* getGlobalCTX();
}
}  // namespace megcc