#pragma once
#include <string>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
namespace megcc {
static inline std::string print_mlir_opr(mlir::Operation* op) {
    std::string type_name;
    llvm::raw_string_ostream raw_os(type_name);
    op->print(raw_os);
    return type_name;
}
template <typename T>
static inline std::string print_mlir(T& op) {
    std::string type_name;
    llvm::raw_string_ostream raw_os(type_name);
    op.print(raw_os);
    return type_name;
}
}  // namespace megcc