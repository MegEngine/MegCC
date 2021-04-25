/**
 * \file compiler/include/compiler/Common/MlirUtils.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
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