/**
 * \file compiler/lib/Dialect/MGB/IR/MGBDialect.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "compiler/Dialect/MGB/IR/MGBDialect.h"

namespace mlir {
namespace MGB {

MGBDialect::MGBDialect(MLIRContext* context)
        : Dialect(getDialectNamespace(), context, TypeID::get<MGBDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "compiler/Dialect/MGB/IR/MGBDialect.cpp.inc"
            >();
    allowUnknownTypes();
}

LogicalResult ParamProvider::verifySymbolUses(
        SymbolTableCollection& symbolTable) {
    // TODO
    return success();
}

}  // namespace MGB
}  // namespace mlir

#define GET_OP_CLASSES
#include "compiler/Dialect/MGB/IR/MGBDialect.cpp.inc"

// vim: syntax=cpp.doxygen
