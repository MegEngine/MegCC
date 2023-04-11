#pragma once

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"

#include "megbrain/comp_node.h"
#include "megbrain/opr/param_defs.h"
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"

namespace mlir {
namespace MGB {

class MGBDialect : public Dialect {
public:
    explicit MGBDialect(MLIRContext* context);
    static StringRef getDialectNamespace() { return "MGB"; }
};

}  // namespace MGB
}  // namespace mlir

#define GET_OP_CLASSES

#include "compiler/Dialect/MGB/IR/MGBDialect.h.inc"
#include "compiler/Dialect/MGB/IR/MGBDialect_attr.h.inc"
#include "compiler/Dialect/MGB/IR/MGBDialect_type.h.inc"

// vim: syntax=cpp.doxygen
