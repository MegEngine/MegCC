#pragma once

#include <map>
#include <vector>

#include "compiler/Common/Logger.h"
#include "compiler/Common/MemoryStatus.h"

#include "megdnn/basic_types.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

#include "compiler/Dialect/MGB/IR/MGBDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "onnx/common/ir.h"

namespace mlir {
namespace ONNX {
static inline mlir::Type elemTypeToType(
        mlir::MLIRContext* context, const int32_t& elem_type) {
    switch (elem_type) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            return mlir::FloatType::getF32(context);
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
            return mlir::IntegerType::get(context, 8, mlir::IntegerType::Unsigned);
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
            return mlir::IntegerType::get(context, 32, mlir::IntegerType::Signed);
        default:
            CC_ABORT << "Unsupported dtype " << elem_type << "\n";
            break;
    }
    return mlir::Type();
}

static inline mlir::ShapedType valueToShapedType(
        mlir::MLIRContext* context, ONNX_NAMESPACE::Value* value) {
    std::vector<int64_t> dims;
    for (auto dim : value->sizes()) {
        dims.emplace_back(dim.dim);
    }
    LOG_DEBUG << "Create RankedTensorType in Value with shape= " << dims << "\n";
    mlir::ShapedType res;
    if (dims.size() > 0) {
        res = mlir::RankedTensorType::get(
                dims, elemTypeToType(context, value->elemType()));
    } else {
        LOG_WARN << "Shape is unknown, compiler just make 1 dim dynamic tensor "
                    "type\n";
        res = mlir::RankedTensorType::get(
                {-1}, elemTypeToType(context, value->elemType()));
    }
    return res;
}

}  // namespace ONNX
}  // namespace mlir