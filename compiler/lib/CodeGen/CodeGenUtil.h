#pragma once
#include <sstream>
#include <vector>
#include "compiler/CodeGen/CodeGen.h"
#include "compiler/Common/Logger.h"
#include "compiler/Common/MemoryStatus.h"
#include "compiler/Common/MlirUtils.h"
#include "compiler/Common/TContext.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"

static inline mlir::Type dtype_to_type(
        mlir::MLIRContext* context, const std::string& dtype) {
    if ("f32" == dtype) {
        return mlir::FloatType::getF32(context);
    } else {
        CC_ABORT << "Unsupport dtype " << dtype << "\n";
    }
    return mlir::Type();
}

static inline std::string print_mlir_module(mlir::OwningOpRef<mlir::ModuleOp>& op) {
    std::string type_name;
    llvm::raw_string_ostream raw_os(type_name);
    op->print(raw_os);
    return type_name;
}

static inline mlir::ShapedType operand_to_shaped_type(
        megcc::TContext* op_ctx, int idx, mlir::MLIRContext* mlir_ctx) {
    auto tensor = op_ctx->getAttrOprand("operand:" + std::to_string(idx));
    std::vector<int64_t> dims(tensor.shape.begin(), tensor.shape.end());

    return mlir::RankedTensorType::get(dims, dtype_to_type(mlir_ctx, tensor.dtype));
}

static inline mlir::MemRefType operand_to_memref_type(
        megcc::TContext* op_ctx, int idx, mlir::MLIRContext* mlir_ctx) {
    auto tensor = op_ctx->getAttrOprand("operand:" + std::to_string(idx));
    std::vector<int64_t> dims(tensor.shape.begin(), tensor.shape.end());
    std::vector<int64_t> stride;
    stride.resize(dims.size());
    size_t offset = 0;
    size_t acc = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
        stride[i] = acc;
        acc *= dims[i];
    }
    megcc::MemoryStatus status;
    return mlir::MemRefType::get(
            dims, dtype_to_type(mlir_ctx, tensor.dtype),
            makeStridedLinearLayoutMap(stride, offset, mlir_ctx),
            static_cast<uint64_t>(status));
}

static inline mlir::MemRefType operand_to_dynamic_memref_type(
        megcc::TContext* op_ctx, int idx, mlir::MLIRContext* mlir_ctx) {
    auto tensor = op_ctx->getAttrOprand("operand:" + std::to_string(idx));
    size_t nr_dims = tensor.shape.size();
    std::vector<int64_t> dims;
    for (size_t i = 0; i < nr_dims; ++i) {
        dims.push_back(-1);
    }
    return mlir::MemRefType::get(dims, dtype_to_type(mlir_ctx, tensor.dtype));
}

static inline mlir::FunctionType get_func_type(
        megcc::TContext* op_ctx, mlir::MLIRContext* mlir_ctx, int result_start_idx) {
    std::vector<mlir::Type> input_type_vec;
    std::vector<mlir::Type> output_type_vec;
    int nr_operand = op_ctx->getAttrInt("nr_operands");
    for (int i = 0; i < result_start_idx; ++i)
        input_type_vec.push_back(operand_to_shaped_type(op_ctx, i, mlir_ctx));
    for (int i = result_start_idx; i < nr_operand; ++i)
        output_type_vec.push_back(operand_to_shaped_type(op_ctx, i, mlir_ctx));
    return mlir::FunctionType::get(mlir_ctx, input_type_vec, output_type_vec);
}

static inline mlir::FunctionType get_func_type_memref(
        megcc::TContext* op_ctx, mlir::MLIRContext* mlir_ctx) {
    std::vector<mlir::Type> input_type_vec;
    int nr_operand = op_ctx->getAttrInt("nr_operands");
    for (int i = 0; i < nr_operand; ++i)
        input_type_vec.push_back(operand_to_dynamic_memref_type(op_ctx, i, mlir_ctx));

    return mlir::FunctionType::get(mlir_ctx, input_type_vec, {});
}

static inline mlir::AffineMap get_affine_map(
        mlir::MemRefType type, mlir::MLIRContext* mlir_ctx) {
    mlir::SmallVector<mlir::AffineExpr, 4> affineExprs;
    for (auto it : llvm::enumerate(type.getShape())) {
        affineExprs.push_back(mlir::getAffineDimExpr(it.index(), mlir_ctx));
    }
    return mlir::AffineMap::get(
            /*dimCount=*/type.getRank(), /*symbolCount=*/0, affineExprs, mlir_ctx);
}

static inline mlir::SmallVector<mlir::StringRef> getNParallelLoopsAttrs(
        unsigned nParallelLoops) {
    return mlir::SmallVector<mlir::StringRef>(
            nParallelLoops, mlir::getParallelIteratorTypeName());
}