/**
 * \file compiler/include/compiler/Target/MGB/helper.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
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

namespace mlir {
namespace MGB {

static inline ::megdnn::DType type_to_dtype(Type type) {
    if (auto floatType = type.dyn_cast<FloatType>()) {
        if (floatType.getWidth() == 32) {
            return ::megdnn::dtype::Float32();
        }
    } else if (auto intType = type.dyn_cast<IntegerType>()) {
        if (intType.isQuant()) {
            CC_ASSERT(intType.isSigned());
            if (intType.getWidth() == 32) {
                return ::megdnn::dtype::QuantizedS32(intType.getScale());
            } else {
                CC_ASSERT(intType.getWidth() == 8);
                return ::megdnn::dtype::QuantizedS8(intType.getScale());
            }
        } else {
            if (intType.getWidth() == 32 && intType.isSigned()) {
                return ::megdnn::dtype::Int32();
            } else if (intType.getWidth() == 8) {
                if (intType.isUnsigned()) {
                    return ::megdnn::dtype::Uint8();
                } else {
                    CC_ASSERT(intType.isSignless() || intType.isSigned());
                    return ::megdnn::dtype::Int8();
                }
            }
        }
    } else {
        std::string type_name;
        llvm::raw_string_ostream raw_os(type_name);
        type.print(raw_os);
        CC_ABORT << "Unsupport type : " << type_name << "\n";
    }
    return {};
}

static inline Type dtype_to_type(MLIRContext* context, const ::megdnn::DType& dtype) {
    switch (dtype.enumv()) {
        case ::megdnn::DTypeEnum::Float32:
            return FloatType::getF32(context);
        case ::megdnn::DTypeEnum::Int32:
            return IntegerType::get(context, 32, IntegerType::Signed);
        case ::megdnn::DTypeEnum::Uint8:
            return IntegerType::get(context, 8, IntegerType::Unsigned);
        case ::megdnn::DTypeEnum::Int8:
            return IntegerType::get(context, 8, IntegerType::Signed);
        case ::megdnn::DTypeEnum::QuantizedS8: {
            constexpr int bit_width = 8;
            auto dtype_param = dtype.param<::megdnn::dtype::QuantizedS8>();
            float scale = dtype_param.scale;
            auto dtype =
                    IntegerType::get(context, bit_width, IntegerType::Signed, scale);
            return dtype;
        };
        case ::megdnn::DTypeEnum::QuantizedS32: {
            constexpr int bit_width = 32;
            auto dtype_param = dtype.param<::megdnn::dtype::QuantizedS32>();
            float scale = dtype_param.scale;
            auto dtype =
                    IntegerType::get(context, bit_width, IntegerType::Signed, scale);
            return dtype;
        };
        case ::megdnn::DTypeEnum::Float16:
            return FloatType::getF16(context);

        default:
            CC_ABORT << "Unsupport dtype " << dtype.name() << "\n";
            break;
    }
    return Type();
}

static inline bool shapedTypeToTensorShape(
        ShapedType shape, ::megdnn::TensorShape& tensorShape, ::megdnn::DType& dtype) {
    if (!shape.hasRank()) {
        return false;
    }

    dtype = type_to_dtype(shape.getElementType());
    if (!dtype.valid()) {
        return false;
    }

    tensorShape.ndim = shape.getRank();
    for (size_t i = 0; i < tensorShape.ndim; ++i)
        tensorShape[i] = shape.getDimSize(i);

    return true;
}

static inline bool memrefTypeToTensorLayout(
        MemRefType memref, ::megdnn::TensorLayout& tensorLayout, int64_t& offset) {
    if (!shapedTypeToTensorShape(memref, tensorLayout, tensorLayout.dtype)) {
        return false;
    }

    llvm::SmallVector<int64_t> stride;
    if (failed(getStridesAndOffset(memref, stride, offset))) {
        return false;
    }

    CC_ASSERT(stride.size() == tensorLayout.ndim);
    for (size_t i = 0; i < tensorLayout.ndim; ++i) {
        tensorLayout.stride[i] = stride[i];
    }

    return true;
}

static inline ShapedType tensorShapeToShapedType(
        MLIRContext* context, const ::megdnn::TensorShape& shape,
        ::megdnn::DType dtype) {
    std::vector<int64_t> dims(shape.shape, shape.shape + shape.ndim);
    LOG_DEBUG << "Create RankedTensorType with shape= " << dims << "\n";
    ShapedType res;
    if (dims.size() > 0) {
        res = RankedTensorType::get(dims, dtype_to_type(context, dtype));
    } else {
        LOG_WARN << "Shape is unknow, compiler just make 1 dim dynamic tensor "
                    "type\n";
        res = RankedTensorType::get({-1}, dtype_to_type(context, dtype));
    }
    return res;
}

static inline MemRefType tensorLayoutToMemref(
        MLIRContext* context, const ::megdnn::TensorLayout& layout, size_t offset = 0,
        megcc::MemoryStatus status = {}) {
    std::vector<int64_t> shape(layout.shape, layout.shape + layout.ndim);
    std::vector<int64_t> stride(layout.stride, layout.stride + layout.ndim);
    return MemRefType::get(
            shape, dtype_to_type(context, layout.dtype),
            makeStridedLinearLayoutMap(stride, offset, context),
            static_cast<uint64_t>(status));
}

}  // namespace MGB
}  // namespace mlir

// vim: syntax=cpp.doxygen
