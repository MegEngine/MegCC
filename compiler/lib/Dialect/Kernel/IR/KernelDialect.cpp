/**
 * \file compiler/lib/Dialect/Kernel/IR/KernelDialect.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Target/MGB/helper.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::Kernel;

KernelDialect::KernelDialect(MLIRContext* context)
        : Dialect(getDialectNamespace(), context,
                  TypeID::get<KernelDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "compiler/Dialect/Kernel/IR/KernelDialect.cpp.inc"
            >();
}

namespace mlir {
namespace Kernel {
bool isContiguous(MemRefType memref) {
    int64_t offset;
    llvm::SmallVector<int64_t> stride;
    if (failed(getStridesAndOffset(memref, stride, offset))) {
        return false;
    }

    auto shape = memref.getShape();

    if (stride.size() != shape.size())
        return false;

    int64_t cur = 1;
    for (int i = stride.size() - 1; i >= 0; --i) {
        auto dim = shape[i];
        if (dim > 1 && cur != stride[i]) {
            return false;
        }
        cur *= shape[i];
    }
    return true;
}
}  // namespace Kernel
}  // namespace mlir

LogicalResult GetWeight::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // TODO
    return success();
}

LogicalResult KernelCall::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // TODO
    return success();
}

MemRefType Reshape::memoryForward(MemRefType inpType) {
    auto oupType = getResult().getType().dyn_cast<ShapedType>();
    if (!oupType) {
        return {};
    }
    ::megdnn::TensorLayout inpLayout;
    int64_t offset = 0;
    if (!MGB::memrefTypeToTensorLayout(inpType, inpLayout, offset)) {
        return {};
    }

    ::megdnn::DType oupDType;
    ::megdnn::TensorShape oupShape;
    if (!MGB::shapedTypeToTensorShape(oupType, oupShape, oupDType)) {
        return {};
    }
    CC_ASSERT(oupDType == inpLayout.dtype);

    ::megdnn::TensorLayout oupLayout;
    if (inpType.getNumDynamicDims() == 0) {
        if (!inpLayout.try_reshape(oupLayout, oupShape)) {
            return {};
            //! if input is dynamic, just set output
        }
    } else {
        oupLayout = inpLayout;
        oupLayout.ndim = oupShape.ndim;
        for (size_t i = 0; i < oupShape.ndim; i++) {
            oupLayout.shape[i] = oupShape[i];
        }
        oupLayout.stride[oupShape.ndim - 1] = 1;
        for (int32_t i = oupShape.ndim - 2; i >= 0; i--) {
            oupLayout.stride[i] =
                    oupLayout.shape[i + 1] * oupLayout.stride[i + 1];
        }
    }
    return MGB::tensorLayoutToMemref(inpType.getContext(), oupLayout, offset);
}

MemRefType Dimshuffle::memoryForward(MemRefType input) {
    llvm::SmallVector<int64_t> stride;
    int64_t offset;
    if (failed(getStridesAndOffset(input, stride, offset))) {
        return {};
    }
    auto pattern =
            llvm::to_vector<4>(this->pattern().getAsRange<IntegerAttr>());
    llvm::SmallVector<int64_t> newStride(pattern.size());
    //! pattern with -1 means add axis
    for (size_t i = 0; i < pattern.size(); ++i) {
        if (pattern[i].getInt() < 0) {
            newStride[i] = i > 0 ? newStride[i - 1]
                                 : newStride[0] * input.getShape()[0];
        } else {
            newStride[i] = stride[pattern[i].getInt()];
        }
    }
    auto output = getResult().getType().dyn_cast<ShapedType>();
    if (!output) {
        return {};
    }
    auto ctx = output.getContext();
    return MemRefType::get(output.getShape(), output.getElementType(),
                           makeStridedLinearLayoutMap(newStride, offset, ctx));
}

MemRefType Subtensor::memoryForward(MemRefType inpType) {
    auto oupType = getResult().getType().dyn_cast<ShapedType>();
    if (!oupType) {
        return {};
    }
    ::megdnn::TensorLayout inpLayout;
    int64_t offset = 0;
    if (!MGB::memrefTypeToTensorLayout(inpType, inpLayout, offset)) {
        return {};
    }
    //! compute the result layout
    auto input_dim = inpLayout.ndim;
    auto dst_layout = inpLayout;
    auto dst_offset = offset;
    auto dtype_size = inpLayout.dtype.size();
    auto descs = llvm::to_vector<4>(this->descs().getAsRange<ArrayAttr>());
    auto flags = llvm::to_vector<4>(this->flags().getAsRange<ArrayAttr>());
    int32_t nr_desc = descs.size();
    std::vector<std::vector<int32_t>> std_desc;
    for (int32_t i = 0; i < nr_desc; i++) {
        auto&& desc = descs[i];
        auto&& flag = flags[i];
        CC_ASSERT(desc.size() == 5 && flag.size() == 5);
        //! assert all desc is const
        auto flag_member = llvm::to_vector<5>(flag.getAsRange<IntegerAttr>());
        for (int j = 0; j < 5; j++) {
            CC_ASSERT(flag_member[j].getInt() != 1);
        }
        int32_t index_flag = static_cast<int32_t>(flag_member[4].getInt());
        int32_t begin_flag = static_cast<int32_t>(flag_member[1].getInt());
        int32_t end_flag = static_cast<int32_t>(flag_member[2].getInt());
        auto member = llvm::to_vector<5>(desc.getAsRange<IntegerAttr>());
        int32_t axis = static_cast<int32_t>(member[0].getInt());
        int32_t begin = static_cast<int32_t>(member[1].getInt());
        int32_t end = static_cast<int32_t>(member[2].getInt());
        int32_t step = static_cast<int32_t>(member[3].getInt());
        int32_t index = static_cast<int32_t>(member[4].getInt());
        int32_t big_end_reverse = 0;

        //! if index is not exist, step < 0, and begin is default value,
        //! set begin to the end of the axis
        if (index_flag == -1 && step < 0) {
            if (begin_flag == -1)
                begin = inpLayout.shape[axis] - 1;
            if (end_flag == -1)
                end = -1;
            big_end_reverse = 1;
        }
        std_desc.push_back({axis, begin, end, step, index, big_end_reverse});
    }
    //! sort in reverse order, so slice would work from low dim to high dim, to
    //! make it contiguous on shape-1 axes
    auto compare = [](const std::vector<int32_t>& v1,
                      const std::vector<int32_t>& v2) {
        auto a0 = v1[0];
        auto a1 = v2[0];
        return (a0 < 0) == (a1 < 0) ? a0 > a1 : a0 < 0;
    };
    std::sort(std_desc.begin(), std_desc.end(), compare);
    //! perform merge
    std::vector<int32_t> axis_to_remove;
    auto merge = [&](const std::vector<int32_t>& desc) {
        int32_t axis = desc[0] < 0 ? desc[0] + input_dim : desc[0];
        int32_t begin = desc[1] < 0 ? desc[1] + inpLayout.shape[axis] : desc[1];
        int32_t end =
                desc[2] < 0 ? desc[2] + inpLayout.shape[axis] + 1 : desc[2];
        auto step = desc[3];
        auto index = desc[4];
        //! reverse big end,
        auto big_end_reverse = desc[5];
        if (big_end_reverse != 0) {
            begin = desc[1];
            end = desc[2];
        }
        if (index != -1) {
            axis_to_remove.push_back(axis);
            begin = index;
            step = 1;
            end = index + step;
        }
        dst_layout.shape[axis] =
                (std::abs(end - begin) + std::abs(step) - 1) / std::abs(step);
        auto origin_stride = dst_layout.stride[axis];
        dst_layout.stride[axis] *= step;

        // make stride as contiguous as possible
        if (dst_layout.shape[axis] != 1 && axis)
            --axis;
        if (dst_layout.shape[axis] == 1) {
            auto stride = dst_layout.stride[axis] =
                    axis + 1 < static_cast<int>(dst_layout.ndim)
                            ? dst_layout.stride[axis + 1] *
                                      dst_layout.shape[axis + 1]
                            : 1;

            for (int i = axis - 1; i >= 0; --i) {
                if (dst_layout.shape[i] == 1) {
                    dst_layout.stride[i] = stride;
                } else {
                    break;
                }
            }
        }
        dst_offset +=
                dst_layout.is_empty() ? 0 : origin_stride * begin * dtype_size;
    };

    for (auto&& desc : std_desc) {
        merge(desc);
    }
    for (auto axis : axis_to_remove) {
        if (dst_layout.ndim > 1) {
            dst_layout.remove_axis_inplace(axis);
        }
    }

    llvm::SmallVector<int64_t> newStride(dst_layout.ndim);
    llvm::SmallVector<int64_t> newShape(dst_layout.ndim);
    for (size_t i = 0; i < dst_layout.ndim; i++) {
        newStride[i] = dst_layout.stride[i];
        newShape[i] = dst_layout.shape[i];
    }
    auto ctx = oupType.getContext();
    return MemRefType::get(
            newShape, oupType.getElementType(),
            makeStridedLinearLayoutMap(newStride, dst_offset, ctx));
}

bool RelayoutKernel::checkInputLayout(MemRefType memref, size_t index) {
    int64_t offset;
    llvm::SmallVector<int64_t> stride;
    if (failed(getStridesAndOffset(memref, stride, offset))) {
        return false;
    }

    // could pass any strided memref to relayout
    return true;
}

#include "compiler/Dialect/Kernel/IR/KernelInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "compiler/Dialect/Kernel/IR/KernelDialect.cpp.inc"

// vim: syntax=cpp.doxygen
