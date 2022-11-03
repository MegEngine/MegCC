/**
 * \file compiler/lib/Conversion/MGBToKernel/MGBToKernelHelper.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#pragma once

#include "compiler/Common/Logger.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "llvm/ADT/StringRef.h"
#include "megbrain/reflection.h"
#include "megdnn/opr_param_defs.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GetParamRename(name_, new_name_)                      \
    if (direct_attr.get(name_)) {                             \
        attrs.push_back({StringAttr::get(context, new_name_), \
                         direct_attr.get(name_)});            \
    }

#define GetParam(name_)                                                     \
    if (direct_attr.get(name_)) {                                           \
        attrs.push_back(                                                    \
                {StringAttr::get(context, name_), direct_attr.get(name_)}); \
    }

#define GetParamEnum(type_, name_)                                       \
    if (direct_attr.get(name_)) {                                        \
        auto attr = mgb::reflection::nameOfEnumValue(static_cast<type_>( \
                direct_attr.getAs<mlir::IntegerAttr>(name_).getInt()));  \
        attrs.push_back({StringAttr::get(context, name_),                \
                         StringAttr::get(context, attr)});               \
    }

SmallVector<NamedAttribute, 4> ConvertConvLikeAttr(DictionaryAttr direct_attr,
                                               Value weight,
                                               MLIRContext* context) {
    using Mode = ::megdnn::param::Convolution::Mode;
    using Sparse = ::megdnn::param::Convolution::Sparse;
    using Format = ::megdnn::param::Convolution::Format;
    using ComputeMode = ::megdnn::param::Convolution::ComputeMode;
    using NonlineMode = ::megdnn::param::ConvBias::NonlineMode;

    auto GetKernelSize = [](const Format format, Value weight, int& kh, int& kw,
                            const Sparse sparse) {
        //! TODO: add more format
        auto type = weight.getType().cast<ShapedType>();
        auto shape = type.getShape();
        auto dims = shape.size();
        switch (format) {
            case Format::NCHW:
                kh = shape[dims - 2];
                kw = shape[dims - 1];
                return;
            case Format::NCHW44_DOT:
            case Format::NCHW44:
                CC_ASSERT(dims > 4);
                if (sparse == Sparse::GROUP) {
                    if (dims == 7) {
                        //! group conv with weight layout is g, ocpg/4, icpg/4,
                        //! fh, fw, 4, 4
                        kh = shape[dims - 4];
                        kw = shape[dims - 3];
                    } else {
                        //! channel wise weight layout is g/4, 1, 1, fh, fw, 4
                        kh = shape[dims - 3];
                        kw = shape[dims - 2];
                    }
                } else {
                    //! dense layout is oc/4, ic/4, fh, fw, 4, 4
                    //! hybrid first layout oc/4, fh, fw, ic, 4
                    kh = shape[dims - 4];
                    kw = shape[dims - 3];
                }
                return;
            default:
                llvm::errs() << "Unsupport Format " << (int)format << "\n";
                abort();
        }
    };

    SmallVector<NamedAttribute, 4> attrs;
    GetParam("stride_h");
    GetParam("stride_w");
    GetParam("pad_h");
    GetParam("pad_w");
    GetParam("dilate_h");
    GetParam("dilate_w");
    GetParam("strategy");
    GetParam("workspace_limit");

    GetParamEnum(Mode, "mode");
    GetParamEnum(ComputeMode, "compute_mode");
    GetParamEnum(Format, "format");
    GetParamEnum(Sparse, "sparse");
    GetParamEnum(NonlineMode, "nonlineMode");

    //! kernel size
    int kh, kw;
    Format format = static_cast<Format>(
            direct_attr.getAs<mlir::IntegerAttr>("format").getInt());
    Sparse sparse = static_cast<Sparse>(
            direct_attr.getAs<mlir::IntegerAttr>("sparse").getInt());
    GetKernelSize(format, weight, kh, kw, sparse);
    auto khAttr =
            IntegerAttr::get(IntegerType::get(context, 32), APInt(32, kh));
    auto kwAttr =
            IntegerAttr::get(IntegerType::get(context, 32), APInt(32, kw));
    attrs.push_back({StringAttr::get(context, "kernel_h"), khAttr});
    attrs.push_back({StringAttr::get(context, "kernel_w"), kwAttr});

    return attrs;
}

template <class Opr>
SmallVector<NamedAttribute, 4> ConvertAttr(DictionaryAttr direct_attr,
                                           MLIRContext* context);

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Pooling>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    using Mode = ::megdnn::param::Pooling::Mode;
    using Format = ::megdnn::param::Pooling::Format;

    SmallVector<NamedAttribute, 4> attrs;
    GetParam("stride_h");
    GetParam("stride_w");
    GetParam("pad_h");
    GetParam("pad_w");
    GetParam("window_h");
    GetParam("window_w");

    GetParamEnum(Mode, "mode");
    GetParamEnum(Format, "format");

    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::MatrixInverse>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Argsort>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    using Order = ::megdnn::param::Argsort::Order;
    GetParamEnum(Order, "order");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::TopK>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    using Mode = ::megdnn::param::TopK::Mode;
    GetParamEnum(Mode, "mode");
    GetParam("k");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Argmax>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("axis");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::IndexingMultiAxisVec>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("axis");
    return attrs;
}
template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::IndexingOneHot>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("axis");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::WarpAffine>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("border_val");

    using IMode = ::megdnn::param::WarpAffine::InterpolationMode;
    using BMode = ::megdnn::param::WarpAffine::BorderMode;
    using Format = ::megdnn::param::WarpAffine::Format;

    GetParamEnum(IMode, "imode");
    GetParamEnum(BMode, "border_mode");
    GetParamEnum(Format, "format");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Resize>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;

    using IMode = ::megdnn::param::Resize::InterpolationMode;
    using Format = ::megdnn::param::Resize::Format;

    GetParamEnum(IMode, "imode");
    GetParamEnum(Format, "format");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Elemwise>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    using Mode = ::megdnn::param::Elemwise::Mode;

    SmallVector<NamedAttribute, 4> attrs;
    GetParamEnum(Mode, "mode");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::ElemwiseMultiType>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    using Mode = ::megdnn::param::ElemwiseMultiType::Mode;

    SmallVector<NamedAttribute, 4> attrs;
    GetParamEnum(Mode, "mode");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::PowC>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("exp");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::MatrixMul>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    using ComputeMode = ::megdnn::param::MatrixMul::ComputeMode;
    using Format = ::megdnn::param::MatrixMul::Format;

    SmallVector<NamedAttribute, 4> attrs;
    GetParam("transposeA");
    GetParam("transposeB");

    GetParamEnum(ComputeMode, "compute_mode");
    GetParamEnum(Format, "format");

    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Reduce>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    using Mode = ::megdnn::param::Reduce::Mode;
    using DataType = ::megdnn::param::Reduce::DataType;

    SmallVector<NamedAttribute, 4> attrs;
    GetParam("axis");

    GetParamEnum(Mode, "mode");
    GetParamEnum(DataType, "data_type");

    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::BatchedMatrixMul>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    return ConvertAttr<MGB::MatrixMul>(direct_attr, context);
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::TypeCvt>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("i_scale");
    GetParam("o_scale");
    GetParam("i_zero");
    GetParam("o_zero");

    CC_ASSERT(direct_attr.get("i_dtype") && direct_attr.get("o_dtype"));

    auto idtype = direct_attr.getAs<mlir::TypeAttr>("i_dtype");
    auto odtype = direct_attr.getAs<mlir::TypeAttr>("o_dtype");

    std::string idtype_s, odtype_s;
    llvm::raw_string_ostream raw_os_idtype(idtype_s);
    llvm::raw_string_ostream raw_os_odtype(odtype_s);
    idtype.print(raw_os_idtype);
    odtype.print(raw_os_odtype);

    attrs.push_back({StringAttr::get(context, "i_dtype"),
                     StringAttr::get(context, idtype_s)});
    attrs.push_back({StringAttr::get(context, "o_dtype"),
                     StringAttr::get(context, odtype_s)});
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::WarpPerspective>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("mat_idx");
    GetParam("border_val");

    using IMode = ::megdnn::param::WarpPerspective::InterpolationMode;
    using BMode = ::megdnn::param::WarpPerspective::BorderMode;
    using Format = ::megdnn::param::WarpPerspective::Format;

    GetParamEnum(IMode, "imode");
    GetParamEnum(BMode, "bmode");
    GetParamEnum(Format, "format");

    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Subtensor>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("descs");
    GetParam("flags");
    auto descs = direct_attr.getAs<mlir::ArrayAttr>("descs");
    auto flags = direct_attr.getAs<mlir::ArrayAttr>("flags");
    CC_ASSERT(descs.size() == flags.size())
            << "subtensor with different size between desc and flag\n";
    for (size_t i = 0; i < descs.size(); i++) {
        auto desc = descs[i];
        auto flag = flags[i];
        auto array_desc = desc.dyn_cast<mlir::ArrayAttr>();
        auto array_flag = flag.dyn_cast<mlir::ArrayAttr>();
        //! every desc contain: axis, start, end, step, index 5 elements
        CC_ASSERT(array_desc && array_desc.size() == 5)
                << "Subtensor desc is not array of 5 elements!\n";
        CC_ASSERT(array_flag && array_flag.size() == 5)
                << "Subtensor flag is not array of 5 elements!\n";
    }
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::SetSubtensor>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    return ConvertAttr<MGB::Subtensor>(direct_attr, context);
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::GetVarShape>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("axis");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Concat>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("axis");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Dimshuffle>(
        DictionaryAttr direct_attr, MLIRContext* context) {
    SmallVector<NamedAttribute, 4> attrs;
    GetParam("pattern");
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Broadcast>(DictionaryAttr,
                                                           MLIRContext*) {
    SmallVector<NamedAttribute, 4> attrs;
    return attrs;
}

template <>
SmallVector<NamedAttribute, 4> ConvertAttr<MGB::Reshape>(DictionaryAttr,
                                                         MLIRContext*) {
    SmallVector<NamedAttribute, 4> attrs;
    return attrs;
}

ArrayAttr MakeConcatArrayAttr(size_t axis, size_t& start, Value input,
                              Value output,
                              ConversionPatternRewriter& rewriter) {
    auto inputType = input.getType().dyn_cast<ShapedType>();
    auto ouputType = output.getType().dyn_cast<ShapedType>();
    auto axis_size = inputType.getDimSize(axis);
    auto axis_end = ouputType.getDimSize(axis);
    //! axis, start, end, step, index
    int32_t axis_ = static_cast<int32_t>(axis);
    int32_t start_ = static_cast<int32_t>(start);
    int32_t end_ = start + axis_size;
    CC_ASSERT(end_ <= axis_end) << "Concat with wrong shape\n";
    int32_t step_ = 1;
    int32_t index_ = -1;
    start += axis_size;
    SmallVector<int32_t> desc{axis_, start_, end_, step_, index_};
    SmallVector<Attribute> array_attrs{rewriter.getI32ArrayAttr(desc)};
    return rewriter.getArrayAttr(array_attrs);
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
