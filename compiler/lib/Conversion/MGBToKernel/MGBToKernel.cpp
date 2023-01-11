/**
 * \file compiler/lib/Conversion/MGBToKernel/MGBToKernel.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"

#include "compiler/Conversion/MGBToKernel/MGBToKernel.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Target/MGB/helper.h"
#include "megbrain/reflection.h"

#include "./MGBToKernelHelper.h"

using namespace mlir::MGB;

namespace mlir {
namespace {
#define GEN_PASS_CLASSES
#include "compiler/Conversion/MGBToKernel/Passes.h.inc"

class MGBToKernelPass final : public MGBToKernelPassBase<MGBToKernelPass> {
    void runOnOperation() override {
        bufferization::BufferizeTypeConverter typeConverter;
        RewritePatternSet patterns(&getContext());
        ConversionTarget target(getContext());

        populateMGBToKernelConversionPatterns(typeConverter, patterns);
        populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                                 typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        target.addLegalDialect<Kernel::KernelDialect, memref::MemRefDialect>();
        target.addIllegalDialect<MGB::MGBDialect>();
        auto isMemRefType = [](Type type) {
            return type.isa<BaseMemRefType>();
        };
        target.addLegalOp<ModuleOp>();
        target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
            auto inputs = op.type().dyn_cast<FunctionType>().getInputs();
            return llvm::all_of(inputs, isMemRefType);
        });
        target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp op) {
            return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                               isMemRefType);
        });

        auto op = getOperation();
        if (failed(applyFullConversion(op, target, std::move(patterns))))
            signalPassFailure();
    }
};

bool isDynamicShape(ValueRange operands) {
    bool is_dynamic_shape = false;
    for (size_t i = 0; i < operands.size(); i++) {
        if (auto shapedType = operands[i].getType().dyn_cast<ShapedType>()) {
            is_dynamic_shape |= shapedType.getNumDynamicDims() > 0;
        }
    }
    return is_dynamic_shape;
}

MemRefType asMemRef(Type type) {
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
        return MemRefType::get(shapedType.getShape(),
                               shapedType.getElementType());
    }
    return {};
}

Value asBuffer(Value value, ConversionPatternRewriter& rewriter) {
    if (auto memref = asMemRef(value.getType())) {
        if (value.getType().dyn_cast<ShapedType>().getNumDynamicDims() == 0) {
            return rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(),
                                                    memref);
        } else {
            LOG_WARN << "Value shape is dynamic, compiler just create place "
                        "holder of DynamicAlloc.\n";
            return rewriter.create<Kernel::DynamicAlloc>(
                    rewriter.getUnknownLoc(), memref);
        }
    }
    return nullptr;
}

LogicalResult prepareOperands(Operation* op, SmallVector<Value>& operands,
                              ConversionPatternRewriter& rewriter) {
    for (auto&& i : op->getResults()) {
        if (auto memref = asBuffer(i, rewriter)) {
            operands.push_back(memref);
        } else {
            return failure();
        }
    }
    return success();
}

template <typename OpType>
LogicalResult createOp(Operation* op, ::mlir::ValueRange operands,
                       ConversionPatternRewriter& rewriter,
                       llvm::ArrayRef<NamedAttribute> attributes = {}) {
    SmallVector<Value> newOperands(operands.begin(), operands.end());
    if (succeeded(prepareOperands(op, newOperands, rewriter))) {
        rewriter.create<OpType>(rewriter.getUnknownLoc(), llvm::None,
                                newOperands, attributes);
        rewriter.replaceOp(
                op, llvm::makeArrayRef(newOperands.begin() + operands.size(),
                                       newOperands.end()));
        return success();
    }
    return failure();
}

void setOperandSegmentAttr(MLIRContext* context,
                           SmallVector<NamedAttribute, 4>& attrs,
                           const SmallVector<int32_t>& value) {
    auto attr = DenseIntElementsAttr::get(
            VectorType::get(value.size(), IntegerType::get(context, 32)),
            value);

    attrs.push_back({StringAttr::get(context, "operand_segment_sizes"), attr});
}

class ConvertParamStorage final
        : public OpConversionPattern<MGB::ParamStorage> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
            MGB::ParamStorage op, MGB::ParamStorage::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert MGB dialect to Abstract kernel WeightStorage of "
                     "opr name: "
                  << op.getName().str() << "\n";
        rewriter.replaceOpWithNewOp<Kernel::WeightStorage>(
                op, op.sym_name(), op.value(), op.type(), op.user_count());
        return success();
    }
};

class ConvertParamProvider final
        : public OpConversionPattern<MGB::ParamProvider> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
            MGB::ParamProvider op, MGB::ParamProvider::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert MGB dialect to Abstract kernel GetWeight of "
                     "opr name: "
                  << op.name().str() << "\n";
        auto memref = asMemRef(op->getResult(0).getType());
        if (!memref)
            return failure();
        rewriter.replaceOpWithNewOp<Kernel::GetWeight>(op, memref, op.name());
        return success();
    }
};

class ConvertElemwise final : public OpConversionPattern<MGB::Elemwise> {
public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
            MGB::Elemwise op, MGB::Elemwise::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        auto operands = adaptor.getOperands();
        LOG_DEBUG << "Convert MGB dialect to Abstract Elemwise kernel of "
                     "opr mode: "
                  << static_cast<int32_t>(op.mode())
                  << ", name: " << op.getOperationName().str() << "\n";
        using Mode = ::megdnn::param::Elemwise::Mode;
        switch (op.mode()) {
            case Mode::RELU:
                return createOp<Kernel::ReluKernel>(op, operands, rewriter);
            case Mode::SIGMOID:
                return createOp<Kernel::SigmoidKernel>(op, operands, rewriter);
            case Mode::ADD:
                return createOp<Kernel::AddKernel>(op, operands, rewriter);
            case Mode::ABS:
                return createOp<Kernel::AbsKernel>(op, operands, rewriter);
            case Mode::H_SWISH:
                return createOp<Kernel::HswishKernel>(op, operands, rewriter);
            case Mode::MUL:
                return createOp<Kernel::MulKernel>(op, operands, rewriter);
            case Mode::NEGATE:
                return createOp<Kernel::NegateKernel>(op, operands, rewriter);
            case Mode::SUB:
                return createOp<Kernel::SubKernel>(op, operands, rewriter);
            case Mode::TRUE_DIV:
                return createOp<Kernel::TrueDivKernel>(op, operands, rewriter);
            case Mode::MAX:
                return createOp<Kernel::MaxKernel>(op, operands, rewriter);
            case Mode::MIN:
                return createOp<Kernel::MinKernel>(op, operands, rewriter);
            case Mode::LT:
                return createOp<Kernel::LtKernel>(op, operands, rewriter);
            case Mode::LEQ:
                return createOp<Kernel::LeqKernel>(op, operands, rewriter);
            case Mode::FLOOR_DIV:
                return createOp<Kernel::FloorDivKernel>(op, operands, rewriter);
            case Mode::ROUND:
                return createOp<Kernel::RoundKernel>(op, operands, rewriter);
            case Mode::EXP:
                return createOp<Kernel::ExpKernel>(op, operands, rewriter);
            case Mode::FUSE_MUL_ADD3:
                return createOp<Kernel::FuseMulAdd3Kernel>(op, operands,
                                                           rewriter);
            case Mode::FUSE_MUL_ADD4:
                return createOp<Kernel::FuseMulAdd4Kernel>(op, operands,
                                                           rewriter);
            case Mode::FUSE_ADD_RELU:
                return createOp<Kernel::FuseAddReluKernel>(op, operands,
                                                           rewriter);
            case Mode::FUSE_ADD_TANH:
                return createOp<Kernel::FuseAddTanhKernel>(op, operands,
                                                           rewriter);
            case Mode::FUSE_ADD_SIGMOID:
                return createOp<Kernel::FuseAddSigmoidKernel>(op, operands,
                                                              rewriter);
            case Mode::LOG:
                return createOp<Kernel::LogKernel>(op, operands, rewriter);

            case Mode::MOD:
                return createOp<Kernel::ModKernel>(op, operands, rewriter);

            case Mode::EQ:
                return createOp<Kernel::EqualKernel>(op, operands, rewriter);
            case Mode::SILU:
                return createOp<Kernel::SILUKernel>(op, operands, rewriter);
            default:
                CC_ABORT << "Unsupport Elemwise mode :"
                         << static_cast<int>(op.mode()) << "\n";
                return failure();
        };
        return success();
    }
};

class ConvertConvLike final : public ConversionPattern {
public:
    ConvertConvLike(TypeConverter& converter, MLIRContext* ctx)
            : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const override {
        SmallVector<int32_t> operand_segment_sizes;
        // operands of Conv2DKernel: input, weight, bias, z, output
        if (isa<MGB::Convolution>(op)) {
            if (operands.size() != 2)
                return failure();
            operand_segment_sizes = {1, 1, 0, 0, 1};
        } else if (isa<MGB::ConvBias>(op)) {
            // FIXME: only conv_bias(input, weight, bias) is supported now
            if (operands.size() != 3)
                return failure();
            operand_segment_sizes = {1, 1, 1, 0, 1};
        } else if (isa<MGB::ConvolutionBackwardData>(op)) {
            if (operands.size() != 2)
                return failure();
        } else {
            LOG_WARN << "convert convlution like opr falied\n";
            return failure();
        }
        LOG_DEBUG << "Convert MGB dialect to Abstract Conv2DKernel or "
                     "ConvBackDataKernel kernel of "
                     "opr name: "
                  << op->getName().getStringRef().str() << "\n";
        CC_ASSERT(!isDynamicShape(operands))
                << "Convolution operands shape should not be dynamic.\n";
        if (isa<MGB::Convolution>(op) || isa<MGB::ConvBias>(op)) {
            auto attrs = ConvertConvLikeAttr(op->getAttrDictionary(),
                                             operands[1], op->getContext());
            setOperandSegmentAttr(op->getContext(), attrs,
                                  operand_segment_sizes);
            return createOp<Kernel::Conv2DKernel>(op, operands, rewriter,
                                                  attrs);
        } else if (isa<MGB::ConvolutionBackwardData>(op)) {
            auto attrs = ConvertConvLikeAttr(op->getAttrDictionary(),
                                             operands[0], op->getContext());
            return createOp<Kernel::ConvBackDataKernel>(op, operands, rewriter,
                                                        attrs);
        } else {
            return failure();
        }
    }
};

class ConvertFusedElemwise final
        : public OpConversionPattern<MGB::FusedElemwise> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::FusedElemwise op, MGB::FusedElemwise::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert FusedElemwise MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        CC_ASSERT(!isDynamicShape(operands))
                << "FusedElemwise operands shape should not be dynamic.\n";

        auto attrs = op->getAttrs();
        return createOp<Kernel::FusedElemwiseKernel>(op, adaptor.getOperands(),
                                                     rewriter, attrs);
    }
};

class ConvertReduce final : public OpConversionPattern<MGB::Reduce> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::Reduce op, MGB::Reduce::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert Reduce MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        CC_ASSERT(!isDynamicShape(operands))
                << "Reduce operands shape should not be dynamic.\n";

        auto attrs = ConvertAttr<MGB::Reduce>(op->getAttrDictionary(),
                                              op->getContext());
        return createOp<Kernel::ReduceKernel>(op, adaptor.getOperands(),
                                              rewriter, attrs);
    }
};

class ConvertSetSubtensor final
        : public OpConversionPattern<MGB::SetSubtensor> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::SetSubtensor op, MGB::SetSubtensor::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert SetSubtensor MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        bool is_dynamic_shape = isDynamicShape(operands);
        if (!is_dynamic_shape && operands.size() == 2) {
            auto src = operands[0];
            auto value = operands[1];
            //! 1. create output tensor, and copy to
            auto dst_value = asBuffer(op->getResult(0), rewriter);
            rewriter.create<Kernel::RelayoutKernel>(op->getLoc(), src,
                                                    dst_value);
            //! 2. create SubtensorView
            auto sub_view = asMemRef(dst_value.getType());
            auto descs = op->getAttrDictionary().getAs<ArrayAttr>("descs");
            auto flags = op->getAttrDictionary().getAs<ArrayAttr>("flags");
            auto subtensor_opr = rewriter.create<Kernel::Subtensor>(
                    op->getLoc(), sub_view, dst_value,
                    rewriter.getBoolAttr(true), descs, flags);
            subtensor_opr->setAttr("determined", rewriter.getBoolAttr(true));
            //! 3. create relayou and replace origin opr
            rewriter.create<Kernel::RelayoutKernel>(
                    op->getLoc(), value, subtensor_opr->getResult(0));
            rewriter.replaceOp(op, dst_value);
            return success();
        } else {
            auto attrs = ConvertAttr<MGB::SetSubtensor>(op->getAttrDictionary(),
                                                        op->getContext());
            return createOp<Kernel::SetSubtensorIns>(op, operands, rewriter,
                                                     attrs);
        }
    }
};

class ConvertSubtensor final : public OpConversionPattern<MGB::Subtensor> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::Subtensor op, MGB::Subtensor::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert Subtensor MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        bool is_dynamic_shape = isDynamicShape(operands);
        if (!is_dynamic_shape && operands.size() == 1) {
            auto resultType = asMemRef(op->getResult(0).getType());
            if (!resultType)
                return failure();
            rewriter.replaceOpWithNewOp<Kernel::Subtensor>(
                    op, resultType, operands, op->getAttrs());
            return success();
        } else {
            auto attrs = ConvertAttr<MGB::Subtensor>(op->getAttrDictionary(),
                                                     op->getContext());
            return createOp<Kernel::SubtensorIns>(op, operands, rewriter,
                                                  attrs);
        }
    }
};

class ConvertConcat final : public OpConversionPattern<MGB::Concat> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::Concat op, MGB::Concat::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert Concat MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        size_t nr_input = operands.size();
        bool is_dynamic_shape = isDynamicShape(operands);
        if (!is_dynamic_shape) {
            //! 1. create output tensor, and copy to
            auto dst_value = asBuffer(op->getResult(0), rewriter);
            //! 2. create every src SubtensorView and create relayout
            auto axis =
                    op->getAttrDictionary().getAs<IntegerAttr>("axis").getInt();
            auto type = operands[0].getType().cast<ShapedType>();
            if (axis < 0) {
                axis = type.getShape().size() + axis;
            }
            CC_ASSERT(axis < type.getShape().size() && axis >= 0);
            size_t start_index = 0;
            for (size_t i = 0; i < nr_input; i++) {
                auto sub_view = asMemRef(dst_value.getType());
                auto input = operands[i];
                auto desc = MakeConcatArrayAttr(axis, start_index, input,
                                                dst_value, rewriter);
                auto subtensor_opr = rewriter.create<Kernel::Subtensor>(
                        op->getLoc(), sub_view, dst_value,
                        rewriter.getBoolAttr(true), desc,
                        rewriter.getArrayAttr(
                                rewriter.getI32ArrayAttr({0, 0, 0, 0, 0})));
                subtensor_opr->setAttr("determined",
                                       rewriter.getBoolAttr(true));
                rewriter.create<Kernel::RelayoutKernel>(
                        op->getLoc(), input, subtensor_opr->getResult(0));
            }
            //! 3. replace origin opr
            rewriter.replaceOp(op, dst_value);
            return success();
        } else {
            auto attrs = ConvertAttr<MGB::Concat>(op->getAttrDictionary(),
                                                  op->getContext());
            return createOp<Kernel::ConcatKernel>(op, operands, rewriter,
                                                  attrs);
        }
    }
};

class ConvertReshape final : public OpConversionPattern<MGB::Reshape> {
public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::Reshape op, MGB::Reshape::Adaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        auto operands = adaptor.getOperands();
        bool is_dynamic_shape = isDynamicShape(operands);
        LOG_DEBUG << "Convert Reshape MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str()
                  << ", dynamic = " << is_dynamic_shape << "\n";
        if (!is_dynamic_shape || operands.size() == 1) {
            auto resultType = asMemRef(op->getResult(0).getType());
            if (!resultType)
                return failure();
            rewriter.replaceOpWithNewOp<Kernel::Reshape>(
                    op, resultType, operands, op->getAttrs());
            return success();
        } else {
            auto attrs = ConvertAttr<MGB::Reshape>(op->getAttrDictionary(),
                                                   op->getContext());
            return createOp<Kernel::ReshapeIns>(op, operands, rewriter, attrs);
        }
    }
};

template <class SrcOp, class DstIns, class DstKernelOp>
class DynamicConverter : public OpConversionPattern<SrcOp> {
public:
    using OpAdaptor = typename SrcOp::Adaptor;
    using OpConversionPattern<SrcOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(
            SrcOp op, OpAdaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        auto operands = adaptor.getOperands();
        auto attrs =
                ConvertAttr<SrcOp>(op->getAttrDictionary(), op->getContext());
        auto out_shapedType =
                op->getResult(0).getType().template dyn_cast<ShapedType>();
        if (out_shapedType.getNumDynamicDims() > 0) {
            return createOp<DstIns>(op, operands, rewriter, attrs);
        } else {
            LOG_DEBUG << "General convert MGB dialect to Abstract kernel of "
                         "opr name: "
                      << op.getOperationName().str() << "\n";
            return createOp<DstKernelOp>(op, operands, rewriter, attrs);
        }
    }
};

template <class SrcOp, class DstOp, class DstKernelOp>
class MemRefConverter final : public OpConversionPattern<SrcOp> {
public:
    using OpAdaptor = typename SrcOp::Adaptor;
    using OpConversionPattern<SrcOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(
            SrcOp op, OpAdaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        auto operands = adaptor.getOperands();
        bool is_dynamic_shape = isDynamicShape(operands);
        LOG_DEBUG << "Convert MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str()
                  << ", dynamic = " << is_dynamic_shape << "\n";
        if (!is_dynamic_shape) {
            auto resultType = asMemRef(op->getResult(0).getType());
            if (!resultType)
                return failure();
            rewriter.replaceOpWithNewOp<DstOp>(op, resultType, operands,
                                               op->getAttrs());
            return success();
        } else {
            auto attrs = ConvertAttr<SrcOp>(op->getAttrDictionary(),
                                            op->getContext());
            return createOp<DstKernelOp>(op, operands, rewriter, attrs);
        }
    }
};

template <class SrcOp, class DstOp>
class GenericConverter : public OpConversionPattern<SrcOp> {
public:
    using OpAdaptor = typename SrcOp::Adaptor;
    using OpConversionPattern<SrcOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(
            SrcOp op, OpAdaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "General convert MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        auto attrs =
                ConvertAttr<SrcOp>(op->getAttrDictionary(), op->getContext());
        return createOp<DstOp>(op, operands, rewriter, attrs);
    }
};

class ExternOprConverter : public OpConversionPattern<MGB::ExternOpr> {
public:
    using OpAdaptor = typename MGB::ExternOpr::Adaptor;
    using OpConversionPattern<MGB::ExternOpr>::OpConversionPattern;
    LogicalResult matchAndRewrite(
            MGB::ExternOpr op, OpAdaptor adaptor,
            ConversionPatternRewriter& rewriter) const override {
        LOG_DEBUG << "Convert ExternOpr MGB dialect to Abstract kernel of "
                     "opr name: "
                  << op.getOperationName().str() << "\n";
        auto operands = adaptor.getOperands();
        CC_ASSERT(!isDynamicShape(operands))
                << "ExternOpr operands shape should not be dynamic.\n";
        auto attrs = ConvertAttr<MGB::ExternOpr>(op->getAttrDictionary(),
                                                 op->getContext());
        setOperandSegmentAttr(op->getContext(), attrs,
                              {op.nr_input(), op.nr_output()});
        return createOp<Kernel::ExternOpr>(op, operands, rewriter, attrs);
    }
};

}  // namespace

void populateMGBToKernelConversionPatterns(TypeConverter& typeConverter,
                                           RewritePatternSet& patterns) {
    patterns.add<
            ConvertParamStorage, ConvertParamProvider, ConvertElemwise,
            ConvertFusedElemwise, ConvertConvLike, ConvertReduce,
            ConvertReshape, ConvertSubtensor, ConvertSetSubtensor,
            ConvertConcat, ExternOprConverter,
            GenericConverter<MGB::WarpPerspective,
                             Kernel::WarpPerspectiveKernel>,
            GenericConverter<MGB::IndexingMultiAxisVec,
                             Kernel::IndexingMultiAxisVecKernel>,
            GenericConverter<MGB::IndexingOneHot, Kernel::IndexingOneHotKernel>,
            MemRefConverter<MGB::Dimshuffle, Kernel::Dimshuffle,
                            Kernel::DimshuffleIns>,
            GenericConverter<MGB::Argsort, Kernel::ArgsortKernel>,
            GenericConverter<MGB::Argmax, Kernel::ArgmaxKernel>,
            GenericConverter<MGB::TopK, Kernel::TopkKernel>,
            GenericConverter<MGB::Broadcast, Kernel::BroadcastIns>,
            GenericConverter<MGB::TypeCvt, Kernel::TypeCvtKernel>,
            GenericConverter<MGB::BatchedMatrixMul,
                             Kernel::BatchedMatrixMulKernel>,
            GenericConverter<MGB::MatrixMul, Kernel::MatrixMulKernel>,
            GenericConverter<MGB::Pooling, Kernel::Pooling2DKernel>,
            GenericConverter<MGB::WarpAffine, Kernel::WarpAffineKernel>,
            GenericConverter<MGB::Resize, Kernel::ResizeKernel>,
            GenericConverter<MGB::MatrixInverse, Kernel::MatrixInvKernel>,

            GenericConverter<MGB::GetVarShape, Kernel::GetVarShapeIns>,
            GenericConverter<MGB::ElemwiseMultiType, Kernel::ElemwiseMultiType>,
            GenericConverter<MGB::PowC, Kernel::PowCKernel>>(
            typeConverter, patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> createMGBToKernelPass() {
    return std::make_unique<MGBToKernelPass>();
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
