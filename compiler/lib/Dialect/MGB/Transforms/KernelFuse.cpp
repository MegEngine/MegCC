
/**
 * \file compiler/lib/Dialect/MGB/Transforms/KernelFuse.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>
#include <unordered_map>
#include "compiler/Common/Logger.h"
#include "compiler/Common/MlirUtils.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Dialect/MGB/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace megcc;
namespace mlir {
namespace {

#define GEN_PASS_CLASSES
#include "compiler/Dialect/MGB/Transforms/Passes.h.inc"

#define CHECK_IT(x)           \
    do {                      \
        if (!(x)) {           \
            return failure(); \
        }                     \
    } while (0)

class FuseTypeCvtPattern final : public OpRewritePattern<MGB::TypeCvt> {
public:
    FuseTypeCvtPattern(MLIRContext* ctx) : OpRewritePattern(ctx) {}
    LogicalResult matchAndRewrite(MGB::TypeCvt op,
                                  PatternRewriter& rewriter) const override {
        auto rst = op.getResult();
        CHECK_IT(rst.hasOneUse());
        auto next_typecvt =
                llvm::dyn_cast<MGB::TypeCvt>(rst.getUses().begin()->getOwner());
        CHECK_IT(next_typecvt);
        auto in_type = op.i_dtype();
        auto in_tensor = op.inputs();
        auto out_type = next_typecvt.o_dtype();
        auto new_typecvt = rewriter.create<MGB::TypeCvt>(
                op->getLoc(), next_typecvt.getResult().getType(), in_tensor,
                in_type, out_type);
        rewriter.replaceOp(next_typecvt, new_typecvt.getResult());
        return failure();
    }
};

class FuseConvHswishPattern final : public OpRewritePattern<MGB::ConvBias> {
public:
    FuseConvHswishPattern(MLIRContext* ctx) : OpRewritePattern(ctx) {}
    LogicalResult matchAndRewrite(MGB::ConvBias op,
                                  PatternRewriter& rewriter) const override {
        CHECK_IT(op.nonlineMode() ==
                 ::megdnn::param::ConvBias::NonlineMode::IDENTITY);
        auto rst = op.getResult();
        CHECK_IT(rst.hasOneUse());

        auto typecvt =
                llvm::dyn_cast<MGB::TypeCvt>(rst.getUses().begin()->getOwner());
        CHECK_IT(typecvt);
        CHECK_IT(typecvt.i_dtype().isInteger(8));
        CHECK_IT(typecvt.i_dtype().cast<IntegerType>().isQuant());
        CHECK_IT(typecvt.o_dtype().isF32());
        auto typecvt_rst = typecvt.getResult();
        CHECK_IT(typecvt_rst.hasOneUse());

        auto elem = llvm::dyn_cast<MGB::Elemwise>(
                typecvt_rst.getUses().begin()->getOwner());
        CHECK_IT(elem);
        CHECK_IT(elem.mode() == ::megdnn::param::Elemwise::Mode::H_SWISH);
        auto elem_rst = elem.getResult();
        CHECK_IT(elem_rst.hasOneUse());

        auto typecvt2 = llvm::dyn_cast<MGB::TypeCvt>(
                elem_rst.getUses().begin()->getOwner());
        CHECK_IT(typecvt2);
        CHECK_IT(typecvt2.i_dtype().isF32());
        CHECK_IT(typecvt2.o_dtype().isInteger(8));
        CHECK_IT(typecvt2.o_dtype().cast<IntegerType>().isQuant());
        auto typecvt_rst2 = typecvt2.getResult();
        CHECK_IT(typecvt_rst2.hasOneUse());

        auto new_conv = rewriter.create<MGB::ConvBias>(
                op->getLoc(), typecvt_rst2.getType(), op.inputs(),
                ::megdnn::param::ConvBias::NonlineMode::H_SWISH, op.mode(),
                op.sparse(), op.format(), op.pad_h(), op.pad_w(), op.stride_h(),
                op.stride_w(), op.dilate_h(), op.dilate_w(), op.compute_mode());
        rewriter.replaceOp(typecvt2, new_conv.getResult());
        return failure();
    }
};
void populateFuseKernelPatterns(RewritePatternSet& patterns) {
    patterns.add(std::make_unique<FuseTypeCvtPattern>(patterns.getContext()));
    patterns.add(
            std::make_unique<FuseConvHswishPattern>(patterns.getContext()));
}

class MGBFuseKernelPass final
        : public MGBFuseKernelPassBase<MGBFuseKernelPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        auto op = getOperation();
        populateFuseKernelPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
            signalPassFailure();
    }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createMGBFuseKernelPass() {
    return std::make_unique<MGBFuseKernelPass>();
}
}  // namespace mlir