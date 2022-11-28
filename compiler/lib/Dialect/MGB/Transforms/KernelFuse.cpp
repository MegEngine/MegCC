
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
#include "llvm/Support/Casting.h"
#include "megbrain/reflection.h"
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

bool isDynamicShape(ValueRange operands) {
    bool is_dynamic_shape = false;
    for (size_t i = 0; i < operands.size(); i++) {
        if (auto shapedType = operands[i].getType().dyn_cast<ShapedType>()) {
            is_dynamic_shape |= shapedType.getNumDynamicDims() > 0;
        }
    }
    return is_dynamic_shape;
}

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
        return success();
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
        return success();
    }
};

class FuseElemwisePattern final : public OpRewritePattern<MGB::Elemwise> {
public:
    FuseElemwisePattern(MLIRContext* ctx) : OpRewritePattern(ctx) {}
    LogicalResult matchAndRewrite(MGB::Elemwise op,
                                  PatternRewriter& rewriter) const override {
        //! find the last elemwise op
        Operation* last_elemwise_op = op;
        Operation* tmp_op = op;
        if (isDynamicShape(op.getOperands())) {
            return failure();
        }
        while (llvm::dyn_cast<MGB::Elemwise>(tmp_op)) {
            last_elemwise_op = llvm::dyn_cast<MGB::Elemwise>(tmp_op);
            auto output = last_elemwise_op->getResult(0);
            if (!output.hasOneUse() || isDynamicShape(output)) {
                break;
            }
            tmp_op = output.getUses().begin()->getOwner();
        }
        //! no other elemwise to be fused to the Operator
        if (last_elemwise_op == op) {
            return failure();
        }
        //! recursively find all inputs and the elemwise mode
        //! record all the inputs and their mode to a vector of string, the
        //! string consist of inputs, mode, outputs, stored in the format of
        //! [I0, I1, Add, O0], the dst data is alias to "D"
        //
        //! for example: relu((a + b)*c -d) will record as
        //! std::vector<llvm::StringRef>{"I0,I1,Add,O0", "O0,I2,Mul,O1",
        //! "O1,I3,Sub,O2", "O2,Relu,D"};
        //!
        std::unordered_map<detail::ValueImpl*, std::string> var_alias;
        std::vector<Value> in_values;
        std::vector<Value> out_values;
        std::vector<std::string> modes;
        std::function<void(MGB::Elemwise op)>
                find_all_elemwise_info;
        auto dst = llvm::cast<MGB::Elemwise>(last_elemwise_op).getResult();
        find_all_elemwise_info = [&](MGB::Elemwise op) {
            auto all_input = op->getOperands();
            if (isDynamicShape(all_input)) {
                return;
            }
            std::string operator_mode;
            for (auto input : all_input) {
                auto owner_opr = input.getDefiningOp<MGB::Elemwise>();
                //! when the op is elemwise and only have one user
                if (owner_opr && input.hasOneUse()) {
                    find_all_elemwise_info(owner_opr);
                }
                if (operator_mode.size() != 0) {
                    operator_mode += ",";
                }
                if (var_alias.find(input.getImpl()) != var_alias.end()) {
                    operator_mode += var_alias[input.getImpl()];
                } else {
                    size_t id = in_values.size();
                    in_values.push_back(input);
                    std::string name = "I" + std::to_string(id);
                    operator_mode += name;
                    var_alias[input.getImpl()] = name;
                }
            }
            operator_mode += ",";
            operator_mode += mgb::reflection::nameOfEnumValue(op.mode());
            operator_mode += ",";
            if (op.getResult() == dst) {
                operator_mode += "D";
            } else {
                size_t id = out_values.size();
                out_values.push_back(op.getResult());
                std::string name = "O" + std::to_string(id);
                operator_mode += name;
                var_alias[op.getResult().getImpl()] = name;
            }
            modes.push_back(operator_mode);
            LOG_DEBUG << "Fuse elemwise operator mode is : " << operator_mode
                      << "\n";
        };

        //! the output var alias to "D", means dst
        var_alias[llvm::cast<MGB::Elemwise>(last_elemwise_op)
                          .getResult()
                          .getImpl()] = "D";
        find_all_elemwise_info(llvm::cast<MGB::Elemwise>(last_elemwise_op));
        CC_ASSERT(out_values.size() > 0)
                << "The fused elemwise with no output var.\n";
        auto dst_var = out_values.back();
        if (modes.size() <= 1) {
            return failure();
        }
        auto ctx = op.getContext();
        SmallVector<Attribute> attributs;
        for (size_t i = 0; i < modes.size(); i++) {
            attributs.push_back(StringAttr::get(ctx, modes[i]));
        }
        auto arry_attrs = mlir::ArrayAttr::get(ctx, attributs);
        SmallVector<NamedAttribute, 4> attrs;
        attrs.push_back({StringAttr::get(ctx, "modes"), arry_attrs});

        auto new_elemwise = rewriter.create<MGB::FusedElemwise>(
                op->getLoc(), dst_var.getType(), in_values, attrs);
        rewriter.replaceOp(last_elemwise_op, new_elemwise.getResult());
        return success();
    }
};

void populateFuseKernelPatterns(RewritePatternSet& patterns) {
    patterns.add(std::make_unique<FuseTypeCvtPattern>(patterns.getContext()));
    patterns.add(
            std::make_unique<FuseConvHswishPattern>(patterns.getContext()));
    patterns.add(std::make_unique<FuseElemwisePattern>(patterns.getContext()));
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
