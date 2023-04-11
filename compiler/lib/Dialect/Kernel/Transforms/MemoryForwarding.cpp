#include "compiler/Common/Logger.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {
#define GEN_PASS_CLASSES
#include "compiler/Dialect/Kernel/Transforms/Passes.h.inc"

class MemoryForwardingPass final
        : public MemoryForwardingPassBase<MemoryForwardingPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        FuncOp func = getOperation();
        populateMemoryForwardingPatterns(patterns);
        FrozenRewritePatternSet frozenPatterns(std::move(patterns));
        func.walk([&](Operation* op) {
            (void)applyOpPatternsAndFold(op, frozenPatterns);
        });

        // rewrite function type
        FunctionType oldFuncType = func.type().dyn_cast<FunctionType>();
        auto ArgumentsType = oldFuncType.getInputs();
        auto returnOp = llvm::dyn_cast<ReturnOp>(&func.getBody().front().back());
        auto ResultsType = returnOp.getOperandTypes();
        function_interface_impl::setFunctionType(
                func, FunctionType::get(func.getContext(), ArgumentsType, ResultsType));
    }
};

LogicalResult tryMemoryForward(
        Kernel::MemFwdInterface op, PatternRewriter& rewriter,
        std::function<LogicalResult(void)> onFailure) {
    CC_ASSERT(op->getNumOperands() == 1 && op->getNumResults() == 1);
    Value input = op->getOperand(0), output = op->getResult(0);
    MemRefType outputType;
    if (auto inputType = input.getType().dyn_cast<MemRefType>()) {
        outputType = op.memoryForward(inputType);
    }

    auto determined_attr = op->getAttr("determined");
    bool determined =
            determined_attr ? determined_attr.dyn_cast<BoolAttr>().getValue() : false;
    if (!outputType) {
        CC_ASSERT(!determined) << "memory forward failed on determined op";
        return onFailure();
    }

    output.setType(outputType);
    if (!determined) {
        op->setAttr("determined", rewriter.getBoolAttr(true));
    }

    for (auto&& use : output.getUses()) {
        auto layoutConstraint =
                llvm::dyn_cast<Kernel::LayoutConstraintInterface>(use.getOwner());
        bool relayout_for_next_kernel =
                layoutConstraint &&
                !layoutConstraint.checkInputLayout(outputType, use.getOperandNumber());
        bool relayout_for_return = mlir::dyn_cast<ReturnOp>(use.getOwner()) &&
                                   !::mlir::Kernel::isContiguous(outputType);
        bool need_relayout_output = relayout_for_next_kernel || relayout_for_return;
        if (need_relayout_output) {
            CC_ASSERT(!determined) << "memory forward failed on determined op";
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointAfter(op);
            Value newOutput = rewriter.create<memref::AllocOp>(
                    op->getLoc(),
                    MemRefType::get(
                            outputType.getShape(), outputType.getElementType()));
            Operation* relayout = rewriter.create<Kernel::RelayoutKernel>(
                    op->getLoc(), output, newOutput);
            output.replaceUsesWithIf(newOutput, [&](mlir::OpOperand& i) {
                return i.getOwner() != relayout;
            });
            break;
        }
    }
    return success();
}

template <typename T>
LogicalResult onMemoryForwardFailure(T /* op */, PatternRewriter& /* rewriter */) {
    return failure();
}

template <>
LogicalResult onMemoryForwardFailure<Kernel::Reshape>(
        Kernel::Reshape op, PatternRewriter& rewriter) {
    Value input = op->getOperand(0), output = op->getResult(0);
    MemRefType outputType = output.getType().dyn_cast<MemRefType>();
    Value newOutput = rewriter.create<memref::AllocOp>(
            op->getLoc(),
            MemRefType::get(outputType.getShape(), outputType.getElementType()));
    MemRefType inputType = input.getType().dyn_cast<MemRefType>();
    Value newOutputReshaped = rewriter.create<Kernel::Reshape>(
            op->getLoc(),
            MemRefType::get(inputType.getShape(), inputType.getElementType()),
            newOutput, true);
    rewriter.create<Kernel::RelayoutKernel>(op->getLoc(), input, newOutputReshaped);
    rewriter.replaceOp(op, newOutput);
    return success();
}

template <typename OpTy>
class MemFwdOpConversion final : public OpRewritePattern<OpTy> {
public:
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy op, PatternRewriter& rewriter) const override {
        return tryMemoryForward(
                op, rewriter, [&] { return onMemoryForwardFailure(op, rewriter); });
    }
};

}  // namespace

void populateMemoryForwardingPatterns(RewritePatternSet& patterns) {
    patterns.add<
            MemFwdOpConversion<Kernel::Reshape>, MemFwdOpConversion<Kernel::Dimshuffle>,
            MemFwdOpConversion<Kernel::Subtensor>>(patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>> createMemoryForwardingPass() {
    return std::make_unique<MemoryForwardingPass>();
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
