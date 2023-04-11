#include <unordered_map>
#include <unordered_set>

#include "./migrate/static_mem_alloc.h"

#include "compiler/Common/Logger.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {
#define GEN_PASS_CLASSES
#include "compiler/Dialect/Kernel/Transforms/Passes.h.inc"

class CleanDimshufflePattern final : public OpRewritePattern<Kernel::Dimshuffle> {
public:
    CleanDimshufflePattern(MLIRContext* ctx) : OpRewritePattern(ctx) {}
    LogicalResult matchAndRewrite(
            Kernel::Dimshuffle op, PatternRewriter& rewriter) const override {
        if (!op->getAttrOfType<BoolAttr>("determined").getValue())
            return failure();
        Value input = op->getOperand(0);
        Value output = op->getResult(0);
        if (input.getDefiningOp<Kernel::MemPlan>())
            return failure();

        if (input.getDefiningOp() &&
            mlir::dyn_cast_or_null<Kernel::Reshape>(input.getDefiningOp())) {
            input.setType(op->getResult(0).getType());
        } else {
            auto input_memref = input.getType().dyn_cast_or_null<MemRefType>();
            auto output_memref =
                    op->getResult(0).getType().dyn_cast_or_null<MemRefType>();
            CC_ASSERT(input_memref);
            CC_ASSERT(output_memref);
            bool io_contig = Kernel::isContiguous(input_memref) &&
                             Kernel::isContiguous(output_memref);
            bool next_reshape =
                    output.hasOneUse() && mlir::dyn_cast_or_null<Kernel::Reshape>(
                                                  output.getUses().begin()->getOwner());
            CC_ASSERT(io_contig || next_reshape);
        }
        op.replaceAllUsesWith(input);
        op.erase();
        return success();
    }
};

void populateFinalCleanPatterns(RewritePatternSet& patterns) {
    patterns.add(std::make_unique<CleanDimshufflePattern>(patterns.getContext()));
}

class KernelFinalCleanPass final
        : public KernelFinalCleanPassBase<KernelFinalCleanPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        auto op = getOperation();
        populateFinalCleanPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
            signalPassFailure();
    }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createKernelFinalCleanPass() {
    return std::make_unique<KernelFinalCleanPass>();
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
