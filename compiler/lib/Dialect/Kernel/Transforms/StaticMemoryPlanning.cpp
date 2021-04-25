/**
 * \file compiler/lib/Dialect/Kernel/Transforms/StaticMemoryPlanning.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <unordered_map>
#include <unordered_set>

#include "./migrate/static_mem_alloc.h"

#include "compiler/Common/Logger.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
using namespace bufferization;
namespace {
#define GEN_PASS_CLASSES
#include "compiler/Dialect/Kernel/Transforms/Passes.h.inc"

class StaticMemoryPlanning : BufferPlacementTransformationBase {
    using Solver = Kernel::migrate::StaticMemAlloc;

public:
    StaticMemoryPlanning(FuncOp op)
            : BufferPlacementTransformationBase(op),
              func(op),
              postDominators(op) {}

    void makePlan() {
        std::vector<Value> allocValues;
        auto solver = constructProblem(allocValues);
        solver->solve();
        applySolution(*solver, allocValues);
    }

private:
    std::unique_ptr<Solver> constructProblem(std::vector<Value>& allocValues) {
        std::unique_ptr<Solver> solver =
                Solver::make(Solver::AllocatorAlgo::PUSHDOWN);
        std::unordered_map<Operation*, size_t> op2idx;
        size_t curIdx = 0;
        for (auto&& op : func.getBody().front()) {
            op2idx[&op] = curIdx++;
        }
        for (const BufferPlacementAllocs::AllocEntry& entry : allocs) {
            Value alloc = std::get<0>(entry);
            if (alloc.getType().dyn_cast<ShapedType>().getNumDynamicDims() >
                0) {
                continue;
            }
            auto aliasesSet = aliases.resolve(alloc);
            assert(aliasesSet.size() > 0 && "must contain at least one alias");

            // Determine the actual block to place the dealloc and get liveness
            // information.
            Block* placementBlock =
                    findCommonDominator(alloc, aliasesSet, postDominators);
            const LivenessBlockInfo* livenessInfo =
                    liveness.getLiveness(placementBlock);

            // We have to ensure that the dealloc will be after the last use of
            // all aliases of the given value. We first assume that there are no
            // uses in the placementBlock and that we can safely place the
            // dealloc at the beginning.
            Operation* endOperation = &placementBlock->front();

            // Iterate over all aliases and ensure that the endOperation will
            // point to the last operation of all potential aliases in the
            // placementBlock.
            for (Value alias : aliasesSet) {
                // Ensure that the start operation is at least the defining
                // operation of the current alias to avoid invalid placement of
                // deallocs for aliases without any uses.
                Operation* beforeOp = endOperation;
                if (alias.getDefiningOp() &&
                    !(beforeOp = placementBlock->findAncestorOpInBlock(
                              *alias.getDefiningOp())))
                    continue;

                Operation* aliasEndOperation =
                        livenessInfo->getEndOperation(alias, beforeOp);
                // Check whether the aliasEndOperation lies in the desired block
                // and whether it is behind the current endOperation. If yes,
                // this will be the new endOperation.
                if (aliasEndOperation->getBlock() == placementBlock &&
                    endOperation->isBeforeInBlock(aliasEndOperation))
                    endOperation = aliasEndOperation;
            }

            // TODO : maybe we should skip the return value and pass its
            // memref by function arguments
            /* Operation* nextOp = endOperation->getNextNode();
            if (!nextOp) {
                continue;
            } */
            size_t begin = op2idx.at(alloc.getDefiningOp()),
                   end = op2idx.at(endOperation) + 1,
                   sizeInBits = alloc.getType()
                                        .dyn_cast<ShapedType>()
                                        .getSizeInBits();
            CC_ASSERT((sizeInBits & 7) == 0);
            solver->add(begin, end, sizeInBits >> 3,
                        alloc.getAsOpaquePointer());
            allocValues.push_back(alloc);
        }
        return solver;
    }

    MemRefType assignOffset(MemRefType oldMemRef, int64_t newOffset) {
        int64_t oldOffset;
        llvm::SmallVector<int64_t> oldStride;
        if (failed(getStridesAndOffset(oldMemRef, oldStride, oldOffset))) {
            CC_ABORT << "only support strided memref type\n";
        }
        CC_ASSERT(oldOffset == 0);
        return MemRefType::get(
                oldMemRef.getShape(), oldMemRef.getElementType(),
                makeStridedLinearLayoutMap(oldStride, newOffset,
                                           oldMemRef.getContext()));
    }

    void applySolution(Solver& solver, const std::vector<Value>& allocValues) {
        // add global buffer to block argument
        Type globalBufferType =
                MemRefType::get({static_cast<int64_t>(solver.tot_alloc())},
                                IntegerType::get(func.getContext(), 8));
        OpBuilder func_builder(func);
        func.getBody().front().addArgument(globalBufferType,
                                           func_builder.getUnknownLoc());
        Value globalBuffer = func.getArguments().back();
        for (auto allocated : allocValues) {
            size_t offset =
                    solver.get_start_addr(allocated.getAsOpaquePointer());
            memref::AllocOp alloc = allocated.getDefiningOp<memref::AllocOp>();
            OpBuilder builder(alloc);
            auto oldMemRef = allocated.getType().dyn_cast<MemRefType>();
            auto newMemRef = assignOffset(oldMemRef, offset);
            auto memplan = builder.create<Kernel::MemPlan>(
                    alloc->getLoc(), newMemRef, globalBuffer);
            allocated.replaceAllUsesWith(memplan);
            alloc->erase();
        }

        removeUnusedMemFwd();

        // rewrite function type
        FunctionType oldFuncType = func.type().dyn_cast<FunctionType>();
        std::vector<Type> ArgumentsType = oldFuncType.getInputs();
        ArgumentsType.push_back(globalBufferType);
        auto returnOp =
                llvm::dyn_cast<ReturnOp>(&func.getBody().front().back());
        auto ResultsType = returnOp.getOperandTypes();
        function_interface_impl::setFunctionType(
                func, FunctionType::get(func.getContext(), ArgumentsType,
                                        ResultsType));
        func.setArgAttr(func.getNumArguments() - 1, "mgb.func_arg_name",
                        StringAttr::get(func.getContext(), "kGlobalBuffer"));
    }

    void removeUnusedMemFwd() {
        Value globalBuffer = func.getArguments().back();
        auto tryFoldMemFwd = [](Kernel::MemFwdInterface op) -> MemRefType {
            Value input = op->getOperand(0);
            if (!input.getDefiningOp<Kernel::MemPlan>())
                return {};

            if (!op->getAttrOfType<BoolAttr>("determined").getValue())
                return {};

            if (auto inputType = input.getType().dyn_cast<MemRefType>()) {
                return op.memoryForward(inputType);
            };

            return {};
        };
        func.walk([&](Kernel::MemFwdInterface op) {
            // fold MemPlan -> MemFwd to MemPlan
            if (auto newMemRef = tryFoldMemFwd(op)) {
                OpBuilder builder(op);
                auto memplan = builder.create<Kernel::MemPlan>(
                        op->getLoc(), newMemRef, globalBuffer);
                op->replaceAllUsesWith(memplan);
                op->erase();
            }
        });
    }

    FuncOp func;
    PostDominanceInfo postDominators;
};

class StaticMemoryPlanningPass final
        : public StaticMemoryPlanningPassBase<StaticMemoryPlanningPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        auto op = getOperation();
        StaticMemoryPlanning planner(op);
        planner.makePlan();
    }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createStaticMemoryPlanningPass() {
    return std::make_unique<StaticMemoryPlanningPass>();
}

}  // namespace mlir

// vim: syntax=cpp.doxygen
