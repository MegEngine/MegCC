/**
 * \file compiler/lib/CodeGen/Elemwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "CodeGenUtil.h"
#include "Elemwise.h"
#include "GlobalCtx.h"

#include "compiler/CodeGen/CodeGen.h"
#include "compiler/Common/Logger.h"
#include "compiler/Common/MemoryStatus.h"
#include "compiler/Common/MlirUtils.h"
#include "compiler/Common/TContext.h"

#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace megcc;
using namespace codegen;
using namespace mlir;

bool ElemwiseKernel::IsAvailable(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    auto nr_operands = context->getAttrInt("nr_operands");
    bool nr_operands_ok = nr_operands == 2;
    bool mode_ok_unary = mode == "RELU";
    bool mode_ok_binary = false;
    bool mode_ok_other = false;
    return nr_operands_ok && (mode_ok_unary || mode_ok_binary || mode_ok_other);
}

std::string ElemwiseKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "mlir_auto_elementwise";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << context->getAttrOprand("operand:0").shape.size();
    return ss.str();
}

void ElemwiseKernel::CreateCompute(Block* entryBlock,
                                   mlir::OpBuilder& op_builder,
                                   mlir::MLIRContext* ctx,
                                   TContext* context) const {
    Value input_val = entryBlock->getArgument(0);
    Value output_val = entryBlock->getArgument(1);
    MemRefType out_memref = output_val.getType().dyn_cast_or_null<MemRefType>();

    SmallVector<AffineMap> indexing_maps;
    SmallVector<StringRef> iter_type =
            getNParallelLoopsAttrs(out_memref.getRank());

    indexing_maps.push_back(get_affine_map(
            input_val.getType().dyn_cast_or_null<MemRefType>(), ctx));

    indexing_maps.push_back(get_affine_map(
            output_val.getType().dyn_cast_or_null<MemRefType>(), ctx));
    SmallVector<Value> inputs_val;
    SmallVector<Value> outputs_val;
    inputs_val.push_back(input_val);
    outputs_val.push_back(output_val);
    op_builder.create<linalg::GenericOp>(
            op_builder.getUnknownLoc(), inputs_val, outputs_val, indexing_maps,
            iter_type,
            [&](OpBuilder& nestedBuilder, Location nestedLoc,
                ValueRange blockArgs) {
                Type act_type = blockArgs[0].getType();
                Value input_val = blockArgs[0];
                // ConstantOp
                Value const_zero_op = op_builder.create<arith::ConstantOp>(
                        op_builder.getUnknownLoc(), act_type,
                        FloatAttr::get(act_type, llvm::APFloat(0.f)));
                Value opResult = op_builder.create<arith::MaxFOp>(
                        op_builder.getUnknownLoc(), act_type, input_val,
                        const_zero_op);
                op_builder.create<linalg::YieldOp>(op_builder.getUnknownLoc(),
                                                   opResult);
            });
    std::vector<Value> results;
    op_builder.create<ReturnOp>(op_builder.getUnknownLoc(), results);
}

void ElemwiseKernel::CreatePass(mlir::PassManager& pm, mlir::MLIRContext* ctx,
                                TContext* context) const {
    tosa::addTosaToLinalgPasses(pm);
    //! used to do complex math with some simple operations
    pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
    //! Linalg To Loops, may not used in gpu
    pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
    //! Loops to CFG
    pm.addNestedPass<FuncOp>(createLowerToCFGPass());
    //! std optimize
    pm.addNestedPass<FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(createCSEPass());
    //! check only memref is used
    pm.addNestedPass<FuncOp>(bufferization::createFinalizingBufferizePass());
    //! memref to llvm
    pm.addPass(createMemRefToLLVMPass());
    //! arith to llvm
    pm.addNestedPass<FuncOp>(arith::createConvertArithmeticToLLVMPass());
    LowerToLLVMOptions std2llvm_opt(ctx);
    std2llvm_opt.emitCWrappers = true;
    //! std to llvm with emit-c
    pm.addPass(createLowerToLLVMPass(std2llvm_opt));
    //! remove all unsupported ir
    pm.addPass(createReconcileUnrealizedCastsPass());
}
