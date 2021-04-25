/**
 * \file compiler/lib/CodeGen/Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "CodeGenUtil.h"
#include "GlobalCtx.h"
#include "Matmul.h"

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

bool MatmulKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "DEFAULT" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:1").shape.size() == 2 &&
                    context->getAttrOprand("operand:0").shape.size() == 2;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;

    return ok_dtype && ok_mode && ok_shape && ok_tran;
}

std::string MatmulKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "mlir_auto_matmul_";
    if (context->getAttrBool("transposeA")) {
        ss << "t";
    } else {
        ss << "n";
    }
    if (context->getAttrBool("transposeB")) {
        ss << "t";
    } else {
        ss << "n";
    }
    return ss.str();
}

void MatmulKernel::CreateCompute(Block* entryBlock, mlir::OpBuilder& op_builder,
                                 mlir::MLIRContext* ctx,
                                 TContext* context) const {
    Value input_a = entryBlock->getArgument(0);
    Value input_b = entryBlock->getArgument(1);
    Value output_c = entryBlock->getArgument(2);

    SmallVector<Value> inputs_val;
    SmallVector<Value> outputs_val;
    inputs_val.push_back(input_a);
    inputs_val.push_back(input_b);
    outputs_val.push_back(output_c);

    op_builder.create<linalg::MatmulOp>(op_builder.getUnknownLoc(), inputs_val,
                                        outputs_val);
    std::vector<Value> results;
    op_builder.create<ReturnOp>(op_builder.getUnknownLoc(), results);
}

void MatmulKernel::CreatePass(mlir::PassManager& pm, mlir::MLIRContext* ctx,
                              TContext* context) const {
    tosa::addTosaToLinalgPasses(pm);
    //! used to do complex math with some simple operations
    pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
    //! below pass not used for dynamic shape, and not divisible
    //! pm.addNestedPass<FuncOp>(createLinalgTilingPass({4, 16, 1}));

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
