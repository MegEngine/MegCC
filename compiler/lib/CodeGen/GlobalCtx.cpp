/**
 * \file compiler/lib/CodeGen/GlobalCtx.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "GlobalCtx.h"

#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
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
using namespace mlir;

class GlobalCtx {
public:
    GlobalCtx() {
        reg_ptr = std::make_unique<mlir::DialectRegistry>();
        reg_ptr->insert<StandardOpsDialect, tensor::TensorDialect,
                        tosa::TosaDialect, linalg::LinalgDialect,
                        bufferization::BufferizationDialect>();

        ctx_ptr = std::make_unique<mlir::MLIRContext>(*reg_ptr);
        ctx_ptr->loadDialect<StandardOpsDialect, tosa::TosaDialect,
                             linalg::LinalgDialect,
                             bufferization::BufferizationDialect>();

        registerLLVMDialectTranslation(*ctx_ptr);
    }
    mlir::MLIRContext* get_ctx() { return ctx_ptr.get(); }

private:
    std::unique_ptr<mlir::MLIRContext> ctx_ptr;
    std::unique_ptr<mlir::DialectRegistry> reg_ptr;
};

mlir::MLIRContext* megcc::codegen::getGlobalCTX() {
    static GlobalCtx ctx;
    return ctx.get_ctx();
}
