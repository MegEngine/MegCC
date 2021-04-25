/**
 * \file compiler/lib/CodeGen/AutoKernelFunc.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <unistd.h>
#include <sstream>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "AutoKernelFunc.h"
#include "CodeGenUtil.h"
#include "GlobalCtx.h"

#include <unistd.h>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Translation.h"

using namespace megcc;
using namespace KernelGen;
using namespace mlir;

KernelGen::KernelObj codegen::AutoKernelFuncInternal::GetKernelObj(
        TContext* context) const {
    auto ctx = codegen::getGlobalCTX();
    mlir::OwningOpRef<mlir::ModuleOp> mod =
            mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx));
    mlir::OpBuilder op_builder(ctx);
    op_builder.setInsertionPointToEnd(mod->getBody());
    auto func_name = GetKernelSymbol(context);
    auto func = op_builder.create<FuncOp>(op_builder.getUnknownLoc(), func_name,
                                          get_func_type_memref(context, ctx));
    Block* entryBlock = func.addEntryBlock();
    op_builder.setInsertionPointToStart(entryBlock);
    CreateCompute(entryBlock, op_builder, ctx, context);

    mlir::PassManager pm(ctx);
    CreatePass(pm, ctx, context);
    if (failed(pm.run(mod.get()))) {
        CC_ABORT << "pass error\n";
    }
    return CompileKernel(mod.get(), func_name);
}

KernelGen::KernelObj codegen::AutoKernelFuncInternal::CompileKernel(
        mlir::ModuleOp mod, std::string func_name) const {
    KernelGen::KernelObj rst;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    std::vector<std::string> libs;
    auto opt_pipeline = mlir::makeOptimizingTransformer(3, 1, 0);
    auto&& mb_engine = mlir::ExecutionEngine::create(
            mod, nullptr, opt_pipeline, llvm::None,
            std::vector<llvm::StringRef>(libs.begin(), libs.end()), true,
            false);

    CC_ASSERT(mb_engine && "Error can't create engine\n");
    std::unique_ptr<mlir::ExecutionEngine> my_engine = std::move(*mb_engine);

    auto lkup_rst = my_engine->lookup(func_name);
    CC_ASSERT(lkup_rst && "lkup_rst ");
    char temp_filename[100] = "/tmp/megcc_obj.XXXXXXX";
    CC_ASSERT(mkstemp(temp_filename));
    CC_ASSERT(unlink(temp_filename) == 0);

    my_engine->dumpToObjectFile(temp_filename);
    std::ifstream obj_input(temp_filename, std::ios::binary);

    // copies all data into buffer
    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(obj_input), {});
    rst.kernel_symbol = func_name;
    rst.kernel_bin = buffer;
    return rst;
}