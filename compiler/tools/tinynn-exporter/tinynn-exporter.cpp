/**
 * \file compiler/tools/tinynn-exporter/tinynn-exporter.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "llvm/Support/CommandLine.h"
#include "mlir/Parser.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Target/TinyNN/export.h"

using namespace llvm;

cl::opt<std::string> InputFile(cl::Positional, cl::Required,
                               cl::desc("<input kernel dialect file>"));
cl::opt<std::string> OutputDir(cl::Positional, cl::Required,
                               cl::desc("<output dir>"));
cl::opt<bool> Verbose(
        "verbose", cl::desc("log more detail information when compiler model"));
cl::opt<bool> SaveModel("save-model", cl::desc("save model to c file"));

cl::opt<bool> EnableCompressWeightToFp16(
        "enable_compress_fp16",
        cl::desc("enable compress model weight from fp32 to fp16, enable this "
                 "may effect model precision."));

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    cl::ParseCommandLineOptions(argc, argv);
    if (Verbose) {
        megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    }
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::Kernel::KernelDialect>();

    llvm::outs() << "Parse Source File from " << InputFile.getValue() << "\n";
    mlir::OwningOpRef<mlir::ModuleOp> mod =
            mlir::parseSourceFile<mlir::ModuleOp>(InputFile.getValue(), &ctx);

    llvm::outs() << "Export tinynn model and kernel to dir "
                 << OutputDir.getValue() << "\n";
    mlir::KernelExporter kernel_exporter;
    mlir::export_tinynn_model(mod.get(), OutputDir.getValue() + "/model.tiny",
                              SaveModel, kernel_exporter,
                              EnableCompressWeightToFp16.getValue());
    llvm::outs() << "Export tinynn model and kernel done.\n";
    return 0;
}

// vim: syntax=cpp.doxygen
