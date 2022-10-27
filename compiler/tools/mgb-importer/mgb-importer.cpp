/**
 * \file compiler/tools/mgb-importer/mgb-importer.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/StringExtras.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Target/MGB/import.h"

using namespace llvm;

cl::opt<std::string> InputFile(cl::Positional, cl::Required,
                               cl::desc("<input mgb model>"));
cl::opt<std::string> OutputFile(cl::Positional, cl::Required,
                                cl::desc("<output mlir file>"));
cl::opt<std::string> InputShapes(
        "input-shapes", cl::Optional, cl::desc("modify input shapes"),
        cl::value_desc("name0=(xx0,yy0);name1=(xx1,yy1,zz1)"));
cl::opt<bool> Verbose(
        "verbose", cl::desc("log more detail information when compiler model"));
cl::opt<bool> Enable_nchw44("enable_nchw44", cl::desc("enable nchw44 trans"));
cl::opt<bool> Enable_nchw44_dot("enable_nchw44_dot",
                                cl::desc("enable nchw44-dot trans"));
cl::opt<bool> Add_nhwc2nchw_to_input(
        "add_nhwc2nchw_to_input",
        cl::desc("add nhwc2nchw dimshuffle to input"));
cl::opt<bool> Enable_convbias_fusez("enable_convbias_fusez",
                                    cl::desc("enable convbias_fusez trans"));

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    cl::ParseCommandLineOptions(argc, argv);
    if (Verbose) {
        megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    }
    mlir::MLIRContext ctx;
    mlir::MGB::MGBImporterOptions options;
    options.graph_opt_level = 2;
    options.use_static_memory_plan = false;
    options.enable_nchw44 = Enable_nchw44;
    options.enable_nchw44_dot = Enable_nchw44_dot;
    options.add_nhwc2nchw_to_input = Add_nhwc2nchw_to_input;
    options.enable_fuse_conv_bias_nonlinearity_z = Enable_convbias_fusez;
    if (failed(parseInputShapes(InputShapes.getValue(), options))) {
        CC_ABORT << "parseInputShapes error\n";
        return -1;
    }
    llvm::SmallVector<llvm::StringRef> names;
    llvm::SplitString(OutputFile, names, ".");
    options.module_name = names[0].str();
    llvm::outs() << "Import mgb/mge model from " << InputFile.getValue()
                 << "\n";
    mlir::OwningOpRef<mlir::ModuleOp> mod =
            mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    auto status =
            mlir::MGB::import_mgb(mod.get(), InputFile.getValue(), options);
    if (mlir::failed(status)) {
        llvm::errs() << "import megengine model failed\n";
        return -1;
    }
    std::error_code EC;
    llvm::raw_fd_stream FileStream(OutputFile.getValue(), EC);
    llvm::outs() << "Export mgb dialect to " << OutputFile.getValue() << "\n";
    mod->print(FileStream);
    llvm::outs() << "Mgb/mge convert to mgb dialect done.\n";
    return 0;
}

// vim: syntax=cpp.doxygen
