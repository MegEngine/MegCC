/**
 * \file compiler/tools/tinynn-exporter/tinynn-exporter.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Conversion/MGBToKernel/MGBToKernel.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Dialect/MGB/Transforms/Passes.h"
#include "compiler/KernelGen/KernelGen.h"
#include "compiler/Target/MGB/import.h"
#include "compiler/Target/TinyNN/export.h"

using namespace llvm;

cl::opt<std::string> InputFile(cl::Positional, cl::Optional,
                               cl::desc("<input megengine cpp model>"));
cl::opt<std::string> OutputDir(
        cl::Positional, cl::Optional,
        cl::desc("<output dir for tinynn model and generated kernels>"));
cl::opt<std::string> dumpDir("dump", cl::Optional,
                             cl::desc("<override output dir in json for tinynn "
                                      "model and generated kernels>"));
cl::opt<std::string> InputShapes(
        "input-shapes", cl::Optional, cl::desc("modify input shapes"),
        cl::value_desc("name0=(xx0,yy0);name1=(xx1,yy1,zz1)"));
cl::opt<bool> Verbose(
        "verbose", cl::desc("log more detail information when compiler model"));
cl::opt<bool> EnableNchw44("enable_nchw44", cl::desc("enable nchw44 trans"));
cl::opt<bool> EnableNchw44Dot("enable_nchw44_dot",
                              cl::desc("enable nchw44-dot trans"));
cl::opt<bool> MGBFuseKernel("mgb_fuse_kernel",
                            cl::desc("fuse mgb kernel as possible"));
cl::opt<bool> SaveModel("save-model", cl::desc("save model to c"));
cl::opt<bool> Add_nhwc2nchw_to_input(
        "add_nhwc2nchw_to_input",
        cl::desc("add nhwc2nchw dimshuffle to input"));

cl::opt<std::string> JsonFile("json", cl::Optional,
                              cl::desc("config app by json"),
                              cl::value_desc("<path/to/json/file>"));

extern llvm::cl::opt<megcc::KernelGen::Arch> target_arch;
struct DumpJson {
    struct ModelJson {
        ModelJson() {
            str_options["model_name"] = "";
            str_options["model_path"] = "";
            str_options["input_shape_str"] = "";
            bool_options["enable_nchw44"] = false;
            bool_options["enable_nchw44_dot"] = false;
            bool_options["add_nhwc2nchw_to_input"] = false;
            bool_options["mgb_fuse_kernel"] = false;
        );
        if (mlir::failed(status)) {
            llvm::outs() << "import megengine model failed\n";
            return -1;
        }
        mlir::PassManager pm(&ctx);
        if (model_mgb_fuse_kernel) {
            pm.addNestedPass<mlir::FuncOp>(mlir::createMGBFuseKernelPass());
        }
        pm.addPass(mlir::createMGBToKernelPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::createMemoryForwardingPass());
        pm.addPass(mlir::createKernelMaterializationPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::createStaticMemoryPlanningPass());
        pm.addNestedPass<mlir::FuncOp>(mlir::createKernelFinalCleanPass());
        //! Now all the memory is allocated in runtime, the Deallocation
        //! instruction is not used.
        // pm.addNestedPass<mlir::FuncOp>(mlir::createBufferDeallocationPass());
        pm.addNestedPass<mlir::FuncOp>(
                mlir::bufferization::createFinalizingBufferizePass());
        llvm::outs() << "Apply createMGBToKernelPass and "
                        "createKernelMaterializationPass to the dialect.\n";
        if (failed(pm.run(mod.get()))) {
            return -1;
        }
        llvm::outs() << "Export tinynn model and kernel to dir " << dump_dir
                     << "\n";
        mlir::export_tinynn_model(
                mod.get(), dump_dir + "/" + options.module_name + ".tiny",
                SaveModel, kernel_exporter);
        llvm::outs() << "Mgb/mge model convert to tinynn model "
                     << options.module_name << " done.\n";
    }
    export_cv_opr(kernel_exporter, dump_info->cv_impl);
    kernel_exporter.write(dump_dir);
    llvm::outs() << "Mgb/mge model convert to tinynn kernel done.\n";
    return 0;
}

// vim: syntax=cpp.doxygen
