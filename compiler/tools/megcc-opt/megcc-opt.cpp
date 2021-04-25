/**
 * \file compiler/tools/megcc-opt/megcc-opt.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/Conversion/MGBToKernel/MGBToKernel.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/Dialect/Kernel/Transforms/Passes.h"
#include "compiler/Dialect/MGB/IR/MGBDialect.h"
#include "compiler/Dialect/MGB/Transforms/Passes.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char** argv) {
    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    registerMGBTransformsPasses();
    registerTransformsPasses();
    registerMGBToKernelPasses();
    registerKernelTransformsPasses();

    bufferization::registerBufferizationPasses();
    DialectRegistry registry;
    registry.insert<StandardOpsDialect, memref::MemRefDialect, MGB::MGBDialect,
                    bufferization::BufferizationDialect,
                    Kernel::KernelDialect>();
    return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                              registry,
                              /*preloadDialectsInContext=*/false));
}
