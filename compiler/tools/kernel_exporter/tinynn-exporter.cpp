/**
 * \file compiler/tools/kernel_exporter/tinynn-exporter.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <fstream>

#include "config_attr.h"
#include "utils.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser.h"

#include "exporter_imp.h"

using namespace llvm;

int main(int argc, char** argv) {
    auto k_name_desc = "input kernel name, valid option:\n" +
                       KernelExporter::support_kernels();
    cl::opt<std::string> KernelName("kernel", cl::Required,
                                    cl::desc(k_name_desc));
    auto arch_desc = "the platform arch, valid options:\n" +
                     KernelExporter::support_archs();
    cl::opt<std::string> kernelArch("arch", cl::Required, cl::desc(arch_desc));
    cl::opt<bool> use_default_attr(
            "use_default_attr",
            cl::desc("Use a default attribute to generate kernel, if not "
                     "config, user need dynamic config it"));
    cl::opt<bool> print_to_console("print_to_console",
                                   cl::desc("Print kernel body to console"));
    cl::opt<bool> Verbose(
            "verbose",
            cl::desc("log more detail information when compiler model"));

    cl::AddExtraVersionPrinter(
            [](raw_ostream& oss) { oss << megcc::getMegccVersionString(); });
    cl::ParseCommandLineOptions(argc, argv);
    if (Verbose) {
        megcc::SetLogLevel(megcc::LogLevel::DEBUG);
    }
    KernelExporter exporter(KernelName.getValue(), kernelArch.getValue(),
                            use_default_attr.getValue(),
                            print_to_console.getValue());
    exporter.gen_kenrels();

    return 0;
}
