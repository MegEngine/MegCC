/**
 * \file compiler/tools/kernel_exporter/exporter_imp.h
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

#include "compiler/Common/Logger.h"
#include "compiler/Common/Version.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace llvm;

class KernelExporter {
    static std::map<std::string, KPT> m_kern_name2type;
    static std::map<std::string, KA> m_name2arch;

    std::string m_kernel_name;
    std::string m_kernel_arch;
    bool m_use_default_attr;
    bool m_print_to_console;

    KPT kernel_name_to_type();
    KA get_arch_type();

    std::pair<std::vector<const megcc::KernelGen::KernelFunc*>,
              const megcc::KernelGen::DeduceFunc*>
    get_kernels();

public:
    KernelExporter(std::string kernel_name, std::string kernel_arch,
                   bool use_default_attr, bool print_to_console)
            : m_kernel_name{kernel_name},
              m_kernel_arch{kernel_arch},
              m_use_default_attr(use_default_attr),
              m_print_to_console(print_to_console) {
        std::string attr = "use kernel default attr";
        if (!m_use_default_attr) {
            attr = "use user config attr";
        }
        llvm::outs() << "try export tinynn kernel of " << m_kernel_name << "("
                     << m_kernel_arch << ")"
                     << "\n";
        llvm::outs() << "kernel attr: " << attr << "\n";
        llvm::outs() << "print to console: " << m_print_to_console << "\n";
        megcc::setAssertThrow(true);
    };

#define MAPKEY2STR(m)   \
    std::string ret;    \
    for (auto i : m) {  \
        ret += i.first; \
        ret += "\n";    \
    }                   \
    return ret;

    static std::string support_kernels() { MAPKEY2STR(m_kern_name2type); }

    static std::string support_archs() { MAPKEY2STR(m_name2arch); }

    void gen_kenrels();
};
