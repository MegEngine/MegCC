/**
 * \file compiler/include/compiler/Target/TinyNN/export.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include <set>
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
extern llvm::cl::opt<megcc::KernelGen::Arch> target_arch;

namespace mlir {

class KernelExporter {
public:
    struct FuncUnit {
        std::string symbol;
        std::string signature;
        std::string body;
        std::string guard_begin;
        std::string guard_end;
    };

    struct Config {
        std::vector<FuncUnit> cvkernels;
        std::vector<FuncUnit> kernels;
        std::vector<FuncUnit> deduce_shape;
        std::vector<FuncUnit> workspaces;
        std::vector<FuncUnit> inits;
        std::vector<FuncUnit> internal_kernels;
    } config;

    void write(std::string save_path);

    void addInst(std::string macro_type) { used_inst.insert(macro_type); }

    void addKernel(llvm::StringRef symbol, llvm::StringRef sig,
                   llvm::StringRef body, llvm::StringRef guard_begin,
                   llvm::StringRef guard_end) {
        if (symbol2kernel_id.find(symbol.str()) == symbol2kernel_id.end()) {
            config.kernels.push_back({symbol.str(), sig.str(), body.str(),
                                      guard_begin.str(), guard_end.str()});
            symbol2kernel_id[symbol.str()] = m_kernel_cnt++;
        }
    }

    void addCVKernel(llvm::StringRef symbol, llvm::StringRef sig,
                     llvm::StringRef body) {
        config.cvkernels.push_back({symbol.str(), sig.str(), body.str()});
    }

    void addInitFunc(llvm::StringRef symbol, llvm::StringRef sig,
                     llvm::StringRef body, llvm::StringRef kern_symbol,
                     llvm::StringRef guard_begin, llvm::StringRef guard_end) {
        if (symbol2init_id.find(kern_symbol.str()) == symbol2init_id.end()) {
            config.inits.push_back({symbol.str(), sig.str(), body.str(),
                                    guard_begin.str(), guard_end.str()});
            symbol2init_id[kern_symbol.str()] = m_init_cnt++;
        }
    }

    void addDeduceShapeKernel(llvm::StringRef symbol, llvm::StringRef sig,
                              llvm::StringRef body,
                              llvm::StringRef kern_symbol) {
        if (symbol2deduce_id.find(symbol.str()) == symbol2deduce_id.end()) {
            config.deduce_shape.push_back(
                    {symbol.str(), sig.str(), body.str()});
            symbol2deduce_id[symbol.str()] = m_deduce_cnt++;
        }
        kern_symbol2deduce_id[kern_symbol.str()] =
                symbol2deduce_id[symbol.str()];
    }

    void addInternalKernel(llvm::StringRef symbol, llvm::StringRef sig,
                           llvm::StringRef body, llvm::StringRef guard_begin,
                           llvm::StringRef guard_end) {
        if (symbol2internel_id.find(symbol.str()) == symbol2internel_id.end()) {
            config.internal_kernels.push_back({symbol.str(), sig.str(),
                                               body.str(), guard_begin.str(),
                                               guard_end.str()});
            symbol2internel_id[symbol.str()] = m_internel_cnt++;
        }
    }
    int get_kernel_id(std::string kernel_name) const {
        CC_ASSERT(symbol2kernel_id.find(kernel_name) != symbol2kernel_id.end())
                << "can not find kernel id of " << kernel_name << "\n";
        return symbol2kernel_id.at(kernel_name);
    }
    int get_init_id(std::string kernel_name) const {
        CC_ASSERT(symbol2init_id.find(kernel_name) != symbol2init_id.end())
                << "can not find init kernel id of " << kernel_name << "\n";
        return symbol2init_id.at(kernel_name);
    }
    int get_deduce_id(std::string kernel_name) const {
        CC_ASSERT(kern_symbol2deduce_id.find(kernel_name) !=
                  kern_symbol2deduce_id.end())
                << "can not find deduce kernel id of " << kernel_name << "\n";
        return kern_symbol2deduce_id.at(kernel_name);
    }

private:
    int m_kernel_cnt{0};
    int m_init_cnt{0};
    int m_deduce_cnt{0};
    int m_internel_cnt{0};

    std::unordered_map<std::string, int> symbol2kernel_id;
    std::unordered_map<std::string, int> symbol2init_id;
    std::unordered_map<std::string, int> kern_symbol2deduce_id;
    std::unordered_map<std::string, int> symbol2deduce_id;
    std::unordered_map<std::string, int> symbol2internel_id;
    std::set<std::string> used_inst;
};

void export_tinynn_model(ModuleOp top_module, std::string save_path,
                         const bool save_model_as_symbol,
                         KernelExporter& kernel_exporter);

}  // namespace mlir

// vim: syntax=cpp.doxygen
