/**
 * \file compiler/tools/kernel_exporter/exporter_imp.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "exporter_imp.h"

KPT KernelExporter::kernel_name_to_type() {
    KPT ret;
    auto m_find = m_kern_name2type.find(m_kernel_name);
    if (m_find == m_kern_name2type.end()) {
        EXPORT_ERR(
                ssprintf("do not support kernel name: %s, support lists:\n%s",
                         m_kernel_name.c_str(), support_kernels().c_str()));
    } else {
        ret = m_find->second;
    }

    return ret;
}

KA KernelExporter::get_arch_type() {
    KA ret;
    auto m_find = m_name2arch.find(m_kernel_arch);
    if (m_find == m_name2arch.end()) {
        EXPORT_ERR(ssprintf("do not support arch: %s, support archs:\n%s",
                            m_kernel_arch.c_str(), support_archs().c_str()));
    } else {
        ret = m_find->second;
    }

    return ret;
}

std::pair<std::vector<const megcc::KernelGen::KernelFunc*>,
          const megcc::KernelGen::DeduceFunc*>
KernelExporter::get_kernels() {
    KPT k_type = kernel_name_to_type();
    KA arch_type = get_arch_type();
    return megcc::KernelGen::KernelPack::GetKernel(k_type, arch_type);
}

void KernelExporter::gen_kenrels() {
    auto kernels = get_kernels().first;
    if (kernels.size() <= 0) {
        EXPORT_ERR(ssprintf("ERR: can not find any KernelFunc for: %s",
                            m_kernel_name.c_str()));
    }

    auto attrs = megcc::exporter::config_attr(
            kernel_name_to_type(), m_kernel_name, m_use_default_attr);
    std::string common_header = R"(
#include <data_struct.h>
#include <math.h>
#include <stdio.h>
)";
    for (auto& i : kernels) {
        for (auto& ctx : attrs) {
            auto gen = [&]() {
                bool is_cv = !i->GetCVKernelSymbol(&ctx).empty();
                auto kernel_file_name = i->GetKernelSymbol(&ctx) + ".c";
                if (is_cv) {
                    kernel_file_name = i->GetCVKernelSymbol(&ctx) + ".c";
                }
                std::stringstream ss;
                auto file_path = kernel_file_name;
                llvm::outs() << "\n";
                llvm::outs() << "\n";
                ss << common_header;
                if (is_cv) {
                    ss << i->GetCVKernelBody(&ctx) << "\n";
                } else {
                    ss << i->GetKernelBody(&ctx) << "\n";
                    for (auto& d : i->GetDependInternalSymbol(&ctx)) {
                        ss << d.kernel_body;
                    }
                }
                if (m_print_to_console) {
                    std::cout << ss.rdbuf() << "\n";
                };
                std::ofstream out_file(file_path);
                out_file << ss.str();
                out_file.close();
                llvm::outs() << "====>get kernel to: " << file_path << "\n";
            };

            try {
                gen();
            } catch (...) {
            }
        }
    }

    llvm::outs() << "Export tinynnkernel done.\n";
}

std::map<std::string, KPT> KernelExporter::m_kern_name2type{
        {"ConvKernel", KPT::ConvKernel},
        {"ElemwiseKernel", KPT::ElemwiseKernel},
        {"ElemwiseMultiKernel", KPT::ElemwiseMultiKernel},
        {"PoolingKernel", KPT::PoolingKernel},
        {"MatrixMulKernel", KPT::MatrixMulKernel},
        {"MatrixInvKernel", KPT::MatrixInvKernel},
        {"RelayoutKernel", KPT::RelayoutKernel},
        {"ReduceKernel", KPT::ReduceKernel},
        {"IndexingMultiAxisKernel", KPT::IndexingMultiAxisKernel},
        {"IndexingOneHotKernel", KPT::IndexingOneHotKernel},
        {"WarpPerspectiveKernel", KPT::WarpPerspectiveKernel},
        {"WarpAffineKernel", KPT::WarpAffineKernel},
        {"TypeCvtKernel", KPT::TypeCvtKernel},
        {"TopK", KPT::TopK},
        {"BatchMatmulKernel", KPT::BatchMatmulKernel},
        {"PowCKernel", KPT::PowCKernel},
        {"CVTransposeKernel", KPT::CVTransposeKernel},
        {"FlipKernel", KPT::FlipKernel},
        {"ResizeKernel", KPT::ResizeKernel},
        {"RotateKernel", KPT::RotateKernel},
        {"RoiCopyKernel", KPT::RoiCopyKernel},
        {"CvtColorKernel", KPT::CvtColorKernel},
        {"ArgSortKernel", KPT::ArgSortKernel},
        {"ArgmaxKernel", KPT::ArgmaxKernel},
        {"ConcatKernel", KPT::ConcatKernel},
        {"ConvBackDataKernel", KPT::ConvBackDataKernel}

};

std::map<std::string, KA> KernelExporter::m_name2arch{
        {"BAREMETAL", KA::BAREMETAL},
        {"ARM64", KA::ARM64},
        {"ARMV7", KA::ARMV7},
};
