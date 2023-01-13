/**
 * \file
 * compiler/lib/KernelGen/KernelGen.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <regex>
#include "Arm/Arm64/KernelPack.h"
#include "Arm/ArmCommon/KernelPack.h"
#include "Arm/Armv7/KernelPack.h"
#if MEGCC_ENABLE_MLIR_KERN_GEN
#include "AutoBareMetal/KernelPack.h"
#endif
#include "BareMetal/KernelPack.h"
#include "Common/DeduceLayoutMap.h"
#include "GeneralIntrinsic/KernelPack.h"
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;

//! DumpHelper
std::string DumpHelper::ARM64V7_COMMON_POSTFIX = "#x#";
std::string DumpHelper::ARM64V7_ARM64_POSTFIX =
        ARM64V7_COMMON_POSTFIX + "_arm64";
std::string DumpHelper::ARM64V7_ARMV7_POSTFIX =
        ARM64V7_COMMON_POSTFIX + "_armv7";

DeduceFunc* GetDeduceLayout(KernelPack::KernType kernel_type) {
    static DeduceLayoutMap deduce_map;
    return deduce_map.map[kernel_type].get();
}

std::pair<std::vector<const KernelFunc*>, const DeduceFunc*>
KernelPack::GetKernel(KernelPack::KernType kernel_type, Arch arch) {
    //! arm64v7 is used by tinycv, nn opr should be armv64 or armv7, not arm64v7
    auto deduce_func = GetDeduceLayout(kernel_type);
    if (arch == Arch::ARM64 || arch == Arch::ARM64V7) {
        auto a64_kerns = Arm64::ArchKernelPack::GetKernel(kernel_type);
        auto armcommon_kerns =
                ArmCommon::ArchKernelPack::GetKernel(kernel_type);
        auto gi_kerns =
                GeneralIntrinsic::ArchKernelPack::GetKernel(kernel_type);
        if (kernel_type == KernelPack::KernType::MatrixMulKernel) {
            armcommon_kerns.insert(armcommon_kerns.end(), a64_kerns.begin(),
                                   a64_kerns.end());
            armcommon_kerns.insert(armcommon_kerns.end(), gi_kerns.begin(),
                                   gi_kerns.end());
            return {armcommon_kerns, deduce_func};
        }

        std::vector<const KernelFunc*> valid_kern;
        if (kernel_type == KernelPack::KernType::ConvKernel) {
            std::vector<const KernelFunc*> sorted_kern(2);
            for (auto&& kern : gi_kerns) {
                auto kern_sym = kern->GetKernelSymbol(nullptr);
                auto is_f63 = std::regex_match(
                        kern_sym, std::regex("^GI.*_winograd_f63.*"));
                auto is_f43 = std::regex_match(
                        kern_sym, std::regex("^GI.*_winograd_f43.*"));
                auto if_match = is_f63 || is_f43;
                if (!if_match) {
                    valid_kern.push_back(kern);
                } else {
                    if (is_f43) {
                        sorted_kern[0] = kern;
                    } else {
                        sorted_kern[1] = kern;
                    }
                }
            }
            //! WARNING: the f63 and f43 must exist in GI kernel
            if (arch == Arch::ARM64) {
                a64_kerns.insert(a64_kerns.begin(), sorted_kern.begin(),
                                 sorted_kern.end());
            }
        } else {
            valid_kern = gi_kerns;
        }

        a64_kerns.insert(a64_kerns.end(), armcommon_kerns.begin(),
                         armcommon_kerns.end());
        a64_kerns.insert(a64_kerns.end(), valid_kern.begin(), valid_kern.end());
        return {a64_kerns, deduce_func};

    } else if (arch == Arch::ARMV7) {
        auto a32_kerns = Armv7::ArchKernelPack::GetKernel(kernel_type);

        auto armcommon_kerns =
                ArmCommon::ArchKernelPack::GetKernel(kernel_type);
        auto gi_kerns =
                GeneralIntrinsic::ArchKernelPack::GetKernel(kernel_type);
        a32_kerns.insert(a32_kerns.end(), armcommon_kerns.begin(),
                         armcommon_kerns.end());
        a32_kerns.insert(a32_kerns.end(), gi_kerns.begin(), gi_kerns.end());
        return {a32_kerns, deduce_func};
    }
#if MEGCC_ENABLE_MLIR_KERN_GEN
    else if (arch == Arch::AUTO_BAREMETAL) {
        auto auto_bare_kerns =
                AutoBareMetal::ArchKernelPack::GetKernel(kernel_type);

        return {auto_bare_kerns, deduce_func};
    }
#endif
    else {
        CC_ASSERT(arch == Arch::BAREMETAL);
        //! FIXME: the f43 f63 winograd matmul is using arm64 asm kernel, it is
        //! invalid for barmetal
        auto gi_kerns =
                GeneralIntrinsic::ArchKernelPack::GetKernel(kernel_type);
        std::vector<const KernelFunc*> valid_kern;
        if (kernel_type == KernelPack::KernType::ConvKernel) {
            for (auto&& kern : gi_kerns) {
                auto kern_sym = kern->GetKernelSymbol(nullptr);
                auto if_match =
                        std::regex_match(kern_sym,
                                         std::regex("^GI.*_winograd_f63.*")) ||
                        std::regex_match(kern_sym,
                                         std::regex("^GI.*_winograd_f43.*"));
                if (!if_match) {
                    valid_kern.push_back(kern);
                }
            }
        } else {
            valid_kern = gi_kerns;
        }

        auto naive_impl = BareMetal::ArchKernelPack::GetKernel(kernel_type);
        naive_impl.insert(naive_impl.begin(), valid_kern.begin(),
                          valid_kern.end());
        return {naive_impl, deduce_func};
    }
}

namespace megcc {
namespace KernelGen {
std::string GenCommonInclude() {
    return "#include <data_struct.h>";
}
std::string GenCommonCall() {
    return R"((
        Tensor** inputs, int nr_input, Tensor** outputs, int nr_output,
        const Workspace* workspace, const RuntimeOpt* opt))";
}
std::string GenCommonInitCall() {
    return R"((
        Tensor** inputs, int nr_input, Tensor* out_weights,
        int* nr_out_weight, const RuntimeOpt* opt))";
}
std::string GenCommonWorkspaceCall() {
    return R"((
        Tensor** inputs, int nr_input, int nr_thread, size_t* workspace))";
}
std::string GenCommonDeduceCall() {
    return R"((
        Tensor** inputs, int nr_input, Tensor** outputs, int nr_output))";
}
std::string GenCommonRet() {
    return "TinyNNStatus";
}
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
