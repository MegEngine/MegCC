/**
 * \file
 * compiler/lib/KernelGen/KernelGen.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

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
        a64_kerns.insert(a64_kerns.end(), armcommon_kerns.begin(),
                         armcommon_kerns.end());
        a64_kerns.insert(a64_kerns.end(), gi_kerns.begin(), gi_kerns.end());
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
        auto gi_kerns =
                GeneralIntrinsic::ArchKernelPack::GetKernel(kernel_type);
        auto naive_impl = BareMetal::ArchKernelPack::GetKernel(kernel_type);
        naive_impl.insert(naive_impl.begin(), gi_kerns.begin(), gi_kerns.end());
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
