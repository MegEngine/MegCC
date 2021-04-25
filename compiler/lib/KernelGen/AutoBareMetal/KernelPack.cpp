/**
 * \file
 * compiler/lib/KernelGen/AutoBareMetal/KernelPack.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <memory>
#include "ElemwiseKernel.h"
#include "KernelPack.h"
#include "MatmulKernel.h"

using namespace megcc;
using namespace KernelGen;
using namespace AutoBareMetal;
namespace {
struct AllAutoKernel {
    AllAutoKernel() {
        inner_map[KernelPack::KernType::ElemwiseKernel] = {
                std::make_shared<AutoBareMetal::ElmwiseKernel>()};
        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<AutoBareMetal::MatmulKernel>()};
    }

    std::unordered_map<KernelPack::KernType,
                       std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};
}  // namespace
std::vector<const KernelFunc*> ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllAutoKernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}

// vim: syntax=cpp.doxygen
