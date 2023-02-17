/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/KernelPack.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Arm/Armv7/KernelPack.h"
#include <memory>
#include "Arm/Armv7/ConvKernel/ConvKernel.h"
#include "InternalKernel/InternalKernel.h"
#include "MatMulKernel/Fp32MatMul.h"

using namespace megcc;
using namespace KernelGen;
using namespace Armv7;

struct AllA32Kernel {
    AllA32Kernel() {
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<Armv7::ConvFloatNCHWNCHW443x3s2>(),
                std::make_shared<Armv7::Conv1x1FloatMk4>(),
                std::make_shared<Armv7::WinogradFloatF23NCHW44>(),
                std::make_shared<Armv7::ConvIm2colFloat>()};

        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<Armv7::Fp32MatMulM4N8K4>(),
                std::make_shared<Armv7::Fp32MatMulM4N12K4>(),
                std::make_shared<Armv7::Fp32MatMulM4N12>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<Armv7::MatmulM4N8MK4Kernel>(),
                std::make_shared<Armv7::MatmulM4N12MK4Kernel>(),
                std::make_shared<Armv7::MatmulM4N12Kernel>()};
    }
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};

std::vector<const KernelFunc*> Armv7::ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllA32Kernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}
// vim: syntax=cpp.doxygen
