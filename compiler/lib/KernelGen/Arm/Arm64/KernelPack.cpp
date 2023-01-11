/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/KernelPack.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <memory>

#include "ConvKernel.h"
#include "Elemwise/Elemwise.h"
#include "InternalKernel/InternalKernel.h"
#include "KernelPack.h"
#include "MatMulKernel/Fp32MatMul.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
namespace {
struct AllA64Kernel {
    AllA64Kernel() {
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<Arm64::Conv1x1FloatMk4>(),
                std::make_shared<Arm64::WinogradFloatF23Nchw44>(),
                std::make_shared<Arm64::ChannelWiseInt8Mk4K3>(),
                std::make_shared<Arm64::ConvIm2colFloat>(),
                std::make_shared<Arm64::Conv1x1DotMk4>(),
                std::make_shared<Arm64::ConvIm2colDot>(),
                std::make_shared<Arm64::ConvDotNCHWNCHW44>()};

        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<Arm64::Fp32MatMulM8N12>(),
                std::make_shared<Arm64::Fp32MatMulM8N12K4>(),
                std::make_shared<Arm64::Fp32MatMulM4N16K4>(),
                std::make_shared<Arm64::Int8DotMatMulM8N12K4>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<Arm64::MatmulM4N16MK4Kernel>(),
                std::make_shared<Arm64::MatmulM8N12Kernel>(),
                std::make_shared<Arm64::MatmulM8N12MK4Kernel>(),
                std::make_shared<Arm64::MatmulInt8DotM8N12MK4Kernel>()};

        inner_map[KernelPack::KernType::ElemwiseKernel] = {
                std::make_shared<Arm64::ElemwiseKernel>()};
    }
    std::unordered_map<KernelPack::KernType,
                       std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};
}  // namespace
std::vector<const KernelFunc*> ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllA64Kernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}

// vim: syntax=cpp.doxygen
