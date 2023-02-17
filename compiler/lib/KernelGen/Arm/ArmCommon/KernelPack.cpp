/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/KernelPack.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "KernelPack.h"
#include <memory>
#include "CVTranspose.h"
#include "ConvKernel.h"
#include "CvtColor.h"
#include "Elemwise/Elemwise.h"
#include "Flip.h"
#include "InternalKernel.h"
#include "MatMulKernel/Fp32Gemv.h"
#include "MatMulKernel/Fp32Gevm.h"
#include "Pooling.h"
#include "Reduce.h"
#include "Relayout.h"
#include "Resize.h"
#include "Rotate.h"
#include "Typecvt.h"
#include "WarpAffine.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;
namespace {
struct AllArmCommonKernel {
    AllArmCommonKernel() {
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<ArmCommon::ChannelWiseFloatMk4>(),
                std::make_shared<ArmCommon::ConvFloatNCHWNCHW44>(),
        };

        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<ArmCommon::Fp32GevmKernel>(),
                std::make_shared<ArmCommon::Fp32GemvKernel>()};

        inner_map[KernelPack::KernType::PoolingKernel] = {
                std::make_shared<ArmCommon::PoolingNchw44Fp32>(),
                std::make_shared<ArmCommon::PoolingNchw44QInt8>()};

        inner_map[KernelPack::KernType::ResizeKernel] = {
                std::make_shared<ArmCommon::ResizeKernel>()};

        inner_map[KernelPack::KernType::RelayoutKernel] = {
                std::make_shared<ArmCommon::RelayoutKernel>()};

        inner_map[KernelPack::KernType::WarpAffineKernel] = {
                std::make_shared<ArmCommon::WarpAffineKernel>()};

        inner_map[KernelPack::KernType::TypeCvtKernel] = {
                std::make_shared<ArmCommon::TypecvtKernel>()};

        inner_map[KernelPack::KernType::ReduceKernel] = {
                std::make_shared<ArmCommon::ReduceKernel>()};

        inner_map[KernelPack::KernType::CvtColorKernel] = {
                std::make_shared<ArmCommon::CvtColorKernel>()};

        inner_map[KernelPack::KernType::CVTransposeKernel] = {
                std::make_shared<ArmCommon::CvTransposeKernel>()};

        inner_map[KernelPack::KernType::FlipKernel] = {
                std::make_shared<ArmCommon::FlipKernel>()};

        inner_map[KernelPack::KernType::RotateKernel] = {
                std::make_shared<ArmCommon::RotateKernel>()};

        inner_map[KernelPack::KernType::ElemwiseKernel] = {
                std::make_shared<ArmCommon::ElemwiseKernel>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<ArmCommon::ExpNeonKernel>()};
    }

    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};
}  // namespace

std::vector<const KernelFunc*> ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllArmCommonKernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}

// vim: syntax=cpp.doxygen
