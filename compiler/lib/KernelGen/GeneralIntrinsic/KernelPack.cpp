/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/KernelPack.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "KernelPack.h"
#include <memory>
#include "CVTranspose.h"
#include "ConvKernel/ConvKernel.h"
#include "CvtColor.h"
#include "Elemwise/Elemwise.h"
#include "Flip.h"
#include "FusedElemwiseKernel.h"
#include "InternalKernel/InternalKernel.h"
#include "MatMulKernel/Fp32MatMul.h"
#include "PoolingKernel/Pooling.h"
#include "Reduce.h"
#include "Relayout.h"
#include "Resize.h"
#include "Rotate.h"
#include "Typecvt.h"
#include "WarpAffine.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
struct AllGICommonKernel {
    AllGICommonKernel() {
        inner_map[KernelPack::KernType::ElemwiseKernel] = {
                std::make_shared<GeneralIntrinsic::ElemwiseKernel>()};
        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<GeneralIntrinsic::Fp32MatMulM4N8K4>(),
                std::make_shared<GeneralIntrinsic::Fp32GevmKernel>(),
                std::make_shared<GeneralIntrinsic::Fp32GemvKernel>(),
                std::make_shared<GeneralIntrinsic::Fp32GemvMk4Kernel>(),
                std::make_shared<GeneralIntrinsic::Fp32MatMulM4N12>(),
                std::make_shared<GeneralIntrinsic::Fp32MatMulM4N12K4>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<GeneralIntrinsic::MatmulM4N8MK4Kernel>(),
                std::make_shared<GeneralIntrinsic::MatmulM4N12Kernel>(),
                std::make_shared<GeneralIntrinsic::MatmulM4N12MK4Kernel>()};
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<GeneralIntrinsic::ChannelWiseFloatMk4>(),
                std::make_shared<GeneralIntrinsic::ConvFloatNCHWNCHW44>(),
                std::make_shared<GeneralIntrinsic::ConvIm2colFloat>(),
                std::make_shared<GeneralIntrinsic::WinogradFloatF23NCHW44>(),
                std::make_shared<GeneralIntrinsic::Conv1x1FloatMk4>()};

        inner_map[KernelPack::KernType::PoolingKernel] = {
                std::make_shared<GeneralIntrinsic::PoolingNchw44Fp32>(),
                std::make_shared<GeneralIntrinsic::PoolingNchw44QInt8>()};

        inner_map[KernelPack::KernType::TypeCvtKernel] = {
                std::make_shared<GeneralIntrinsic::TypecvtKernel>()};

        inner_map[KernelPack::KernType::ReduceKernel] = {
                std::make_shared<GeneralIntrinsic::ReduceKernel>()};

        inner_map[KernelPack::KernType::FlipKernel] = {
                std::make_shared<GeneralIntrinsic::FlipKernel>()};

        inner_map[KernelPack::KernType::ResizeKernel] = {
                std::make_shared<GeneralIntrinsic::ResizeKernel>()};

        inner_map[KernelPack::KernType::RelayoutKernel] = {
                std::make_shared<GeneralIntrinsic::RelayoutKernel>()};

        inner_map[KernelPack::KernType::CVTransposeKernel] = {
                std::make_shared<GeneralIntrinsic::CvTransposeKernel>()};

        inner_map[KernelPack::KernType::RotateKernel] = {
                std::make_shared<GeneralIntrinsic::RotateKernel>()};

        inner_map[KernelPack::KernType::CvtColorKernel] = {
                std::make_shared<GeneralIntrinsic::CvtColorKernel>()};

        inner_map[KernelPack::KernType::WarpAffineKernel] = {
                std::make_shared<GeneralIntrinsic::WarpAffineKernel>()};

        inner_map[KernelPack::KernType::FusedElemwiseKernel] = {
                std::make_shared<GeneralIntrinsic::FusedElmwiseKernel>()};
    }

    std::unordered_map<KernelPack::KernType,
                       std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};
}  // namespace

std::vector<const KernelFunc*> ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllGICommonKernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}

// vim: syntax=cpp.doxygen
