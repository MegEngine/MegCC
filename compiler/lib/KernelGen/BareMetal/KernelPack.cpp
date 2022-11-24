/**
 * \file
 * compiler/lib/KernelGen/BareMetal/KernelPack.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <memory>
#include "Argmax.h"
#include "Argsort.h"
#include "BatchedMatmul.h"
#include "CVTranspose.h"
#include "Concat.h"
#include "ConvKernel.h"
#include "CvtColor.h"
#include "ElemwiseKernel.h"
#include "ElemwiseMultiType.h"
#include "Flip.h"
#include "Fp32Gemv.h"
#include "Fp32Gevm.h"
#include "FusedElemwiseKernel.h"
#include "IndexingMultiAxisVec.h"
#include "IndexingOneHot.h"
#include "KernelPack.h"
#include "MatrixInv.h"
#include "MatrixMul.h"
#include "Pooling.h"
#include "PowC.h"
#include "Reduce.h"
#include "Relayout.h"
#include "Resize.h"
#include "RoiCopy.h"
#include "Rotate.h"
#include "Topk.h"
#include "Typecvt.h"
#include "WarpAffine.h"
#include "WarpPerspective.h"
#include "ConvBackDataKernel.h"
using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;
namespace {
struct AllBareKernel {
    AllBareKernel() {
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<BareMetal::ConvGeneral>()};
        inner_map[KernelPack::KernType::ElemwiseKernel] = {
                std::make_shared<BareMetal::ElmwiseKernel>()};
        inner_map[KernelPack::KernType::ElemwiseMultiKernel] = {
                std::make_shared<BareMetal::ElemwiseMultiTypeKernel>()};
        inner_map[KernelPack::KernType::PoolingKernel] = {
                std::make_shared<BareMetal::PoolingKernel>()};
        inner_map[KernelPack::KernType::MatrixInvKernel] = {
                std::make_shared<BareMetal::MatrixInvKernel>()};
        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<BareMetal::Fp32GemvKernel>(),
                std::make_shared<BareMetal::Fp32GevmKernel>(),
                std::make_shared<BareMetal::MatrixMulKernel>()};
        inner_map[KernelPack::KernType::IndexingMultiAxisKernel] = {
                std::make_shared<BareMetal::IndexingMultiAxisKernel>()};
        inner_map[KernelPack::KernType::IndexingOneHotKernel] = {
                std::make_shared<BareMetal::IndexingOneHotKernel>()};
        inner_map[KernelPack::KernType::BatchMatmulKernel] = {
                std::make_shared<BareMetal::BatchedMatrixMulKernel>()};
        inner_map[KernelPack::KernType::ReduceKernel] = {
                std::make_shared<BareMetal::ReduceKernel>()};
        inner_map[KernelPack::KernType::TypeCvtKernel] = {
                std::make_shared<BareMetal::TypecvtKernel>()};
        inner_map[KernelPack::KernType::TopK] = {
                std::make_shared<BareMetal::TopkKernel>()};
        inner_map[KernelPack::KernType::PowCKernel] = {
                std::make_shared<BareMetal::PowCKernel>()};
        inner_map[KernelPack::KernType::WarpPerspectiveKernel] = {
                std::make_shared<BareMetal::WarpPerspectiveKernel>()};
        inner_map[KernelPack::KernType::WarpAffineKernel] = {
                std::make_shared<BareMetal::WarpAffineKernel>()};
        inner_map[KernelPack::KernType::RelayoutKernel] = {
                std::make_shared<BareMetal::RelayoutKernel>()};
        inner_map[KernelPack::KernType::CVTransposeKernel] = {
                std::make_shared<BareMetal::CvTransposeKernel>()};
        inner_map[KernelPack::KernType::FlipKernel] = {
                std::make_shared<BareMetal::FlipKernel>()};
        inner_map[KernelPack::KernType::ResizeKernel] = {
                std::make_shared<BareMetal::ResizeKernel>()};
        inner_map[KernelPack::KernType::RotateKernel] = {
                std::make_shared<BareMetal::RotateKernel>()};
        inner_map[KernelPack::KernType::RoiCopyKernel] = {
                std::make_shared<BareMetal::RoiCopyKernel>()};
        inner_map[KernelPack::KernType::CvtColorKernel] = {
                std::make_shared<BareMetal::CvtColorKernel>()};
        inner_map[KernelPack::KernType::ArgSortKernel] = {
                std::make_shared<BareMetal::ArgSortKernel>()};
        inner_map[KernelPack::KernType::ConcatKernel] = {
                std::make_shared<BareMetal::ConcatKernel>()};
        inner_map[KernelPack::KernType::ArgmaxKernel] = {
                std::make_shared<BareMetal::ArgmaxKernel>()};
        inner_map[KernelPack::KernType::ConvBackDataKernel] = {
                std::make_shared<BareMetal::ConvBackDataGeneral>()};
        inner_map[KernelPack::KernType::FusedElemwiseKernel] = {
                std::make_shared<BareMetal::FusedElmwiseKernel>()};
    }

    std::unordered_map<KernelPack::KernType,
                       std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
};
}  // namespace
std::vector<const KernelFunc*> ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllBareKernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}

// vim: syntax=cpp.doxygen
