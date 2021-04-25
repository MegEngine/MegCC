/**
 * \file
 * compiler/lib/Conversion/MGBToKernel/KernelRegister.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "KernelTemplate.h"
#include "compiler/Dialect/Kernel/IR/KernelDialect.h"
#include "compiler/KernelGen/KernelGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using KernType = megcc::KernelGen::KernelPack::KernType;

template <class T>
std::pair<std::vector<const megcc::KernelGen::KernelFunc*>,
          const megcc::KernelGen::DeduceFunc*>
GetKernels(megcc::KernelGen::Arch platform) {
    llvm::errs() << "no implement yet\n";
    abort();
}
#define INSTANCE_GET_KERNELS(kern_opr, kern_type)                            \
    template <>                                                              \
    std::pair<std::vector<const megcc::KernelGen::KernelFunc*>,              \
              const megcc::KernelGen::DeduceFunc*>                           \
    GetKernels<kern_opr>(megcc::KernelGen::Arch platform) {                  \
        return megcc::KernelGen::KernelPack::GetKernel(kern_type, platform); \
    }

INSTANCE_GET_KERNELS(mlir::Kernel::Conv2DKernel, KernType::ConvKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ConvBackDataKernel, KernType::ConvBackDataKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::Pooling2DKernel, KernType::PoolingKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ConcatKernel, KernType::ConcatKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::MatrixMulKernel, KernType::MatrixMulKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::MatrixInvKernel, KernType::MatrixInvKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::WarpAffineKernel, KernType::WarpAffineKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::IndexingMultiAxisVecKernel,
                     KernType::IndexingMultiAxisKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ReduceKernel, KernType::ReduceKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::BatchedMatrixMulKernel,
                     KernType::BatchMatmulKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::PowCKernel, KernType::PowCKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::TypeCvtKernel, KernType::TypeCvtKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::TopkKernel, KernType::TopK)
INSTANCE_GET_KERNELS(mlir::Kernel::WarpPerspectiveKernel,
                     KernType::WarpPerspectiveKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::RelayoutKernel, KernType::RelayoutKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ResizeKernel, KernType::ResizeKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ElemwiseKernelInterface,
                     KernType::ElemwiseKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ElemwiseMultiType,
                     KernType::ElemwiseMultiKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ArgsortKernel, KernType::ArgSortKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::ArgmaxKernel, KernType::ArgmaxKernel)
INSTANCE_GET_KERNELS(mlir::Kernel::IndexingOneHotKernel,
                     KernType::IndexingOneHotKernel)

template <class T, typename... Args>
void addBuiltinTemplatesOpr(mlir::Kernel::KernelTemplateRegistry& registry,
                            megcc::KernelGen::Arch arch, Args&&... args) {
    auto kernels = GetKernels<T>(arch);
    for (const auto& kernel : kernels.first) {
        registry.create<T>(kernel, kernels.second, std::forward<Args>(args)...);
    }
}

}  // namespace

namespace mlir {
namespace Kernel {
void addBuiltinTemplatesByOperator(
        mlir::Kernel::KernelTemplateRegistry& registry,
        megcc::KernelGen::Arch arch) {
    //! add internal kernel to builtin templates

    addBuiltinTemplatesOpr<mlir::Kernel::Conv2DKernel>(registry, arch);
    //! TODO: should refactor the elemwise kernel get
    addBuiltinTemplatesOpr<mlir::Kernel::ElemwiseKernelInterface>(
            registry, arch,
            // add extra template attribute 'mode' into TContext
            [&](mlir::Operation* op, megcc::TContext* ctx) {
                auto elem =
                        llvm::dyn_cast<mlir::Kernel::ElemwiseKernelInterface>(
                                op);
                ctx->setAttr("mode", elem.getMode().str());
            });
    addBuiltinTemplatesOpr<mlir::Kernel::ElemwiseMultiType>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::PowCKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::Pooling2DKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::ConcatKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::ResizeKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::MatrixMulKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::ReduceKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::BatchedMatrixMulKernel>(registry,
                                                                 arch);
    addBuiltinTemplatesOpr<mlir::Kernel::TypeCvtKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::TopkKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::WarpPerspectiveKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::RelayoutKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::MatrixInvKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::WarpAffineKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::IndexingMultiAxisVecKernel>(registry,
                                                                     arch);
    addBuiltinTemplatesOpr<mlir::Kernel::IndexingOneHotKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::ArgsortKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::ArgmaxKernel>(registry, arch);
    addBuiltinTemplatesOpr<mlir::Kernel::ConvBackDataKernel>(registry, arch); 
}
}  // namespace Kernel
}  // namespace mlir

// vim: syntax=cpp.doxygen
