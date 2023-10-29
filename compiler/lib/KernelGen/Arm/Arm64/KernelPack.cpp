#include <memory>

#include "BatchedMatmul/BatchedMatmul.h"
#include "ConvKernel.h"
#include "Elemwise/Elemwise.h"
#include "InternalKernel/InternalKernel.h"
#include "KernelPack.h"
#include "MatMulKernel/MatMul.h"
#include "Rotate.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
namespace {
struct AllA64Kernel {
    AllA64Kernel() {
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<Arm64::Conv1x1FloatMk4>(),
                std::make_shared<Arm64::WinogradFloatF23Nchw44>(),
                std::make_shared<Arm64::ConvIm2colFloat>(),
                std::make_shared<Arm64::Conv1x1DotMk4>(),
                std::make_shared<Arm64::ConvIm2colDot>(),
                std::make_shared<Arm64::ConvDotNCHWNCHW44Stride2>(),
                std::make_shared<Arm64::ConvDotNCHWNCHW44Stride1>()};
        inner_i8mm_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<Arm64::ConvBiasIm2colI8mmNCHW44>(),
        };

        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<Arm64::Fp16MatMulM8N8K8>(),
                std::make_shared<Arm64::Fp32MatMulM8N12>(),
                std::make_shared<Arm64::Fp32MatMulM8N12K4>(),
                std::make_shared<Arm64::Fp32MatMulM4N16K4>(),
                std::make_shared<Arm64::Int8DotMatMulM8N12K4>()};
        inner_i8mm_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<Arm64::Int8I8mmMatMulM8N12K8MK4>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<Arm64::MatmulM4N16MK4Kernel>(),
                std::make_shared<Arm64::MatmulM8N12Kernel>(),
                std::make_shared<Arm64::MatmulM8N12MK4Kernel>(),
                std::make_shared<Arm64::MatmulInt8DotM8N12MK4Kernel>()};

        inner_map[KernelPack::KernType::ElemwiseKernel] = {
                std::make_shared<Arm64::ElemwiseKernel>()};

        inner_map[KernelPack::KernType::BatchMatmulKernel] = {
                std::make_shared<Arm64::Fp32BatchedMatmul>()};

        inner_map[KernelPack::KernType::RotateKernel] = {
                std::make_shared<Arm64::RotateKernel>()};
    }
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_i8mm_map;
};
}  // namespace
std::vector<const KernelFunc*> ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type, const bool& with_i8mm) {
    static AllA64Kernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    if (with_i8mm) {
        for (auto& kernel : all_kernel.inner_i8mm_map[kernel_type]) {
            ret_kernels.push_back(kernel.get());
        }
    }
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}

// vim: syntax=cpp.doxygen
