#include "Arm/Armv7/KernelPack.h"
#include <memory>
#include "Arm/Armv7/ConvKernel/ConvKernel.h"
#include "InternalKernel/InternalKernel.h"
#include "MatMulKernel/Fp32MatMul.h"
#include "MatMulKernel/Int8MatMul.h"

using namespace megcc;
using namespace KernelGen;
using namespace Armv7;

struct AllA32Kernel {
    AllA32Kernel() {
        inner_map[KernelPack::KernType::ConvKernel] = {
                std::make_shared<Armv7::ConvFloatNCHWNCHW443x3s2>(),
                std::make_shared<Armv7::Conv1x1FloatMk4>(),
                std::make_shared<Armv7::WinogradFloatF23NCHW44>(),
                std::make_shared<Armv7::ConvIm2colFloat>(),
                std::make_shared<Armv7::Int8Conv1x1NCHW44>(),
                std::make_shared<Armv7::Int8Conv5x5S1DirectNCHW>(),
        };
        inner_map_with_dot[KernelPack::KernType::ConvKernel] = {
                std::make_shared<Armv7::DotInt8Conv1x1NCHWM6N8K4>(),
                std::make_shared<Armv7::DotInt8Conv5x5S2DirectNCHW>(),
        };

        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<Armv7::Fp32MatMulM4N8K4>(),
                std::make_shared<Armv7::Fp32MatMulM4N12K4>(),
                std::make_shared<Armv7::Fp32MatMulM4N12>(),
                std::make_shared<Armv7::Int8x8x32MatMulMK4>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<Armv7::MatmulM4N8MK4Kernel>(),
                std::make_shared<Armv7::MatmulM4N12MK4Kernel>(),
                std::make_shared<Armv7::MatmulM4N12Kernel>()};
    }
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map_with_dot;
};

std::vector<const KernelFunc*> Armv7::ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type, const bool with_dot) {
    static AllA32Kernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    if (with_dot) {
        for (auto& kernel : all_kernel.inner_map_with_dot[kernel_type]) {
            ret_kernels.push_back(kernel.get());
        }
    }
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}
// vim: syntax=cpp.doxygen
