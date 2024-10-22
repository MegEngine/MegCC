#include "WebAssembly/KernelPack.h"
#include <memory>
#include "InternalKernel/InternalKernel.h"
#include "MatMulKernel/Fp32MatMul.h"

using namespace megcc;
using namespace KernelGen;
using namespace WebAssembly;

namespace {
struct AllWebAssemblyKernel {
    AllWebAssemblyKernel() {
        inner_map[KernelPack::KernType::MatrixMulKernel] = {
                std::make_shared<WebAssembly::Fp32MatMulM4N12>()};

        inner_map[KernelPack::KernType::InternelKernel] = {
                std::make_shared<WebAssembly::MatmulM4N12Kernel>()};
    }
    std::unordered_map<KernelPack::KernType, std::vector<std::shared_ptr<KernelFunc>>>
            inner_map;

};
}
std::vector<const KernelFunc*> WebAssembly::ArchKernelPack::GetKernel(
        KernelPack::KernType kernel_type) {
    static AllWebAssemblyKernel all_kernel;
    std::vector<const KernelFunc*> ret_kernels;
    for (auto& kernel : all_kernel.inner_map[kernel_type]) {
        ret_kernels.push_back(kernel.get());
    }
    return ret_kernels;
}