#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK
TEST(GI, BENCHMARK_TYPECVT) {
    Benchmarker<TypeCvtForward> benchmarker(Arch::BAREMETAL);
    for (auto kernel_filter : {"GI_kernel_typecvt.+"}) {
        benchmarker.set_kernel_symbol(kernel_filter);
        printf("kernel_filter=%s\n", kernel_filter);
        std::vector<std::pair<megdnn::DType, megdnn::DType>> types = {
                {dtype::Uint8(), dtype::Float32()},
                {dtype::QuantizedS8(0.3f), dtype::Float32()},
                {dtype::Float32(), dtype::QuantizedS8(1.7f)},
                {dtype::QuantizedS8(1.7f), dtype::QuantizedS8(0.3f)}};

        for (auto type : types) {
            benchmarker.set_dtype(0, type.first);
            benchmarker.set_dtype(1, type.second);
            benchmarker.execs({{1, 16, 1024, 1024}, {1, 16, 1024, 1024}}).print();
        }
    }
}
#endif