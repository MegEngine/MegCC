#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK

TEST(AUTONAIVE, BenchmarkFP32GEMM) {
    Benchmarker<MatrixMulForward> benchmarker(Arch::AUTO_BAREMETAL);
    size_t m = 256, k = 256, n = 256;
    size_t a0 = m, a1 = k, b0 = k, b1 = n;
    BenchmarkOption option;
    option.valid_megcc_performance = true;
    benchmarker.set_benchmark_option(option);
    MatrixMulForward::Param param;
    param.transposeA = false;
    param.transposeB = false;
    benchmarker.set_param(param);
    if (param.transposeA) {
        a0 = k, a1 = m;
    }
    if (param.transposeB) {
        b0 = n, b1 = k;
    }
    benchmarker.execs({{a0, a1}, {b0, b1}, {}}).print();
}

#endif
