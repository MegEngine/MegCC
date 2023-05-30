#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK

TEST(ARMCOMMON, BenchmarkInt16MatMulM8N8K8) {
#ifdef __aarch64__
    Benchmarker<MatrixMulForward> benchmarker(Arch::ARM64);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<MatrixMulForward>("AARCH64_INT16X16X32_MK8_8X8"));
#else
    Benchmarker<MatrixMulForward> benchmarker(Arch::ARMV7);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<MatrixMulForward>("ARMV7_INT16X16X32_MK8_4X8"));
#endif
    MatrixMulForward::Param param;
    UniformIntRNG rng(-32767, 32767);
    benchmarker.set_rng(0, &rng);
    benchmarker.set_rng(1, &rng);
    benchmarker.set_dtype(0, dtype::Int16())
            .set_dtype(1, dtype::Int16())
            .set_dtype(2, dtype::Int32());
    benchmarker.set_kernel_symbol("ArmCommon_kernel_int16_matmul_m8_n8_k8_.*");
    for (size_t m : {64, 128})
        for (size_t n : {256, 384})
            for (size_t k : {128, 256}) {
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK8;
                benchmarker.set_param(param);
                auto result =
                        benchmarker.execs({{m / 8, k / 8, 8, 8}, {k / 8, n, 8}, {}});
                printf("m=%zu, n=%zu, k=%zu\n", m, n, k);
                result.print();
            }
}
#endif