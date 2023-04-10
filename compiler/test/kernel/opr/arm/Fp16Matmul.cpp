#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#if ENABLE_KERNEL_FP16
TEST(AARCH64, Fp16MatMulM8N8K8) {
    Checker<MatrixMulForward> checker(Arch::ARM64);
    MatrixMulForward::Param param;
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    // megcc::test::SequenceRNG rng;
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_epsilon(5e-3);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16());

    checker.set_kernel_symbol("Arm64_kernel_fp16_matmul_8x8mk8_.*");
    for (size_t m : {8, 16, 64})
        for (size_t n : {3, 8, 15, 56})
            for (size_t k : {8, 16, 64}) {
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK8;
                checker.set_param(param);
                checker.execs({{m / 8, k / 8, 8, 8}, {k / 8, n, 8}, {}});
            }
}
#endif
// vim: syntax=cpp.doxygen
