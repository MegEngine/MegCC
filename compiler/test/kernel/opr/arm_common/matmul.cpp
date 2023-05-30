#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(ARMCOMMON, Int16MatMulM8N8K8) {
#ifdef __aarch64__
    Checker<MatrixMulForward> checker(Arch::ARM64);
#else
    Checker<MatrixMulForward> checker(Arch::ARMV7);
#endif
    MatrixMulForward::Param param;
    UniformIntRNG rng(-32767, 32767);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Int16())
            .set_dtype(1, dtype::Int16())
            .set_dtype(2, dtype::Int32());

    checker.set_kernel_symbol("ArmCommon_kernel_int16_matmul_m8_n8_k8_.*");
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