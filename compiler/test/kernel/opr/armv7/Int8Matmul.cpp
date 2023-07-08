#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(ARMV7, Int8x8x32MatMulMK4) {
    Checker<MatrixMulForward> checker(Arch::ARMV7);
    MatrixMulForward::Param param;
    UniformIntRNG rng(-127, 127);
    checker.set_epsilon(1e-4);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());

    checker.set_kernel_symbol("Armv7_kernel_int8x8x32_matmul_mk4_.*");
    size_t m = 16, k = 64, n = 8;
    param.transposeA = false;
    param.transposeB = false;
    param.format = param::MatrixMul::Format::MK4;
    checker.set_param(param);
    checker.execs({{m / 4, k / 4, 4, 4}, {k / 4, n, 4}, {}});
}

// vim: syntax=cpp.doxygen