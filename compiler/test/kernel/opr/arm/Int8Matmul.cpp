#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(AARCH64, Int8MatMulM8N12K4Dot) {
    Checker<MatrixMulForward> checker(Arch::ARM64);
    MatrixMulForward::Param param;
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);

    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int32());
    checker.set_kernel_symbol("Arm64_kernel_int8_dot_matmul_8x12mk4_.*");
    for (size_t m : {4, 8, 16, 64})
        for (size_t n : {3, 8, 15, 56})
            for (size_t k : {4, 8, 16, 64}) {
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK4_DOT;
                checker.set_param(param);
                checker.execs({{m / 4, k / 4, 4, 4}, {k / 4, n, 4}, {}});
            }
}

TEST(AARCH64, Int8MatMulM8N12K8MK4I8mm) {
    Checker<MatrixMulForward> checker(Arch::ARM64_WITH_I8MM);
    MatrixMulForward::Param param;
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);

    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int32());
    checker.set_kernel_symbol("Arm64_kernel_int8_i8mm_matmul_8x8x12mk4_.*");
    param.transposeA = false;
    param.transposeB = false;
    param.format = param::MatrixMul::Format::MK4;
    checker.set_param(param);
    for (size_t m : {1, 4, 7, 8})
        for (size_t n : {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35})
            for (size_t k : {8, 9, 11}) {
                checker.execs({{m, k, 4, 4}, {k, n, 4}, {}});
            }
}

// vim: syntax=cpp.doxygen
