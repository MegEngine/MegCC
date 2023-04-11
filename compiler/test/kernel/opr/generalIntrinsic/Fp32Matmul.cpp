#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(GI, Fp32MatMulM4N12) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_epsilon(2e-4);
    checker.set_kernel_symbol("GI_kernel_fp32_matmul_4x12_.*");
    for (bool trans_a : {false, true})
        for (bool trans_b : {true, false})
            for (size_t m : {1, 3, 58, 9, 10, 12, 23, 67})
                for (size_t n : {1, 5, 6, 7, 8, 9, 10, 12, 24, 33})
                    for (size_t k : {1, 3, 5}) {
                        size_t a0 = m;
                        size_t a1 = k;
                        size_t b0 = k;
                        size_t b1 = n;
                        if (trans_a) {
                            a0 = k, a1 = m;
                        }
                        if (trans_b) {
                            b0 = n, b1 = k;
                        }
                        param.transposeA = trans_a;
                        param.transposeB = trans_b;
                        checker.set_param(param);
                        checker.execs({{a0, a1}, {b0, b1}, {}});
                    }
}

TEST(GI, Fp32MatMulM4N8K4) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_fp32_matmul_4x8mk4_.*");
    MatrixMulForward::Param param;
    checker.set_epsilon(1e-4);

    for (size_t m : {4, 8, 16, 64})
        for (size_t n : {3, 8, 15, 56})
            for (size_t k : {4, 8, 16, 64}) {
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK4;
                checker.set_param(param);
                checker.execs({{m / 4, k / 4, 4, 4}, {k / 4, n, 4}, {}});
            }
}

TEST(GI, FP32GEVM) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_kernel_symbol("GI_kernel_gevm.*");
    for (bool trans_a : {false})
        for (bool trans_b : {false, true})
            for (size_t m : {1})
                for (size_t n : {3, 8, 11, 15, 33, 56})
                    for (size_t k : {3, 8, 11, 15, 33, 14}) {
                        size_t a0 = m;
                        size_t a1 = k;
                        size_t b0 = k;
                        size_t b1 = n;
                        if (trans_b) {
                            b0 = n, b1 = k;
                        }
                        param.transposeA = trans_a;
                        param.transposeB = trans_b;
                        checker.set_param(param);
                        checker.execs({{a0, a1}, {b0, b1}, {}});
                    }
}

TEST(GI, FP32GEMV) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_kernel_symbol("GI_kernel_gemv_nn.*");
    for (size_t m : {1, 2, 3, 4})
        for (size_t n : {1})
            for (size_t k : {3, 8, 11, 15, 33, 14}) {
                size_t a0 = m;
                size_t a1 = k;
                size_t b0 = k;
                size_t b1 = n;
                param.transposeA = false;
                param.transposeB = false;
                checker.set_param(param);
                checker.execs({{a0, a1}, {b0, b1}, {}});
            }
}

TEST(GI, FP32GEMVMK4) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("GI_kernel_gemv_MK4_nn.*");
    for (size_t m : {1, 2, 3, 4})
        for (size_t n : {1})
            for (size_t k : {3, 8, 11, 15, 33}) {
                m = m * 4;
                k = k * 4;
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK4;
                checker.set_param(param);
                checker.execs({{m / 4, k / 4, 4, 4}, {k / 4, n, 4}, {}});
            }
}

TEST(GI, Fp32MatMulM4N12K4) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("GI_kernel_fp32_matmul_4x12mk4_.*");
    for (size_t m : {4, 8, 16, 64})
        for (size_t n : {3, 8, 15, 56})
            for (size_t k : {4, 8, 16, 64}) {
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK4;
                checker.set_param(param);
                checker.execs({{m / 4, k / 4, 4, 4}, {k / 4, n, 4}, {}});
            }
}

// vim: syntax=cpp.doxygen
