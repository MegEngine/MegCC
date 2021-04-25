/**
 * \file
 * compiler/test/kernel/opr/arm/Fp32MatMul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(ARMV7, Fp32MatMulM4N12) {
    Checker<MatrixMulForward> checker(Arch::ARMV7);
    MatrixMulForward::Param param;
    checker.set_epsilon(2e-4);
    checker.set_kernel_symbol("Armv7_kernel_fp32_matmul_4x12_.*");
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

TEST(ARMV7, Fp32MatMulM4N12K4) {
    Checker<MatrixMulForward> checker(Arch::ARMV7);
    MatrixMulForward::Param param;
    checker.set_kernel_symbol("Armv7_kernel_fp32_matmul_4x12mk4_.*");

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

TEST(ARMV7, Fp32MatMulM4N8K4) {
    Checker<MatrixMulForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol("Armv7_kernel_fp32_matmul_4x8mk4_.*");
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

// vim: syntax=cpp.doxygen
