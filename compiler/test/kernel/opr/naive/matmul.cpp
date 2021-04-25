/**
 * \file
 * compiler/test/kernel/opr/naive/matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
TEST(NAIVE, MatMul) {
    Checker<MatrixMulForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    MatrixMulForward::Param param;

    for (bool trans_a : {false, true})
        for (bool trans_b : {false, true})
            for (size_t m : {3, 8, 11})
                for (size_t n : {3, 8, 11})
                    for (size_t k : {3, 8, 11}) {
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

TEST(NAIVE, BatchedMatMul) {
    Checker<BatchedMatrixMulForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    BatchedMatrixMulForward::Param param;

    for (bool trans_a : {false, true})
        for (bool trans_b : {false, true})
            for (size_t b : {1, 8, 23})
                for (size_t m : {3, 8, 11})
                    for (size_t n : {3, 8, 11})
                        for (size_t k : {3, 8, 11}) {
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
                            checker.execs({{b, a0, a1}, {b, b0, b1}, {}});
                        }
}