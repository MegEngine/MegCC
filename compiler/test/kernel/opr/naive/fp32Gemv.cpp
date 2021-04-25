/**
 * \file
 * compiler/test/kernel/opr/naive/fp32Gemv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
TEST(NAIVE, FP32GEMV) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_gemv.*");
    MatrixMulForward::Param param;

    for (bool trans_a : {false})
        for (bool trans_b : {false})
            for (size_t m : {3, 8, 11, 15, 33, 56})
                for (size_t n : {1, 2, 3, 4})
                    for (size_t k : {3, 8, 11, 14, 15, 33}) {
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