/**
 * \file
 * compiler/test/kernel/opr/arm/Int8Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

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

// vim: syntax=cpp.doxygen
