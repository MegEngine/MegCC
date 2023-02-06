/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/Fp32MatMul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(GI, Fp16MatMulM8N8K8) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_fp16_matmul_8x8mk8_.*");
    MatrixMulForward::Param param;
    megcc::test::UniformRNG rng(-1.0, 1.0);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16());

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

// vim: syntax=cpp.doxygen
