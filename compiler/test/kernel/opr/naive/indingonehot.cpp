/**
 * \file
 * compiler/test/kernel/opr/naive/indingonehot.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, IndexingOneHot) {
    Checker<IndexingOneHot> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    UniformIntRNG rng_idx{0, 7};
    checker.set_param({2}).set_dtype(1, dtype::Int32{}).set_rng(1, &rng_idx);
    checker.execs({{10, 4, 8, 9}, {10, 4, 9}, {}});
    checker.execs({{10, 4, 8, 9, 7}, {10, 4, 9, 7}, {}});
    UniformIntRNG rng_idx2{0, 3};
    checker.set_param({1}).set_dtype(1, dtype::Int32{}).set_rng(1, &rng_idx2);
    checker.execs({{10, 4, 8, 9, 7}, {10, 8, 9, 7}, {}});
}
