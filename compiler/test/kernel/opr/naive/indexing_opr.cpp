/**
 * \file
 * compiler/test/kernel/opr/naive/indexing_opr.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, IndexingMultiAxisVec) {
    Checker<IndexingMultiAxisVec> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    UniformIntRNG idx_rng(0, 5);
    ConstRNG zero(0);
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(2, &zero)
            .set_rng(3, &idx_rng)
            .set_rng(4, &idx_rng);

    checker.set_proxy({{0}});
    checker.execs({{10}, {5}, {5}});
    checker.execs({{10, 7}, {5, 7}, {5}});

    checker.set_proxy({{1, 2}});
    checker.execs({{10, 7, 11, 9}, {10, 5, 9}, {5}, {5}});

    checker.set_proxy({{0, 1, 2}});
    checker.execs({{1, 8, 8, 32}, {60, 32}, {60}, {60}, {60}});

    checker.set_proxy({{1, 2, 3}});
    checker.execs({
            {1, 8, 8, 32, 4},
            {1, 60, 4},
            {60},
            {60},
            {60},
    });
}
