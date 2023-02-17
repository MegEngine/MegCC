/**
 * \file
 * compiler/test/kernel/opr/naive/argsort.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, Argsort) {
    Checker<ArgsortForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    ArgsortForward::Param param;
    for (auto order :
         {ArgsortForward::Param::Order::ASCENDING,
          ArgsortForward::Param::Order::DESCENDING})
        for (size_t batch_size : {1, 3, 4})
            for (size_t vec_len = 1; vec_len < 77; vec_len++) {
                param.order = order;
                checker.set_param(param);
                checker.execs({{batch_size, vec_len}, {}, {}});
            }
}