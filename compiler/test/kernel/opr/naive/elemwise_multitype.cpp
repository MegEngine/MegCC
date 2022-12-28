/**
 * \file
 * compiler/test/kernel/opr/naive/elemwise_multitype.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using MODE = ElemwiseMultiType::Param::Mode;

TEST(NAIVE, ElementwiseMultitypeBinary) {
    Checker<ElemwiseMultiType> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_epsilon(1e-4);
    checker.set_dtype(0, dtype::QuantizedS8(1.f));
    checker.set_dtype(1, dtype::QuantizedS8(2.f));
    checker.set_dtype(2, dtype::QuantizedS8(3.f));
    ElemwiseMultiType::Param param;

    for (auto mode : {MODE::QADD, MODE::QFUSE_ADD_RELU}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 10}, {1, 10}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
    }
}