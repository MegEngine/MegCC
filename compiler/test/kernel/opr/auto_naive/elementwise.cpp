/**
 * \file
 * compiler/test/kernel/opr/auto/elementwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
#include "test/kernel/opr/common/elemwise.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;

TEST(AUTONAIVE, ElementwiseUnique) {
    Checker<ElemwiseForward> checker(Arch::AUTO_BAREMETAL);
    ElemwiseForward::Param param;
    for (auto mode : {MODE::RELU}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{3, 10}, {}});
        checker.execs({{2, 3, 4}, {}});
    }
}

// vim: syntax=cpp.doxygen
