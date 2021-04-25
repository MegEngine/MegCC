/**
 * \file
 * compiler/test/kernel/opr/naive/argmax.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2021 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, Argmax) {
    Checker<Argmax> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    for (DType dtype : {(DType)dtype::Float32()}) {
        checker.set_dtype(0, dtype);
            for (auto src : {TensorShape{2, 3}, TensorShape{3, 4, 5},
                             TensorShape{4, 5, 6, 7}})
                for (size_t axis = 0; axis < 4; ++axis) {
                    if (axis < src.ndim) {
                        ArgmaxForward::Param param;
                        param.axis = axis;
                        checker.set_param(param);
                        checker.execs({src, {}});
                    }
                }
    }
}
