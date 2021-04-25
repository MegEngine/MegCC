/**
 * \file
 * compiler/test/kernel/opr/arm/reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = ReduceForward::Param::Mode;
TEST(AARCH64, Reduce) {
    Checker<Reduce> checker(Arch::ARM64);
    for (auto mode : {Mode::SUM, Mode::MEAN, Mode::MAX, Mode::MIN,
                      Mode::PRODUCT, Mode::SUM_SQR})
        for (auto src : {TensorShape{2, 3}, {3, 4, 5}, {4, 5, 6, 7}})
            for (size_t axis = 0; axis < 4; ++axis) {
                if (axis < src.ndim) {
                    ReduceForward::Param param;
                    param.axis = axis;
                    param.mode = mode;
                    checker.set_param(param);
                    checker.execs({src, {}});
                }
            }
}
