/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "megbrain/reflection.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = ReduceForward::Param::Mode;
TEST(GI, Reduce) {
    Checker<Reduce> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_reduce.*");
    for (auto mode :
         {Mode::MIN, Mode::MAX, Mode::SUM, Mode::SUM_SQR, Mode::MEAN, Mode::PRODUCT}) {
        printf("mode=%s\n", mgb::reflection::nameOfEnumValue<Mode>(mode).c_str());
        for (auto src :
             {TensorShape{2, 3}, TensorShape{3, 4, 5}, TensorShape{4, 5, 6, 7}})
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
}
#if ENABLE_KERNEL_FP16
TEST(GI, ReduceFp16) {
    Checker<Reduce> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_reduce.*");
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::Float16());
    checker.set_epsilon(1e-2);
    for (auto mode :
         {Mode::MIN, Mode::MAX, Mode::SUM, Mode::SUM_SQR, Mode::MEAN, Mode::PRODUCT}) {
        printf("mode=%s\n", mgb::reflection::nameOfEnumValue<Mode>(mode).c_str());
        for (auto src :
             {TensorShape{2, 3}, TensorShape{3, 4, 5}, TensorShape{4, 5, 6, 7}})
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
}
#endif