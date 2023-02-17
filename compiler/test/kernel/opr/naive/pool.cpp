/**
 * \file
 * compiler/test/kernel/opr/naive/pool.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = PoolingForward::Param::Mode;
TEST(NAIVE, PoolingNCHW) {
    Checker<Pooling> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    PoolingForward::Param param;
    checker.set_param(param);
    for (auto mode : {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING})
        for (size_t window : {2, 3, 5})
            for (size_t stride : {(size_t)1, window})
                for (size_t pad : {(size_t)0, window / 2})
                    for (size_t n : {1, 3})
                        for (size_t c : {1, 3})
                            for (size_t hw : {5, 23}) {
                                param.mode = mode;
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.window_h = window;
                                param.window_w = window;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs({{n, c, hw, hw}, {}});
                            }
}

TEST(NAIVE, PoolingNCHWQuant) {
    Checker<Pooling> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    PoolingForward::Param param;
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::QuantizedS8(0.7f));
    checker.set_dtype(1, dtype::QuantizedS8(0.7f));
    checker.set_param(param);
    for (auto mode : {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING})
        for (size_t window : {2, 3, 5})
            for (size_t stride : {(size_t)1, window})
                for (size_t pad : {(size_t)0, window / 2})
                    for (size_t n : {1, 3})
                        for (size_t c : {1, 3})
                            for (size_t hw : {5, 23}) {
                                param.mode = mode;
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.window_h = window;
                                param.window_w = window;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs({{n, c, hw, hw}, {}});
                            }
}
