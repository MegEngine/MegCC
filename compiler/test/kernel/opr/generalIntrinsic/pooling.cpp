/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/pooling.cpp
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
TEST(GI, PoolingNCHW44) {
    Checker<Pooling> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_pooling.*");
    PoolingForward::Param param;
    param.format = PoolingForward::Param::Format::NCHW44;
    checker.set_param(param);
    checker.set_epsilon(1e-4);
    for (auto mode : {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING})
        for (size_t window : {2, 3, 5})
            for (size_t stride : {(size_t)1, window})
                for (size_t pad : {(size_t)0, window / 2})
                    for (size_t n : {1, 3})
                        for (size_t c : {4, 12})
                            for (size_t hw : {5, 23}) {
                                param.mode = mode;
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.window_h = window;
                                param.window_w = window;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs({{n, c / 4, hw, hw, 4}, {}});
                            }
}

TEST(GI, PoolingNCHW44QInt8) {
    Checker<Pooling> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_pooling.*");
    PoolingForward::Param param;
    param.format = PoolingForward::Param::Format::NCHW44;
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    checker.set_param(param);
    checker.set_epsilon(1e-4);
    for (auto scale : {0.35f, 0.7f, 1.6f})
        for (auto mode :
             {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING})
            for (size_t window : {2, 3, 5})
                for (size_t stride : {(size_t)1, window})
                    for (size_t pad : {(size_t)0, window / 2})
                        for (size_t n : {1, 3})
                            for (size_t c : {4, 12})
                                for (size_t hw : {5, 23}) {
                                    checker.set_dtype(0, dtype::QuantizedS8(scale));
                                    checker.set_dtype(1, dtype::QuantizedS8(scale));
                                    param.mode = mode;
                                    param.pad_h = pad;
                                    param.pad_w = pad;
                                    param.window_h = window;
                                    param.window_w = window;
                                    param.stride_h = stride;
                                    param.stride_w = stride;
                                    checker.set_param(param);
                                    checker.execs({{n, c / 4, hw, hw, 4}, {}});
                                }
}
#if ENABLE_KERNEL_FP16
TEST(GI, PoolingNCHW88) {
    Checker<Pooling> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_pooling.*");
    PoolingForward::Param param;
    param.format = PoolingForward::Param::Format::NCHW88;
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::Float16());
    checker.set_param(param);
    checker.set_epsilon(1e-3);
    for (auto mode : {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING})
        for (size_t window : {2, 3, 5})
            for (size_t stride : {(size_t)1, window})
                for (size_t pad : {(size_t)0, window / 2})
                    for (size_t n : {1, 3})
                        for (size_t c : {8, 16})
                            for (size_t hw : {5, 23}) {
                                param.mode = mode;
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.window_h = window;
                                param.window_w = window;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs({{n, c / 8, hw, hw, 8}, {}});
                            }
}
#endif
