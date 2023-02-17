/**
 * \file
 * compiler/test/kernel/opr/naive/warpaffine.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Format = WarpPerspective::Param::Format;
using BorderMode = WarpPerspective::Param::BorderMode;
using InterpolationMode = WarpPerspective::Param::InterpolationMode;

TEST(NAIVE, WarpAffine) {
    Checker<WarpAffineForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_epsilon(1e-4);
    WarpAffineForward::Param param;
    param.imode = InterpolationMode::LINEAR;
    for (auto fmt : {Format::NCHW, Format::NHWC})
        for (auto bmode :
             {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
              WarpPerspective::BorderMode::REPLICATE,
              WarpPerspective::BorderMode::CONSTANT}) {
            param.format = fmt;
            param.border_val = 1.25;
            param.border_mode = bmode;
            checker.set_param(param);
            checker.set_dtype(0, dtype::Float32());
            checker.execs({{1, 13, 13, 17}, {1, 2, 3}, {1, 13, 7, 17}});
            checker.execs({{2, 13, 22, 17}, {2, 2, 3}, {2, 13, 7, 17}});
            checker.execs({{5, 13, 33, 17}, {5, 2, 3}, {5, 13, 7, 17}});
        }
}

TEST(NAIVE, WarpAffineU8) {
    Checker<WarpAffineForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_epsilon(1 + 1e-4);
    UniformIntRNG rng(0, 255);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(2, dtype::Uint8());
    WarpAffineForward::Param param;
    param.imode = InterpolationMode::LINEAR;
    for (auto fmt : {Format::NHWC})
        for (auto bmode : {WarpPerspective::BorderMode::REPLICATE}) {
            param.format = fmt;
            param.border_val = 5;
            param.border_mode = bmode;
            checker.set_param(param);
            checker.execs({{1, 13, 13, 17}, {1, 2, 3}, {1, 13, 7, 17}});
            checker.execs({{2, 13, 22, 17}, {2, 2, 3}, {2, 13, 7, 17}});
            checker.execs({{5, 13, 33, 17}, {5, 2, 3}, {5, 13, 7, 17}});
        }
}