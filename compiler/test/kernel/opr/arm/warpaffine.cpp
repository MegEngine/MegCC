/**
 * \file
 * compiler/test/kernel/opr/arm/warpaffine.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "megbrain/reflection.h"
#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Format = WarpPerspective::Param::Format;
using BorderMode = WarpPerspective::Param::BorderMode;
using InterpolationMode = WarpPerspective::Param::InterpolationMode;

TEST(AARCH64, WarpAffine) {
    Checker<WarpAffineForward> checker(Arch::ARM64);
    NormalRNG rng;
    checker.set_rng(1, &rng);
    checker.set_epsilon(1e-3);
    checker.set_dtype(0, dtype::Float32());
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

            checker.execs({{1, 13, 13, 17}, {1, 2, 3}, {1, 13, 7, 17}});
            checker.execs({{2, 13, 22, 17}, {2, 2, 3}, {2, 13, 7, 17}});
            checker.execs({{5, 13, 33, 17}, {5, 2, 3}, {5, 13, 7, 17}});
        }
}

TEST(AARCH64, WarpAffineU8) {
    Checker<WarpAffineForward> checker(Arch::ARM64);
    UniformIntRNG rng(0, 255);
    checker.set_epsilon(1 + 1e-4);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(2, dtype::Uint8());
    WarpAffineForward::Param param;
    param.imode = InterpolationMode::LINEAR;
    for (auto fmt : {Format::NHWC})
        for (auto bmode :
             {WarpPerspective::BorderMode::WRAP, WarpPerspective::BorderMode::REFLECT,
              WarpPerspective::BorderMode::REPLICATE,
              WarpPerspective::BorderMode::CONSTANT}) {
            param.format = fmt;
            param.border_val = 0;
            param.border_mode = bmode;
            printf("bmode=%s\n",
                   mgb::reflection::nameOfEnumValue<WarpPerspective::BorderMode>(bmode)
                           .c_str());
            checker.set_param(param);
            checker.execs({{1, 13, 13, 1}, {1, 2, 3}, {1, 13, 7, 1}});
            checker.execs({{2, 13, 22, 2}, {2, 2, 3}, {2, 13, 7, 2}});
            checker.execs({{2, 288, 153, 3}, {2, 2, 3}, {2, 63, 77, 3}});
            checker.execs({{5, 13, 33, 3}, {5, 2, 3}, {5, 13, 7, 3}});
        }
}
#ifdef ENABLE_KERNEL_BENCHMARK
TEST(AARCH64, BenchmarkWarpAffine) {
    Benchmarker<WarpAffineForward> benchmarker(Arch::ARM64);
    WarpAffineForward::Param param;
    param.format = WarpAffineForward::Param::Format::NHWC;
    param.imode = WarpAffineForward::Param::InterpolationMode::LINEAR;
    param.border_mode = WarpAffineForward::BorderMode::REPLICATE;
    param.border_val = 5;
    benchmarker.set_param(param);
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(2, dtype::Uint8());

    benchmarker.execs({{1, 1080, 1920, 3}, {1, 2, 3}, {1, 720, 1280, 3}}).print();

    benchmarker.execs({{1, 1080, 1920, 1}, {1, 2, 3}, {1, 720, 1280, 1}}).print();

    benchmarker.execs({{1, 1080, 1920, 2}, {1, 2, 3}, {1, 720, 1280, 2}}).print();
}
#endif