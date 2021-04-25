/**
 * \file
 * compiler/test/kernel/opr/naive/relayout.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
#include "test/kernel/common/cv_opr.h"
using namespace megcc::test;
using namespace megdnn;
using namespace megcc::KernelGen;
//! CV default hwc format
TEST(NAIVE, CVtranspose) {
    Checker<megdnn::CVtranspose> checker;
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    checker.exec({{1, 3, 5, 4}, {}});
    checker.exec({{1, 7, 3, 1}, {}});
}

TEST(NAIVE, CVflip) {
    Checker<megdnn::CVflip> checker;
    megdnn::CVflip::Param param;
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    for (bool vertical : {false, true})
        for (bool horizontal : {false, true}) {
            param.vertical = vertical;
            param.horizontal = horizontal;
            checker.set_param(param);
            checker.exec({{1, 3, 5, 3}, {}});
        }
}

TEST(NAIVE, CVresize) {
    Checker<megdnn::CVResize> checker;
    megdnn::CVResize::Param param;
    param.format = megdnn::CVResize::Param::Format::NHWC;
    param.imode = megdnn::CVResize::Param::InterpolationMode::LINEAR;
    checker.set_param(param);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    UniformIntRNG rng(0, 255);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1 + 1e-4);

    checker.exec({{1, 3, 5, 3}, {1, 6, 8, 3}});
    checker.exec({{1, 13, 11, 1}, {1, 3, 8, 1}});
}

TEST(NAIVE, CVrotate) {
    Checker<megdnn::CVRotate> checker;
    megdnn::CVRotate::Param param;

    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    for (bool clockwise : {false, true}) {
        param.clockwise = clockwise;
        checker.set_param(param);
        checker.exec({{1, 3, 5, 3}, {}});
        checker.exec({{1, 7, 4, 1}, {}});
    }
}

TEST(NAIVE, CVroicopy) {
    Checker<megdnn::CVRoicopy> checker;
    megdnn::CVRoicopy::Param param;

    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());

    param.row_from = 3;
    param.row_to = 15;
    param.col_from = 7;
    param.col_to = 31;
    checker.set_param(param);
    checker.exec({{1, 17, 31, 3}, {}});
    checker.exec({{1, 17, 31, 1}, {}});
}

TEST(NAIVE, CVcvtcolor) {
    Checker<megdnn::CVCvtColor> checker;
    using CvtMode = megdnn::CVCvtColor::Param::Mode;
    megdnn::CVCvtColor::Param param;
    UniformIntRNG rng(0, 255);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());

    for (auto mode : {CvtMode::RGB2YUV, CvtMode::RGB2GRAY, CvtMode::RGB2BGR}) {
        param.mode = mode;
        checker.set_param(param);
        checker.exec({{1, 17, 31, 3}, {}});
    }
    for (auto mode : {CvtMode::YUV2BGR_NV21}) {
        param.mode = mode;
        checker.set_param(param);
        checker.exec({{1, 3, 18, 1}, {}});
        checker.exec({{1, 18, 18, 1}, {}});
    }
}

TEST(NAIVE, CVWarpAffine) {
    using Format = megdnn::CVWarpAffine::Param::Format;
    using BorderMode = megdnn::CVWarpAffine::Param::BorderMode;
    using InterpolationMode = megdnn::CVWarpAffine::Param::InterpolationMode;
    Checker<CVWarpAffine> checker;
    checker.set_epsilon(1 + 1e-4);
    UniformIntRNG rng(0, 255);
    checker.set_rng(0, &rng);
    ListRNG l_rng({1.f, 2.f, -10.f, 2.f, 1.f, -10.f});
    checker.set_rng(1, &l_rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(2, dtype::Uint8());
    CVWarpAffine::Param param;
    param.imode = InterpolationMode::LINEAR;
    for (auto fmt : {Format::NHWC})
        for (auto bmode : {BorderMode::WRAP, BorderMode::REFLECT,
                           BorderMode::REPLICATE, BorderMode::CONSTANT}) {
            param.format = fmt;
            param.border_val = 5;
            param.border_mode = bmode;
            checker.set_param(param);
            checker.execs({{1, 13, 13, 17}, {1, 2, 3}, {1, 13, 7, 17}});
        }
}