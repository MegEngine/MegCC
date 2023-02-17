/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/benchmark_cv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/cv_opr.h"
using namespace megcc::test;
using namespace megdnn;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK

TEST(GI, CVBenchmarkresize) {
    Benchmarker<CVResize> benchmarker(Arch::BAREMETAL);
    megdnn::CVResize::Param param;
    param.format = megdnn::CVResize::Param::Format::NHWC;
    param.imode = megdnn::CVResize::Param::InterpolationMode::LINEAR;
    benchmarker.set_param(param);
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(1, dtype::Uint8());

    benchmarker.execs({{1, 1080, 1920, 3}, {1, 720, 1280, 3}}).print();
    benchmarker.execs({{1, 1080, 1920, 1}, {1, 720, 1280, 1}}).print();
    benchmarker.execs({{1, 1080, 1920, 2}, {1, 720, 1280, 2}}).print();
}

TEST(GI, CVBenchmarkflip) {
    Benchmarker<CVflip> benchmarker(Arch::BAREMETAL);
    megdnn::CVflip::Param param;
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(1, dtype::Uint8());
    for (auto vec : {true, false})
        for (auto horn : {true, false}) {
            param.vertical = vec;
            param.horizontal = horn;
            benchmarker.set_param(param);
            benchmarker.exec({{1, 1025, 516, 1}, {}}).print();
            // FIXME: dnn only support channel== 1 and channel == 3
            // benchmarker.exec({{1, 1025, 516, 2}, {}}).print();
            benchmarker.exec({{1, 1025, 516, 3}, {}}).print();
        }
}

TEST(GI, CVBenchmarktranspose) {
    Benchmarker<CVtranspose> benchmarker(Arch::BAREMETAL);
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(1, dtype::Uint8());

    benchmarker.execs({{1, 1080, 1920, 1}, {1, 1920, 1080, 1}}).print();
    benchmarker.execs({{1, 1024, 1920, 1}, {1, 1920, 1024, 1}}).print();
    benchmarker.execs({{1, 1080, 1920, 2}, {1, 1920, 1080, 2}}).print();
    benchmarker.execs({{1, 1080, 1920, 3}, {1, 1920, 1080, 3}}).print();
}

TEST(GI, CVBenchmarkrotate) {
    Benchmarker<CVRotate> benchmarker(Arch::BAREMETAL);
    megdnn::CVRotate::Param param;
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(1, dtype::Uint8());
    for (auto clockwise : {true, false}) {
        param.clockwise = clockwise;
        benchmarker.set_param(param);
        benchmarker.exec({{1, 1031, 519, 1}, {}}).print();
        // FIXME: dnn only support channel== 1 and channel == 3
        // benchmarker.exec({{1, 1031, 519, 2}, {}}).print();
        benchmarker.exec({{1, 1031, 519, 3}, {}}).print();
    }
}
TEST(GI, CVBenchmarkWarpAffine) {
    Benchmarker<CVWarpAffine> benchmarker(Arch::BAREMETAL);
    WarpAffineForward::Param param;

    param.format = WarpAffineForward::Param::Format::NHWC;
    param.imode = WarpAffineForward::Param::InterpolationMode::LINEAR;
    param.border_mode = WarpAffineForward::BorderMode::REPLICATE;
    param.border_val = 1.25;
    ListRNG rng({1.f, 2.f, -10.f, 2.f, 1.f, -10.f});
    benchmarker.set_rng(1, &rng);
    benchmarker.set_param(param);
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(2, dtype::Uint8());

    benchmarker.execs({{1, 1080, 1920, 3}, {1, 2, 3}, {1, 720, 1280, 3}}).print();

    benchmarker.execs({{1, 1080, 1920, 1}, {1, 2, 3}, {1, 720, 1280, 1}}).print();

    benchmarker.execs({{1, 1080, 1920, 2}, {1, 2, 3}, {1, 720, 1280, 2}}).print();
}

TEST(GI, CVBenchmarkCvtColor) {
    Benchmarker<CVCvtColor> benchmarker(Arch::BAREMETAL);
    megdnn::CVCvtColor::Param param;
    using CvtMode = megdnn::CVCvtColor::Param::Mode;
    param.mode = CvtMode::RGB2YUV;
    benchmarker.set_param(param);
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(1, dtype::Uint8());

    benchmarker.execs({{1, 1080, 1920, 3}, {}}).print();

    param.mode = CvtMode::YUV2BGR_NV21;
    benchmarker.set_param(param);
    benchmarker.execs({{1, 1080, 1920, 1}, {}}).print();

    param.mode = CvtMode::RGB2BGR;
    benchmarker.set_param(param);
    benchmarker.execs({{1, 1080, 1920, 3}, {}}).print();
}
#endif