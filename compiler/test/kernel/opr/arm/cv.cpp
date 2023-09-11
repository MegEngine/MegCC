#include "test/kernel/common/checker.h"
#include "test/kernel/common/cv_opr.h"
using namespace megcc::test;
using namespace megdnn;
using namespace megcc::KernelGen;

TEST(AARCH64, CVresize) {
    Checker<megdnn::CVResize> checker(Arch::ARM64);
    megdnn::CVResize::Param param;
    param.format = megdnn::CVResize::Param::Format::NHWC;
    param.imode = megdnn::CVResize::Param::InterpolationMode::LINEAR;
    UniformIntRNG rng(0, 254);
    checker.set_rng(0, &rng);
    checker.set_param(param);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    checker.set_epsilon(1 + 1e-4);
    for (size_t h : {2, 3, 5, 6, 7, 9, 10})
        for (size_t w : {2, 3, 5, 7, 8, 9, 10, 11}) {
            checker.exec({{1, h, w, 3}, {1, w, h, 3}});
            checker.exec({{1, h, w, 1}, {1, w, h, 1}});
            checker.exec({{1, h, w, 2}, {1, w, h, 2}});
            checker.exec({{1, h, w, 5}, {1, w, h, 5}});
        }
    checker.exec({{1, 3, 5, 3}, {1, 6, 8, 3}});
    checker.exec({{1, 13, 11, 1}, {1, 3, 8, 1}});
}

TEST(AARCH64, CVcvtcolor) {
    Checker<megdnn::CVCvtColor> checker(Arch::ARM64);
    using CvtMode = megdnn::CVCvtColor::Param::Mode;
    megdnn::CVCvtColor::Param param;
    UniformIntRNG rng(0, 255);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());

    for (auto mode : {CvtMode::RGB2YUV, CvtMode::RGB2BGR}) {
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

TEST(AARCH64, CVWarpAffine) {
    using Format = megdnn::CVWarpAffine::Param::Format;
    using BorderMode = megdnn::CVWarpAffine::Param::BorderMode;
    using InterpolationMode = megdnn::CVWarpAffine::Param::InterpolationMode;
    Checker<CVWarpAffine> checker(Arch::ARM64);
    UniformIntRNG n_rng(0, 255);
    checker.set_rng(0, &n_rng);

    ListRNG rng({1.f, 2.f, -10.f, 2.f, 1.f, -10.f});
    checker.set_rng(1, &rng);

    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(2, dtype::Uint8());
    CVWarpAffine::Param param;
    param.imode = InterpolationMode::LINEAR;
    for (auto fmt : {Format::NHWC})
        for (auto bmode :
             {BorderMode::WRAP, BorderMode::REFLECT, BorderMode::REPLICATE,
              BorderMode::CONSTANT})
            for (size_t c : {1, 2, 3, 4})
                for (size_t h : {3, 7, 8, 13, 16, 23, 32})
                    for (size_t w : {3, 7, 8, 13, 16, 23, 32}) {
                        param.format = fmt;
                        param.border_val = 5;
                        param.border_mode = bmode;
                        checker.set_param(param);
                        checker.execs({{1, h, w, c}, {1, 2, 3}, {1, w, h, c}});
                    }
}

TEST(AARCH64, CVtranspose) {
    Checker<megdnn::CVtranspose> checker(Arch::ARM64);
    UniformIntRNG n_rng(0, 255);
    checker.set_rng(0, &n_rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    for (size_t h : {35, 47})
        for (size_t w : {37, 51})
            for (size_t c : {1, 2, 3, 4}) {
                checker.exec({{1, h, w, c}, {}});
            }
}

TEST(AARCH64, CVflip) {
    Checker<megdnn::CVflip> checker(Arch::ARM64);
    megdnn::CVflip::Param param;
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    for (bool vertical : {false, true})
        for (bool horizontal : {false, true}) {
            param.vertical = vertical;
            param.horizontal = horizontal;
            checker.set_param(param);
            checker.exec({{1, 3, 5, 3}, {}});
            checker.exec({{1, 3, 5, 1}, {}});
            checker.exec({{1, 111, 105, 3}, {}});
            checker.exec({{1, 111, 105, 1}, {}});
            checker.exec({{1, 1025, 516, 3}, {}});
            checker.exec({{1, 1025, 516, 1}, {}});
        }
}

TEST(AARCH64, CVrotate) {
    Checker<megdnn::CVRotate> checker(Arch::ARM64);
    megdnn::CVRotate::Param param;
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());

    for (bool clockwise : {false, true}) {
        param.clockwise = clockwise;
        checker.set_param(param);
        checker.exec({{1, 3, 5, 1}, {}});
        // FIXME: dnn rotate only support channel==1 and channel == 3
        // checker.exec({{1, 3, 5, 2}, {}});
        checker.exec({{1, 3, 5, 3}, {}});
        checker.exec({{1, 16, 16, 1}, {}});

        checker.exec({{1, 16, 16, 3}, {}});
        checker.exec({{1, 16, 19, 1}, {}});

        checker.exec({{1, 16, 19, 3}, {}});
        checker.exec({{1, 19, 19, 1}, {}});

        checker.exec({{1, 19, 19, 3}, {}});
    }
}
#if ENABLE_KERNEL_FP16
TEST(AARCH64, CVrotateFp16) {
    Checker<megdnn::CVRotate> checker(Arch::ARM64);
    megdnn::CVRotate::Param param;
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());

    for (bool clockwise : {false, true}) {
        param.clockwise = clockwise;
        checker.set_param(param);
        checker.exec({{1, 3, 5, 1}, {}});
        // FIXME: dnn rotate only support channel==1 and channel == 3
        // checker.exec({{1, 3, 5, 2}, {}});
        checker.exec({{1, 3, 5, 3}, {}});
        checker.exec({{1, 16, 16, 1}, {}});

        checker.exec({{1, 16, 16, 3}, {}});
        checker.exec({{1, 16, 19, 1}, {}});

        checker.exec({{1, 16, 19, 3}, {}});
        checker.exec({{1, 19, 19, 1}, {}});

        checker.exec({{1, 19, 19, 3}, {}});
    } 
}
#endif
