/**
 * \file
 * compiler/test/kernel/opr/naive/benchmark_cv.cpp
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

TEST(NAIVE, CVBenchmarkGaussianBlur) {
    Benchmarker<megdnn::CVGaussianBlur> benchmarker(Arch::BAREMETAL);
    megdnn::CVGaussianBlur::Param param;
    using BorderMode = megdnn::CVGaussianBlur::Param::BorderMode;
    UniformIntRNG seq(0, 255);
    benchmarker.set_rng(0, &seq);
    benchmarker.set_dtype(0, dtype::Uint8());
    benchmarker.set_dtype(1, dtype::Uint8());

    for (auto mode : {BorderMode::CONSTANT, BorderMode::REFLECT}) {
        for (int kh : {3, 5}) {
            for (int kw : {3, 5, 7}) {
                for (double sigma1 : {0.8}) {
                    for (double sigma2 : {0.5}) {
                        param.border_mode = mode;
                        param.kernel_height = kh;
                        param.kernel_width = kw;
                        param.sigma_x = sigma1;
                        param.sigma_y = sigma2;

                        benchmarker.set_param(param);
                        std::string m = "constant";
                        if (mode == BorderMode::REFLECT)
                            m = "reflect";
                        printf("mode: %s, kh: %d, kw: %d, {n, ih, iw, ic}: {1, 1080, "
                               "1920, 3}: \n",
                               m.c_str(), kh, kw);
                        benchmarker.execs({{1, 1080, 1920, 3}, {}}).print();
                        printf("mode: %s, kh: %d, kw: %d, {n, ih, iw, ic}: {1, 1080, "
                               "1920, 1}: \n",
                               m.c_str(), kh, kw);
                        benchmarker.execs({{1, 1080, 1920, 1}, {}}).print();
                        printf("mode: %s, kh: %d, kw: %d, {n, ih, iw, ic}: {1, 224, "
                               "224, 3}: \n",
                               m.c_str(), kh, kw);
                        benchmarker.execs({{1, 224, 224, 3}, {}}).print();
                        printf("mode: %s, kh: %d, kw: %d, {n, ih, iw, ic}: {1, 224, "
                               "224, 1}: \n",
                               m.c_str(), kh, kw);
                        benchmarker.execs({{1, 224, 224, 1}, {}}).print();
                    }
                }
            }
        }
    }
}
#endif