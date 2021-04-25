/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/benchmark_reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = ReduceForward::Param::Mode;
#ifdef ENABLE_KERNEL_BENCHMARK
TEST(GI, BENCHMARK_Reduce) {
    Benchmarker<Reduce> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("GI_kernel_reduce.*");
    for (auto mode : {Mode::MIN, Mode::MAX, Mode::SUM, Mode::SUM_SQR, Mode::MEAN, Mode::PRODUCT}){
        printf("mode=%d\n", mode);
        for (auto src : {TensorShape{200, 300}, TensorShape{3, 200, 300}, TensorShape{1, 3, 200, 300}}){
            for (size_t axis = 0; axis < 4; ++axis) {
                if (axis < src.ndim) {
                    printf("reduce axis=%ld\n", axis);
                    ReduceForward::Param param;
                    param.axis = axis;
                    param.mode = mode;
                    benchmarker.set_param(param);
                    benchmarker.execs({src, {}}).print();
                }
            }
        }
    }
}
#endif