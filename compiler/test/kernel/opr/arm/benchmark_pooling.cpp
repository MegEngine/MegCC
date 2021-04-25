/**
 * \file
 * compiler/test/kernel/opr/arm/benchmark_conv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = PoolingForward::Param::Mode;
#ifdef ENABLE_KERNEL_BENCHMARK
TEST(AARCH64, BenchmarkPoolingNCHW44) {
    Benchmarker<PoolingForward> benchmarker(Arch::ARM64);
    PoolingForward::Param param;
    for (auto mode :
       {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING}) {
       printf("pooling mode=%d\n", mode);
        param.mode = mode;
        param.pad_h = 1;
        param.pad_w = 1;
        param.window_h = 3;
        param.window_w = 3;
        param.stride_h = 2;
        param.stride_w = 2;
        param.format = PoolingForward::Param::Format::NCHW44;
        benchmarker.set_param(param);
        benchmarker.execs({{1, 16, 112, 112, 4}, {}}).print();
    }
}
TEST(AARCH64, BenchmarkPoolingNchw44QInt8) {
    Benchmarker<PoolingForward> benchmarker(Arch::ARM64);
    PoolingForward::Param param;
    for (auto mode :
       {Mode::MAX, Mode::AVERAGE, Mode::AVERAGE_COUNT_EXCLUDE_PADDING}) {
       printf("pooling mode=%d\n", mode);
       param.mode = mode;
       param.pad_h = 1;
       param.pad_w = 1;
       param.window_h = 3;
       param.window_w = 3;
       param.stride_h = 2;
       param.stride_w = 2;
       param.format = PoolingForward::Param::Format::NCHW44;
       benchmarker.set_param(param);
       benchmarker.set_dtype(0, dtype::QuantizedS8(0.7f));
       benchmarker.set_dtype(1, dtype::QuantizedS8(0.7f));
       benchmarker.execs({{1, 16, 112, 112, 4}, {}}).print();
    }
}
#endif