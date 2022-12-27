/**
 * \file
 * compiler/test/kernel/opr/arm/benchmark_elemwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;
#ifdef ENABLE_KERNEL_BENCHMARK

TEST(AARCH64, BenchmarkElemwise) {
    Benchmarker<ElemwiseForward> benchmarker(Arch::ARM64);
    benchmarker.set_kernel_symbol("ArmCommon.*");
    ElemwiseForward::Param param;
    param.mode = MODE::SIGMOID;
    benchmarker.set_param(param);
    benchmarker.execs({{1, 3, 160, 160}, {}}).print();
    benchmarker.execs({{1, 3, 160, 160}, {}}).print();
}
TEST(AARCH64, BenchmarkElemwise_asm) {
    Benchmarker<ElemwiseForward> benchmarker(Arch::ARM64);
    benchmarker.set_kernel_symbol("Arm64.*");
    ElemwiseForward::Param param;
    param.mode = MODE::SIGMOID;
    benchmarker.set_param(param);
    benchmarker.execs({{1, 3, 160, 160}, {}}).print();
    benchmarker.execs({{1, 3, 160, 160}, {}}).print();
}
#endif
