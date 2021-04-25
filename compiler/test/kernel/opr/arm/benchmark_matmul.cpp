/**
 * \file
 * compiler/test/kernel/opr/arm/benchmark_matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK
TEST(ARMCOMMON, BenchmarkFP32GEVM_NN) {
    Benchmarker<MatrixMulForward> benchmarker(Arch::ARM64);
    benchmarker.set_kernel_symbol("ArmCommon_kernel_gevm.*");
    size_t m = 1, k = 1000, n = 1024;
    size_t a0 = m, a1 = k, b0 = k, b1 = n;
    MatrixMulForward::Param param;
    param.transposeA = false;
    param.transposeB = false;
    benchmarker.set_param(param);
    if (param.transposeA) {
        a0 = k, a1 = m;
    }
    if (param.transposeB) {
        b0 = n, b1 = k;
    }
    benchmarker.execs({{a0, a1}, {b0, b1}, {}}).print();
}

TEST(ARMCOMMON, BenchmarkFP32GEVM_NT) {
    Benchmarker<MatrixMulForward> benchmarker(Arch::ARM64);
    benchmarker.set_kernel_symbol("ArmCommon_kernel_gevm.*");
    size_t m = 1, k = 1000, n = 1024;
    size_t a0 = m, a1 = k, b0 = k, b1 = n;
    MatrixMulForward::Param param;
    param.transposeA = false;
    param.transposeB = true;
    benchmarker.set_param(param);
    if (param.transposeA) {
        a0 = k, a1 = m;
    }
    if (param.transposeB) {
        b0 = n, b1 = k;
    }
    benchmarker.execs({{a0, a1}, {b0, b1}, {}}).print();
}

TEST(AARCH64, BenchmarkFP32M4N16K4) {
    Benchmarker<MatrixMulForward> benchmarker(Arch::ARM64);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<MatrixMulForward>(
                    "AARCH64_F32_MK4_4x16"));
    for (size_t m : {64, 128})
        for (size_t n : {256, 384})
            for (size_t k : {128, 256}) {
                MatrixMulForward::Param param;
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK4;
                benchmarker.set_param(param);
                auto result = benchmarker.execs(
                        {{m / 4, k / 4, 4, 4}, {k / 4, n, 4}, {}});
                printf("megcc result time = %f, throughput %f Gops, %f mbps\n",
                       result.megcc_performance.kernel_time_ms,
                       result.megcc_performance.compute_throughput_gops,
                       result.megcc_performance.memory_throughput_mbps);
                printf("dnn result time = %f, throughput %f Gops, %f mbps\n",
                       result.dnn_performance.kernel_time_ms,
                       result.dnn_performance.compute_throughput_gops,
                       result.dnn_performance.memory_throughput_mbps);
            }
}

TEST(AARCH64, BenchmarkFP32GEMM) {
    Benchmarker<MatrixMulForward> benchmarker(Arch::ARM64);
    benchmarker.set_kernel_symbol("Arm64_kernel_fp32_matmul_8x12.*");
    size_t m = 1000, k = 1000, n = 1024;
    size_t a0 = m, a1 = k, b0 = k, b1 = n;
    MatrixMulForward::Param param;
    param.transposeA = false;
    param.transposeB = false;
    benchmarker.set_param(param);
    if (param.transposeA) {
        a0 = k, a1 = m;
    }
    if (param.transposeB) {
        b0 = n, b1 = k;
    }
    benchmarker.execs({{a0, a1}, {b0, b1}, {}}).print();
}

#endif
