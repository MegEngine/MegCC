/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/benchmark_conv.cpp
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

static void run_conv(size_t n, size_t ic, size_t hw, size_t oc,
                     size_t filter_size, int stride, int pad,
                     std::string cc_algo_name, std::string dnn_algo_name,
                     ConvBiasForward::Param::Format fmt =
                             ConvBiasForward::Param::Format::NCHW,
                     bool qint8 = false,
                     ConvBiasForward::Param::NonlineMode noline =
                             ConvBiasForward::Param::NonlineMode::IDENTITY) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::BAREMETAL);
    if (!cc_algo_name.empty()) {
        benchmarker.set_kernel_symbol(cc_algo_name);
    }
    if (qint8) {
        benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(40.25f));
    }
    ConvBiasForward::Param param;
    param.pad_h = pad;
    param.pad_w = pad;
    param.stride_h = stride;
    param.stride_w = stride;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = fmt;
    param.nonlineMode = noline;

    benchmarker.set_param(param);
    if (!dnn_algo_name.empty()) {
        benchmarker.set_before_exec_callback(
                megdnn::test::AlgoChecker<ConvBiasForward>(
                        dnn_algo_name.c_str()));
    }
    if (fmt == ConvBiasForward::Param::Format::NCHW) {
        benchmarker.execs({{n, ic, hw, hw},
                                    {oc, ic, filter_size, filter_size},
                                    {1, oc, 1, 1},
                                    {},
                                    {}}).print();

    } else {
        mgb_assert(fmt == ConvBiasForward::Param::Format::NCHW44 ||
                   fmt == ConvBiasForward::Param::Format::NCHW44_DOT);
        mgb_assert(oc % 4 == 0);
        mgb_assert(ic % 4 == 0);
        benchmarker.execs(
                {{n, ic / 4, hw, hw, 4},
                 {oc / 4, ic / 4, filter_size, filter_size, 4, 4},
                 {1, oc / 4, 1, 1, 4},
                 {},
                 {}}).print();
    }
}

TEST(GI, BenchmarkConv1x1NCHW4) {
    std::string cc_algo = "GI_kernel_conv2d_conv1x1_NCHW44.+";
    std::string dnn_algo = "CONV1x1:FB_GI_F32_MK4_PACK_4x12:24";
    run_conv(1, 32, 32, 32, 1, 1, 0, cc_algo, dnn_algo,
             ConvBiasForward::Param::Format::NCHW44);
}

TEST(GI, BenchmarkChannelWiseNCHW4) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("GI_chanwise_kernel_conv2d.*_NCHW44.*");
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    benchmarker.set_param(param);

    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>(
                    "F32_CHANNEL_WISE_NCHW44"));
    for (size_t k : {3, 5})
        for (size_t h : {112, 56, 28, 14}) {
            for (size_t channel : {32, 64}) {
                auto result = benchmarker.execs({{1, channel, h, h, 4},
                                                 {channel, 1, 1, k, k, 4},
                                                 {1, channel, 1, 1, 4},
                                                 {},
                                                 {}});
                printf("Bench kernel %zu channel=%zu, hxw=%zux%zu\n", k,
                       channel, h, h);
                result.print();
            }
        }
}

TEST(GI, BenchmarkConvNCHWNCHW44) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("^GI_.*nchw_nchw44.*");
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>(
                    "F32_CONV_NCHW_NCHW44"));
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    benchmarker.set_param(param);
    benchmarker.execs(
            {{1, 3, 224, 224}, {8, 3, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}}).print();
}

TEST(GI, BenchmarkConvF32Winograd) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("^GI.*_winograd_f23.*");

    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    benchmarker.set_param(param);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>(
                    "WINOGRAD_NCHW44:FB_GI_F32_MK4_4x8:4:2"));
    for (size_t Channel : {32, 256}) {
        for (size_t HW : {56, 28, 14}) {
                    benchmarker.execs({{1, Channel / 4, HW, HW, 4},
                                       {Channel / 4, Channel / 4, 3, 3, 4, 4},
                                       {1, Channel / 4, 1, 1, 4},
                                       {},
                                       {}}).print();
        }
    }
}

TEST(GI, BenchmarkConvBiasIm2col) {
    std::string cc_algo = "GI_kernel_conv2d_im2col.*";
    std::string dnn_algo = "IM2COLMATMUL:FB_GI_F32_4x12";
    run_conv(1, 64, 56, 64, 3, 2, 1, cc_algo, dnn_algo);
    run_conv(1, 64, 56, 64, 3, 1, 1, cc_algo, dnn_algo);
}

TEST(GI, BenchmarkConvBiasIm2colNCHW44) {
    std::string cc_algo = "GI_kernel_conv2d_im2col.*";
    std::string dnn_algo = "IM2COLMATMUL:FB_GI_F32_MK4_PACK_4x12";
    auto fmt = ConvBiasForward::Param::Format::NCHW44;
    run_conv(1, 64, 56, 64, 3, 2, 1, cc_algo, dnn_algo, fmt);
    run_conv(1, 64, 56, 64, 3, 1, 1, cc_algo, dnn_algo, fmt);
}

#endif
