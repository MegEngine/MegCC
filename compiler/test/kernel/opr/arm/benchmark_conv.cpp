#include "megbrain/reflection.h"
#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK

static void run_conv(
        size_t n, size_t ic, size_t hw, size_t oc, size_t filter_size, int stride,
        int pad, std::string cc_algo_name, std::string dnn_algo_name,
        ConvBiasForward::Param::Format fmt = ConvBiasForward::Param::Format::NCHW,
        bool qint8 = false,
        ConvBiasForward::Param::NonlineMode noline =
                ConvBiasForward::Param::NonlineMode::IDENTITY,
        const size_t group = 0) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
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
    if (group > 0) {
        param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    }

    benchmarker.set_param(param);
    if (!dnn_algo_name.empty()) {
        benchmarker.set_before_exec_callback(
                megdnn::test::AlgoChecker<ConvBiasForward>(dnn_algo_name.c_str()));
    }
    PerformanceResultPair result;
    if (fmt == ConvBiasForward::Param::Format::NCHW) {
        if (group == 0) {
            result = benchmarker.execs(
                    {{n, ic, hw, hw},
                     {oc, ic, filter_size, filter_size},
                     {1, oc, 1, 1},
                     {},
                     {}});
        } else {
            mgb_assert(oc % group == 0);
            mgb_assert(ic % group == 0);
            result = benchmarker.execs(
                    {{n, ic, hw, hw},
                     {group, oc / group, ic / group, filter_size, filter_size},
                     {1, oc, 1, 1},
                     {},
                     {}});
        }
    } else {
        mgb_assert(
                fmt == ConvBiasForward::Param::Format::NCHW44 ||
                fmt == ConvBiasForward::Param::Format::NCHW44_DOT);
        if (group == 0) {
            mgb_assert(oc % 4 == 0);
            mgb_assert(ic % 4 == 0);
            result = benchmarker.execs(
                    {{n, ic / 4, hw, hw, 4},
                     {oc / 4, ic / 4, filter_size, filter_size, 4, 4},
                     {1, oc / 4, 1, 1, 4},
                     {},
                     {}});
        } else {
            mgb_assert(oc % (group * 4) == 0);
            mgb_assert(ic % (group * 4) == 0);
            result = benchmarker.execs(
                    {{n, ic / 4, hw, hw, 4},
                     {group, oc / group / 4, ic / group / 4, filter_size, filter_size,
                      4, 4},
                     {1, oc / 4, 1, 1, 4},
                     {},
                     {}});
        }
    }
    printf("%s\n", benchmarker.format_result(result).c_str());
}

TEST(AARCH64, BenchmarkConv1x1NCHW4) {
    std::string cc_algo = "";
    std::string dnn_algo = "";
    run_conv(
            1, 32, 32, 32, 1, 1, 0, cc_algo, dnn_algo,
            ConvBiasForward::Param::Format::NCHW44);
}

TEST(AARCH64, BenchmarkConv1x1GroupNCHW4) {
    std::string cc_algo = "";
    std::string dnn_algo = "";
    run_conv(
            1, 32, 256, 32, 1, 1, 0, cc_algo, dnn_algo,
            ConvBiasForward::Param::Format::NCHW44, false,
            ConvBiasForward::Param::NonlineMode::RELU, 2);
}

TEST(AARCH64, BenchmarkConv1x1NCHW4Dot) {
    std::string cc_algo = "Arm64_kernel_dot_conv2d_conv1x1_.*";
    std::string dnn_algo = "";
    run_conv(
            1, 120, 120, 96, 1, 1, 0, cc_algo, dnn_algo,
            ConvBiasForward::Param::Format::NCHW44_DOT, true,
            ConvBiasForward::Param ::NonlineMode::RELU);
}

TEST(AARCH64, BenchmarkConvDotNCHWNCHW44Stride1) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44_DOT;
    benchmarker.set_param(param);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>("ARMDOTS8_NCHW_NCHW44"));
    benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    benchmarker.execs({{1, 3, 224, 224}, {8, 3, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}})
            .print();
}

TEST(AARCH64, BenchmarkConvF32Winograd) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);

    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    benchmarker.set_param(param);
    std::vector<std::vector<std::string>> algo_pairs = {
            {".*_winograd_f23", "WINOGRAD_NCHW44:AARCH64_F32_MK4_4x16:4:2:24"},
            {".*_winograd_f43", "WINOGRAD_NCHW44:AARCH64_F32_MK4_4x16:4:4:68"},
            {".*_winograd_f63", "WINOGRAD_NCHW44:AARCH64_F32_MK4_4x16:4:6:16"}};

    for (auto algo : algo_pairs) {
        printf("megcc algo: %s VS megdnn algo: %s\n", algo[0].c_str(), algo[1].c_str());
        for (size_t Channel : {32, 256}) {
            for (size_t HW : {56, 28, 14}) {
                benchmarker.set_kernel_symbol(algo[0]);
                benchmarker.set_before_exec_callback(
                        megdnn::test::AlgoChecker<ConvBiasForward>(algo[1].c_str()));

                auto result = benchmarker.execs(
                        {{1, Channel / 4, HW, HW, 4},
                         {Channel / 4, Channel / 4, 3, 3, 4, 4},
                         {1, Channel / 4, 1, 1, 4},
                         {},
                         {}});
                result.print();
            }
        }
    }
}

TEST(AARCH64, BenchmarkConvBiasIm2col) {
    std::string cc_algo = "Arm64_kernel_conv2d_im2col.*";
    std::string dnn_algo = "IM2COLMATMUL:AARCH64_F32K8X12X1:192";
    run_conv(1, 64, 56, 64, 3, 2, 1, cc_algo, dnn_algo);
    run_conv(1, 64, 56, 64, 3, 1, 1, cc_algo, dnn_algo);
}

TEST(AARCH64, BenchmarkConvBiasIm2colNCHW44) {
    std::string cc_algo = "Arm64_kernel_conv2d_im2col.*";
    std::string dnn_algo = "IM2COLMATMUL:AARCH64_F32_MK4_K8X12X1:192";
    auto fmt = ConvBiasForward::Param::Format::NCHW44;
    run_conv(1, 64, 56, 64, 3, 2, 1, cc_algo, dnn_algo, fmt);
    run_conv(1, 64, 56, 64, 3, 1, 1, cc_algo, dnn_algo, fmt);
}

TEST(AARCH64, BenchmarkConvBias1x1NCHW44I8mm) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64_WITH_I8MM);
    UniformIntRNG rng(-127, 127);
    benchmarker.set_rng(0, &rng);
    benchmarker.set_rng(1, &rng);
    benchmarker.set_rng(2, &rng);
    benchmarker.set_kernel_symbol("Arm64_kernel_im2col_i8mm_m8n12k8_.+");
    benchmarker.set_before_exec_callback(megdnn::test::AlgoChecker<ConvBiasForward>(
            "CONV1x1:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD.*"));

    benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));

    ConvBiasForward::Param param, dnn_param;
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44;

    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    printf("----------Dense----------\n");
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        printf("%s:\n", mgb::reflection::nameOfEnumValue(noline).c_str());
        benchmarker.set_param(param);
        dnn_param = param;
        dnn_param.format = ConvolutionForward::Param::Format::NCHW44_DOT;
        benchmarker.set_dnn_param(dnn_param);
        for (size_t ic : {16, 64}) {
            for (size_t oc : {32}) {
                for (size_t hw : {224}) {
                    benchmarker
                            .execs({{2, ic, hw, hw, 4},
                                    {oc, ic, 1, 1, 4, 4},
                                    {},
                                    {},
                                    {}})
                            .print();
                    benchmarker
                            .execs({{1, ic, hw, hw, 4},
                                    {oc, ic, 1, 1, 4, 4},
                                    {1, oc, 1, 1, 4},
                                    {},
                                    {}})
                            .print();
                }
            }
        }
    }

    printf("----------Group----------\n");
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    const int group = 3;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        printf("%s:\n", mgb::reflection::nameOfEnumValue(noline).c_str());
        benchmarker.set_param(param);
        dnn_param = param;
        dnn_param.format = ConvolutionForward::Param::Format::NCHW44_DOT;
        benchmarker.set_dnn_param(dnn_param);
        for (size_t ic : {32, 64}) {
            for (size_t oc : {32}) {
                for (size_t hw : {128}) {
                    benchmarker
                            .execs({{2, ic * group, hw, hw, 4},
                                    {group, oc, ic, 1, 1, 4, 4},
                                    {},
                                    {},
                                    {}})
                            .print();
                    benchmarker
                            .execs({{1, ic * group, hw, hw, 4},
                                    {group, oc, ic, 1, 1, 4, 4},
                                    {1, oc * group, 1, 1, 4},
                                    {},
                                    {}})
                            .print();
                }
            }
        }
    }
}

TEST(AARCH64, BenchmarkConvBiasIm2colNCHW44I8mm) {
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64_WITH_I8MM);
    UniformIntRNG rng(-127, 127);
    benchmarker.set_rng(0, &rng);
    benchmarker.set_rng(1, &rng);
    benchmarker.set_rng(2, &rng);
    benchmarker.set_kernel_symbol("Arm64_kernel_im2col_i8mm_m8n12k8_.+");
    benchmarker.set_before_exec_callback(megdnn::test::AlgoChecker<ConvBiasForward>(
            "IM2COLMATMUL:AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD.*"));

    benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));

    ConvBiasForward::Param param, dnn_param;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44;

    for (size_t stride : {2, 1}) {
        printf("stride: %zu\n", stride);
        param.stride_h = stride;
        param.stride_w = stride;
        param.sparse = ConvBiasForward::Param::Sparse::DENSE;
        printf("----------Dense----------\n");
        for (size_t kernel : {2, 3}) {
            param.pad_h = kernel / 2;
            param.pad_w = kernel / 2;
            for (auto noline : {ConvBiasForward::Param::NonlineMode::RELU}) {
                param.nonlineMode = noline;
                printf("%s:\n", mgb::reflection::nameOfEnumValue(noline).c_str());
                benchmarker.set_param(param);
                dnn_param = param;
                dnn_param.format = ConvolutionForward::Param::Format::NCHW44_DOT;
                benchmarker.set_dnn_param(dnn_param);
                for (size_t ic : {16, 32}) {
                    for (size_t oc : {32}) {
                        for (size_t hw : {224}) {
                            benchmarker
                                    .execs({{2, ic, hw, hw, 4},
                                            {oc, ic, kernel, kernel, 4, 4},
                                            {},
                                            {},
                                            {}})
                                    .print();
                            benchmarker
                                    .execs({{1, ic, hw, hw, 4},
                                            {oc, ic, kernel, kernel, 4, 4},
                                            {1, oc, 1, 1, 4},
                                            {},
                                            {}})
                                    .print();
                        }
                    }
                }
            }
        }

        printf("----------Group----------\n");
        param.sparse = ConvBiasForward::Param::Sparse::GROUP;
        const int group = 3;
        for (size_t kernel : {5}) {
            param.pad_h = kernel / 2;
            param.pad_w = kernel / 2;
            for (auto noline : {ConvBiasForward::Param::NonlineMode::IDENTITY}) {
                param.nonlineMode = noline;
                printf("%s:\n", mgb::reflection::nameOfEnumValue(noline).c_str());
                benchmarker.set_param(param);
                dnn_param = param;
                dnn_param.format = ConvolutionForward::Param::Format::NCHW44_DOT;
                benchmarker.set_dnn_param(dnn_param);
                for (size_t ic : {16}) {
                    for (size_t oc : {32}) {
                        for (size_t hw : {128}) {
                            benchmarker
                                    .execs({{2, ic * group, hw, hw, 4},
                                            {group, oc, ic, kernel, kernel, 4, 4},
                                            {},
                                            {},
                                            {}})
                                    .print();
                            benchmarker
                                    .execs({{1, ic * group, hw, hw, 4},
                                            {group, oc, ic, kernel, kernel, 4, 4},
                                            {1, oc * group, 1, 1, 4},
                                            {},
                                            {}})
                                    .print();
                        }
                    }
                }
            }
        }
    }
}

#endif
