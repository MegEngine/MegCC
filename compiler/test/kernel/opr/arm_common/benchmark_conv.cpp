#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_BENCHMARK

TEST(ARMCOMMON, BenchmarkChannelWiseNCHW4Int8) {
#ifdef __aarch64__
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
#else
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARMV7);
#endif
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));

    auto run = [&]() {
        for (size_t k : {3, 5})
            for (size_t h : {112, 56, 28, 14}) {
                for (size_t channel : {32, 64}) {
                    auto result = benchmarker.execs(
                            {{1, channel, h, h, 4},
                             {channel, 1, 1, k, k, 4},
                             {1, channel, 1, 1, 4},
                             {},
                             {}});
                    printf("Bench kernel %zu channel=%zu, hxw=%zux%zu\n", k, channel, h,
                           h);
                    result.print();
                }
            }
    };

    param.stride_h = 1;
    param.stride_w = 1;
    benchmarker.set_param(param);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>("S8_CHAN_WISE_STRD1_NCHW44"));
    printf("-----------stride: 1-----------\n");
    run();

    param.stride_h = 2;
    param.stride_w = 2;
    benchmarker.set_param(param);
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>("S8_CHAN_WISE_STRD2_NCHW44"));
    printf("-----------stride: 2-----------\n");
    run();
}

TEST(ARMCOMMON, BenchmarkDirectNCHW4Int8) {
#ifdef __aarch64__
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
#else
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARMV7);
#endif
    benchmarker.set_kernel_symbol("ArmCommon_direct.+");
    ConvBiasForward::Param param;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    param.pad_h = 1;
    param.pad_w = 1;
    param.nonlineMode = ConvBiasForward::Param::NonlineMode::IDENTITY;
    benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    benchmarker.set_before_exec_callback(
            megdnn::test::AlgoChecker<ConvBiasForward>("S8_NCHW44_DIRECT"));
    for (size_t s : {1, 2})
        for (size_t k : {2, 3, 5, 7})
            for (size_t h : {112, 56, 28, 14})
                for (size_t cdiv4 : {8, 16}) {
                    param.stride_h = s;
                    param.stride_w = s;
                    benchmarker.set_param(param);
                    auto result = benchmarker.execs(
                            {{1, cdiv4, h, h, 4},
                             {cdiv4, cdiv4, k, k, 4, 4},
                             {1, cdiv4, 1, 1, 4},
                             {},
                             {}});
                    printf("Bench kernel %zu, stride=%zu, ic/4=%zu, oc/4=%zu, "
                           "hxw=%zux%zu\n",
                           k, s, cdiv4, cdiv4, h, h);
                    result.print();
                }
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (size_t s : {1, 2})
        for (size_t k : {2, 3, 5, 7})
            for (size_t h : {112, 56, 28, 14})
                for (size_t cdiv4 : {8, 16})
                    for (size_t g : {2, 8}) {
                        param.stride_h = s;
                        param.stride_w = s;
                        benchmarker.set_param(param);
                        auto result = benchmarker.execs(
                                {{1, cdiv4, h, h, 4},
                                 {g, cdiv4 / g, cdiv4 / g, k, k, 4, 4},
                                 {1, cdiv4, 1, 1, 4},
                                 {},
                                 {}});
                        printf("Bench kernel %zu, group=%zu, stride=%zu, ic/4=%zu, "
                               "oc/4=%zu, "
                               "hxw=%zux%zu\n",
                               k, g, s, cdiv4, cdiv4, h, h);
                        result.print();
                    }
}

TEST(ARMCOMMON, BenchmarkChannelWiseNCHW4) {
#ifdef __aarch64__
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
#else
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARMV7);
#endif
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
            megdnn::test::AlgoChecker<ConvBiasForward>("F32_CHANNEL_WISE_NCHW44"));
    for (size_t k : {3, 5})
        for (size_t h : {112, 56, 28, 14}) {
            for (size_t channel : {32, 64}) {
                auto result = benchmarker.execs(
                        {{1, channel, h, h, 4},
                         {channel, 1, 1, k, k, 4},
                         {1, channel, 1, 1, 4},
                         {},
                         {}});
                printf("Bench kernel %zu channel=%zu, hxw=%zux%zu\n", k, channel, h, h);
                result.print();
            }
        }
}

TEST(ARMCOMMON, BenchmarkConvNCHWNCHW44) {
#ifdef __aarch64__
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
#else
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARMV7);
#endif
    benchmarker.set_kernel_symbol("ArmCommon_.+");
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    benchmarker.set_param(param);
    benchmarker.execs({{1, 3, 224, 224}, {8, 3, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}})
            .print();
    benchmarker.execs({{2, 3, 256, 160}, {6, 3, 3, 3, 4}, {1, 6, 1, 1, 4}, {}, {}})
            .print();
    ;
}

TEST(ARMCOMMON, BenchmarkNchwNchw44Int8) {
#ifdef __aarch64__
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARM64);
#else
    Benchmarker<ConvBiasForward> benchmarker(Arch::ARMV7);
#endif
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    benchmarker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    auto run = [&](const int stride, const std::vector<size_t>& filter_size_v) {
        param.stride_h = stride;
        param.stride_w = stride;
        benchmarker.set_param(param);
        for (size_t filter_size : filter_size_v) {
            printf("stride: %d, {1, 3, 224, 224}, {8, %zu, %zu, 3, 4}, {1, 8, 1, 1, "
                   "4}\n",
                   stride, filter_size, filter_size);
            benchmarker
                    .execs({{1, 3, 224, 224},
                            {8, filter_size, filter_size, 3, 4},
                            {1, 8, 1, 1, 4},
                            {},
                            {}})
                    .print();
        }
    };
    run(1, {1, 2, 3, 5, 7});
    run(2, {2, 3, 5, 7});
}

#endif
