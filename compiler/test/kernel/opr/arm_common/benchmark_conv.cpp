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

#endif
