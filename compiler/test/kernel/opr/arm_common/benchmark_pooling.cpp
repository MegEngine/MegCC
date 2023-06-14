#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = PoolingForward::Param::Mode;
#ifdef ENABLE_KERNEL_BENCHMARK

TEST(ARMCOMMON, BenchmarkPoolingNchw44Int8) {
#ifdef __aarch64__
    Benchmarker<PoolingForward> benchmarker(Arch::ARM64, 0);
#else
    Benchmarker<PoolingForward> benchmarker(Arch::ARMV7, 0);
#endif
    PoolingForward::Param param;
    UniformIntRNG rng(-127, 127);
    benchmarker.set_rng(0, &rng);
    benchmarker.set_kernel_symbol("ArmCommon_FilterX_modeX_.*");
    param.format = param::Pooling::Format::NCHW44;
    auto run = [&](megdnn::DType dtype, std::string dtype_name, Mode mode,
                   std::string mode_name) {
        for (size_t window : {2, 3, 4, 5})
            for (size_t stride : {1, 2}) {
                param.mode = mode;
                benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
                param.pad_h = 1;
                param.pad_w = 1;
                param.window_h = window;
                param.window_w = window;
                param.stride_h = stride;
                param.stride_w = stride;
                benchmarker.set_param(param);
                printf("Bench hw=112, window= %zu, dtype=%s, mode=%s stride=%zu \n",
                       window, dtype_name.c_str(), mode_name.c_str(), stride);
                benchmarker.set_before_exec_callback(
                        megdnn::test::AlgoChecker<PoolingForward>(
                                ("ARM_POOLING_FILTER" + std::to_string(window) +
                                 "_MODEX_STRIDEX_NCHW44")
                                        .c_str()));
                benchmarker.execs({{1, 1, 112, 112, 4}, {}}).print();
            }
    };
    run(dtype::Int8(), "int8", Mode::MAX, "max");
    run(dtype::QuantizedS8(1.6f), "qint8", Mode::MAX, "max");
    run(dtype::QuantizedS8(1.6f), "qint8", Mode::AVERAGE, "avg");
}
#endif