#include <list>
#include "test/kernel/common/checker.h"
#include "test/kernel/common/dnn_proxy_algo.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = PoolingForward::Param::Mode;

TEST(ARMCOMMON, PoolingNchw44Int8) {
#ifdef __aarch64__
    Checker<PoolingForward> checker(Arch::ARM64, 0);
#else
    Checker<PoolingForward> checker(Arch::ARMV7, 0);
#endif
    PoolingForward::Param param;
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    checker.set_kernel_symbol("ArmCommon_FilterX_modeX_.*");
    param.format = param::Pooling::Format::NCHW44;

    auto run = [&](std::list<megdnn::DType> dtypes, std::list<Mode> modes) {
        for (auto dtype : dtypes)
            for (auto mode : modes)
                for (size_t window : {2, 3, 4, 5})
                    for (size_t stride : {1, 2})
                        for (size_t pad : {size_t(0), size_t(window / 2)}) {
                            param.mode = mode;
                            checker.set_dtype(0, dtype).set_dtype(1, dtype);
                            param.pad_h = pad;
                            param.pad_w = pad;
                            param.window_h = window;
                            param.window_w = window;
                            param.stride_h = stride;
                            param.stride_w = stride;
                            checker.set_param(param);
                            checker.set_before_exec_callback(
                                    megdnn::test::AlgoChecker<PoolingForward>(
                                            ("ARM_POOLING_FILTER" +
                                             std::to_string(window) +
                                             "_MODEX_STRIDEX_NCHW44")
                                                    .c_str()));
                            checker.execs({{2, 3, 5, 5, 4}, {}});
                            checker.execs({{1, 2, 7, 7, 4}, {}});
                        }
    };

    run({dtype::Int8()}, {Mode::MAX});
    run({dtype::QuantizedS8(0.35f), dtype::QuantizedS8(1.6f)},
        {Mode::AVERAGE, Mode::MAX});
}