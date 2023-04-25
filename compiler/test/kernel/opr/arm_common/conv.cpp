#include "test/kernel/common/checker.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(ARMCOMMON, ConvBiasChannelWiseNCHW4Int8) {
#ifdef __aarch64__
    Checker<ConvBiasForward> checker(Arch::ARM64);
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7);
#endif
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    checker.set_kernel_symbol("ArmCommon_chanwise.+");
    ConvBiasForward::Param param;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (size_t pad : {0, 1, 2})
        for (size_t stride : {1, 2}) {
            for (size_t kernel : {3, 5}) {
                param.pad_h = pad;
                param.pad_w = pad;
                param.nonlineMode = ConvBiasForward::Param::NonlineMode::RELU;
                param.stride_h = stride;
                param.stride_w = stride;
                checker.set_param(param);
                checker.execs(
                        {{2, 8, 13, 23, 4},
                         {8, 1, 1, kernel, kernel, 4},
                         {1, 8, 1, 1, 4},
                         {},
                         {}});
                checker.execs(
                        {{2, 8, 14, 28, 4}, {8, 1, 1, kernel, kernel, 4}, {}, {}, {}});
                param.nonlineMode = ConvBiasForward::Param::NonlineMode::IDENTITY;
                checker.set_param(param);
                checker.execs(
                        {{4, 3, 5, 11, 4},
                         {3, 1, 1, kernel, kernel, 4},
                         {1, 3, 1, 1, 4},
                         {},
                         {}});
                checker.execs(
                        {{4, 3, 5, 11, 4}, {3, 1, 1, kernel, kernel, 4}, {}, {}, {}});
            }
        }
}

// vim: syntax=cpp.doxygen
