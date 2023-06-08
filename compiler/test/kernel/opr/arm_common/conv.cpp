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

TEST(ARMCOMMON, ConvBiasChannelWiseNCHW4K3) {
#ifdef __aarch64__
    Checker<ConvBiasForward> checker(Arch::ARM64);
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7);
#endif
    checker.set_kernel_symbol("ArmCommon_.+");
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        for (size_t stride : {1, 2}) {
            param.nonlineMode = noline;
            param.stride_h = stride;
            param.stride_w = stride;
            checker.set_param(param);
            checker.execs(
                    {{2, 8, 28, 28, 4}, {8, 1, 1, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}});
            checker.execs({{2, 8, 14, 28, 4}, {8, 1, 1, 3, 3, 4}, {}, {}, {}});
            checker.execs(
                    {{4, 3, 5, 11, 4}, {3, 1, 1, 3, 3, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs({{4, 3, 5, 11, 4}, {3, 1, 1, 3, 3, 4}, {}, {}, {}});
            checker.execs(
                    {{2, 8, 24, 28, 4}, {8, 1, 1, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{1, 3, 23, 23, 4}, {3, 1, 1, 3, 3, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{3, 3, 23, 23, 4}, {3, 1, 1, 3, 3, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{1, 3, 14, 14, 4}, {3, 1, 1, 3, 3, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{4, 3, 14, 14, 4}, {3, 1, 1, 3, 3, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{4, 5, 34, 7, 4}, {5, 1, 1, 3, 3, 4}, {1, 5, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{2, 8, 14, 28, 4}, {8, 1, 1, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{2, 8, 28, 28, 4}, {8, 1, 1, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}});
        }
    }
}

TEST(ARMCOMMON, ConvBiasChannelWiseNCHW4K5) {
#ifdef __aarch64__
    Checker<ConvBiasForward> checker(Arch::ARM64);
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7);
#endif
    checker.set_kernel_symbol("ArmCommon_.+");
    ConvBiasForward::Param param;
    param.pad_h = 2;
    param.pad_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (size_t stride : {1, 2})
        for (auto noline :
             {ConvBiasForward::Param::NonlineMode::IDENTITY,
              ConvBiasForward::Param::NonlineMode::RELU,
              ConvBiasForward::Param::NonlineMode::H_SWISH}) {
            param.stride_h = stride;
            param.stride_w = stride;
            param.nonlineMode = noline;
            checker.set_param(param);
            checker.execs(
                    {{2, 3, 6, 6, 4}, {3, 1, 1, 5, 5, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{2, 3, 6, 7, 4}, {3, 1, 1, 5, 5, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{2, 3, 7, 6, 4}, {3, 1, 1, 5, 5, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{2, 3, 7, 7, 4}, {3, 1, 1, 5, 5, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs(
                    {{2, 3, 17, 17, 4}, {3, 1, 1, 5, 5, 4}, {1, 3, 1, 1, 4}, {}, {}});
        }
}

TEST(ARMCOMMON, ConvBiasNCHWNCHW44) {
#ifdef __aarch64
    Checker<ConvBiasForward> checker(Arch::ARM64);
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7);
#endif
    ConvBiasForward::Param param;
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("ArmCommon_.+");
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    checker.set_param(param);
    for (size_t stride : {1, 2})
        for (auto mode :
             {ConvBiasForward::Param::NonlineMode::IDENTITY,
              ConvBiasForward::Param::NonlineMode::RELU,
              ConvBiasForward::Param::NonlineMode::H_SWISH})
            for (size_t filter_size : {2, 3, 5})
                for (size_t ic : {3})
                    for (size_t iw = 11; iw < 33; iw++) {
                        size_t pad = filter_size / 2;
                        size_t oc_div4 = 3;
                        param.stride_h = stride;
                        param.stride_w = stride;
                        param.pad_h = pad;
                        param.pad_w = pad;
                        param.nonlineMode = mode;
                        checker.set_param(param);
                        checker.execs(
                                {{2, ic, 11, iw},
                                 {oc_div4, filter_size, filter_size, ic, 4},
                                 {1, oc_div4, 1, 1, 4},
                                 {},
                                 {}});
                    }
}

TEST(ARMCOMMON, ConvBiasDirectNCHW44Int8) {
#ifdef __aarch64__
    Checker<ConvBiasForward> checker(Arch::ARM64);
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7);
#endif
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("ArmCommon_direct.+");
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    param.nonlineMode = ConvBiasForward::Param::NonlineMode::IDENTITY;
    for (size_t stride : {1, 2})
        for (size_t kernel : {2, 3, 5, 7}) {
            param.stride_h = stride;
            param.stride_w = stride;
            checker.set_param(param);
            checker.execs(
                    {{2, 6, 10, 10, 4},
                     {2, 3, 3, kernel, kernel, 4, 4},
                     {1, 6, 1, 1, 4},
                     {},
                     {}});
            checker.execs(
                    {{2, 2, 10, 10, 4},
                     {2, 3, 1, kernel, kernel, 4, 4},
                     {1, 6, 1, 1, 4},
                     {},
                     {}});
        }
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    for (size_t stride : {1, 2})
        for (size_t kernel : {2, 3, 5, 7}) {
            param.stride_h = stride;
            param.stride_w = stride;
            checker.set_param(param);
            checker.execs(
                    {{2, 7, 10, 10, 4},
                     {7, 7, kernel, kernel, 4, 4},
                     {1, 7, 1, 1, 4},
                     {},
                     {}});
            checker.execs(
                    {{2, 3, 13, 10, 4},
                     {13, 3, kernel, kernel, 4, 4},
                     {1, 13, 1, 1, 4},
                     {},
                     {}});
        }
}

TEST(ARMCOMMON, WinogradF23NCHW44MK8Int8) {
#ifdef __aarch64__
    Checker<ConvBiasForward> checker(Arch::ARM64, 0);
    checker.set_before_exec_callback(megdnn::test::AlgoChecker<ConvBiasForward>(
            "WINOGRAD_NCHW44:AARCH64_INT16X16X32_MK8_8X8:8:2:32"));
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7, 0);
    checker.set_before_exec_callback(megdnn::test::AlgoChecker<ConvBiasForward>(
            "WINOGRAD_NCHW44:ARMV7_INT16X16X32_MK8_4X8:8:2:32"));
#endif
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol(".+_winograd_f23_int8_nchw44_mk8");
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.nonlineMode = ConvBiasForward::Param::NonlineMode::IDENTITY;
    size_t kernel = 3;
    param.stride_h = 1;
    param.stride_w = 1;

    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    checker.set_param(param);
    checker.execs(
            {{1, 2, 4, 4, 4}, {2, 2, kernel, kernel, 4, 4}, {1, 2, 1, 1, 4}, {}, {}});
    checker.execs(
            {{2, 8, 10, 10, 4}, {8, 8, kernel, kernel, 4, 4}, {1, 8, 1, 1, 4}, {}, {}});
    checker.execs(
            {{1, 2, 73, 11, 4},
             {12, 2, kernel, kernel, 4, 4},
             {1, 12, 1, 1, 4},
             {},
             {}});

    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.execs(
            {{1, 8, 5, 3, 4},
             {2, 4, 4, kernel, kernel, 4, 4},
             {1, 8, 1, 1, 4},
             {},
             {}});
    checker.execs(
            {{2, 16, 10, 10, 4},
             {4, 2, 4, kernel, kernel, 4, 4},
             {1, 8, 1, 1, 4},
             {},
             {}});
    checker.execs(
            {{1, 4, 73, 11, 4},
             {2, 4, 2, kernel, kernel, 4, 4},
             {1, 8, 1, 1, 4},
             {},
             {}});
}

TEST(ARMCOMMON, ConvBiasNCHWNCHW44Int8) {
#ifdef __aarch64
    Checker<ConvBiasForward> checker(Arch::ARM64);
#else
    Checker<ConvBiasForward> checker(Arch::ARMV7);
#endif
    ConvBiasForward::Param param;
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("ArmCommon_.+");
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    checker.set_param(param);
    auto run = [&](const int stride, const std::vector<size_t>& filter_size_v) {
        for (auto mode :
             {ConvBiasForward::Param::NonlineMode::IDENTITY,
              ConvBiasForward::Param::NonlineMode::RELU})
            for (size_t iw : {8, 16, 24, 32, 9, 19, 29, 39})
                for (size_t ic : {1, 2, 3})
                    for (size_t filter_size : filter_size_v) {
                        size_t pad = filter_size / 2;
                        size_t oc_div4 = 5;
                        param.stride_h = stride;
                        param.stride_w = stride;
                        param.pad_h = pad;
                        param.pad_w = pad;
                        param.nonlineMode = mode;
                        checker.set_param(param);
                        checker.execs(
                                {{1, ic, 10, iw},
                                 {oc_div4, filter_size, filter_size, ic, 4},
                                 {1, oc_div4, 1, 1, 4},
                                 {},
                                 {}});
                        checker.execs(
                                {{1, ic, 21, iw},
                                 {oc_div4, filter_size, filter_size, ic, 4},
                                 {1, oc_div4, 1, 1, 4},
                                 {},
                                 {}});
                    }
    };
    run(1, {1, 2, 3, 5, 7});
    run(2, {2, 3, 5, 7});
}

// vim: syntax=cpp.doxygen
