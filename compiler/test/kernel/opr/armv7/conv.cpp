#include "test/kernel/common/checker.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

//! Test implement in `armv7' directory
TEST(ARMV7, ConvBiasNCHWNCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    ConvBiasForward::Param param;
    checker.set_epsilon(1e-4);
    param.stride_h = 2;
    param.stride_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    checker.set_param(param);
    for (auto mode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU})
        for (size_t filter_size : {3})
            for (size_t ic : {3})
                for (size_t iw = 13; iw < 33; iw++) {
                    size_t pad = filter_size / 2;
                    size_t oc_div4 = 3;
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

TEST(ARMV7, Conv1x1NCHW44) {
    Checker<ConvolutionForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol("Armv7_kernel_conv2d_conv1x1_NCHW44.+");
    ConvolutionForward::Param param;
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44;
    checker.set_param(param);
    checker.execs({{2, 3, 5, 11, 4}, {3, 3, 1, 1, 4, 4}, {}});
}

TEST(ARMV7, ConvBiasIm2col) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol("Armv7_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID})
        for (size_t n : {1, 3})
            for (size_t oc : {7, 13})
                for (size_t ic : {7, 13})
                    for (size_t stride : {1, 2, 3})
                        for (size_t filter_size : {2, 3, 5})
                            for (size_t hw : {7, 13, 23}) {
                                param.nonlineMode = noline;
                                param.pad_h = filter_size / 2;
                                param.pad_w = filter_size / 2;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs(
                                        {{n, ic, hw, hw},
                                         {oc, ic, filter_size, filter_size},
                                         {1, oc, 1, 1},
                                         {},
                                         {}});
                            }
    {
        param.pad_h = 1;
        param.pad_w = 1;
        param.stride_h = 1;
        param.stride_w = 1;
        checker.set_param(param);
        checker.execs({{1, 64, 56, 56}, {64, 64, 3, 3}, {1, 64, 1, 1}, {}, {}});
    }
}

TEST(ARMV7, ConvBiasIm2colGroup) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol("Armv7_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;

    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID})
        for (size_t n : {1, 3})
            for (size_t group : {3, 7})
                for (size_t ocpg : {3, 13})
                    for (size_t icpg : {1, 7})
                        for (size_t stride : {1, 2})
                            for (size_t filter_size : {3})
                                for (size_t hw : {22, 33}) {
                                    param.nonlineMode = noline;
                                    param.sparse =
                                            ConvBiasForward::Param::Sparse::GROUP;
                                    param.pad_h = filter_size / 2;
                                    param.pad_w = filter_size / 2;
                                    param.stride_h = stride;
                                    param.stride_w = stride;
                                    checker.set_param(param);
                                    checker.execs(
                                            {{n, group * icpg, hw, hw},
                                             {group, ocpg, icpg, filter_size,
                                              filter_size},
                                             {1, group * ocpg, 1, 1},
                                             {},
                                             {}});
                                }
}

TEST(ARMV7, ConvBiasIm2colNCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol("Armv7_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;

    for (auto noline : {
                 ConvBiasForward::Param::NonlineMode::RELU,
                 ConvBiasForward::Param::NonlineMode::IDENTITY,
                 ConvBiasForward::Param::NonlineMode::SIGMOID,
         })
        for (size_t n : {1, 3})
            for (size_t oc : {4, 8, 20})
                for (size_t ic : {4, 20})
                    for (size_t stride : {1, 2, 3})
                        for (size_t filter_size : {2, 3, 5})
                            for (size_t hw : {7, 13, 23}) {
                                param.nonlineMode = noline;
                                param.pad_h = filter_size / 2;
                                param.pad_w = filter_size / 2;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                checker.set_param(param);
                                checker.execs(
                                        {{n, ic / 4, hw, hw, 4},
                                         {oc / 4, ic / 4, filter_size, filter_size, 4,
                                          4},
                                         {1, oc / 4, 1, 1, 4},
                                         {},
                                         {}});
                            }
    {
        param.pad_h = 1;
        param.pad_w = 1;
        param.stride_h = 1;
        param.stride_w = 1;
        checker.set_param(param);
        checker.execs(
                {{1, 64 / 4, 56, 56, 4},
                 {64 / 4, 64 / 4, 3, 3, 4, 4},
                 {1, 64 / 4, 1, 1, 4},
                 {},
                 {}});
    }
    {
        param.pad_h = 0;
        param.pad_w = 1;
        param.stride_h = 1;
        param.stride_w = 1;
        checker.set_param(param);
        checker.execs({{1, 6, 24, 48, 4}, {6, 6, 1, 3, 4, 4}, {1, 6, 1, 1, 4}, {}, {}});
    }
}

TEST(ARMV7, ConvWinogradNCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol(".*_winograd_f23");
    checker.set_epsilon(1e-3);
    ConvBiasForward::Param param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;

    for (size_t Channel : {32, 64, 256}) {
        for (size_t HW : {28, 14}) {
            param.pad_h = 1;
            param.pad_w = 1;
            checker.set_param(param);
            checker.execs(
                    {{1, Channel / 4, HW, HW, 4},
                     {Channel / 4, Channel / 4, 3, 3, 4, 4},
                     {1, Channel / 4, 1, 1, 4},
                     {},
                     {}});
        }
    }

    // clang-format off
    for(size_t P:{0, 1})
    for(size_t IC : {1, 3, 8})
    for(size_t OC : {1, 4})
    for(size_t IH: {3, 5, 22, 32})
    for(size_t IW : {22, 56})
    for(auto mode : {ConvBiasForward::Param::NonlineMode::IDENTITY,
                      ConvBiasForward::Param::NonlineMode::RELU})
                        // clang-format on
                        {
                            param.pad_h = P;
                            param.pad_w = P;
                            param.nonlineMode = mode;
                            checker.set_param(param);
                            checker.execs(
                                    {{1, IC, IH, IW, 4},
                                     {OC, IC, 3, 3, 4, 4},
                                     {},
                                     {},
                                     {}});
                            checker.execs(
                                    {{2, IC, IH, IW, 4},
                                     {OC, IC, 3, 3, 4, 4},
                                     {1, OC, 1, 1, 4},
                                     {},
                                     {}});
                        }
}

TEST(ARMV7, ConvBiasConv1x1NCHW44Int8) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("Armv7_int8_Conv1x1.*");
    ConvBiasForward::Param param;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    param.stride_h = 1;
    param.stride_w = 1;
    size_t kernel = 1;
    for (auto nonlinemode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = nonlinemode;
        checker.set_param(param);
        checker.execs({{2, 7, 11, 11, 4}, {7, 7, kernel, kernel, 4, 4}, {}, {}, {}});
        checker.execs(
                {{2, 3, 13, 10, 4},
                 {13, 3, kernel, kernel, 4, 4},
                 {1, 13, 1, 1, 4},
                 {},
                 {}});
    }
}

TEST(ARMV7, ConvBiasConv1x1NCHWDotInt8M6N8K4) {
    Checker<ConvBiasForward> checker(Arch::ARMV7_WITH_DOT);
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    UniformIntRNG rng(-4, 4);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("Armv7_int8_dot_Conv1x1_M6N8K4.*");
    ConvBiasForward::Param param;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW;
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    param.stride_h = 1;
    param.stride_w = 1;
    size_t kernel = 1;
    for (auto nonlinemode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = nonlinemode;
        checker.set_param(param);
        checker.execs({{2, 7, 11, 11}, {7, 7, kernel, kernel}, {}, {}, {}});
        checker.execs({{2, 3, 13, 10}, {13, 3, kernel, kernel}, {1, 13, 1, 1}, {}, {}});
        checker.execs({{1, 5, 12, 10}, {7, 5, kernel, kernel}, {1, 7, 1, 1}, {}, {}});
    }
}

TEST(ARMV7, DotInt8Conv5x5S2DirectNCHW) {
    Checker<ConvBiasForward> checker(Arch::ARMV7_WITH_DOT);
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("Armv7_dot_int8_direct_.*");
    ConvBiasForward::Param param;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW;
    param.stride_h = 2;
    param.stride_w = 2;
    param.pad_h = 2;
    param.pad_w = 2;
    size_t kernel = 5;

    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (auto nonlinemode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = nonlinemode;
        checker.set_param(param);
        checker.execs(
                {{2, 7, 11, 11}, {7, 1, 1, kernel, kernel}, {1, 7, 1, 1}, {}, {}});
        checker.execs({{1, 4, 13, 16}, {2, 3, 2, kernel, kernel}, {}, {}, {}});
        checker.execs({{1, 3, 37, 63}, {1, 5, 3, kernel, kernel}, {}, {}, {}});
    }

    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    for (auto nonlinemode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = nonlinemode;
        checker.set_param(param);
        checker.execs({{2, 2, 9, 23}, {3, 2, kernel, kernel}, {1, 3, 1, 1}, {}, {}});
        checker.execs({{1, 4, 13, 16}, {6, 4, kernel, kernel}, {}, {}, {}});
        checker.execs({{1, 3, 37, 63}, {5, 3, kernel, kernel}, {}, {}, {}});
    }
}

TEST(ARMV7, Int8Conv5x5S1DirectNCHW) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    checker.set_epsilon(1e-4);
    checker.set_kernel_symbol("Armv7_int8_direct_.*");
    ConvBiasForward::Param param;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW;
    param.stride_h = 1;
    param.stride_w = 1;
    param.pad_h = 2;
    param.pad_w = 2;
    size_t kernel = 5;

    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (auto nonlinemode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = nonlinemode;
        checker.set_param(param);
        checker.execs(
                {{2, 7, 11, 11}, {7, 1, 1, kernel, kernel}, {1, 7, 1, 1}, {}, {}});
        checker.execs({{1, 4, 13, 16}, {2, 3, 2, kernel, kernel}, {}, {}, {}});
        checker.execs({{3, 5, 51, 37}, {1, 3, 5, kernel, kernel}, {}, {}, {}});
    }

    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    for (auto nonlinemode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = nonlinemode;
        checker.set_param(param);
        checker.execs({{2, 2, 9, 23}, {3, 2, kernel, kernel}, {1, 3, 1, 1}, {}, {}});
        checker.execs({{1, 4, 13, 16}, {6, 4, kernel, kernel}, {}, {}, {}});
        checker.execs({{3, 5, 51, 37}, {3, 5, kernel, kernel}, {}, {}, {}});
    }
}

TEST(ARMV7, ConvBiasChannelBroadcastBias) {
    Checker<ConvBiasForward> checker(Arch::ARMV7);
    checker.set_kernel_symbol(".*");
    ConvBiasForward::Param param;

    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    for (size_t kernel : {1, 2, 3, 5, 7}) {
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        for (ConvBiasForward::Param::Format format :
             {param.format = ConvBiasForward::Param::Format::NCHW44,
             param.format = ConvBiasForward::Param::Format::NCHW44_DOT}) {
            param.format = format;
            param.sparse = ConvolutionForward::Param::Sparse::DENSE;
            checker.set_param(param);
#ifdef __x86_64__
            EXPECT_DEATH(
                    checker.execs(
                            {{2, 3, 10, 10, 4},
                             {2, 3, kernel, kernel, 4, 4},
                             {1, 2, 10, 10, 4},
                             {},
                             {}}),
                    "gen kernel failed, available 0");
#endif

            param.sparse = ConvolutionForward::Param::Sparse::GROUP;
            checker.set_param(param);
#ifdef __x86_64__
            EXPECT_DEATH(
                    checker.execs(
                            {{2, 3, 10, 10, 4},
                             {3, 2, 1, kernel, kernel, 4, 4},
                             {1, 6, 10, 10, 4},
                             {},
                             {}}),
                    "gen kernel failed, available 0");
#endif
        }
        param.format = param.format = ConvBiasForward::Param::Format::NCHW;
        param.sparse = ConvolutionForward::Param::Sparse::DENSE;
        checker.set_param(param);
#ifdef __x86_64__
        EXPECT_DEATH(
                checker.execs(
                        {{2, 3, 10, 10},
                         {2, 3, kernel, kernel},
                         {1, 2, 10, 10},
                         {},
                         {}}),
                "gen kernel failed, available 0");
#endif

        param.sparse = ConvolutionForward::Param::Sparse::GROUP;
        checker.set_param(param);
#ifdef __x86_64__
        EXPECT_DEATH(
                checker.execs(
                        {{2, 3, 10, 10},
                         {3, 2, 1, kernel, kernel},
                         {1, 6, 10, 10},
                         {},
                         {}}),
                "gen kernel failed, available 0");
#endif
    }
}
// vim: syntax=cpp.doxygen
