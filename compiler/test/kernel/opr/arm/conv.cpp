#include "megbrain/reflection.h"
#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(AARCH64, Conv1x1NCHW44) {
    Checker<ConvolutionForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_conv1x1_NCHW44.+");
    ConvolutionForward::Param param;
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44;
    checker.set_param(param);
    checker.execs({{2, 3, 5, 11, 4}, {3, 3, 1, 1, 4, 4}, {}});

    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.execs({{2, 6, 17, 19, 4}, {2, 4, 3, 1, 1, 4, 4}, {}});
}
#if !MEGCC_WITHOUT_DOT
TEST(AARCH64, ConvBias1x1NCHW44Dot) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.set_kernel_symbol("Arm64_kernel_dot_conv2d_conv1x1_.+");

    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));

    ConvBiasForward::Param param;
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44_DOT;

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        printf("mode=%s\n",
               mgb::reflection::nameOfEnumValue<ConvBiasForward::Param::NonlineMode>(
                       noline)
                       .c_str());
        checker.set_param(param);
        for (size_t ic : {3, 4, 5}) {
            for (size_t ohw = 7; ohw < 27; ++ohw) {
                checker.execs(
                        {{2, ic, 1, ohw, 4},
                         {5, ic, 1, 1, 4, 4},
                         {1, 5, 1, 1, 4},
                         {},
                         {}});
            }
        }
    }

    checker.set_param(param);

    checker.execs({{2, 33, 1, 23, 4}, {5, 33, 1, 1, 4, 4}, {1, 5, 1, 1, 4}, {}, {}});
}

TEST(AARCH64, ConvBias1x1NCHW44DotNCHWNCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    UniformIntRNG rng(-30, 30);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.set_kernel_symbol(".+dot_nchw_nchw44.+");

    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));

    ConvBiasForward::Param param;

    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44_DOT;

    for (auto mode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU})
        for (size_t filter_size : {2, 3, 5})
            for (size_t ic : {3})
                for (size_t iw = 13; iw < 33; iw++)
                    for (int stride : {1, 2}) {
                        size_t pad = filter_size / 2;
                        size_t oc_div4 = 3;
                        param.pad_h = pad;
                        param.pad_w = pad;
                        param.stride_h = stride;
                        param.stride_w = stride;
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
#endif

TEST(AARCH64, ConvBias1x1NCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_conv1x1.+");
    ConvBiasForward::Param param;

    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        checker.set_param(param);
        checker.execs({{2, 3, 5, 11, 4}, {5, 3, 1, 1, 4, 4}, {1, 5, 1, 1, 4}, {}, {}});
    }

    param.sparse = ConvolutionForward::Param::Sparse::GROUP;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        checker.set_param(param);
        checker.execs(
                {{2, 6, 17, 19, 4}, {2, 4, 3, 1, 1, 4, 4}, {1, 8, 1, 1, 4}, {}, {}});
    }
}

TEST(AARCH64, ConvWinogradNCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_epsilon(1e-2);
    ConvBiasForward::Param param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;
    for (auto name :
         {".*_winograd_f23", "^GI.*_winograd_f43.*", "^GI.*_winograd_f63.*"}) {
        checker.set_kernel_symbol(name);
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
                        ConvBiasForward::Param::NonlineMode::RELU,
                        ConvBiasForward::Param::NonlineMode::H_SWISH})
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
}

TEST(AARCH64, ConvBiasIm2col) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;

    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
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
#if !MEGCC_WITHOUT_DOT
TEST(AARCH64, ConvBiasIm2colDot) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_im2col_dot_.*");
    checker.set_epsilon(5e-4);
    UniformIntRNG rng(-50, 50);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);

    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(40.25f));
    ConvBiasForward::Param param;
    param.format = ConvBiasForward::Param::Format::NCHW44_DOT;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.nonlineMode = ConvBiasForward::Param::NonlineMode::H_SWISH;

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
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
        checker.execs({{1, 2, 22, 33, 4}, {2, 2, 3, 3, 4, 4}, {1, 2, 1, 1, 4}, {}, {}});
    }
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
        for (size_t n : {1, 3})
            for (size_t group : {3, 7})
                for (size_t ocpg : {4, 8, 20})
                    for (size_t icpg : {4, 8})
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
                                            {{n, group * icpg / 4, hw, hw, 4},
                                             {group, ocpg / 4, icpg / 4, filter_size,
                                              filter_size, 4, 4},
                                             {1, group * ocpg / 4, 1, 1, 4},
                                             {},
                                             {}});
                                }
}
#endif

TEST(AARCH64, ConvBiasIm2colGroup) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_im2col.*");
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

TEST(AARCH64, ConvBiasIm2colNCHW44) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_im2col.*");
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

TEST(AARCH64, ConvBiasIm2colNCHW44Group) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID})
        for (size_t n : {1, 3})
            for (size_t group : {3, 7})
                for (size_t ocpg : {4, 8, 20})
                    for (size_t icpg : {4, 8})
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
                                            {{n, group * icpg / 4, hw, hw, 4},
                                             {group, ocpg / 4, icpg / 4, filter_size,
                                              filter_size, 4, 4},
                                             {1, group * ocpg / 4, 1, 1, 4},
                                             {},
                                             {}});
                                }
}

TEST(AARCH64, ConvBiasChannelBroadcastBias) {
    Checker<ConvBiasForward> checker(Arch::ARM64);
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
