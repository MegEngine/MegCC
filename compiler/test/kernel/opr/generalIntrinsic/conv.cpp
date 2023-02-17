/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/conv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(GI, ConvBiasIm2col) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;
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

TEST(GI, ConvBiasIm2colGroup) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;

    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
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

TEST(GI, ConvBiasChannelWiseNCHW4K3) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_chanwise_kernel_conv2d_3x3_NCHW44.*");
    ConvBiasForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (auto mode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
        for (size_t stride : {1, 2}) {
            param.nonlineMode = mode;
            param.stride_h = stride;
            param.stride_w = stride;
            checker.set_param(param);
            checker.execs(
                    {{2, 8, 28, 28, 4}, {8, 1, 1, 3, 3, 4}, {1, 8, 1, 1, 4}, {}, {}});
            checker.execs({{2, 8, 14, 28, 4}, {8, 1, 1, 3, 3, 4}, {}, {}, {}});
            checker.set_param(param);
            checker.execs(
                    {{4, 3, 5, 11, 4}, {3, 1, 1, 3, 3, 4}, {1, 3, 1, 1, 4}, {}, {}});
            checker.execs({{4, 3, 5, 11, 4}, {3, 1, 1, 3, 3, 4}, {}, {}, {}});
        }
}

TEST(GI, ConvChannelWiseNCHW4K3) {
    Checker<ConvolutionForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_chanwise_kernel_conv2d_3x3_NCHW44.*");
    ConvolutionForward::Param param;
    param.pad_h = 1;
    param.pad_w = 1;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (size_t stride : {1, 2}) {
        param.stride_h = stride;
        param.stride_w = stride;
        checker.set_param(param);
        checker.execs({{2, 8, 24, 28, 4}, {8, 1, 1, 3, 3, 4}, {}});
        checker.execs({{1, 3, 23, 23, 4}, {3, 1, 1, 3, 3, 4}, {}});
        checker.execs({{3, 3, 23, 23, 4}, {3, 1, 1, 3, 3, 4}, {}});
        checker.execs({{1, 3, 14, 14, 4}, {3, 1, 1, 3, 3, 4}, {}});
        checker.execs({{4, 3, 14, 14, 4}, {3, 1, 1, 3, 3, 4}, {}});
        checker.execs({{4, 5, 34, 7, 4}, {5, 1, 1, 3, 3, 4}, {}});
        checker.execs({{2, 8, 14, 28, 4}, {8, 1, 1, 3, 3, 4}, {}});
        checker.execs({{2, 8, 28, 28, 4}, {8, 1, 1, 3, 3, 4}, {}});
    }
}

TEST(GI, ConvChannelWiseNCHW4K5) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_chanwise_kernel_conv2d_5x5_NCHW44.*");
    ConvBiasForward::Param param;
    param.pad_h = 2;
    param.pad_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.sparse = ConvBiasForward::Param::Sparse::GROUP;
    for (auto mode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
        for (size_t stride : {1}) {
            param.stride_h = stride;
            param.stride_w = stride;
            param.nonlineMode = mode;
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

TEST(GI, ConvBiasNCHWNCHW44) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    ConvBiasForward::Param param;
    checker.set_kernel_symbol("GI_.*nchw_nchw44.*");
    checker.set_epsilon(1e-4);
    param.stride_h = 2;
    param.stride_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    checker.set_param(param);
    for (auto mode :
         {ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
        for (size_t filter_size : {2, 3, 5})
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

TEST(GI, ConvWinogradNCHW44) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("^GI.*_winograd_f23.*");
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

TEST(GI, Conv1x1NCHW44) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_conv2d_conv1x1_NCHW44.+");
    ConvBiasForward::Param param;
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW44;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        checker.set_param(param);
        checker.execs({{2, 3, 5, 11, 4}, {3, 3, 1, 1, 4, 4}, {}, {}, {}});
    }
}

TEST(GI, ConvBiasIm2colNCHW44) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_conv2d_im2col.*");
    checker.set_epsilon(5e-4);
    ConvBiasForward::Param param;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID,
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

// vim: syntax=cpp.doxygen
