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
#if ENABLE_KERNEL_FP16
TEST(GI, Fp16ConvWinogradNCHW88) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL, 1);
    checker.set_epsilon(1e-3);
    ConvBiasForward::Param param;
    param.stride_h = 1;
    param.stride_w = 1;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW88;
    param.sparse = ConvBiasForward::Param::Sparse::DENSE;

    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);

    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(3, dtype::Float16())
            .set_dtype(4, dtype::Float16());
    //! FIXME: DNN WINOGRAD_NCHW88:FB_GI_F16_MK8_8x8:8:4: algo has some problem,add
    //! {"^GI.*_winograd_f43.*", "WINOGRAD_NCHW88:FB_GI_F16_MK8_8x8:8:2:68:3"} after
    //! problem fixed
    std::vector<std::pair<std::string, std::string>> algo_names = {
            {"^GI.*_winograd_f23", "WINOGRAD_NCHW88:FB_GI_F16_MK8_8x8:8:2:32:3"},
            {"^GI.*_winograd_f63.*", "WINOGRAD_NCHW88:FB_GI_F16_MK8_8x8:8:6:16:3"}};
    for (auto name : algo_names) {
        checker.set_kernel_symbol(name.first.c_str());
        checker.set_before_exec_callback(
                megdnn::test::AlgoChecker<ConvBiasForward>(name.second.c_str()));
        for (size_t Channel : {32, 64, 256}) {
            for (size_t HW : {28, 14}) {
                param.pad_h = 1;
                param.pad_w = 1;
                checker.set_param(param);
                checker.execs(
                        {{1, Channel / 8, HW, HW, 8},
                         {Channel / 8, Channel / 8, 3, 3, 8, 8},
                         {1, Channel / 8, 1, 1, 8},
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
                        ConvBiasForward::Param::NonlineMode::H_SWISH
                        })
                            // clang-format on
                            {
                                param.pad_h = P;
                                param.pad_w = P;
                                param.nonlineMode = mode;
                                checker.set_param(param);
                                checker.execs(
                                        {{1, IC, IH, IW, 8},
                                         {OC, IC, 3, 3, 8, 8},
                                         {},
                                         {},
                                         {}});
                                checker.execs(
                                        {{2, IC, IH, IW, 8},
                                         {OC, IC, 3, 3, 8, 8},
                                         {1, OC, 1, 1, 8},
                                         {},
                                         {}});
                            }
    }
}

TEST(GI, ConvBiasIm2colNCHW88) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_conv2d_im2colm8n8_fp16.*");
    checker.set_epsilon(5e-2);
    ConvBiasForward::Param param;
    param.format = ConvBiasForward::Param::Format::NCHW88;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);

    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(3, dtype::Float16())
            .set_dtype(4, dtype::Float16());

    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID,
          ConvBiasForward::Param::NonlineMode::H_SWISH})
        for (size_t n : {1, 3})
            for (size_t oc : {8, 16})
                for (size_t ic : {8, 16})
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
                                        {{n, ic / 8, hw, hw, 8},
                                         {oc / 8, ic / 8, filter_size, filter_size, 8,
                                          8},
                                         {1, oc / 8, 1, 1, 8},
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
                {{1, 64 / 8, 28, 28, 8},
                 {64 / 8, 64 / 8, 3, 3, 8, 8},
                 {1, 64 / 8, 1, 1, 8},
                 {},
                 {}});
    }
    {
        param.pad_h = 0;
        param.pad_w = 1;
        param.stride_h = 1;
        param.stride_w = 1;
        checker.set_param(param);
        checker.execs({{1, 6, 12, 24, 8}, {6, 6, 1, 3, 8, 8}, {1, 6, 1, 1, 8}, {}, {}});
    }
}

TEST(GI, ConvBiasIm2colNCHW88Group) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_epsilon(5e-2);
    ConvBiasForward::Param param;
    param.format = ConvBiasForward::Param::Format::NCHW88;
    checker.set_kernel_symbol("GI_kernel_conv2d_im2colm8n8_fp16.*");
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);

    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(3, dtype::Float16())
            .set_dtype(4, dtype::Float16());
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID})
        for (size_t n : {1, 3})
            for (size_t group : {3, 7})
                for (size_t ocpg : {8, 16})
                    for (size_t icpg : {8})
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
                                            {{n, group * icpg / 8, hw, hw, 8},
                                             {group, ocpg / 8, icpg / 8, filter_size,
                                              filter_size, 8, 8},
                                             {1, group * ocpg / 8, 1, 1, 8},
                                             {},
                                             {}});
                                }
}

TEST(GI, Conv1x1NCHW88) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_conv2d_conv1x1_fp16.+");
    ConvBiasForward::Param param;
    checker.set_epsilon(5e-2);
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_dtype(3, dtype::Float16())
            .set_dtype(4, dtype::Float16());
    param.compute_mode = ConvolutionForward::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionForward::Param::Format::NCHW88;
    for (auto noline :
         {ConvBiasForward::Param::NonlineMode::RELU,
          ConvBiasForward::Param::NonlineMode::IDENTITY,
          ConvBiasForward::Param::NonlineMode::SIGMOID,
          ConvBiasForward::Param::NonlineMode::H_SWISH}) {
        param.nonlineMode = noline;
        checker.set_param(param);
        checker.execs({{2, 3, 5, 11, 8}, {3, 3, 1, 1, 8, 8}, {1, 3, 1, 1, 8}, {}, {}});
    }
}
#endif
// vim: syntax=cpp.doxygen
