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
#endif
// vim: syntax=cpp.doxygen
