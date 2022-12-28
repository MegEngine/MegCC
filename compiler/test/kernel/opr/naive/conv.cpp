/**
 * \file
 * compiler/test/kernel/opr/naive/conv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
namespace {
void nchw_backdata(Checker<ConvolutionBackwardData>& checker) {
    ConvolutionBackwardData::Param param;
    param.compute_mode = ConvolutionBackwardData::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionBackwardData::Param::Format::NCHW;
    checker.set_epsilon(1e-4);
    for (size_t n : {2})
        for (size_t oc : {1, 4})
            for (size_t ic : {1, 4})
                for (size_t hw : {7, 12})
                    for (size_t kernel : {1, 3})
                        for (size_t pad : {(size_t)0, kernel / 2})
                            for (size_t stride : {1, 2}) {
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                param.sparse =
                                        ConvBiasForward::Param::Sparse::DENSE;
                                checker.set_param(param);
                                checker.execs({{oc, ic, kernel, kernel},
                                               {n, oc, hw, hw},
                                               {n, ic,
                                                (hw - 1) * stride +
                                                        (kernel - 1) *
                                                                param.dilate_h +
                                                        1 - pad * 2,
                                                (hw - 1) * stride +
                                                        (kernel - 1) *
                                                                param.dilate_w +
                                                        1 - pad * 2}});
                                if (ic == oc) {
                                    size_t group = oc;
                                    param.sparse = ConvolutionBackwardData::
                                            Param::Sparse::GROUP;
                                    checker.set_param(param);
                                    checker.execs(
                                            {{group, 1, 1, kernel, kernel},
                                             {n, oc, hw, hw},
                                             {n, ic,
                                              (hw - 1) * stride +
                                                      (kernel - 1) *
                                                              param.dilate_h +
                                                      1 - pad * 2,
                                              (hw - 1) * stride +
                                                      (kernel - 1) *
                                                              param.dilate_w +
                                                      1 - pad * 2}});
                                }
                            }
}
}  // namespace
TEST(NAIVE, ConvBiasNCHWQS8) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    UniformIntRNG qint_rng(-3, 3);
    ConvBiasForward::Param param;
    checker.set_rng(0, &qint_rng);
    checker.set_rng(1, &qint_rng);
    checker.set_rng(2, &qint_rng);
    checker.set_epsilon(1 + 1e-3);
    checker.set_dtype(0, dtype::QuantizedS8(1.0f));
    checker.set_dtype(1, dtype::QuantizedS8(2.0f));
    checker.set_dtype(2, dtype::QuantizedS32(2.0f));
    checker.set_dtype(4, dtype::QuantizedS8(4.0f));
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW;
    for (auto noline : {ConvBiasForward::Param::NonlineMode::IDENTITY,
                        ConvBiasForward::Param::NonlineMode::RELU})
        for (size_t n : {2})
            for (size_t oc : {1, 4})
                for (size_t ic : {1, 4})
                    for (size_t hw : {7, 12})
                        for (size_t kernel : {1, 3})
                            for (size_t pad : {(size_t)0, kernel / 2})
                                for (size_t stride : {1, 2}) {
                                    param.pad_h = pad;
                                    param.pad_w = pad;
                                    param.stride_h = stride;
                                    param.stride_w = stride;
                                    param.nonlineMode = noline;
                                    param.sparse = ConvBiasForward::Param::
                                            Sparse::DENSE;
                                    checker.set_param(param);
                                    checker.execs({{n, ic, hw, hw},
                                                   {oc, ic, kernel, kernel},
                                                   {1, oc, 1, 1},
                                                   {},
                                                   {}});
                                    if (ic == oc) {
                                        size_t group = oc;
                                        param.sparse = ConvBiasForward::Param::
                                                Sparse::GROUP;
                                        checker.set_param(param);
                                        checker.execs(
                                                {{n, ic, hw, hw},
                                                 {group, 1, 1, kernel, kernel},
                                                 {1, oc, 1, 1},
                                                 {},
                                                 {}});
                                    }
                                }
}

TEST(NAIVE, ConvBiasNCHWQS8Overflow) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    UniformIntRNG qint_rng(30, 50);

    ConvBiasForward::Param param;
    checker.set_rng(0, &qint_rng);
    checker.set_rng(1, &qint_rng);
    checker.set_rng(2, &qint_rng);
    checker.set_epsilon(1 + 1e-3);

    checker.set_dtype(0, dtype::QuantizedS8(1.0f));
    checker.set_dtype(1, dtype::QuantizedS8(2.0f));
    checker.set_dtype(2, dtype::QuantizedS32(2.0f));
    checker.set_dtype(4, dtype::QuantizedS8(4.0f));
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW;
    param.nonlineMode = ConvBiasForward::Param::NonlineMode::IDENTITY;
    checker.execs({{1, 20, 10, 10}, {20, 20, 3, 3}, {1, 20, 1, 1}, {}, {}});

    UniformIntRNG qint_neg_rng(-50, -30);
    checker.set_rng(0, &qint_neg_rng);
    checker.set_rng(2, &qint_neg_rng);
    checker.execs({{1, 20, 10, 10}, {20, 20, 3, 3}, {1, 20, 1, 1}, {}, {}});
}

TEST(NAIVE, ConvBiasNCHW) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvBiasForward::Param param;

    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW;
    checker.set_epsilon(1e-4);
    for (auto noline : {ConvBiasForward::Param::NonlineMode::IDENTITY,
                        ConvBiasForward::Param::NonlineMode::RELU,
                        ConvBiasForward::Param::NonlineMode::H_SWISH,
                        ConvBiasForward::Param::NonlineMode::SIGMOID})
        for (size_t n : {2})
            for (size_t oc : {1, 4})
                for (size_t ic : {1, 4})
                    for (size_t hw : {7, 12})
                        for (size_t kernel : {1, 3})
                            for (size_t pad : {(size_t)0, kernel / 2})
                                for (size_t stride : {1, 2}) {
                                    param.pad_h = pad;
                                    param.pad_w = pad;
                                    param.stride_h = stride;
                                    param.stride_w = stride;
                                    param.nonlineMode = noline;
                                    param.sparse = ConvBiasForward::Param::
                                            Sparse::DENSE;
                                    checker.set_param(param);
                                    checker.execs({{n, ic, hw, hw},
                                                   {oc, ic, kernel, kernel},
                                                   {1, oc, 1, 1},
                                                   {},
                                                   {}});
                                    if (ic == oc) {
                                        size_t group = oc;
                                        param.sparse = ConvBiasForward::Param::
                                                Sparse::GROUP;
                                        checker.set_param(param);
                                        checker.execs(
                                                {{n, ic, hw, hw},
                                                 {group, 1, 1, kernel, kernel},
                                                 {1, oc, 1, 1},
                                                 {},
                                                 {}});
                                    }
                                }
}

TEST(NAIVE, ConvBiasNCHW44) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvBiasForward::Param param;
    checker.set_epsilon(1e-4);
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    for (auto noline : {ConvBiasForward::Param::NonlineMode::IDENTITY,
                        ConvBiasForward::Param::NonlineMode::RELU,
                        ConvBiasForward::Param::NonlineMode::H_SWISH,
                        ConvBiasForward::Param::NonlineMode::SIGMOID})
        for (size_t n : {12})
            for (size_t oc : {4, 12})
                for (size_t ic : {4, 12})
                    for (size_t hw : {7, 12})
                        for (size_t kernel : {1, 3})
                            for (size_t pad : {(size_t)0, kernel / 2})
                                for (size_t stride : {1, 2}) {
                                    param.pad_h = pad;
                                    param.pad_w = pad;
                                    param.stride_h = stride;
                                    param.stride_w = stride;
                                    param.nonlineMode = noline;
                                    param.sparse = ConvBiasForward::Param::
                                            Sparse::DENSE;
                                    checker.set_param(param);
                                    checker.execs({{n, ic / 4, hw, hw, 4},
                                                   {oc / 4, ic / 4, kernel,
                                                    kernel, 4, 4},
                                                   {1, oc / 4, 1, 1, 4},
                                                   {},
                                                   {}});
                                    if (ic == oc) {
                                        size_t group = oc;
                                        param.sparse = ConvBiasForward::Param::
                                                Sparse::GROUP;
                                        checker.set_param(param);
                                        checker.execs({{n, ic / 4, hw, hw, 4},
                                                       {group / 4, 1, 1, kernel,
                                                        kernel, 4},
                                                       {1, oc / 4, 1, 1, 4},
                                                       {},
                                                       {}});
                                    }
                                }
    {
        param.sparse = ConvBiasForward::Param::Sparse::DENSE;
        checker.set_param(param);
        checker.execs({{2, 2, 5, 5, 4}, {4, 2, 3, 3, 4, 4}, {}, {}, {}});
    }
}

TEST(NAIVE, ConvBiasNCHWNCHW44) {
    Checker<ConvBiasForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvBiasForward::Param param;
    checker.set_epsilon(1e-4);
    param.stride_h = 2;
    param.stride_w = 2;
    param.compute_mode = ConvBiasForward::Param::ComputeMode::DEFAULT;
    param.format = ConvBiasForward::Param::Format::NCHW44;
    checker.set_param(param);
    for (auto mode : {ConvBiasForward::Param::NonlineMode::IDENTITY,
                      ConvBiasForward::Param::NonlineMode::RELU})
        for (size_t filter_size : {2, 3, 5})
            for (size_t ic : {3})
                for (size_t iw = 13; iw < 33; iw++) {
                    size_t pad = filter_size / 2;
                    size_t oc_div4 = 3;
                    param.pad_h = pad;
                    param.pad_w = pad;
                    param.nonlineMode = mode;
                    checker.set_param(param);
                    checker.execs({{2, ic, 11, iw},
                                   {oc_div4, filter_size, filter_size, ic, 4},
                                   {1, oc_div4, 1, 1, 4},
                                   {},
                                   {}});
                }
}

TEST(NAIVE, ConvBackDataNCHW) {
    Checker<ConvolutionBackwardData> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvolutionBackwardData::Param param;
    nchw_backdata(checker);
}

TEST(NAIVE, ConvBackDataNCHW44) {
    Checker<ConvolutionBackwardData> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvolutionBackwardData::Param param;
    checker.set_epsilon(1e-4);
    param.compute_mode = ConvolutionBackwardData::Param::ComputeMode::DEFAULT;
    param.format = ConvolutionBackwardData::Param::Format::NCHW44;
    for (size_t n : {12})
        for (size_t oc : {4, 12})
            for (size_t ic : {4, 12})
                for (size_t hw : {7, 12})
                    for (size_t kernel : {1, 3})
                        for (size_t pad : {(size_t)0, kernel / 2})
                            for (size_t stride : {1, 2}) {
                                param.pad_h = pad;
                                param.pad_w = pad;
                                param.stride_h = stride;
                                param.stride_w = stride;
                                param.sparse = ConvolutionBackwardData::Param::
                                        Sparse::DENSE;
                                checker.set_param(param);
                                checker.execs(
                                        {{oc / 4, ic / 4, kernel, kernel, 4, 4},
                                         {n, oc / 4, hw, hw, 4},
                                         {n, ic / 4,
                                          (hw - 1) * stride +
                                                  (kernel - 1) *
                                                          param.dilate_h +
                                                  1 - pad * 2,
                                          (hw - 1) * stride +
                                                  (kernel - 1) *
                                                          param.dilate_w +
                                                  1 - pad * 2,
                                          4}});
                                if (ic == oc) {
                                    size_t group = oc;
                                    param.sparse = ConvolutionBackwardData::
                                            Param::Sparse::GROUP;
                                    checker.set_param(param);
                                    checker.execs(
                                            {{group / 4, 1, 1, kernel, kernel,
                                              4},
                                             {n, oc / 4, hw, hw, 4},
                                             {n, ic / 4,
                                              (hw - 1) * stride +
                                                      (kernel - 1) *
                                                              param.dilate_h +
                                                      1 - pad * 2,
                                              (hw - 1) * stride +
                                                      (kernel - 1) *
                                                              param.dilate_w +
                                                      1 - pad * 2,
                                              4}});
                                }
                            }
}

TEST(NAIVE, ConvBackDataNCHWQS8) {
    Checker<ConvolutionBackwardData> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvolutionBackwardData::Param param;

    checker.set_dtype(0, dtype::QuantizedS8(1.0f));
    checker.set_dtype(1, dtype::QuantizedS8(2.0f));
    checker.set_dtype(2, dtype::QuantizedS8(2.0f));
    nchw_backdata(checker);
}

TEST(NAIVE, ConvBackDataNCHWQS8Overflow) {
    Checker<ConvolutionBackwardData> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    ConvolutionBackwardData::Param param;
    UniformIntRNG qint_rng(30, 50);
    checker.set_rng(0, &qint_rng);
    checker.set_rng(1, &qint_rng);

    checker.set_dtype(0, dtype::QuantizedS8(1.0f));
    checker.set_dtype(1, dtype::QuantizedS8(2.0f));
    checker.set_dtype(2, dtype::QuantizedS8(2.0f));
    nchw_backdata(checker);
}
