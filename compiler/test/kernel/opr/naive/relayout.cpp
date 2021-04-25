/**
 * \file
 * compiler/test/kernel/opr/naive/relayout.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
#include "test/kernel/opr/common/relayout.h"
using namespace megdnn;
using namespace megcc::test;
using Mode = RelayoutFormat::Param::Mode;

TEST(NAIVE, Relayout) {
    Checker<RelayoutForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    for (DType dtype : {static_cast<DType>(dtype::Float32()),
                        static_cast<DType>(dtype::Int8())})
        for (size_t n : {1, 3})
            for (size_t c : {8, 16})
                for (size_t hw : {3, 5})
                    for (auto mode : {Mode::NCHW4_NCHW, Mode::NCHW_NCHW4}) {
                        auto layout_pair =
                                concreat_layout(n, c, hw, hw, dtype, mode);
                        checker.execl({layout_pair.first, layout_pair.second});
                    }
}

TEST(NAIVE, RelayoutNeg) {
    Checker<RelayoutForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    size_t n = 1;
    size_t c = 2;
    size_t h = 3;
    size_t w = 4;
    for (DType dtype : {static_cast<DType>(dtype::Float32()),
                        static_cast<DType>(dtype::Int8())}) {
        TensorLayout dst({n, c, h, w}, dtype);
        TensorLayout src({n, c, h, w},
                         {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(h * w),
                          (std::ptrdiff_t)w, (std::ptrdiff_t)(-1)},
                         dtype);
        checker.execl({src, dst});
    }
    for (DType dtype : {static_cast<DType>(dtype::Float32()),
                        static_cast<DType>(dtype::Int8())}) {
        TensorLayout dst({n, c, h, w}, dtype);
        TensorLayout src({n, c, h, w},
                         {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(h * w),
                          (std::ptrdiff_t)(-w), (std::ptrdiff_t)(1)},
                         dtype);
        checker.execl({src, dst});
    }
}

TEST(NAIVE, RelayoutFast) {
    Checker<RelayoutForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    size_t n = 2;
    size_t c = 8;
    size_t h = 2;
    size_t w = 3;
    for (DType dtype : {static_cast<DType>(dtype::Float32()),
                        static_cast<DType>(dtype::Int8())}) {
        {
            TensorLayout dst({n, c, h, w}, dtype);
            TensorLayout src({n, h, w, c},
                             {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(w),
                              (std::ptrdiff_t)(1), (std::ptrdiff_t)(h * w)},
                             dtype);
            checker.execl({src, dst});
            checker.execl({dst, src});
        }
        {
            TensorLayout dst({n, h, w, c}, dtype);
            TensorLayout src({n, c, h, w},
                             {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(1),
                              (std::ptrdiff_t)(c * w), (std::ptrdiff_t)(c)},
                             dtype);
            checker.execl({src, dst});
            checker.execl({dst, src});
        }
        {
            TensorLayout dst({n, c / 4, 4, h, w}, dtype);
            TensorLayout src({n, c / 4, h, w, 4},
                             {(std::ptrdiff_t)(c * h * w),
                              (std::ptrdiff_t)(h * w * 4), (std::ptrdiff_t)(w),
                              (std::ptrdiff_t)(1), (std::ptrdiff_t)(h * w)},
                             dtype);
            checker.execl({src, dst});
            checker.execl({dst, src});
        }
        {
            TensorLayout dst({n, c / 4, h, w, 4}, dtype);
            TensorLayout src({n, c / 4, 4, h, w},
                             {(std::ptrdiff_t)(c * h * w),
                              (std::ptrdiff_t)(h * w * 4), (std::ptrdiff_t)(1),
                              (std::ptrdiff_t)(w * 4), (std::ptrdiff_t)(4)},
                             dtype);
            checker.execl({src, dst});
            checker.execl({dst, src});
        }
    }
}

TEST(NAIVE, RelayoutConcat) {
    Checker<RelayoutForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    size_t n = 2;
    size_t c = 8;
    size_t h = 2;
    size_t w = 3;
    size_t multi = 3;
    for (DType dtype : {static_cast<DType>(dtype::Float32()),
                        static_cast<DType>(dtype::Int8())}) {
        {
            TensorLayout src({n, c, h, w}, dtype);
            TensorLayout dst({n, c, h, w},
                             {(std::ptrdiff_t)(c * multi * h * w),
                              (std::ptrdiff_t)(multi * h * w),
                              (std::ptrdiff_t)(w), (std::ptrdiff_t)(1)},
                             dtype);
            checker.execl({src, dst});
        }
    }
}

TEST(NAIVE, RelayoutPermute) {
    Checker<RelayoutForward> checker;
    UniformIntRNG seq(1, 127);
    checker.set_rng(0, &seq);
    run_relayout_permute(
            [&](const TensorLayout& a, const TensorLayout& b) {
                checker.execl({a, b});
            },
            dtype::Float32());
}

TEST(NAIVE, RelayoutCase) {
    Checker<RelayoutForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    UniformIntRNG seq(1, 127);
    checker.set_rng(0, &seq);
    for (DType dtype : {static_cast<DType>(dtype::Float32()),
                        static_cast<DType>(dtype::Int8()),
                        static_cast<DType>(dtype::QuantizedS8(0.5f))}) {
        // copy most part contiguous block to dst
        auto check_args = get_relyout_common_case(dtype);
        checker.set_dtype(0, dtype);
        checker.set_dtype(1, dtype);
        for (auto&& item : check_args) {
            checker.execl({item.first, item.second});
        }
    }
}