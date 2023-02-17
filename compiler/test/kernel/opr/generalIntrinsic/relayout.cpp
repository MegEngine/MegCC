/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/relayout.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/opr/common/relayout.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = RelayoutFormat::Param::Mode;

TEST(GI, Relayout) {
    Checker<RelayoutForward> checker(Arch::BAREMETAL);
    megcc::test::UniformIntRNG seq(-128, 127);
    checker.set_kernel_symbol("GI_kernel_relayout_.*");
    checker.set_rng(0, &seq);
    for (DType dtype :
         {static_cast<DType>(dtype::Float32()), static_cast<DType>(dtype::Int8()),
          static_cast<DType>(dtype::QuantizedS8(0.5f))}) {
        auto check_args = get_relyout_common_case(dtype);
        checker.set_dtype(0, dtype);
        checker.set_dtype(1, dtype);
        for (auto&& item : check_args) {
            printf("src=%s, dst=%s\n", item.first.to_string().c_str(),
                   item.second.to_string().c_str());
            checker.execl({item.first, item.second});
        }
    }
}

TEST(GI, Relayout44) {
    Checker<RelayoutForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_relayout_.*");
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    for (DType dtype :
         {static_cast<DType>(dtype::Float32()), static_cast<DType>(dtype::Int8())})
        for (size_t n : {1, 3})
            for (size_t c : {8, 16})
                for (size_t hw : {3, 5})
                    for (auto mode : {Mode::NCHW4_NCHW, Mode::NCHW_NCHW4}) {
                        auto layout_pair = concreat_layout(n, c, hw, hw, dtype, mode);
                        checker.execl({layout_pair.first, layout_pair.second});
                    }
}

TEST(GI, RelayoutPermute) {
    Checker<RelayoutForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_relayout_.*");
    UniformIntRNG seq(1, 127);
    checker.set_rng(0, &seq);

    run_relayout_permute(
            [&](const TensorLayout& a, const TensorLayout& b) {
                checker.execl({a, b});
            },
            dtype::Float32());
}

TEST(GI, RelayoutConcatSubtensor) {
    Checker<RelayoutForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_relayout_.*");
    UniformIntRNG rng(1, 127);
    checker.set_rng(0, &rng);
    size_t n = 2;
    size_t c = 8;
    size_t h = 2;
    size_t w = 3;
    size_t multi = 3;

    {
        TensorLayout src({n, c, h, w}, dtype::Float32());
        TensorLayout dst(
                {n, c, h, w},
                {(std::ptrdiff_t)(c * multi * h * w), (std::ptrdiff_t)(multi * h * w),
                 (std::ptrdiff_t)(w), (std::ptrdiff_t)(1)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        TensorLayout src({n, c, h, w}, dtype::Float32());
        TensorLayout dst(
                {n, c, h, w},
                {(std::ptrdiff_t)(c * multi * h * w * 2),
                 (std::ptrdiff_t)(multi * h * w * 2), (std::ptrdiff_t)(w * 2),
                 (std::ptrdiff_t)(2)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        TensorLayout src({n, c, h, w}, dtype::Float32());
        TensorLayout dst(
                {n, c, h, w},
                {(std::ptrdiff_t)(c * multi * multi * h * w),
                 (std::ptrdiff_t)(multi * multi * h * w), (std::ptrdiff_t)(multi * w),
                 (std::ptrdiff_t)(1)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        TensorLayout src({n, c, h, w}, dtype::Float32());
        TensorLayout dst(
                {n, c, h, w},
                {(std::ptrdiff_t)(c * multi * multi * multi * h * w),
                 (std::ptrdiff_t)(multi * multi * h * w), (std::ptrdiff_t)(multi * w),
                 (std::ptrdiff_t)(1)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        TensorLayout src({n}, dtype::Float32());
        TensorLayout dst(
                {n}, {(std::ptrdiff_t)(c * multi * multi * multi * h * w)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        TensorLayout src({n}, dtype::Float32());
        TensorLayout dst({n}, {(std::ptrdiff_t)(1)}, dtype::Float32());
        checker.execl({dst, src});
    }

    {
        TensorLayout src({n, 1, h, 1}, dtype::Float32());
        TensorLayout dst(
                {n, 1, h, 1},
                {(std::ptrdiff_t)(c * multi * multi * multi * h * w),
                 (std::ptrdiff_t)(multi * multi * h * w), (std::ptrdiff_t)(multi * w),
                 (std::ptrdiff_t)(1)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        TensorLayout src({n, 1, h, w}, dtype::Float32());
        TensorLayout dst(
                {n, 1, h, w},
                {(std::ptrdiff_t)(c * multi * multi * multi * h * w),
                 (std::ptrdiff_t)(multi * multi * h * w), (std::ptrdiff_t)(multi * w),
                 (std::ptrdiff_t)(1)},
                dtype::Float32());
        checker.execl({dst, src});
    }
    {
        auto dtype = dtype::Float32();
        TensorLayout src = {{2, 32, 32, 32}, {32768, 32, 1, 1024}, dtype};
        TensorLayout dst = {{2, 32, 32, 32}, {32768, 1024, 32, 1}, dtype};
        checker.execl({src, dst});
    }

    for (DType dtype :
         {static_cast<DType>(dtype::Float32()), static_cast<DType>(dtype::Int8())}) {
        {
            TensorLayout src({n, c, h, w}, dtype);
            TensorLayout dst(
                    {n, c, h, w},
                    {(std::ptrdiff_t)(c * multi * h * w),
                     (std::ptrdiff_t)(multi * h * w), (std::ptrdiff_t)(w),
                     (std::ptrdiff_t)(1)},
                    dtype);
            checker.execl({src, dst});
            checker.execl({dst, src});
        }
        {
            TensorLayout src({n, c, h, w}, dtype);
            TensorLayout dst(
                    {n, c, h, w},
                    {(std::ptrdiff_t)(c * multi * multi * h * w),
                     (std::ptrdiff_t)(multi * multi * h * w),
                     (std::ptrdiff_t)(multi * w), (std::ptrdiff_t)(1)},
                    dtype);
            checker.execl({src, dst});
            checker.execl({dst, src});
        }
    }
}

TEST(GI, RelayoutNeg) {
    Checker<RelayoutForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_relayout_.*");
    SequenceRNG seq;
    checker.set_rng(0, &seq);
    size_t n = 1;
    size_t c = 2;
    size_t h = 3;
    size_t w = 4;
    for (DType dtype :
         {static_cast<DType>(dtype::Float32()), static_cast<DType>(dtype::Int8())}) {
        TensorLayout dst({n, c, h, w}, dtype);
        TensorLayout src(
                {n, c, h, w},
                {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(h * w),
                 (std::ptrdiff_t)w, (std::ptrdiff_t)(-1)},
                dtype);
        checker.set_dtype(0, dtype);
        checker.set_dtype(1, dtype);
        checker.execl({src, dst});
    }
    for (DType dtype :
         {static_cast<DType>(dtype::Float32()), static_cast<DType>(dtype::Int8())}) {
        TensorLayout dst({n, c, h, w}, dtype);
        TensorLayout src(
                {n, c, h, w},
                {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(h * w),
                 (std::ptrdiff_t)(-w), (std::ptrdiff_t)(1)},
                dtype);
        checker.set_dtype(0, dtype);
        checker.set_dtype(1, dtype);
        checker.execl({src, dst});
    }
}