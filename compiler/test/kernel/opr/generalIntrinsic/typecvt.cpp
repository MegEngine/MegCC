/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/typecvt.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(GI, TYPECVT) {
    Checker<TypeCvtForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_typecvt.*");
    UniformIntRNG rng(-128, 127);
    checker.set_rng(0, &rng);
    std::vector<std::pair<megdnn::DType, megdnn::DType>> types = {
            {dtype::QuantizedS8(0.3f), dtype::Float32()},
            {dtype::Float32(), dtype::QuantizedS8(1.7f)},
            {dtype::QuantizedS8(1.7f), dtype::QuantizedS8(0.3f)},
            {dtype::Uint8(), dtype::Float32()}};
    for (auto type : types) {
        checker.set_dtype(0, type.first);
        checker.set_dtype(1, type.second);

        checker.execs({{2, 10}, {2, 10}});
        checker.execs({{2, 10, 4}, {2, 10, 4}});
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});
    }

    UniformIntRNG rng_uint8(0, 255);
    checker.set_rng(0, &rng_uint8);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Float32());
    checker.execs({{1, 1, 128, 128}, {1, 1, 128, 128}});
}