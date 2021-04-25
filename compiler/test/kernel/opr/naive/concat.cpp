/**
 * \file
 * compiler/test/kernel/opr/naive/concat.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;

TEST(NAIVE, Concat) {
    Checker<Concat> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    using Param = Concat::Param;
    for (auto dtype :
         std::vector<DType>{dtype::Float32(), dtype::Int32(), dtype::Int16(),
                            dtype::Int8(), dtype::Uint8()}) {
        for (size_t axis = 0; axis < 4; ++axis) {
            Param param;
            param.axis = axis;
            TensorShapeArray shapes(4, TensorShape({12, 13, 14, 15}));
            for (size_t i = 0; i < 4; ++i) {
                shapes[i].shape[axis] = i + 1;
            }
            shapes.emplace_back();
            for (size_t i = 0; i < shapes.size(); ++i)
                checker.set_dtype(i, dtype);
            checker.set_param(param).exec(shapes);
        }
    }
}