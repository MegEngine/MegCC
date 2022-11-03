/**
 * \file
 * compiler/test/kernel/opr/common/elemwise.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <functional>
#include "test/kernel/common/checker.h"
namespace megcc {
namespace test {
namespace {

using ShapeCases = megdnn::SmallVector<TensorShapeArray>;
ShapeCases get_elewise_binary_case() {
    ShapeCases res;
    res.push_back({{1, 10}, {1, 10}, {}});
    res.push_back({{10, 10}, {10, 10}, {}});
    res.push_back({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
    //! vec_scalar
    res.push_back({{1, 10}, {1, 10}, {}});
    res.push_back({{10, 10}, {1}, {}});
    res.push_back({{100}, {1}, {}});
    //! scalar_vec
    res.push_back({{1, 10}, {1, 10}, {}});
    res.push_back({{1}, {100}, {}});
    res.push_back({{1}, {10, 32}, {}});
    res.push_back({{1, 10}, {1, 10}, {}});
    //! vec_bcast101
    res.push_back({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
    res.push_back({{5, 6, 7, 8}, {1, 6, 1, 1}, {}});
    res.push_back({{1, 81, 2}, {1, 1, 2}, {}});
    res.push_back({{2, 7}, {1, 7}, {}});
    res.push_back({{2, 4}, {1, 4}, {}});
    //! bcast101_vec
    res.push_back({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    res.push_back({{1, 6, 1, 1}, {5, 6, 7, 8}, {}});
    res.push_back({{1, 7}, {2, 7}, {}});
    res.push_back({{1, 4}, {2, 4}, {}});
    //! vec_bcast101x4
    res.push_back({{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {}});
    res.push_back({{5, 6, 7, 8, 4}, {1, 6, 1, 1, 4}, {}});
    //! bcast101x4_vec
    res.push_back({{1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
    res.push_back({{1, 6, 1, 1, 4}, {5, 6, 7, 8, 4}, {}});
    return res;
}

ShapeCases get_elewise_binary_bound_case() {
    ShapeCases res;
    megdnn::TensorShape dst_shape_temp{2, 3, 4, 5, 6};
    for (int ndim = 2; ndim < 6; ++ndim) {
        megdnn::TensorShape dst_shape;
        dst_shape.ndim = ndim;
        for (int i = 0; i < ndim; ++i) {
            dst_shape[i] = dst_shape_temp[i];
        }
        for (int i = 0; i < (1 << ndim); ++i) {
            megdnn::TensorShape sample_shape = dst_shape;
            for (int bit = 0; bit < ndim; ++bit) {
                if (((i >> bit) & 0x1) != 1) {
                    sample_shape[bit] = 1;
                }
            }
            res.push_back({dst_shape, sample_shape, dst_shape});
            res.push_back({sample_shape, dst_shape, dst_shape});
        }
    }
    return res;
}

ShapeCases get_elewise_tenary_bound_case() {
    ShapeCases res;
    megdnn::TensorShape dst_shape_temp{2, 3, 4, 5, 6};
    for (int ndim = 2; ndim < 6; ++ndim) {
        megdnn::TensorShape dst_shape;
        dst_shape.ndim = ndim;
        for (int i = 0; i < ndim; ++i) {
            dst_shape[i] = dst_shape_temp[i];
        }
        for (int i = 0; i < (1 << ndim); ++i) {
            megdnn::TensorShape sample_shape = dst_shape;
            for (int bit = 0; bit < ndim; ++bit) {
                if (((i >> bit) & 0x1) != 1) {
                    sample_shape[bit] = 1;
                }
            }
            res.push_back({dst_shape, sample_shape, dst_shape, dst_shape});
            res.push_back({dst_shape, sample_shape, sample_shape, dst_shape});
            res.push_back({sample_shape, dst_shape, sample_shape, dst_shape});
            res.push_back({sample_shape, dst_shape, dst_shape, dst_shape});
        }
    }
    return res;
}

}  // namespace
}  // namespace test
}  // namespace megcc