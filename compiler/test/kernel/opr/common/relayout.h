/**
 * \file
 * compiler/test/kernel/opr/common/relayout.h
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

using namespace megdnn;
using Mode = RelayoutFormat::Param::Mode;
std::pair<TensorLayout, TensorLayout> concreat_layout(
        size_t n, size_t c, size_t h, size_t w, DType dtype = dtype::Float32(),
        Mode mode = Mode::NCHW_NCHW4) {
    if (mode == Mode::NCHW_NCHW4 || mode == Mode::NCHW4_NCHW) {
        mgb_assert(c % 4 == 0);
        TensorLayout dst({n, c / 4, h, w, 4}, dtype);
        TensorLayout src(
                {n, c / 4, h, w, 4},
                {(std::ptrdiff_t)(c * h * w), (std::ptrdiff_t)(h * w * 4),
                 (std::ptrdiff_t)w, (std::ptrdiff_t)1, (std::ptrdiff_t)(h * w)},
                dtype);
        if (mode == Mode::NCHW_NCHW4) {
            return {src, dst};
        } else {
            return {dst, src};
        }
    } else {
        mgb_assert(0, "not support mode %d", (int)mode);
    }
}

static inline int get_index_of_vec(const std::vector<size_t> vec, int i) {
    int res = 0;
    for (auto x : vec) {
        if (x == (size_t)i) {
            return res;
        } else {
            ++res;
        }
    }
    return -1;
}

static inline void print_vec(const std::vector<size_t>& transpose) {
    for (auto x : transpose) {
        printf("%zu, ", x);
    }
    printf("\n");
}

static inline std::pair<TensorLayout, TensorLayout> transpose_calc(
        const std::vector<size_t>& shape, const std::vector<size_t>& transpose,
        DType dtype) {
    mgb_assert(shape.size() == transpose.size());
    mgb_assert(shape.size() <= TensorShape::MAX_NDIM);
    TensorShape tshape;
    tshape.ndim = shape.size();
    for (size_t i = 0; i < tshape.ndim; ++i) {
        tshape[i] = shape[i];
    }
    TensorLayout src(tshape, dtype);
    TensorLayout dst(tshape, dtype);
    int stride = 1;
    for (int i = tshape.ndim - 1; i >= 0; --i) {
        int axis = get_index_of_vec(transpose, i);
        mgb_assert(axis >= 0, "%d from %zu", i, transpose.size());
        src.stride[axis] = stride;
        stride *= src.shape[axis];
    }
    return std::make_pair(src, dst);
}

static inline void run_relayout_permute(
        std::function<void(const TensorLayout& a, const TensorLayout& b)> func,
        DType dtype) {
    auto run_permutation = [&](std::function<int(int)> cal_shape) {
        for (int ndim = 2; ndim < 6; ++ndim) {
            std::vector<size_t> transpose;
            std::vector<size_t> shape;
            for (int i = 0; i < ndim; ++i) {
                transpose.push_back(i);
                int shape_size = cal_shape(i);
                shape.push_back(shape_size > 4 ? 4 : shape_size);
            }
            do {
                auto testcase = transpose_calc(shape, transpose, dtype);
                func(testcase.first, testcase.second);
                func(testcase.second, testcase.first);
            } while (std::next_permutation(transpose.begin(), transpose.end()));
        }
    };
    run_permutation([](int x) { return x + 1; });
    run_permutation([](int x) { return x + 2; });
}

std::vector<std::pair<TensorLayout, TensorLayout>> get_relyout_common_case(
        DType dtype) {
    std::vector<std::pair<TensorLayout, TensorLayout>> check_args;
    TensorLayout src({1, 128, 32, 32}, {1024, 2048, 32, 1}, dtype);
    TensorLayout dst({1, 128, 32, 32}, {131072, 1024, 32, 1}, dtype);
    check_args.push_back({src, dst});
    // concat
    src = {{1, 32, 32, 32}, {65536, 2048, 32, 1}, dtype};
    dst = {{1, 32, 32, 32}, {32768, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});
    // nchw->nhwc
    src = {{1, 32, 32, 32}, {32768, 32, 1, 1024}, dtype};
    dst = {{1, 32, 32, 32}, {32768, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});
    // nhwc->nchw
    src = {{1, 32, 32, 32}, {32768, 1, 1024, 32}, dtype};
    dst = {{1, 32, 32, 32}, {32768, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});
    // nchw44->ncc4hw(nchw)
    src = {{1, 64, 4, 32, 32}, {262144, 4096, 1, 128, 4}, dtype};
    dst = {{1, 64, 4, 32, 32}, {262144, 4096, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});
    // ncc4hw(nchw)->nchw44
    src = {{1, 32, 32, 32, 4}, {131072, 4096, 32, 1, 1024}, dtype};
    dst = {{1, 32, 32, 32, 4}, {131072, 4096, 128, 4, 1}, dtype};
    check_args.push_back({src, dst});
    // ncc4hw(nchw)->nchw44 + concat
    src = {{1, 32, 32, 32, 4}, {262144, 8192, 32, 1, 2048}, dtype};
    dst = {{1, 32, 32, 32, 4}, {131072, 4096, 128, 4, 1}, dtype};
    check_args.push_back({src, dst});
    // noncontiguous -> noncontiguous
    src = {{1, 32, 32, 32}, {131072, 4096, 64, 1}, dtype};
    dst = {{1, 32, 32, 32}, {4096, 131072, 1, 64}, dtype};
    check_args.push_back({src, dst});
    // transpose 0, 3, 2, 1
    src = {{1, 7, 4, 4}, {112, 1, 7, 28}, dtype};
    dst = {{1, 7, 4, 4}, {112, 16, 4, 1}, dtype};
    check_args.push_back({src, dst});

    // transpose 0, 4, 3, 2, 1
    src = {{1, 7, 2, 3, 4}, {168, 1, 7, 14, 42}, dtype};
    dst = {{1, 7, 2, 3, 4}, {168, 24, 12, 4, 1}, dtype};
    check_args.push_back({src, dst});

    // transpose 0, 1, 4, 3, 2
    src = {{1, 7, 2, 3, 4}, {168, 24, 1, 2, 6}, dtype};
    dst = {{1, 7, 2, 3, 4}, {168, 24, 12, 4, 1}, dtype};
    check_args.push_back({src, dst});

    // transpose 2, 0, 3, 4, 1
    src = {{2, 3, 4, 4, 4}, {16, 128, 4, 1, 32}, dtype};
    dst = {{2, 3, 4, 4, 4}, {192, 64, 16, 4, 1}, dtype};
    check_args.push_back({src, dst});
    check_args.push_back({dst, src});

    check_args.push_back(transpose_calc({1, 7, 2, 3, 4}, {4, 0, 2, 1, 3}, dtype));

    // nchw->nhwc
    src = {{1, 32, 32, 32}, {32768, 32, 1, 1024}, dtype};
    dst = {{1, 32, 32, 32}, {32768, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});
    src = {{2, 32, 32, 32}, {65536, 32, 1, 2048}, dtype};
    dst = {{2, 32, 32, 32}, {32768, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});
    src = {{1, 64, 4, 32, 32}, {262144, 4096, 1, 128, 4}, dtype};
    dst = {{1, 64, 4, 32, 32}, {262144, 4096, 1024, 32, 1}, dtype};
    check_args.push_back({src, dst});

    src = {{1, 2, 3}, {6, 3, 1}, dtype};
    dst = {{1, 2, 3}, {9, 1, 3}, dtype};
    check_args.push_back({src, dst});

    src = {{2, 3, 4}, {12, 4, 1}, dtype};
    dst = {{2, 3, 4}, {1, 2, 6}, dtype};
    check_args.push_back({src, dst});

    src = {{1, 2307, 4, 4}, {36912, 1, 2307, 9228}, dtype};
    dst = {{1, 2307, 4, 4}, {36912, 16, 4, 1}, dtype};
    check_args.push_back({src, dst});

    src = {{1, 8, 3, 4}, {96, 1, 8, 24}, dtype};
    dst = {{1, 8, 3, 4}, {96, 12, 4, 1}, dtype};
    check_args.push_back({src, dst});

    src = {{1, 3, 1, 224}, {672, 1, 672, 3}, dtype};
    dst = {{1, 3, 1, 224}, {672, 224, 224, 1}, dtype};
    check_args.push_back({src, dst});

    src = {{2, 32, 4, 32, 20}, {81920, 2560, 1, 80, 4}, dtype};
    dst = {{2, 32, 4, 32, 20}, {81920, 2560, 640, 20, 1}, dtype};
    check_args.push_back({src, dst});
    return check_args;
}

}  // namespace
}  // namespace test
}  // namespace megcc