/**
 * \file
 * compiler/test/kernel/opr/naive/topk.cpp
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
Checker<TopK>::OutputCanonizer OutputCanonizer(TopK::Param::Mode mode) {
    auto output_canonizer = [=](const TensorNDArray& arr) {
        if (mode == TopK::Param::Mode::KTH_ONLY) {
            return;
        }
        auto pval = arr[1].ptr<float>();
        auto pidx = arr.at(2).ptr<int>();
        size_t m = arr[1].layout[0], n = arr[1].layout[1];
        using idx_val = std::pair<int, float>;
        std::vector<idx_val> data(n);
        auto compare = [](const idx_val& it1, const idx_val& it2) {
            return (it1.second > it2.second);
        };
        for (size_t i = 0; i < m; ++i) {
            if (mode == TopK::Param::Mode::VALUE_IDX_NOSORT) {
                // sort output pairs to canonize
                for (size_t j = 0; j < n; ++j) {
                    data[j].first = pidx[i * n + j];
                    data[j].second = pval[i * n + j];
                }
                std::sort(data.begin(), data.end(), compare);
                for (size_t j = 0; j < n; ++j) {
                    pidx[i * n + j] = data[j].first;
                    pval[i * n + j] = data[j].second;
                }
            }
        }
    };
    return output_canonizer;
}
}  // namespace

TEST(NAIVE, Topk) {
    Checker<TopK> checker;
    checker.set_kernel_symbol("kernel_.*");

    TopK::Param param;
    using Mode = TopK::Param::Mode;
    for (int k : {-5, 7})
        for (auto mode : {
                     Mode::KTH_ONLY,
                     Mode::VALUE_IDX_NOSORT,
                     Mode::VALUE_IDX_SORTED,
             }) {
            checker.set_output_canonizer(OutputCanonizer(mode));
            checker.set_proxy(k);
            for (size_t batch_size : {1, 3})
                for (size_t vec_len = std::abs(k); vec_len < 27; vec_len++) {
                    param.mode = mode;
                    checker.set_param(param);
                    if (mode == Mode::KTH_ONLY) {
                        checker.execs({{batch_size, vec_len}, {}});
                    } else {
                        checker.execs({{batch_size, vec_len}, {}, {}});
                    }
                }
        }
}