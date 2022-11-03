/**
 * \file
 * compiler/test/kernel/commonproxy.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include "compiler/KernelGen/KernelGen.h"
#include "megdnn/oprs.h"
#include "test/kernel/common/performance.h"
using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;

namespace megcc {
namespace test {
namespace {
struct OutputScope {
    //! [start, end]
    int start;
    int end;
    void normalize(int len) {
        start = start < 0 ? start + len : start;
        end = end < 0 ? end + len : end;
    }
};

}  // namespace
template <typename Opr>
struct CCOprProxy {
    PerformanceResult exec(
            Opr* opr, const TensorNDArray& tensors, KernelGen::Arch arch,
            const BenchmarkOption& benchmark_option,
            const std::string& kernel_symbol,
            const std::unordered_map<std::string, CCAttr>& proxy_attr,
            bool gen_dynamic);

    OutputScope get_output_idx(Opr*);
};

}  // namespace test
}  // namespace megcc