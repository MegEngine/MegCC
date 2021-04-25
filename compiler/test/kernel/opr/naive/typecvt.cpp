/**
 * \file
 * compiler/test/kernel/opr/naive/typecvt.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;

TEST(NAIVE, Typecvt) {
    Checker<TypeCvtForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    for (auto src_dtype : {dtype::Uint8()})
        for (auto dst_dtype : {dtype::Float32()}) {
            checker.set_dtype(0, src_dtype);
            checker.set_dtype(1, dst_dtype);
            checker.execs({{2, 10}, {2, 10}});
            checker.execs({{2, 10, 4}, {2, 10, 4}});
            checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});
        }
}

TEST(NAIVE, Typecvt2Int8) {
    Checker<TypeCvtForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    megcc::test::UniformRNG rng(100, 600);
    checker.set_rng(0, &rng);
    for (auto src_dtype : {dtype::Float32()})
        for (DType dst_dtype : {(DType)dtype::Int8(), (DType)dtype::Uint8()}) {
            checker.set_dtype(0, src_dtype);
            checker.set_dtype(1, dst_dtype);
            checker.execs({{2, 10}, {2, 10}});
            checker.execs({{2, 10, 4}, {2, 10, 4}});
            checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});
        }
}

TEST(NAIVE, TypecvtQuant) {
    Checker<TypeCvtForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    megcc::test::UniformIntRNG uni_flt(-300, 300);
    checker.set_rng(0, &uni_flt);

    std::vector<::megdnn::DType> dtype_list;
    dtype_list.push_back(dtype::QuantizedS8(0.7f));
    dtype_list.push_back(dtype::QuantizedS8(0.5f));
    dtype_list.push_back(dtype::Float32());
    dtype_list.push_back(dtype::QuantizedS32(0.3f));

    for (auto src_dtype : dtype_list)
        for (auto dst_dtype : dtype_list) {
            checker.set_dtype(0, src_dtype);
            checker.set_dtype(1, dst_dtype);
            checker.execs({{2, 10}, {2, 10}});
            checker.execs({{2, 10, 4}, {2, 10, 4}});
            checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});
        }

    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::QuantizedS8(0.1f));
    checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Float32());
    checker.execs({{2, 10}, {2, 10}});
    checker.execs({{2, 10, 4}, {2, 10, 4}});
    checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});
}
