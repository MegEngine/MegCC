/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/elementwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
#include "test/kernel/opr/common/elemwise.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;
TEST(GI, ElementwiseUnique) {
    Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    ElemwiseForward::Param param;
    checker.set_kernel_symbol("GI_kernel_elementwise.+");
    for (auto mode : {MODE::RELU, MODE::SIGMOID, MODE::EXP, MODE::H_SWISH}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
#ifdef ENABLE_KERNEL_FP16
    megcc::test::UniformRNG rng(-1.0, 1.0);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::Float16());
    for (auto mode : {MODE::RELU, MODE::SIGMOID, MODE::EXP, MODE::H_SWISH}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
#endif
}

TEST(GI, ElementwiseBinary) {
    //! only support 1x11 broadcast
    Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_elementwise.+");

    ElemwiseForward::Param param;
    for (auto mode :
         {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU, MODE::MAX, MODE::MIN}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 10}, {1, 10}, {}});
        checker.execs({{1, 10}, {1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
        checker.execs({{13}, {3, 9, 13}, {}});
        checker.execs({{3, 9, 13}, {13}, {}});
        checker.execs({{1, 1, 1, 13}, {2, 3, 4, 13}, {}});
        checker.execs({{2, 100, 3, 96}, {1, 1, 1, 96}, {}});
    }
    checker.set_epsilon(1e-4);
    megcc::test::UniformRNG rng(3, 12);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    for (auto mode : {MODE::TRUE_DIV}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 10}, {1, 10}, {}});
        checker.execs({{1, 10}, {1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4}, {1}, {}});
        checker.execs({{1}, {2, 3, 4}, {}});
        checker.execs({{57}, {10, 50, 57}, {}});
        checker.execs({{10, 50, 57}, {57}, {}});
        checker.execs({{1, 1, 1, 33}, {2, 30, 4, 33}, {}});
        checker.execs({{2, 100, 3, 69}, {1, 1, 1, 69}, {}});
    }
}

TEST(GI, ElementwiseBinaryDynamic) {
    Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    //! as TRUE_DIV will case precision error when compile with -Ofast, set
    //! epsilon to 1e-4
    checker.set_epsilon(1e-4);
    checker.set_dynamic_megcc(true);
    ElemwiseForward::Param param;
    param.mode = MODE::SUB;
    checker.set_param(param);
    auto normal_cases = get_elewise_binary_case();
    for (auto&& shapes : normal_cases) {
        checker.execs(shapes);
    }

    auto bound_case = get_elewise_binary_bound_case();
    checker.set_param(param);
    for (auto&& shapes : bound_case) {
        checker.execs(shapes);
    }
}

TEST(GI, ElementwiseTernary) {
    Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    ElemwiseForward::Param param;
    checker.set_kernel_symbol("GI_kernel_elementwise.+");
    for (auto mode : {MODE::FUSE_MUL_ADD3}) {
        param.mode = mode;
        checker.set_param(param);
        //! vec_vec
        checker.execs({{1, 13}, {1, 13}, {1, 13}, {}});
        checker.execs({{1, 1}, {1, 1}, {1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        // // //! vec_bcast101_vec
        checker.execs({{2, 3, 4, 5}, {2, 3, 1, 1}, {2, 3, 4, 5}, {}});
        checker.execs({{5, 6, 7, 8}, {1, 6, 7, 1}, {5, 6, 7, 8}, {}});
        // // //! vec_bcast101x4_vec
        checker.execs({{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
        checker.execs({{5, 6, 7, 8, 4}, {5, 6, 1, 1, 4}, {5, 6, 7, 8, 4}, {}});
    }
}

// vim: syntax=cpp.doxygen
