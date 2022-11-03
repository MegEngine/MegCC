/**
 * \file
 * compiler/test/kernel/opr/arm/Elementwise.cpp
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

TEST(AARCH64, ElementwiseUnique) {
    Checker<ElemwiseForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("ArmCommon_kernel_elementwise.+");
    ElemwiseForward::Param param;
    for (auto mode : {MODE::RELU, MODE::EXP, MODE::SIGMOID, MODE::H_SWISH}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
        checker.execs({{10, 8, 2, 1}, {}});
    }
}

TEST(AARCH64, ElementwiseBinary) {
    Checker<ElemwiseForward> checker(Arch::ARM64);
    ElemwiseForward::Param param;
    auto normal_cases = get_elewise_binary_case();
    for (auto mode : {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU}) {
        param.mode = mode;
        checker.set_param(param);
        for (auto&& shapes : normal_cases) {
            checker.execs(shapes);
        }
    }

    auto bound_case = get_elewise_binary_bound_case();
    param.mode = MODE::SUB;
    checker.set_param(param);
    for (auto&& shapes : bound_case) {
        checker.execs(shapes);
    }
    {
        //! as TRUE_DIV will case precision error when compile with -Ofast, set
        //! epsilon to 1e-4
        checker.set_epsilon(1e-4);
        megcc::test::UniformRNG rng(3, 12);
        checker.set_rng(0, &rng);
        checker.set_rng(1, &rng);
        param.mode = MODE::TRUE_DIV;
        checker.set_param(param);

        for (auto&& shapes : normal_cases) {
            checker.execs(shapes);
        }
    }
}

TEST(AARCH64, GIElementwiseUnique) {
    Checker<ElemwiseForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("GI_kernel_elementwise.+");
    ElemwiseForward::Param param;
    for (auto mode : {MODE::RELU}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
        checker.execs({{10, 8, 2, 1}, {}});
    }
}

TEST(AARCH64, GIElementwiseBinary) {
    Checker<ElemwiseForward> checker(Arch::ARM64);
    checker.set_kernel_symbol("GI_kernel_elementwise.+");
    ElemwiseForward::Param param;
    auto normal_cases = get_elewise_binary_case();
    for (auto mode : {MODE::ADD}) {
        param.mode = mode;
        checker.set_param(param);
        for (auto&& shapes : normal_cases) {
            checker.execs(shapes);
        }
    }
}

TEST(AARCH64, ElementwiseBinaryDynamic) {
    Checker<ElemwiseForward> checker(Arch::ARM64);
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

TEST(AARCH64, ElementwiseTernary) {
    Checker<ElemwiseForward> checker(Arch::ARM64);
    //! as TRUE_DIV will case precision error when compile with -Ofast, set
    //! epsilon to 1e-4

    ElemwiseForward::Param param;
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