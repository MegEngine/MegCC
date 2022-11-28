/**
 * \file
 * compiler/test/kernel/opr/naive/fused_elementwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string>
#include "megdnn/handle.h"
#include "test/kernel/common/checker.h"
#include "test/kernel/opr/common/fused_elemwise.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = ElemwiseForward::Param::Mode;

TEST(NAIVE, FusedElemwiseKernel) {
    //! check: 2 * (x+y)-z
    std::vector<std::string> modes;
    modes.push_back("I0,I1,ADD,O0");
    modes.push_back("I2,O0,MUL,O1");
    modes.push_back("O0,I3,SUB,D");

    auto test = [&](TensorShapeArray shapes) {
        check_fuse_elemwise(shapes, modes, megcc::KernelGen::Arch::BAREMETAL,
                            "kernel_naive_.*");
    };

    test({{4, 6, 9, 2}, {4, 6, 9, 2}, {1}, {4, 6, 9, 2}});
    test({{1, 6, 1, 1}, {4, 6, 9, 2}, {1}, {4, 6, 9, 2}});
    test({{1, 6, 1, 1}, {1, 6, 1, 1}, {1}, {4, 6, 9, 2}});
}

TEST(NAIVE, FusedMoreElemwise) {
    //! check: max(sigmoid(x + y), z) / z 
    std::vector<std::string> modes;
    modes.push_back("I0,I1,ADD,O0");
    modes.push_back("O0,SIGMOID,O1");
    modes.push_back("O1,I2,MAX,O2");
    modes.push_back("O2,I2,TRUE_DIV,D");

    auto test = [&](TensorShapeArray shapes) {
        check_fuse_elemwise(shapes, modes, megcc::KernelGen::Arch::BAREMETAL,
                            "kernel_naive_.*", 1e-1);
    };

    test({{3, 10, 10, 5, 4},{1, 10, 1, 1, 4}, {1}});
    test({{3, 10, 10, 5, 4},{3, 10, 10, 5, 4}, {3, 10, 10, 5, 4}});
    test({{3, 10, 10, 5, 4},{1, 10, 1, 1, 4}, {1, 10, 1, 1, 4}});
}

TEST(NAIVE, FusedAllMode) {
    std::vector<std::string> modes;
    modes.push_back("I0,RELU,O0");
    modes.push_back("I1,EXP,O1");
    modes.push_back("O0,O1,FUSE_ADD_RELU,D");

    auto test = [&](TensorShapeArray shapes) {
        check_fuse_elemwise(shapes, modes, megcc::KernelGen::Arch::BAREMETAL,
                            "kernel_naive_.*");
    };
    test({{3, 10, 10, 5, 4}, {1, 10, 1, 1, 4}});
}

TEST(NAIVE, FusedAllMode2) {
    std::vector<std::string> modes;
    modes.push_back("I0,NEGATE,O0");
    modes.push_back("I1,H_SWISH,O1");
    modes.push_back("O0,O1,FUSE_ADD_SIGMOID,D");

    auto test = [&](TensorShapeArray shapes) {
        check_fuse_elemwise(shapes, modes, megcc::KernelGen::Arch::BAREMETAL,
                            "kernel_naive_.*");
    };
    test({{3, 10, 10, 5, 4}, {1, 10, 1, 1, 4}});
}

TEST(NAIVE, FusedMulAnd3) {
    //! relu(fuse_mul_add3(x, y, z))
    std::vector<std::string> modes;
    modes.push_back("I0,I1,I2,FUSE_MUL_ADD3,O0");
    modes.push_back("O0,RELU,D");

    auto test = [&](TensorShapeArray shapes) {
        check_fuse_elemwise(shapes, modes, megcc::KernelGen::Arch::BAREMETAL,
                            "kernel_naive_.*");
    };

    test({{3, 10, 10, 5}, {3, 10, 10, 5}, {3, 10, 10, 5}});
}

TEST(NAIVE, FusedMulAnd4) {
    //! relu(fuse_mul_add4(x, y, z, w))
    //
    std::vector<std::string> modes;
    modes.push_back("I0,I1,I2,I3,FUSE_MUL_ADD4,O0");
    modes.push_back("O0,RELU,D");

    auto test = [&](TensorShapeArray shapes) {
        check_fuse_elemwise(shapes, modes, megcc::KernelGen::Arch::BAREMETAL,
                            "kernel_naive_.*");
    };

    test({{3, 10, 10, 5},{3, 10, 10, 5}, {3, 10, 10, 5}, {3, 10, 10, 5}});
}

// vim: syntax=cpp.doxygen
