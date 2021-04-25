/**
 * \file
 * compiler/test/kernel/opr/generalIntrinsic/benchmark_elemwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;
#ifdef ENABLE_KERNEL_BENCHMARK
TEST(GI, ElementwiseUnique_BMK) {
    Benchmarker<ElemwiseForward> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("GI_kernel_elementwise.+");
    ElemwiseForward::Param param;
    for (auto mode : {MODE::RELU, MODE::SIGMOID, MODE::EXP, MODE::H_SWISH}) {
        printf("mode=%d\n", mode);
        param.mode = mode;
        benchmarker.set_param(param);
        benchmarker.execs({{10000}, {}}).print();
        benchmarker.execs({{1, 10, 400, 500}, {}}).print();
    }
}

TEST(GI, ElementwiseBinary_BMK) {
    //! only support 1x11 broadcast
    Benchmarker<ElemwiseForward> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("GI_kernel_elementwise.+");
    megcc::test::UniformRNG rng(3, 12);
    benchmarker.set_rng(0, &rng);
    benchmarker.set_rng(1, &rng);
    ElemwiseForward::Param param;
    for (auto mode : {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU,
                      MODE::TRUE_DIV}) {
        printf("mode=%d\n", mode);
        param.mode = mode;
        benchmarker.set_param(param);
        benchmarker.execs({{10000}, {10000}, {}}).print();
        benchmarker.execs({{1, 1}, {1, 10000}, {}}).print();
        benchmarker.execs({{1, 10000}, {1, 1}, {}}).print();
        benchmarker.execs({{2, 3, 400, 500}, {2, 3, 400, 500}, {}}).print();
        benchmarker.execs({{2, 3, 400, 500}, {1, 3, 1, 1}, {}}).print();
        benchmarker.execs({{1, 3, 1, 1}, {2, 3, 400, 500}, {}}).print();
    }
}

TEST(GI, ElementwiseTernary_BMK) {
    Benchmarker<ElemwiseForward> benchmarker(Arch::BAREMETAL);
    ElemwiseForward::Param param;
    benchmarker.set_kernel_symbol("GI_kernel_elementwise.+");
    for (auto mode : {MODE::FUSE_MUL_ADD3}) {
        printf("mode=%d\n", mode);
        param.mode = mode;
        benchmarker.set_param(param);
        //! vec_vec
        benchmarker.execs({{1, 13000}, {1, 13000}, {1, 13000}, {}}).print();
        benchmarker
                .execs({{2, 3, 400, 500},
                        {2, 3, 400, 500},
                        {2, 3, 400, 500},
                        {}})
                .print();
        //! vec_bcast101_vec
        benchmarker
                .execs({{2, 3, 400, 500}, {2, 3, 1, 1}, {2, 3, 400, 500}, {}})
                .print();
        benchmarker
                .execs({{5, 6, 700, 800}, {1, 6, 700, 1}, {5, 6, 700, 800}, {}})
                .print();
        //! vec_bcast101x4_vec
        benchmarker
                .execs({{2, 3, 400, 500, 4},
                        {1, 3, 1, 1, 4},
                        {2, 3, 400, 500, 4},
                        {}})
                .print();
        benchmarker
                .execs({{5, 6, 700, 800, 4},
                        {5, 6, 1, 1, 4},
                        {5, 6, 700, 800, 4},
                        {}})
                .print();
    }
}
#endif
