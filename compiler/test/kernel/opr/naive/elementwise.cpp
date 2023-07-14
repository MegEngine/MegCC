#include "test/kernel/common/checker.h"
#include "test/kernel/common/rng.h"
#include "test/kernel/opr/common/elemwise.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;
TEST(NAIVE, ElementwiseUnique) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    ElemwiseForward::Param param;
    for (auto mode :
         {MODE::RELU, MODE::SIGMOID, MODE::EXP, MODE::NEGATE, MODE::ROUND,
          MODE::H_SWISH, MODE::ABS}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
    {
        param.mode = MODE::LOG;
        checker.set_param(param);
        megcc::test::UniformRNG rng(1e-5, 3);
        checker.set_rng(0, &rng);
        checker.execs({{1, 10}, {}});
    }
}

TEST(NAIVE, ElementwiseUniqueQuant) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    ElemwiseForward::Param param;
    //! NOTE: maybe bug for -128 with negate mode
    UniformIntRNG rng(-128, 127);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::Int8());
    for (auto mode : {MODE::RELU, MODE::NEGATE}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
}

TEST(NAIVE, ElementwiseBinary) {
    //! only support 1x11 broadcast
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(1e-4);

    ElemwiseForward::Param param;
    for (auto mode :
         {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU, MODE::FUSE_ADD_SIGMOID,
          MODE::MAX, MODE::MIN, MODE::LEQ, MODE::LT, MODE::EQ, MODE::FUSE_ADD_TANH}) {
        param.mode = mode;
        checker.set_param(param);
        // scalar_scalar
        checker.execs({{1}, {1}, {}});
        // vec_vec
        checker.execs({{1, 10}, {1, 10}, {}});
        // vec_broadcast101
        checker.execs({{2, 10}, {1}, {}});
        checker.execs({{1}, {2, 10}, {}});
        // vec_broadcast101
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
        // vec_broadcast101x4
        checker.execs({{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
        // vec_scalar
        checker.execs({{2, 3, 4}, {1}, {}});
        // sclar_vec
        checker.execs({{1}, {2, 3, 4}, {}});
    }

    megcc::test::UniformRNG rng(3, 12);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    for (auto mode : {MODE::TRUE_DIV, MODE::FLOOR_DIV}) {
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
    }
}

#if ENABLE_KERNEL_FP16

TEST(NAIVE, ElementwiseUniqueFp16) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(4e-3);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());
    ElemwiseForward::Param param;
    for (auto mode :
         {MODE::RELU, MODE::SIGMOID, MODE::EXP, MODE::NEGATE, MODE::ROUND,
          MODE::H_SWISH, MODE::ABS}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
    {
        param.mode = MODE::LOG;
        checker.set_param(param);
        megcc::test::UniformRNG rng(1e-5, 3);
        checker.set_rng(0, &rng);
        checker.execs({{1, 10}, {}});
    }
}

TEST(NAIVE, ElementwiseBinaryFp16) {
    //! only support 1x11 broadcast
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(4e-3);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());
    checker.set_dtype(2, dtype::Float16());

    ElemwiseForward::Param param;
    for (auto mode :
         {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU, MODE::FUSE_ADD_SIGMOID,
          MODE::MAX, MODE::MIN, MODE::LEQ, MODE::LT, MODE::EQ, MODE::FUSE_ADD_TANH}) {
        param.mode = mode;
        checker.set_param(param);
        // scalar_scalar
        checker.execs({{1}, {1}, {}});
        // vec_vec
        checker.execs({{1, 10}, {1, 10}, {}});
        // vec_broadcast101
        checker.execs({{2, 10}, {1}, {}});
        checker.execs({{1}, {2, 10}, {}});
        // vec_broadcast101
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
        // vec_broadcast101x4
        checker.execs({{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
        // vec_scalar
        checker.execs({{2, 3, 4}, {1}, {}});
        // sclar_vec
        checker.execs({{1}, {2, 3, 4}, {}});
    }

    megcc::test::UniformRNG rng(3, 12);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    for (auto mode : {MODE::TRUE_DIV, MODE::FLOOR_DIV}) {
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
    }
}

TEST(NAIVE, ElementwiseTernaryFp16) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    ElemwiseForward::Param param;
    checker.set_dynamic_megcc(true);
    param.mode = MODE::FUSE_MUL_ADD3;
    checker.set_param(param);
    checker.set_epsilon(4e-3);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());
    checker.set_dtype(2, dtype::Float16());
    checker.set_dtype(3, dtype::Float16());

    checker.execs({{1}, {1}, {1}, {}});
    checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {}});
    checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {1}, {}});
    checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {1, 3, 1, 1}, {}});
    // broadcast_vector_broadcast
    checker.execs({{1, 3, 1, 1, 4}, {2, 3, 4, 4, 4}, {1, 3, 1, 1, 4}, {}});

    checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    checker.execs({{2, 3, 4, 4}, {1, 3, 1, 1}, {2, 3, 4, 4}, {}});
    checker.execs({{2, 3, 4, 5}, {1}, {2, 3, 4, 5}, {}});
    checker.execs({{2, 3, 4, 5}, {1}, {1}, {}});
    checker.execs({{1}, {2, 3, 4, 5}, {1}, {}});
    // multi batch broadcast
    checker.execs({{2, 32, 32, 20}, {2, 32, 1, 1}, {2, 32, 32, 20}, {}});
    checker.execs({{2, 32, 32, 20, 4}, {2, 32, 1, 1, 4}, {2, 32, 32, 20, 4}, {}});
}

TEST(NAIVE, ElementwiseQuaterFp16) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    ElemwiseForward::Param param;
    checker.set_dynamic_megcc(true);
    param.mode = MODE::FUSE_MUL_ADD4;
    checker.set_param(param);

    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());
    checker.set_dtype(2, dtype::Float16());
    checker.set_dtype(3, dtype::Float16());
    checker.set_dtype(4, dtype::Float16());

    checker.execs({{1}, {1}, {1}, {1}, {}});
    checker.execs({{3, 3}, {3, 3}, {3, 3}, {3, 3}, {}});
    checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {}});
    //! dnn require layout 0==2, 1==3
    checker.execs({{1}, {2, 3, 4, 5}, {1}, {2, 3, 4, 5}, {}});
    checker.execs({{1}, {1, 33}, {1}, {1, 33}, {}});
    checker.execs({{1}, {1, 3, 11}, {1}, {1, 3, 11}, {}});
    checker.execs({{2, 3, 4, 5}, {1}, {2, 3, 4, 5}, {1}, {}});
    checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}, {1, 3, 1, 1}, {}});
    checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    checker.execs(
            {{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {}});
    checker.execs(
            {{1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
    // multi batch broadcast
    checker.execs(
            {{2, 32, 32, 20, 4},
             {2, 32, 1, 1, 4},
             {2, 32, 32, 20, 4},
             {2, 32, 1, 1, 4},
             {}});
}
#endif

TEST(NAIVE, ElementwiseBinaryInt) {
    //! only support 1x11 broadcast
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(1e-4);
    megcc::test::UniformIntRNG rng(-20, 20);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    ElemwiseForward::Param param;
    for (auto mode :
         {MODE::ADD, MODE::SUB, MODE::MUL, MODE::MAX, MODE::MIN, MODE::LEQ, MODE::LT}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 10}, {1, 10}, {}});
        checker.execs({{1, 10}, {1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    }

    megcc::test::UniformIntRNG rng2(3, 12);
    checker.set_rng(0, &rng2);
    checker.set_rng(1, &rng2);
    for (auto mode : {MODE::FLOOR_DIV, MODE::MOD}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 10}, {1, 10}, {}});
        checker.execs({{1, 10}, {1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    }
}

TEST(NAIVE, ElementwiseTernary) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    ElemwiseForward::Param param;
    checker.set_dynamic_megcc(true);
    param.mode = MODE::FUSE_MUL_ADD3;
    checker.set_param(param);
    checker.execs({{1}, {1}, {1}, {}});
    checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {}});
    checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {1}, {}});
    checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {1, 3, 1, 1}, {}});
    // broadcast_vector_broadcast
    checker.execs({{1, 3, 1, 1, 4}, {2, 3, 4, 4, 4}, {1, 3, 1, 1, 4}, {}});

    checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    checker.execs({{2, 3, 4, 4}, {1, 3, 1, 1}, {2, 3, 4, 4}, {}});
    checker.execs({{2, 3, 4, 5}, {1}, {2, 3, 4, 5}, {}});
    checker.execs({{2, 3, 4, 5}, {1}, {1}, {}});
    checker.execs({{1}, {2, 3, 4, 5}, {1}, {}});
    // multi batch broadcast
    checker.execs({{2, 32, 32, 20}, {2, 32, 1, 1}, {2, 32, 32, 20}, {}});
    checker.execs({{2, 32, 32, 20, 4}, {2, 32, 1, 1, 4}, {2, 32, 32, 20, 4}, {}});
}

TEST(NAIVE, ElementwiseQuater) {
    Checker<ElemwiseForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    ElemwiseForward::Param param;
    checker.set_dynamic_megcc(true);
    param.mode = MODE::FUSE_MUL_ADD4;
    checker.set_param(param);
    checker.execs({{1}, {1}, {1}, {1}, {}});
    checker.execs({{3, 3}, {3, 3}, {3, 3}, {3, 3}, {}});
    checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, {}});
    //! dnn require layout 0==2, 1==3
    checker.execs({{1}, {2, 3, 4, 5}, {1}, {2, 3, 4, 5}, {}});
    checker.execs({{1}, {1, 33}, {1}, {1, 33}, {}});
    checker.execs({{1}, {1, 3, 11}, {1}, {1, 3, 11}, {}});
    checker.execs({{2, 3, 4, 5}, {1}, {2, 3, 4, 5}, {1}, {}});
    checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}, {1, 3, 1, 1}, {}});
    checker.execs({{1, 3, 1, 1}, {2, 3, 4, 5}, {1, 3, 1, 1}, {2, 3, 4, 5}, {}});
    checker.execs(
            {{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {}});
    checker.execs(
            {{1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
    // multi batch broadcast
    checker.execs(
            {{2, 32, 32, 20, 4},
             {2, 32, 1, 1, 4},
             {2, 32, 32, 20, 4},
             {2, 32, 1, 1, 4},
             {}});
}

TEST(NAIVE, ElementwiseBinary_Boundary_test) {
    megcc::test::Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(1e-4);
    megcc::test::UniformRNG rng(-32, 32);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    TensorShape dst_shape_temp{2, 3, 4, 5, 6};
    ElemwiseForward::Param param;
    auto bound_case = get_elewise_binary_bound_case();
    for (auto mode :
         {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU, MODE::FUSE_ADD_SIGMOID,
          MODE::MAX, MODE::MIN, MODE::LEQ, MODE::LT, MODE::FUSE_ADD_TANH}) {
        param.mode = mode;
        checker.set_param(param);
        for (auto&& shapes : bound_case) {
            checker.execs(shapes);
        }
    }
}

TEST(NAIVE, ElementwiseTenary_Boundary_test) {
    megcc::test::Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(1e-4);
    megcc::test::UniformRNG rng(-32, 32);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    TensorShape dst_shape_temp{2, 3, 4, 5, 6};
    ElemwiseForward::Param param;
    auto mode = MODE::FUSE_MUL_ADD3;
    param.mode = mode;
    checker.set_param(param);
    auto bound_cases = get_elewise_tenary_bound_case();
    for (auto&& arg : bound_cases) {
        checker.execs(arg);
    }
}

TEST(NAIVE, ElementwiseQunary_Boundary_test) {
    megcc::test::Checker<ElemwiseForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    checker.set_epsilon(1e-4);
    megcc::test::UniformRNG rng(-32, 32);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    TensorShape dst_shape_temp{2, 3, 4, 5, 6};
    ElemwiseForward::Param param;
    auto mode = MODE::FUSE_MUL_ADD4;
    param.mode = mode;
    checker.set_param(param);
    for (int ndim = 2; ndim < 6; ++ndim) {
        TensorShape dst_shape;
        dst_shape.ndim = ndim;
        for (int i = 0; i < ndim; ++i) {
            dst_shape[i] = dst_shape_temp[i];
        }
        for (int i = 0; i < (1 << ndim); ++i) {
            TensorShape sample_shape = dst_shape;
            for (int bit = 0; bit < ndim; ++bit) {
                if (((i >> bit) & 0x1) != 1) {
                    sample_shape[bit] = 1;
                }
            }
            checker.execs(
                    {dst_shape, sample_shape, dst_shape, sample_shape, dst_shape});
            checker.execs(
                    {dst_shape, sample_shape, sample_shape, dst_shape, dst_shape});
            checker.execs(
                    {sample_shape, dst_shape, sample_shape, dst_shape, dst_shape});
            checker.execs(
                    {sample_shape, dst_shape, dst_shape, sample_shape, dst_shape});
        }
    }
}

// vim: syntax=cpp.doxygen
