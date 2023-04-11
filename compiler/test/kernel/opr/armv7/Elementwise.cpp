#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;

TEST(ARMV7, ElementwiseUnique) {
    Checker<ElemwiseForward> checker(Arch::ARMV7);
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

TEST(ARMV7, ElementwiseBinary) {
    Checker<ElemwiseForward> checker(Arch::ARMV7);
    ElemwiseForward::Param param;
    checker.set_epsilon(3e-4);
    std::vector<TensorShapeArray> shape_vec;
    {
        shape_vec.push_back({{1, 10}, {1, 10}, {}});
        shape_vec.push_back({{1, 10}, {1, 10}, {}});
        //! vec_vec
        shape_vec.push_back({{1, 10}, {1, 10}, {}});
        shape_vec.push_back({{10, 10}, {10, 10}, {}});
        shape_vec.push_back({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        //! vec_scalar
        shape_vec.push_back({{1, 10}, {1, 10}, {}});
        shape_vec.push_back({{10, 10}, {1}, {}});
        shape_vec.push_back({{100}, {1}, {}});
        //! scalar_vec
        shape_vec.push_back({{1, 10}, {1, 10}, {}});
        shape_vec.push_back({{1}, {100}, {}});
        shape_vec.push_back({{1}, {10, 32}, {}});
        shape_vec.push_back({{1, 10}, {1, 10}, {}});
        //! vec_bcast101
        shape_vec.push_back({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        shape_vec.push_back({{5, 6, 7, 8}, {1, 6, 1, 1}, {}});
        shape_vec.push_back({{1, 81, 2}, {1, 1, 2}, {}});
        shape_vec.push_back({{2, 7}, {1, 7}, {}});
        shape_vec.push_back({{2, 4}, {1, 4}, {}});
        //! bcast101_vec
        shape_vec.push_back({{1, 3, 1, 1}, {2, 3, 4, 5}, {}});
        shape_vec.push_back({{1, 6, 1, 1}, {5, 6, 7, 8}, {}});
        shape_vec.push_back({{1, 7}, {2, 7}, {}});
        shape_vec.push_back({{1, 4}, {2, 4}, {}});
        //! vec_bcast101x4
        shape_vec.push_back({{2, 3, 4, 5, 4}, {1, 3, 1, 1, 4}, {}});
        shape_vec.push_back({{5, 6, 7, 8, 4}, {1, 6, 1, 1, 4}, {}});
        //! bcast101x4_vec
        shape_vec.push_back({{1, 3, 1, 1, 4}, {2, 3, 4, 5, 4}, {}});
        shape_vec.push_back({{1, 6, 1, 1, 4}, {5, 6, 7, 8, 4}, {}});
    }
    for (auto mode : {MODE::ADD, MODE::SUB, MODE::MUL, MODE::FUSE_ADD_RELU}) {
        param.mode = mode;
        checker.set_param(param);
        for (auto& testcase : shape_vec) {
            checker.execs(testcase);
        }
    }
    {
        param.mode = MODE::TRUE_DIV;
        megcc::test::UniformRNG one2three(1, 3);
        megcc::test::UniformRNG one2six(1, 6);
        checker.set_rng(0, &one2three);
        checker.set_rng(1, &one2six);
        checker.set_param(param);
        for (auto& testcase : shape_vec) {
            checker.execs(testcase);
        }
    }
}

TEST(ARMV7, ElementwiseTernary) {
    Checker<ElemwiseForward> checker(Arch::ARMV7);
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