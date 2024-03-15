#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using MODE = ElemwiseMultiType::Param::Mode;

TEST(AARCH64, ElementwiseMultitypeUnary) {
    Checker<ElemwiseMultiType> checker(megcc::KernelGen::Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_.*");
    checker.set_epsilon(1e-4);
    checker.set_dtype(0, dtype::QuantizedS32(1.f));
    checker.set_dtype(1, dtype::QuantizedS8(2.f));
    ElemwiseMultiType::Param param;
    for (auto mode : {MODE::QRELU}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 33}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }

    checker.set_dtype(0, dtype::QuantizedS8(1.f));
    checker.set_dtype(1, dtype::QuantizedS8(3.f));
    for (auto mode : {MODE::QRELU}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {}});
        checker.execs({{1, 33}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
}

TEST(AARCH64, ElementwiseMultitypeBinary) {
    Checker<ElemwiseMultiType> checker(megcc::KernelGen::Arch::ARM64);
    checker.set_kernel_symbol("Arm64_kernel_.*");
    checker.set_epsilon(1e-4);
    checker.set_dtype(0, dtype::QuantizedS8(1.f));
    checker.set_dtype(1, dtype::QuantizedS8(2.f));
    checker.set_dtype(2, dtype::QuantizedS8(3.f));
    ElemwiseMultiType::Param param;

    for (auto mode : {MODE::QADD, MODE::QFUSE_ADD_RELU, MODE::QMUL}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 18}, {1, 18}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {1}, {}});
    }

    checker.set_dtype(0, dtype::QuantizedS32(0.73f));
    checker.set_dtype(1, dtype::QuantizedS32(2.21f));

    for (auto mode : {MODE::QADD, MODE::QFUSE_ADD_RELU, MODE::QMUL}) {
        param.mode = mode;
        checker.set_param(param);
        checker.execs({{1}, {1}, {}});
        checker.execs({{1, 18}, {1, 18}, {}});
        checker.execs({{2, 3, 4, 5}, {2, 3, 4, 5}, {}});
        checker.execs({{2, 3, 4, 5}, {1, 3, 1, 1}, {}});
        checker.execs({{2, 3, 4, 5}, {1}, {}});
    }
}