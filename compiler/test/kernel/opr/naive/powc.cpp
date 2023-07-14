#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;

TEST(NAIVE, PowC) {
    Checker<PowC> checker;
    checker.set_kernel_symbol("kernel_.*");
    ::megcc::test::UniformRNG rng(0, 5);
    checker.set_rng(0, &rng);

    PowC::Param param;
    param.exp = 2.5f;
    checker.set_param(param);
#if ENABLE_KERNEL_FP16
    checker.set_epsilon(5e-3);
    for (auto dtype : {(DType)dtype::Float32(), (DType)dtype::Float16()})
#else
    checker.set_epsilon(1e-4);
    for (auto dtype : {(DType)dtype::Float32()})
#endif
    {
        checker.set_dtype(0, dtype);
        checker.set_dtype(1, dtype);
        checker.execs({{1}, {}});
        checker.execs({{1, 10}, {}});
        checker.execs({{1, 10, 12, 13}, {}});
    }
}
