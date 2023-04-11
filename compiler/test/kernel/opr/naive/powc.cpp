#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;

TEST(NAIVE, PowC) {
    Checker<PowC> checker;
    checker.set_kernel_symbol("kernel_.*");
    ::megcc::test::UniformRNG rng(0, 5);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-4);

    PowC::Param param;
    param.exp = 2.5f;
    checker.set_param(param);
    checker.execs({{1}, {}});
    checker.execs({{1, 10}, {}});
    checker.execs({{1, 10, 12, 13}, {}});
}
