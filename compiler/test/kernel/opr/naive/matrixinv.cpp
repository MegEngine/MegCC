#include "test/kernel/common/checker.h"
#include "test/kernel/common/rng.h"
using namespace megdnn;
using namespace megcc::test;

TEST(NAIVE, MatInv) {
    Checker<MatrixInverse> checker;
    checker.set_kernel_symbol("kernel_.*");
    InvertibleMatrixRNG rng;
    checker.set_rng(0, &rng);
    checker.set_epsilon(3e-3);
    checker.exec({{1, 1}, {}});
    checker.exec({{3, 3}, {}});
    checker.exec({{1, 7, 7}, {}});
    checker.exec({{3, 7, 7}, {}});
    checker.exec({{3, 3, 7, 7}, {}});
}
