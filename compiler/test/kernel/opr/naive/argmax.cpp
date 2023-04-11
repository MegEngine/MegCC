#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, Argmax) {
    Checker<Argmax> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("kernel_.*");
    for (DType dtype : {(DType)dtype::Float32()}) {
        checker.set_dtype(0, dtype);
        for (auto src :
             {TensorShape{2, 3}, TensorShape{3, 4, 5}, TensorShape{4, 5, 6, 7}})
            for (size_t axis = 0; axis < 4; ++axis) {
                if (axis < src.ndim) {
                    ArgmaxForward::Param param;
                    param.axis = axis;
                    checker.set_param(param);
                    checker.execs({src, {}});
                }
            }
    }
}
