#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = ReduceForward::Param::Mode;
TEST(ARMV7, Reduce) {
    Checker<Reduce> checker(Arch::ARMV7);
    for (auto mode :
         {Mode::SUM, Mode::MEAN, Mode::MAX, Mode::MIN, Mode::PRODUCT, Mode::SUM_SQR})
        for (auto src : {TensorShape{2, 3}, {3, 4, 5}, {4, 5, 6, 7}})
            for (size_t axis = 0; axis < 4; ++axis) {
                if (axis < src.ndim) {
                    ReduceForward::Param param;
                    param.axis = axis;
                    param.mode = mode;
                    checker.set_param(param);
                    checker.execs({src, {}});
                }
            }
}
