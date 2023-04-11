#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(GI, Resize) {
    Checker<ResizeForward> checker(Arch::BAREMETAL);
    megcc::test::UniformRNG rng(-30, 30);
    checker.set_rng(0, &rng);
    checker.set_epsilon(3e-4);
    megdnn::ResizeForward::Param param;
    param.format = megdnn::ResizeForward::Param::Format::NCHW;
    param.imode = megdnn::ResizeForward::Param::InterpolationMode::LINEAR;
    checker.set_param(param);
    checker.execs({{1, 1, 5, 6}, {1, 1, 7, 13}});
    checker.execs({{1, 4, 5, 6}, {1, 4, 9, 12}});
    checker.execs({{2, 3, 15, 16}, {2, 3, 9, 12}});
}