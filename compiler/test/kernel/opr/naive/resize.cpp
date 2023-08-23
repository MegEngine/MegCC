#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, Resize) {
    Checker<ResizeForward> checker;
    checker.set_kernel_symbol("kernel_.*");
    using Param = megdnn::ResizeForward::Param;
    megcc::test::UniformRNG rng(-30, 30);
    checker.set_rng(0, &rng);
    checker.set_epsilon(5e-4);
    Param param;
    param.format = Param::Format::NCHW;
    param.imode = Param::InterpolationMode::LINEAR;
    checker.set_param(param);
    checker.execs({{1, 1, 5, 6}, {1, 1, 7, 13}});
    checker.execs({{1, 4, 5, 6}, {1, 4, 9, 12}});
    checker.execs({{2, 3, 15, 16}, {2, 3, 9, 12}});
    param.format = Param::Format::NCHW44;
    param.imode = Param::InterpolationMode::LINEAR;
    checker.set_param(param);
    checker.execs({{1, 1, 5, 6, 4}, {1, 1, 7, 13, 4}});
    checker.execs({{1, 4, 5, 6, 4}, {1, 4, 9, 12, 4}});
    checker.execs({{2, 3, 15, 16, 4}, {2, 3, 9, 12, 4}});
    param.format = Param::Format::NCHW;
    param.imode = Param::InterpolationMode::NEAREST;
    checker.set_param(param);
    checker.execs({{1, 1, 5, 6}, {1, 1, 7, 13}});
    checker.execs({{1, 4, 5, 6}, {1, 4, 9, 12}});
    checker.execs({{2, 3, 15, 16}, {2, 3, 9, 12}});
}