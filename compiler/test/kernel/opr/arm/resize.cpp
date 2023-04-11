#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(AARCH64, Resize) {
    Checker<ResizeForward> checker(Arch::ARM64);
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

#ifdef ENABLE_KERNEL_BENCHMARK
TEST(AARCH64, BENCHMARK_Resize) {
    Benchmarker<ResizeForward> benchmarker(Arch::ARM64);
    megdnn::ResizeForward::Param param;
    param.format = megdnn::ResizeForward::Param::Format::NCHW;
    param.imode = megdnn::ResizeForward::Param::InterpolationMode::LINEAR;
    benchmarker.set_param(param);
    benchmarker.execs({{1, 1, 1080, 1920}, {1, 1, 720, 1280}}).print();
    benchmarker.execs({{1, 3, 1080, 1920}, {1, 3, 720, 1280}}).print();
    benchmarker.execs({{1, 32, 112, 112}, {1, 32, 224, 224}}).print();
}
#endif