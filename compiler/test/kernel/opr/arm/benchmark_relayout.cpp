#include "test/kernel/common/benchmark.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using Mode = RelayoutFormat::Param::Mode;

#ifdef ENABLE_KERNEL_BENCHMARK
TEST(AARCH64, BENCHMARK_Relayout) {
    Benchmarker<RelayoutForward> benchmarker(Arch::ARM64);
    megdnn::DType dtype = dtype::Int8();

    TensorLayout src({1, 128, 32, 32}, {1024, 2048, 32, 1}, dtype);
    TensorLayout dst({1, 128, 32, 32}, {131072, 1024, 32, 1}, dtype);
    benchmarker.execl({src, dst});
    benchmarker.execl({src, dst}).print();

    src = {{1, 32, 32, 32, 4}, {262144, 8192, 32, 1, 2048}, dtype};
    dst = {{1, 32, 32, 32, 4}, {131072, 4096, 128, 4, 1}, dtype};
    benchmarker.execl({src, dst}).print();

    src = {{1, 64, 4, 32, 32}, {262144, 4096, 1, 128, 4}, dtype};
    dst = {{1, 64, 4, 32, 32}, {262144, 4096, 1024, 32, 1}, dtype};
    benchmarker.execl({src, dst}).print();

    src = {{1, 3, 256, 320}, {245760, 1, 960, 3}, dtype};
    dst = {{1, 3, 256, 320}, {245760, 81920, 320, 1}, dtype};
    benchmarker.execl({src, dst}).print();

    src = {{1, 1, 256, 256}, {65536, 65536, 1, 256}, dtype::Float32()};
    dst = {{1, 1, 256, 256}, {65536, 65536, 256, 1}, dtype::Float32()};
    benchmarker.execl({src, dst}).print();

    src = {{1, 2307, 4, 4}, {36912, 1, 2307, 9228}, dtype::Float32()};
    dst = {{1, 2307, 4, 4}, {36912, 16, 4, 1}, dtype::Float32()};
    benchmarker.execl({src, dst}).print();

    src = {{1, 64, 4, 32, 32}, {262144, 4096, 1, 128, 4}, dtype::Float32()};
    dst = {{1, 64, 4, 32, 32}, {262144, 4096, 1024, 32, 1}, dtype::Float32()};
    benchmarker.execl({src, dst}).print();

    src = {{2, 32, 4, 32, 20}, {81920, 2560, 1, 80, 4}, dtype::Float32()};
    dst = {{2, 32, 4, 32, 20}, {81920, 2560, 640, 20, 1}, dtype::Float32()};
    benchmarker.execl({src, dst}).print();
}
#endif
//*/