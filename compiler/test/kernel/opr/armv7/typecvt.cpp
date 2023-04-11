#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(ARMV7, TYPECVT) {
    Checker<TypeCvtForward> checker(Arch::ARMV7);
    UniformIntRNG rng(-127, 127);
    checker.set_rng(0, &rng);
    std::vector<std::pair<megdnn::DType, megdnn::DType>> types = {
            {dtype::QuantizedS8(0.3f), dtype::Float32()},
            {dtype::Float32(), dtype::QuantizedS8(1.7f)},
            {dtype::QuantizedS8(1.7f), dtype::QuantizedS8(0.3f)},
            {dtype::Uint8(), dtype::Float32()}};
    for (auto type : types) {
        checker.set_dtype(0, type.first);
        checker.set_dtype(1, type.second);

        checker.execs({{2, 10}, {2, 10}});
        checker.execs({{2, 10, 4}, {2, 10, 4}});
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}});
    }
}