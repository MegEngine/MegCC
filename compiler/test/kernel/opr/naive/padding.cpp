#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, Padding) {
    Checker<megdnn::Padding> checker;
    megdnn::Padding::Param param;
    using PaddingMode = megdnn::Padding::Param::PaddingMode;
    auto run = [&checker, &param]() {
        for (auto mode :
             {PaddingMode::CONSTANT, PaddingMode::REFLECT, PaddingMode::REPLICATE}) {
            for (int offset0 : {3, 5}) {
                for (int offset1 : {5, 7}) {
                    param.back_offset_dim0 = 0;
                    param.back_offset_dim1 = 0;
                    param.back_offset_dim2 = offset0;
                    param.back_offset_dim3 = offset1;

                    param.front_offset_dim0 = 0;
                    param.front_offset_dim1 = 0;
                    param.front_offset_dim2 = offset0;
                    param.front_offset_dim3 = offset1;

                    param.padding_mode = mode;
                    param.padding_val = 2.f;

                    checker.set_param(param);
                    checker.exec({{1, 1, 30, 30}, {}});
                    checker.exec({{1, 3, 30, 30}, {}});
                    checker.exec({{3, 3, 30, 30}, {}});
                }
            }
        }
    };
    UniformIntRNG seq(0, 255);
    checker.set_rng(0, &seq);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(1, dtype::Uint8());
    run();

    megcc::test::UniformRNG rng(-30, 30);
    checker.set_rng(0, &rng);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    run();

#if ENABLE_KERNEL_FP16
    megcc::test::Float16PeriodicalRNG rng1;
    checker.set_rng(0, &rng1);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());
    run();
#endif
}