#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, GaussianBlur) {
    Checker<megdnn::GaussianBlur> checker;
    megdnn::GaussianBlur::Param param;
    using BorderMode = megdnn::GaussianBlur::Param::BorderMode;
    auto run = [&checker, &param]() {
        for (auto mode :
             {BorderMode::CONSTANT, BorderMode::REFLECT, BorderMode::REFLECT_101,
              BorderMode::REPLICATE}) {
            for (int kh = 3; kh <= 9; kh += 2) {
                for (int kw = 3; kw <= 9; kw += 2) {
                    for (double sigma1 : {0., 0.8}) {
                        for (double sigma2 : {0., 0.5}) {
                            param.border_mode = mode;
                            param.kernel_height = kh;
                            param.kernel_width = kw;
                            param.sigma_x = sigma1;
                            param.sigma_y = sigma2;

                            checker.set_param(param);
                            checker.exec({{1, 3, 5, 1}, {}});
                            checker.exec({{1, 3, 5, 3}, {}});
                            checker.exec({{3, 16, 16, 1}, {}});
                            checker.exec({{1, 16, 16, 3}, {}});
                            checker.exec({{1, 16, 19, 1}, {}});
                            checker.exec({{1, 16, 19, 3}, {}});
                            checker.exec({{4, 19, 19, 1}, {}});
                            checker.exec({{1, 19, 19, 3}, {}});
                        }
                    }
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
    checker.set_epsilon(1e-4);
    run();
}