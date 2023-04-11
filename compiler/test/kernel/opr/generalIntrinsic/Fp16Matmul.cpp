#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
#ifdef ENABLE_KERNEL_FP16
TEST(GI, Fp16MatMulM8N8K8) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    checker.set_kernel_symbol("GI_kernel_fp16_matmul_8x8mk8_.*");
    MatrixMulForward::Param param;
    megcc::test::UniformRNG rng(-1.0, 1.0);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16());

    for (size_t m : {8, 16, 64})
        for (size_t n : {3, 8, 15, 56})
            for (size_t k : {8, 16, 64}) {
                param.transposeA = false;
                param.transposeB = false;
                param.format = param::MatrixMul::Format::MK8;
                checker.set_param(param);
                checker.execs({{m / 8, k / 8, 8, 8}, {k / 8, n, 8}, {}});
            }
}
TEST(GI, FP16GEVM) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_kernel_symbol("GI_kernel_gevm_fp16.*");
    checker.set_epsilon(8e-3);
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16());
    for (bool trans_a : {false})
        for (bool trans_b : {false, true})
            for (size_t m : {1})
                for (size_t n : {3, 8, 11, 15, 33, 56})
                    for (size_t k : {3, 8, 11, 15, 33, 14}) {
                        size_t a0 = m;
                        size_t a1 = k;
                        size_t b0 = k;
                        size_t b1 = n;
                        if (trans_b) {
                            b0 = n, b1 = k;
                        }
                        param.transposeA = trans_a;
                        param.transposeB = trans_b;
                        checker.set_param(param);
                        checker.execs({{a0, a1}, {b0, b1}, {}});
                    }
}

TEST(GI, FP16GEMV) {
    Checker<MatrixMulForward> checker(Arch::BAREMETAL);
    MatrixMulForward::Param param;
    checker.set_kernel_symbol("GI_kernel_gemv_fp16_nn.*");
    checker.set_epsilon(8e-3);
    megcc::test::Float16PeriodicalRNG rng(0x3c00);
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16());
    for (size_t m : {1, 2, 3, 4})
        for (size_t n : {1})
            for (size_t k : {3, 8, 11, 15, 33, 14}) {
                size_t a0 = m;
                size_t a1 = k;
                size_t b0 = k;
                size_t b1 = n;
                param.transposeA = false;
                param.transposeB = false;
                checker.set_param(param);
                checker.execs({{a0, a1}, {b0, b1}, {}});
            }
}

#endif
// vim: syntax=cpp.doxygen
