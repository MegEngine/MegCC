#include "test/kernel/common/checker.h"
#include "test/kernel/opr/common/elemwise.h"

using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
using MODE = ElemwiseForward::Param::Mode;

TEST(AUTONAIVE, Matmul) {
    Checker<MatrixMulForward> checker(Arch::AUTO_BAREMETAL);
    MatrixMulForward::Param param;
    for (bool trans_a : {false})
        for (bool trans_b : {false})
            for (size_t m : {17, 21})
                for (size_t n : {23, 13})
                    for (size_t k : {33, 17}) {
                        size_t a0 = m;
                        size_t a1 = k;
                        size_t b0 = k;
                        size_t b1 = n;
                        if (trans_a) {
                            a0 = k, a1 = m;
                        }
                        if (trans_b) {
                            b0 = n, b1 = k;
                        }
                        param.transposeA = trans_a;
                        param.transposeB = trans_b;
                        checker.set_param(param);
                        checker.execs({{a0, a1}, {b0, b1}, {}});
                    }
}

// vim: syntax=cpp.doxygen
