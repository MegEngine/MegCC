#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;

TEST(NAIVE, Argsort) {
    Checker<ArgsortForward> checker;
    megcc::test::SequenceRNG rng;
    checker.set_rng(0, &rng);
    checker.set_kernel_symbol("kernel_.*");
    checker.set_dynamic_megcc(true);
    ArgsortForward::Param param;
#if ENABLE_KERNEL_FP16
    for (auto type : {(DType)dtype::Float16(), (DType)dtype::Float32()})
#else
    for (auto type : {(DType)dtype::Float32()})
#endif
    {
        checker.set_dtype(0, type);
        checker.set_dtype(1, type);
        checker.set_dtype(2, dtype::Int32());
        for (auto order :
             {ArgsortForward::Param::Order::ASCENDING,
              ArgsortForward::Param::Order::DESCENDING})
            for (size_t batch_size : {1, 3, 4})
                for (size_t vec_len = 1; vec_len < 77; vec_len++) {
                    param.order = order;
                    checker.set_param(param);
                    checker.execs({{batch_size, vec_len}, {}, {}});
                }
    }
}