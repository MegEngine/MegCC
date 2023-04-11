#pragma once
#include <cstddef>
#include "compiler/KernelGen/KernelGen.h"
#include "megbrain/common.h"
#include "megdnn/oprs.h"
#include "test/kernel/common/performance.h"
using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;

namespace megcc {
namespace test {

template <typename Opr>
struct WorkloadOprProxy {
    static size_t get_compute_workload(Opr* opr, const TensorNDArray& tensors);
    static KernelWorkload get_workload(Opr* opr, const TensorNDArray& tensors) {
        KernelWorkload res;
        res.compute_workload_go = (float)get_compute_workload(opr, tensors) / 1e9;
        res.memory_workload_mb = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            res.memory_workload_mb +=
                    tensors[i].layout.total_nr_elems() * tensors[i].layout.dtype.size();
        }
        res.memory_workload_mb = res.memory_workload_mb / 1e6;
        return res;
    }
};

}  // namespace test
}  // namespace megcc