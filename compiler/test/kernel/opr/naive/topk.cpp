#include "test/kernel/common/benchmark.h"
#include "test/kernel/common/checker.h"
using namespace megdnn;
using namespace megcc::test;
using namespace megcc::KernelGen;
namespace {
template <typename T>
Checker<TopK>::OutputCanonizer OutputCanonizer(TopK::Param::Mode mode) {
    auto output_canonizer = [=](const TensorNDArray& arr) {
        if (mode == TopK::Param::Mode::KTH_ONLY) {
            return;
        }
        auto pval = arr[1].ptr<T>();
        auto pidx = arr.at(2).ptr<int>();
        size_t m = arr[1].layout[0], n = arr[1].layout[1];
        using idx_val = std::pair<int, T>;
        std::vector<idx_val> data(n);
        auto compare = [](const idx_val& it1, const idx_val& it2) {
            if (it2.second == it1.second) {
                return it1.first > it2.first;
            }
            return (it1.second > it2.second);
        };
        for (size_t i = 0; i < m; ++i) {
            if (mode == TopK::Param::Mode::VALUE_IDX_NOSORT) {
                // sort output pairs to canonize
                for (size_t j = 0; j < n; ++j) {
                    data[j].first = pidx[i * n + j];
                    data[j].second = pval[i * n + j];
                }
                std::sort(data.begin(), data.end(), compare);
                for (size_t j = 0; j < n; ++j) {
                    pidx[i * n + j] = data[j].first;
                    pval[i * n + j] = data[j].second;
                }
            } else if (mode == TopK::Param::Mode::VALUE_IDX_SORTED) {
                for (size_t j = 1; j < n; ++j) {
                    if (pval[i * n + j - 1] == pval[i * n + j] &&
                        pidx[i * n + j - 1] > pidx[i * n + j])
                        std::swap(pidx[i * n + j - 1], pidx[i * n + j]);
                }
            }
        }
    };
    return output_canonizer;
}
}  // namespace

TEST(NAIVE, Topk) {
    Checker<TopK> checker;
    checker.set_kernel_symbol("kernel_.*");

    TopK::Param param;
    using Mode = TopK::Param::Mode;
    for (int k : {-5, 7})
        for (auto mode : {
                     Mode::KTH_ONLY,
                     Mode::VALUE_IDX_NOSORT,
                     Mode::VALUE_IDX_SORTED,
             }) {
            checker.set_proxy(k);
            auto run = [&]() {
                for (size_t batch_size : {1, 3})
                    for (size_t vec_len = std::abs(k); vec_len < 27; vec_len++) {
                        param.mode = mode;
                        checker.set_param(param);
                        if (mode == Mode::KTH_ONLY) {
                            checker.execs({{batch_size, vec_len}, {}});
                        } else {
                            checker.execs({{batch_size, vec_len}, {}, {}});
                        }
                    }
            };
#if ENABLE_KERNEL_FP16
            checker.set_dtype(0, dtype::Float16());
            checker.set_dtype(1, dtype::Float16());
            checker.set_output_canonizer(OutputCanonizer<dt_float16>(mode));
            run();
#endif
            checker.set_dtype(0, dtype::Float32());
            checker.set_dtype(1, dtype::Float32());
            checker.set_output_canonizer(OutputCanonizer<float>(mode));
            run();
        }
}

#ifdef ENABLE_KERNEL_BENCHMARK

TEST(NAIVE, BenchmarkTopK) {
    Benchmarker<TopK> benchmarker(Arch::BAREMETAL);
    benchmarker.set_kernel_symbol("kernel_.*");
    TopK::Param param;
    using Mode = TopK::Param::Mode;
    for (int k : {-50, 70})
        for (auto mode : {
                     Mode::KTH_ONLY,
                     Mode::VALUE_IDX_NOSORT,
                     Mode::VALUE_IDX_SORTED,
             }) {
            benchmarker.set_proxy(k);
            for (size_t batch_size : {1})
                for (size_t vec_len : {100000}) {
                    param.mode = mode;
                    benchmarker.set_param(param);
                    if (mode == Mode::KTH_ONLY) {
                        benchmarker.execs({{batch_size, vec_len}, {}}).print();
                    } else {
                        benchmarker.execs({{batch_size, vec_len}, {}, {}}).print();
                    }
                }
        }
}
#endif