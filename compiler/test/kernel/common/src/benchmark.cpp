#include "test/kernel/common/benchmark.h"
#include "megcc_test_config.h"
#include "test/kernel/common/cc_proxy.h"
#include "test/kernel/common/timer.h"
#include "test/kernel/common/workload_proxy.h"
using namespace megdnn;
using namespace megdnn::test;
using TensorNDArray = SmallVector<TensorND>;
using TensorLayoutArray = SmallVector<TensorLayout>;
using namespace megcc::test;

static inline void fill_performance_result(
        PerformanceResult& result, const KernelWorkload& workload) {
    if (result.valid) {
        result.compute_throughput_gops =
                workload.compute_workload_go / result.kernel_time_ms * 1e3;
        result.memory_throughput_mbps =
                workload.memory_workload_mb / result.kernel_time_ms * 1e3;
    }
}
namespace {
template <typename Opr>
void fix_addition_attr_map(
        std::unordered_map<std::string, megcc::CCAttr>& proxy_attr,
        megdnn::test::DnnOprProxy<Opr>& dnn_proxy, TensorNDArray& tensor_array) {}

template <>
void fix_addition_attr_map<megdnn::TopK>(
        std::unordered_map<std::string, megcc::CCAttr>& proxy_attr,
        megdnn::test::DnnOprProxy<megdnn::TopK>& dnn_proxy,
        TensorNDArray& tensor_array) {
    proxy_attr["k"] = megcc::CCAttr(dnn_proxy.get_k());
}
}  // namespace

template <typename Opr>
PerformanceResultPair Benchmarker<Opr>::exec(TensorLayoutArray all_layouts) {
    using CCProxy = CCOprProxy<Opr>;
    auto dnn_handle = Runner<Opr>::get_dnn_handle();
    auto opr = dnn_handle->template create_operator<Opr>();
    opr->param() = m_param;
    m_dnn_proxy.deduce_layout(opr.get(), all_layouts);

    auto tensor_array_storage = dnn_alloc_tensors(dnn_handle, all_layouts, 0);
    auto tensor_array_naive_storage = dnn_alloc_tensors(dnn_handle, all_layouts, 0);
    auto tensor_array_dnn = *tensor_array_naive_storage;
    auto tensor_array = *tensor_array_storage;
    Runner<Opr>::init_tensor(tensor_array_dnn, m_rng);
    dnn_copy_tensors(tensor_array, tensor_array_dnn);
    PerformanceResultPair res;
    std::stringstream ss;
    for (auto& tensor : tensor_array) {
        ss << tensor.layout.to_string();
    }
    res.args = ss.str();
    //! test mode
    CCProxy cc_proxy;
    std::unordered_map<std::string, CCAttr> proxy_attr;
    fix_addition_attr_map<Opr>(proxy_attr, m_dnn_proxy, tensor_array_dnn);
    auto megcc_perf = cc_proxy.exec(
            opr.get(), tensor_array, m_arch, m_benchmark_option, m_kernel_symbol,
            proxy_attr, false);
    if (m_benchmark_option.valid_megcc_performance) {
        auto workload = WorkloadOprProxy<Opr>::get_workload(opr.get(), tensor_array);
        fill_performance_result(megcc_perf, workload);
        res.megcc_performance = megcc_perf;
    }
#if !MEGCC_TEST_GEN
    if (m_benchmark_option.disable_check && !m_benchmark_option.valid_dnn_performance) {
        //! fast return;
        return res;
    }
    //! run dnn
    if (m_before_exec_callback) {
        m_before_exec_callback(opr.get(), tensor_array_dnn);
    }
    m_dnn_proxy.exec(opr.get(), tensor_array_dnn);
    if (m_benchmark_option.valid_dnn_performance) {
        for (int i = 0; i < m_benchmark_option.warmup_iter; ++i) {
            m_dnn_proxy.exec(opr.get(), tensor_array_dnn);
        }
        mgb_assert(m_benchmark_option.test_iter > 0);
        Timer timer;
        timer.start();
        for (int i = 0; i < m_benchmark_option.test_iter; ++i) {
            m_dnn_proxy.exec(opr.get(), tensor_array_dnn);
        }
        timer.stop();
        PerformanceResult dnn_perf;
        dnn_perf.valid = true;
        dnn_perf.kernel_time_ms =
                timer.get_time_in_us() / 1e3 / m_benchmark_option.test_iter;
        auto workload = WorkloadOprProxy<Opr>::get_workload(opr.get(), tensor_array);
        fill_performance_result(dnn_perf, workload);
        res.dnn_performance = dnn_perf;
    }

#endif
    return res;
}

#if !MEGCC_TEST_GEN
#define CV_BENCHMARK_BODY()                                                          \
    {                                                                                \
        if (m_benchmark_option.disable_check &&                                      \
            !m_benchmark_option.valid_dnn_performance) {                             \
            return res;                                                              \
        }                                                                            \
        dnn_proxy.exec(opr.get(), tensor_array_dnn);                                 \
        if (m_benchmark_option.valid_dnn_performance) {                              \
            for (int i = 0; i < m_benchmark_option.warmup_iter; ++i) {               \
                dnn_proxy.exec(opr.get(), tensor_array_dnn);                         \
            }                                                                        \
            mgb_assert(m_benchmark_option.test_iter > 0);                            \
            Timer timer;                                                             \
            timer.start();                                                           \
            for (int i = 0; i < m_benchmark_option.test_iter; ++i) {                 \
                dnn_proxy.exec(opr.get(), tensor_array_dnn);                         \
            }                                                                        \
            timer.stop();                                                            \
            PerformanceResult dnn_perf;                                              \
            dnn_perf.valid = true;                                                   \
            dnn_perf.kernel_time_ms =                                                \
                    timer.get_time_in_us() / 1e3 / m_benchmark_option.test_iter;     \
            auto workload =                                                          \
                    WorkloadOprProxy<DnnOpr>::get_workload(opr.get(), tensor_array); \
            fill_performance_result(dnn_perf, workload);                             \
            res.dnn_performance = dnn_perf;                                          \
        }                                                                            \
    }
#else
#define CV_BENCHMARK_BODY() \
    {}
#endif

#define DEF_CV_BENCHMARK(_Opr)                                                       \
    template <>                                                                      \
    PerformanceResultPair Benchmarker<_Opr>::exec(TensorLayoutArray all_layouts) {   \
        using CvOpr = _Opr;                                                          \
        using DnnOpr = CvOpr::DnnOpr;                                                \
        using DnnProxy = DnnOprProxy<DnnOpr>;                                        \
        using CCProxy = CCOprProxy<CvOpr>;                                           \
        Runner<DnnOpr> runner(m_arch, 0);                                            \
        auto dnn_handle = runner.get_dnn_handle();                                   \
        auto opr = dnn_handle->template create_operator<DnnOpr>();                   \
        CvOpr cv_opr;                                                                \
        cv_opr.param() = m_param;                                                    \
        cv_opr.reformat_layout(&cv_opr, all_layouts);                                \
        opr->param() = cv_opr.dnn_param(m_param);                                    \
        DnnProxy dnn_proxy;                                                          \
        dnn_proxy.deduce_layout(opr.get(), all_layouts);                             \
                                                                                     \
        auto tensor_array_storage = dnn_alloc_tensors(dnn_handle, all_layouts, 0);   \
        auto tensor_array_naive_storage =                                            \
                dnn_alloc_tensors(dnn_handle, all_layouts, 0);                       \
        auto tensor_array_dnn = *tensor_array_naive_storage;                         \
        auto tensor_array = *tensor_array_storage;                                   \
        runner.init_tensor(tensor_array_dnn, m_rng);                                 \
        dnn_copy_tensors(tensor_array, tensor_array_dnn);                            \
        PerformanceResultPair res;                                                   \
        std::stringstream ss;                                                        \
        for (auto& tensor : tensor_array) {                                          \
            ss << tensor.layout.to_string();                                         \
        }                                                                            \
        res.args = ss.str();                                                         \
        CCProxy cc_proxy;                                                            \
        auto megcc_perf = cc_proxy.exec(                                             \
                &cv_opr, tensor_array, m_arch, m_benchmark_option, m_kernel_symbol,  \
                {}, false);                                                          \
        if (m_benchmark_option.valid_megcc_performance) {                            \
            auto workload =                                                          \
                    WorkloadOprProxy<DnnOpr>::get_workload(opr.get(), tensor_array); \
            fill_performance_result(megcc_perf, workload);                           \
            res.megcc_performance = megcc_perf;                                      \
        }                                                                            \
        CV_BENCHMARK_BODY();                                                         \
        return res;                                                                  \
    }

template <typename Opr>
std::string Benchmarker<Opr>::format_result(const PerformanceResultPair& perf_result) {
    std::stringstream ss;
    char buffer[200];

    ss << "case: " << perf_result.args << "\n";
    snprintf(
            buffer, sizeof(buffer),
            "megcc result time = %f ms, throughput %f Gops, %f mbps\n",
            perf_result.megcc_performance.kernel_time_ms,
            perf_result.megcc_performance.compute_throughput_gops,
            perf_result.megcc_performance.memory_throughput_mbps);
    ss << std::string(buffer);
    snprintf(
            buffer, sizeof(buffer),
            "dnn result time = %f ms, throughput %f Gops, %f mbps, speedup %f\n",
            perf_result.dnn_performance.kernel_time_ms,
            perf_result.dnn_performance.compute_throughput_gops,
            perf_result.dnn_performance.memory_throughput_mbps,
            perf_result.megcc_performance.compute_throughput_gops /
                    perf_result.dnn_performance.compute_throughput_gops);
    ss << std::string(buffer);
    return ss.str();
}
template class megcc::test::Benchmarker<megdnn::ElemwiseForward>;
template class megcc::test::Benchmarker<megdnn::ConvBiasForward>;
template class megcc::test::Benchmarker<megdnn::ConvolutionForward>;
template class megcc::test::Benchmarker<megdnn::ConvolutionBackwardData>;
template class megcc::test::Benchmarker<megdnn::PoolingForward>;
template class megcc::test::Benchmarker<megdnn::MatrixMulForward>;
template class megcc::test::Benchmarker<megdnn::TypeCvtForward>;
template class megcc::test::Benchmarker<megdnn::RelayoutForward>;
template class megcc::test::Benchmarker<megdnn::WarpAffine>;
template class megcc::test::Benchmarker<megdnn::ResizeForward>;
template class megcc::test::Benchmarker<megdnn::ReduceForward>;
template class megcc::test::Benchmarker<megdnn::TopK>;
namespace megcc {
namespace test {

DEF_CV_BENCHMARK(megdnn::CVResize);
DEF_CV_BENCHMARK(megdnn::CVCvtColor);
DEF_CV_BENCHMARK(megdnn::CVWarpAffine);
DEF_CV_BENCHMARK(megdnn::CVtranspose);

DEF_CV_BENCHMARK(megdnn::CVflip);
DEF_CV_BENCHMARK(megdnn::CVRotate);
DEF_CV_BENCHMARK(megdnn::CVGaussianBlur);
}  // namespace test
}  // namespace megcc
