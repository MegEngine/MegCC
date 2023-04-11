#pragma once
#include <stdio.h>
namespace megcc {
namespace test {
struct PerformanceResult {
    bool valid{false};

    float kernel_time_ms{0.f};
    //! G Operations Per Seconed
    float compute_throughput_gops{0.f};
    //! Million Bytes Per Seconed
    float memory_throughput_mbps{0.f};
};
struct KernelWorkload {
    //! G Operations
    float compute_workload_go{0.f};
    //! Million Bytes
    float memory_workload_mb{0.f};
};
struct PerformanceResultPair {
    std::string args;
    PerformanceResult megcc_performance;
    PerformanceResult dnn_performance;
    void print() {
        float speed_up = 0;
        if (megcc_performance.compute_throughput_gops > 0 &&
            dnn_performance.compute_throughput_gops > 0) {
            speed_up = megcc_performance.compute_throughput_gops /
                       dnn_performance.compute_throughput_gops;
        }
        printf("megcc result time = %f ms, throughput %f Gops, %f mbps, speedup "
               "%f\n",
               megcc_performance.kernel_time_ms,
               megcc_performance.compute_throughput_gops,
               megcc_performance.memory_throughput_mbps, speed_up);
        printf("dnn result time = %f ms, throughput %f Gops, %f mbps\n",
               dnn_performance.kernel_time_ms, dnn_performance.compute_throughput_gops,
               dnn_performance.memory_throughput_mbps);
    }
};
struct BenchmarkOption {
    bool valid_megcc_performance{false};
    bool valid_dnn_performance{false};
    bool disable_check{false};
    int warmup_iter{50};
    int test_iter{100};
};
}  // namespace test
}  // namespace megcc