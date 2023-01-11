/**
 * \file benchmark/src/CCbenchmark.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include <vector>
#include "benchmark.h"
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
namespace megcc {
namespace Benchmark {
class CCBenchmarker final : public Benchmarker {
public:
    CCBenchmarker(std::string model, int log_level)
            : m_model_path(model), m_log_level(log_level) {
        LITE_set_log_level(static_cast<LiteLogLevel>(log_level));
    };
    virtual void load_model() override;
    virtual void profile() override;
    ~CCBenchmarker();

private:
    int m_log_level;
    std::string m_model_path;
    LiteNetwork m_model;
};
}  // namespace Benchmark

}  // namespace megcc
