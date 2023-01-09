/**
 * \file benchmark/src/MGEbenchmark.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include "build_config.h"
#if ENABLE_MEGENGINE_FRAMEWORK
#include <string>
#include <vector>
#include "benchmark.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/serialization/serializer.h"
namespace megcc {
namespace Benchmark {
class MGEBenchmarker final : public Benchmarker {
public:
    MGEBenchmarker(std::string model, int log_level)
            : m_model_path(model), m_log_level(log_level) {
        m_load_config.comp_graph = mgb::ComputingGraph::make();
    };
    virtual void load_model() override;
    virtual void profile() override;

private:
    int m_log_level;
    std::string m_model_path;
    mgb::serialization::GraphLoadConfig m_load_config;
    mgb::serialization::GraphLoader::LoadResult m_model;
    std::unique_ptr<mgb::cg::AsyncExecutable> m_func;
    std::unique_ptr<mgb::GraphProfiler> m_profiler;
    mgb::cg::ComputingGraph::OutputSpec m_output_spec;
};
}  // namespace Benchmark
}  // namespace megcc
#endif