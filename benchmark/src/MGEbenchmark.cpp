/**
 * \file benchmark/src/MGEbenchmark.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "MGEbenchmark.h"
#if ENABLE_MEGENGINE_FRAMEWORK
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <map>
#include "megbrain/gopt/inference.h"
using namespace mgb;
using namespace megcc;
using namespace Benchmark;
const int number = 50;
const int warmup = 10;

void MGEBenchmarker::load_model() {
    std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_fs(m_model_path.c_str());
    auto format =
            serialization::GraphLoader::identify_graph_dump_format(*inp_file);
    mgb_assert(format.valid(), "invalid model: unknown model format");
    auto loader =
            serialization::GraphLoader::make(std::move(inp_file), format.val());
    if (m_log_level == 0) {
        m_profiler = std::move(std::make_unique<mgb::GraphProfiler>(
                m_load_config.comp_graph.get()));
    } else {
        m_load_config.comp_graph->options().comp_node_seq_record_level = 1;
    }
    m_load_config.comp_graph->options().var_sanity_check_first_run = false;
    m_load_config.comp_graph->options()
            .graph_opt.enable_fuse_conv_bias_nonlinearity();
    m_load_config.comp_graph->options().graph_opt.enable_weight_preprocess();

    m_model = loader->load(m_load_config, false);
}

void MGEBenchmarker::profile() {
    //! optimize for inference
    auto& output_vars = m_model.output_var_list;

    using Strategy =
            mgb::opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    auto strategy = static_cast<Strategy>(0);
    strategy = Strategy::PROFILE | Strategy::OPTIMIZED | strategy;
    mgb::gopt::modify_opr_algo_strategy_inplace(output_vars, strategy);
    mgb::gopt::OptimizeForInferenceOptions opt_for_inference;
#ifdef __ANDROID__
#if __ARM_FEATURE_DOTPROD
    opt_for_inference.enable_nchw44_dot();
#else
    opt_for_inference.enable_nchw44();
#endif
#else
    output_vars = mgb::gopt::layout_transform(
            output_vars, mgb::gopt::GraphTuningOptions::Target::CPU);
#endif
    size_t nr_output = output_vars.size();

    output_vars =
            mgb::gopt::optimize_for_inference(output_vars, opt_for_inference);
    std::vector<std::map<std::string, megdnn::TensorShape>> input_map_vec;
    auto cg = m_model.output_var_list[0].node()->owner_graph();
    for (auto&& i : output_vars) {
        mgb::ComputingGraph::Callback cb;
        m_output_spec.emplace_back(i, std::move(cb));
    }
    m_func = cg->compile(m_output_spec);
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < warmup; ++i) {
        m_func->execute().wait();
    }
    gettimeofday(&end, NULL);
    unsigned long diff =
            1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

    gettimeofday(&start, NULL);
    for (int i = 0; i < number; ++i) {
        m_func->execute().wait();
    }
    gettimeofday(&end, NULL);
    diff = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    float average_time = ((float)diff) / number / 1000;
    if (m_log_level == 0) {
        std::string profile_ret;
        m_profiler->to_json_full(m_func.get())->writeto(profile_ret, 4);
        printf("%s\n", profile_ret.c_str());
    } else {
        printf("the inference average time=%.3f ms\n", average_time);
    }
}
#endif