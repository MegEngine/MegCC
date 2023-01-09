/**
 * \file benchmark/main.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <cmath>
#include <cstdio>
#include <memory>
#include "src/CCbenchmark.h"
#include "src/MGEbenchmark.h"

using namespace megcc;
using namespace Benchmark;
int main(int argc, char** argv) {
    if (argc < 2 && argc > 4) {
        fprintf(stderr, "cmdline error, please run with:\n");
        fprintf(stderr, "benchmarker <input_model> [options] ... \n");
        fprintf(stderr,
                "tips:\n\t you can use --profile and --mge to profile model "
                "and enable megengine framework (\"megcc\" is default)\n");
        return -1;
    }
    int log_level = 3;
    std::string framework = "megcc";
    std::string model_path = argv[1];
    int idx = 2;
    while (idx < argc) {
        std::string args = argv[idx];
        if (args == "--profile") {
            log_level = 0;
        } else if (args == "--mge") {
            framework = "mge";
        } else {
            fprintf(stderr, "invalid option: %s\n", argv[idx]);
        }
        ++idx;
    }
    std::vector<std::shared_ptr<Benchmarker>> benchmarkers;
    if (framework == "megcc") {
        benchmarkers.push_back(
                std::make_shared<CCBenchmarker>(model_path, log_level));
    }
#if ENABLE_MEGENGINE_FRAMEWORK
    else if (framework == "mge") {
        benchmarkers.push_back(
                std::make_shared<MGEBenchmarker>(model_path, log_level));
    }
#endif
    else {
        fprintf(stderr,
                "unsupport framework: %s, megcc, mge(export "
                "ENABLE_MEGENGINE_FRAMEWORK=ON) is supported\n",
                framework.c_str());
    }

    for (size_t i = 0; i < benchmarkers.size(); ++i) {
        benchmarkers[i]->load_model();
        benchmarkers[i]->profile();
    }

    return 0;
}