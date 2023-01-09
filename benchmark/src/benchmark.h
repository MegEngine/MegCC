/**
 * \file benchmark/src/benchmark.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include <vector>
namespace megcc {
namespace Benchmark {
/**
 * Benchmarker interface
 *
 */
class Benchmarker {
public:
    virtual void load_model() = 0;
    virtual void profile() = 0;
    virtual ~Benchmarker() = default;
};
}  // namespace Benchmark

}  // namespace megcc
