/**
 * \file
 * compiler/test/kernel/common/runner.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <unordered_map>
#include "compiler/KernelGen/KernelGen.h"
#include "gtest/gtest.h"
#include "megbrain/common.h"
#include "megcore.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/linalg.h"
#include "megdnn/oprs/nn.h"
#include "megdnn/tensor_iter.h"
#include "megdnn/thin/small_vector.h"
#include "test/kernel/common/dnn_helper.h"
#include "test/kernel/common/rng.h"

namespace megcc {
namespace test {

using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;
using TensorLayoutArray = megdnn::SmallVector<megdnn::TensorLayout>;
using TensorShapeArray = megdnn::SmallVector<megdnn::TensorShape>;
template <typename Opr>
class Runner {
public:
    using Param = typename Opr::Param;
    Runner(KernelGen::Arch arch = KernelGen::Arch::BAREMETAL,
           const int dnn_level = 2)
            : m_arch(arch) {
        megcore_check(megcoreCreateDeviceHandle(&m_device_handle,
                                                megcorePlatformCPU));
        megcore_check(megcoreCreateComputingHandle(&m_compute_handle,
                                                   m_device_handle));
        m_dnn_handle =
                std::move(megdnn::Handle::make(m_compute_handle, dnn_level));
        m_default_rng = std::move(std::make_unique<NormalRNG>());
    }

    ~Runner() {
        megcore_check(megcoreDestroyComputingHandle(m_compute_handle));
        megcore_check(megcoreDestroyDeviceHandle(m_device_handle));
    }
    TensorLayoutArray make_layouts(
            const TensorShapeArray& shapes,
            const std::unordered_map<size_t, megdnn::DType>& dtype_map,
            const std::unordered_map<size_t, megdnn::TensorFormat>& fmt_map) {
        TensorLayoutArray layouts(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            auto dtype_iter = dtype_map.find(i);
            megdnn::DType dt =
                    (dtype_iter != dtype_map.end() ? dtype_iter->second
                                                   : megdnn::dtype::Float32());
            auto fmt_iter = fmt_map.find(i);
            megdnn::TensorFormat fmt =
                    (fmt_iter != fmt_map.end() ? fmt_iter->second
                                               : megdnn::TensorFormat{});
            layouts[i] = megdnn::TensorLayout(shapes[i], dt, fmt);
        }
        return layouts;
    }
    void init_tensor(TensorNDArray& tensor_array,
                     const std::unordered_map<size_t, RNG*>& rng_map) {
        for (size_t idx = 0; idx < tensor_array.size(); ++idx) {
            auto rng_iter = rng_map.find(idx);
            if (rng_iter != rng_map.end()) {
                rng_iter->second->gen(tensor_array[idx]);
                rng_iter->second->reset();
            } else {
                m_default_rng->gen(tensor_array[idx]);
                m_default_rng->reset();
            }
        }
    }
    megdnn::Handle* get_dnn_handle() { return m_dnn_handle.get(); }

protected:
    megcoreDeviceHandle_t m_device_handle;
    megcoreComputingHandle_t m_compute_handle;
    std::unique_ptr<megdnn::Handle> m_dnn_handle;
    std::unique_ptr<RNG> m_default_rng;
    KernelGen::Arch m_arch;
};
}  // namespace test
}  // namespace megcc