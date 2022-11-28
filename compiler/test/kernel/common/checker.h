/**
 * \file
 * compiler/test/kernel/common/checker.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <functional>
#include "test/kernel/common/dnn_proxy.h"
#include "test/kernel/common/runner.h"

#include "dnn_algo_checker.h"
namespace megcc {
namespace test {

using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;
using TensorLayoutArray = megdnn::SmallVector<megdnn::TensorLayout>;
using TensorShapeArray = megdnn::SmallVector<megdnn::TensorShape>;

template <typename Opr>
class Checker : public Runner<Opr> {
public:
    using DnnProxy = megdnn::test::DnnOprProxy<Opr>;
    using Param = typename Opr::Param;
    using BeforeExecCallback = std::function<void(Opr*, const TensorNDArray&)>;
    using OutputCanonizer = std::function<void(const TensorNDArray&)>;
    Checker(KernelGen::Arch arch = KernelGen::Arch::BAREMETAL)
            : Runner<Opr>(arch), m_arch(arch), m_kernel_symbol(".*") {}

    /*!
     * \brief execute opr on current param/dtype/rng config
     * \param shapes input/output shapes, which would be passed as
     *      arguments to Opr::deduce_layout
     *
     * Checker would construct TensorLayout vectors from shapes and dtypes,
     * and call exec(TensorLayoutArray &).
     */
    void exec(const TensorShapeArray& shapes) {
        exec(Runner<Opr>::make_layouts(shapes, m_dtype, m_fmt));
    }

    void exec(TensorLayoutArray layouts);

    //! explicitly require argument to be TensorShape
    void execs(const TensorShapeArray& shapes) {
        if (m_run_cc_only) {
            mgb_assert(shapes.back().ndim != 0,
                       "only run megcc test should give all shapes");
        }
        exec(shapes);
    }

    //! explicitly require argument to be TensorLayout
    void execl(const TensorLayoutArray& layouts) {
        if (m_run_cc_only) {
            mgb_assert(layouts.back().ndim != 0,
                       "only run megcc test should give all shapes");
        }
        exec(layouts);
    }

    Checker& set_param(Param param) {
        m_param = param;
        return *this;
    }
    Checker& set_dtype(size_t idx, megdnn::DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }

    Checker& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }
    Checker& reset_rng() {
        m_rng.clear();
        return *this;
    }
    Checker& set_fmt(size_t idx, megdnn::TensorFormat fmt) {
        m_fmt[idx] = fmt;
        return *this;
    }

    //! max error of a single element
    Checker& set_epsilon(float epsilon) {
        m_epsilon = epsilon;
        m_max_avg_error = epsilon;
        m_max_avg_biased_error = epsilon;
        return *this;
    }
    //! max average error; defaults to epsilon
    Checker& set_max_avg_error(float error) {
        m_max_avg_error = error;
        return *this;
    }
    //! max average biased error; defaults to epsilon
    Checker& set_max_avg_biased_error(float error) {
        m_max_avg_biased_error = error;
        return *this;
    }

    // select the specify algo
    Checker& set_kernel_symbol(std::string kernel_symbol) {
        m_kernel_symbol = kernel_symbol;
        return *this;
    }
    Checker& set_before_exec_callback(const BeforeExecCallback& cb) {
        m_before_exec_callback = cb;
        return *this;
    }

    /*!
     * \brief set a function to canonize the outputs
     *
     * For some oprs maybe multiple outputs can be accepted; we can use a
     * function to transform them into a canonized form before comparing.
     *
     * The arguments are tensors on CPU and should be modified in-place.
     */
    Checker& set_output_canonizer(OutputCanonizer canonizer) {
        m_output_canonizer = std::move(canonizer);
        return *this;
    }

    Checker& set_proxy(const DnnProxy& proxy) {
        m_dnn_proxy = proxy;
        return *this;
    }

    Checker& set_only_megcc(bool flag) {
        m_run_cc_only = flag;
        return *this;
    }

    Checker& set_dynamic_megcc(bool flag) {
        m_run_cc_dynamic = flag;
        return *this;
    }

    void check_tensors(const TensorNDArray& expected,
                       const TensorNDArray& computed, float epsilon,
                       float max_avg_error, float max_avg_biased_error);

private:
    bool m_run_cc_only = false;
    bool m_run_cc_dynamic = false;
    Param m_param;
    float m_epsilon{1e-5};
    float m_max_avg_error{1e-5};
    float m_max_avg_biased_error{1e-5};
    std::unique_ptr<Opr> m_dnn_opr;
    std::unordered_map<size_t, RNG*> m_rng;
    std::unordered_map<size_t, megdnn::DType> m_dtype;
    std::unordered_map<size_t, megdnn::TensorFormat> m_fmt;
    OutputCanonizer m_output_canonizer;
    KernelGen::Arch m_arch;
    std::string m_kernel_symbol;
    DnnProxy m_dnn_proxy;
    BeforeExecCallback m_before_exec_callback = nullptr;
};

}  // namespace test
}  // namespace megcc
