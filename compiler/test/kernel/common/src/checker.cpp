/**
 * \file
 * compiler/test/kernel/common/src/checker.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "test/kernel/common/checker.h"
#include "megcc_test_config.h"
#include "test/kernel/common/cc_proxy.h"
#include "test/kernel/common/dnn_proxy.h"
#include "test/kernel/common/timer.h"

using namespace megdnn;
using namespace megdnn::test;
using TensorNDArray = SmallVector<TensorND>;
using TensorLayoutArray = SmallVector<TensorLayout>;
using namespace megcc::test;
namespace {

class Index {
public:
    Index(TensorLayout layout, size_t linear);
    Index(TensorLayout layout, TensorShape array);

    std::string to_string() const;

    TensorShape array() const { return m_array; }
    TensorLayout layout() const { return m_layout; }
    size_t linear_index() const { return m_linear; }
    ptrdiff_t offset() const { return m_offset; }
    /**
     * Add a universal offset to all return values to make the minimal
     * offset zero.
     */
    size_t positive_offset() const {
        return m_offset - m_layout.span().low_elem;
    }

private:
    TensorLayout m_layout;
    size_t m_linear;
    TensorShape m_array;
    ptrdiff_t m_offset;

    void linear_to_array();
    void array_to_offset();
};

Index::Index(TensorLayout layout, size_t linear)
        : m_layout(layout), m_linear(linear) {
    linear_to_array();
    array_to_offset();
}

void Index::linear_to_array() {
    auto linear = m_linear;
    auto& array = m_array;
    array.ndim = m_layout.ndim;
    for (size_t j = m_layout.ndim; j > 0; --j) {
        size_t i = j - 1;
        array[i] = linear % m_layout[i];
        linear /= m_layout[i];
    }
    mgb_assert(linear == 0);
}

void Index::array_to_offset() {
    auto& offset = m_offset;
    mgb_assert(m_array.ndim == m_layout.ndim);
    offset = 0;
    for (size_t i = 0; i < m_array.ndim; ++i) {
        mgb_assert(m_array[i] < m_layout[i]);
        offset += m_array[i] * m_layout.stride[i];
    }
}

std::string Index::to_string() const {
    std::string res = "";
    res.append("{");
    res.append("array=");
    res.append(m_array.to_string());
    res.append(",linear=");
    res.append(std::to_string(m_linear));
    res.append(",offset=");
    res.append(std::to_string(m_offset));
    res.append("}");
    return res;
}
static inline float diff(float x, float y) {
    return x - y;
}
static inline int diff(int x, int y) {
    return x - y;
}

static inline int diff(dt_quint8 x, dt_quint8 y) {
    return x.as_uint8() - y.as_uint8();
}

static inline int diff(dt_qint32 x, dt_qint32 y) {
    return x.as_int32() - y.as_int32();
}

static inline int diff(dt_qint8 x, dt_qint8 y) {
    return x.as_int8() - y.as_int8();
}
static inline bool good_float(float val) {
    return std::isfinite(val);
}

static inline bool good_float(int) {
    return true;
}

static inline bool good_float(dt_qint8) {
    return true;
}

static inline bool good_float(dt_quint8) {
    return true;
}

static inline bool good_float(dt_qint32) {
    return true;
}
// A hack for the (x+0) promote to int trick on dt_quint8.
static inline int operator+(dt_quint8 lhs, int rhs) {
    mgb_assert(rhs == 0, "unexpected rhs");
    return lhs.as_uint8();
}

static inline int operator+(dt_qint32 lhs, int rhs) {
    mgb_assert(rhs == 0, "unexpected rhs");
    return lhs.as_int32();
}

static inline int operator+(dt_qint8 lhs, int rhs) {
    mgb_assert(rhs == 0, "unexpected rhs");
    return int8_t(lhs);
}

template <typename ctype, class Iter>
::testing::AssertionResult assert_tensor_eq_with_iter(
        const char* expr0, const char* expr1, Iter it0, Iter it1,
        const TensorLayout& layout, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    auto nr_elem = layout.total_nr_elems();
    double error_sum = 0;
    double error_sum_biased = 0;
    for (size_t i = 0; i < nr_elem; ++i) {
        ctype iv0 = *it0, iv1 = *it1;
        float err = diff(iv0, iv1);
        error_sum += std::abs(err);
        error_sum_biased += err;
        if (!good_float(iv0) || !good_float(iv1) || std::abs(err) > maxerr) {
            Index index(layout, i);
            return ::testing::AssertionFailure()
                   << "Unequal value\n"
                   << "Value of: " << expr1 << "\n"
                   << "  Actual: " << (iv1 + 0) << "\n"
                   << "Expected: " << expr0 << "\n"
                   << "Which is: " << (iv0 + 0) << "\n"
                   << "At index: " << index.to_string() << "/"
                   << layout.TensorShape::to_string() << "\n"
                   << "   DType: " << layout.dtype.name() << "\n"
                   << "   error: " << std::abs(err) << "/" << maxerr;
        }

        ++it0;
        ++it1;
    }

    float error_avg = error_sum / nr_elem;
    if (error_avg > maxerr_avg) {
        return ::testing::AssertionFailure()
               << "Average error exceeds the upper limit\n"
               << "Value of: " << expr1 << "\n"
               << "Expected: " << expr0 << "\n"
               << "Average error: " << error_avg << "/" << maxerr_avg << "\n"
               << "Num of elements: " << nr_elem;
    }

    float error_avg_biased = error_sum_biased / nr_elem;
    if (std::abs(error_avg_biased) > maxerr_avg_biased) {
        return ::testing::AssertionFailure()
               << "Average biased error exceeds the upper limit\n"
               << "Value of: " << expr1 << "\n"
               << "Expected: " << expr0 << "\n"
               << "Average biased error: " << error_avg_biased << "/"
               << maxerr_avg_biased << "\n"
               << "Num of elements: " << nr_elem;
    }

    return ::testing::AssertionSuccess();
}

template <typename ctype>
::testing::AssertionResult assert_tensor_eq_with_dtype(
        const char* expr0, const char* expr1, const TensorND& v0,
        const TensorND& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    if (v0.layout.is_physical_contiguous() &&
        v1.layout.is_physical_contiguous()) {
        return assert_tensor_eq_with_iter<ctype>(
                expr0, expr1, v0.ptr<ctype>(), v1.ptr<ctype>(), v0.layout,
                maxerr, maxerr_avg, maxerr_avg_biased);
    }

    auto it0 = megdnn::tensor_iter_valonly<ctype>(v0).begin(),
         it1 = megdnn::tensor_iter_valonly<ctype>(v1).begin();

    return assert_tensor_eq_with_iter<ctype>(expr0, expr1, it0, it1, v0.layout,
                                             maxerr, maxerr_avg,
                                             maxerr_avg_biased);
}

::testing::AssertionResult assert_tensor_eq(
        const char* expr0, const char* expr1, const char* /*expr_maxerr*/,
        const char* /*expr_maxerr_avg*/, const char* /*expr_maxerr_avg*/,
        const TensorND& v0, const TensorND& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    if (!v0.layout.eq_shape(v1.layout)) {
        return ::testing::AssertionFailure()
               << "Shape mismatch\n"
               << "Value of: " << expr1 << "\n"
               << "  Actual: " << v1.layout.TensorShape::to_string() << "\n"
               << "Expected: " << expr0 << "\n"
               << "Which is: " << v0.layout.TensorShape::to_string() << "\n";
    }
    auto dtype = v0.layout.dtype;
    if (dtype != v1.layout.dtype) {
        return ::testing::AssertionFailure()
               << "Data type mismatch\n"
               << "Value of: " << expr1 << "\n"
               << "  Actual: " << v1.layout.dtype.name() << "\n"
               << "Expected: " << expr0 << "\n"
               << "Which is: " << v0.layout.dtype.name() << "\n";
    }

    switch (dtype.enumv()) {
#define cb(_dt)                                                     \
    case DTypeTrait<_dt>::enumv:                                    \
        return assert_tensor_eq_with_dtype<DTypeTrait<_dt>::ctype>( \
                expr0, expr1, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
        default:
            megdnn_trap();
    }
}
#define MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr_avg,   \
                                        maxerr_avg_biased)            \
    ASSERT_PRED_FORMAT5(assert_tensor_eq, v0, v1, maxerr, maxerr_avg, \
                        maxerr_avg_biased)

}




template <typename Opr>
void fix_addition_attr_map(
        std::unordered_map<std::string, megcc::CCAttr>& proxy_attr,
        megdnn::test::DnnOprProxy<Opr>& dnn_proxy,
        TensorNDArray& tensor_array) {}

template <>
void fix_addition_attr_map<megdnn::IndexingMultiAxisVec>(
        std::unordered_map<std::string, megcc::CCAttr>& proxy_attr,
        megdnn::test::DnnOprProxy<megdnn::IndexingMultiAxisVec>& dnn_proxy,
        TensorNDArray& tensor_array) {
    std::vector<size_t> axes_vec;
    for (size_t i = 0; i < tensor_array.size() - 2; i++) {
        axes_vec.push_back(dnn_proxy.axes[i]);
    }
    proxy_attr["axis"] = megcc::CCAttr({axes_vec});
}

template <>
void fix_addition_attr_map<megdnn::TopK>(
        std::unordered_map<std::string, megcc::CCAttr>& proxy_attr,
        megdnn::test::DnnOprProxy<megdnn::TopK>& dnn_proxy,
        TensorNDArray& tensor_array) {
    proxy_attr["k"] = megcc::CCAttr(dnn_proxy.get_k());
}

template <typename Opr>
void Checker<Opr>::check_tensors(const TensorNDArray& expected,
                                 const TensorNDArray& computed, float epsilon,
                                 float max_avg_error,
                                 float max_avg_biased_error) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].layout.ndim == 0)
            continue;
        MEGDNN_ASSERT_TENSOR_EQ_EPS_AVG(expected[i], computed[i], epsilon,
                                        max_avg_error, max_avg_biased_error);
    }
}

template <typename Opr>
void Checker<Opr>::exec(TensorLayoutArray all_layouts) {
    using CCProxy = CCOprProxy<Opr>;
    auto dnn_handle = Runner<Opr>::get_dnn_handle();
    auto opr = dnn_handle->template create_operator<Opr>();
    opr->param() = m_param;
    if (!m_run_cc_only)
        m_dnn_proxy.deduce_layout(opr.get(), all_layouts);

    auto tensor_array_storage = dnn_alloc_tensors(dnn_handle, all_layouts, 0);
    auto tensor_array_naive_storage =
            dnn_alloc_tensors(dnn_handle, all_layouts, 0);
    auto tensor_array_dnn = *tensor_array_naive_storage;
    auto tensor_array = *tensor_array_storage;
#if !MEGCC_TEST_GEN
    Runner<Opr>::init_tensor(tensor_array_dnn, m_rng);
    dnn_copy_tensors(tensor_array, tensor_array_dnn);
#endif
    //! test mode
    CCProxy cc_proxy;
    std::unordered_map<std::string, CCAttr> proxy_attr;
    fix_addition_attr_map<Opr>(proxy_attr, m_dnn_proxy, tensor_array_dnn);
    cc_proxy.exec(opr.get(), tensor_array, m_arch, {}, m_kernel_symbol,
                  proxy_attr, m_run_cc_dynamic);
#if !MEGCC_TEST_GEN
    //! run dnn
    if (m_before_exec_callback) {
        m_before_exec_callback(opr.get(), tensor_array_dnn);
    }
    if (!m_run_cc_only) {
        m_dnn_proxy.exec(opr.get(), tensor_array_dnn);
        if (m_output_canonizer) {
            m_output_canonizer(tensor_array);
            m_output_canonizer(tensor_array_dnn);
        }
        check_tensors(tensor_array_dnn, tensor_array, m_epsilon,
                      m_max_avg_error, m_max_avg_biased_error);
    }
#endif
}
namespace megcc {
namespace test {

#if !MEGCC_TEST_GEN
#define INIT_TENSOR_MACRO(...)                            \
    {                                                     \
        runner.init_tensor(tensor_array_dnn, m_rng);      \
        dnn_copy_tensors(tensor_array, tensor_array_dnn); \
    }

#define RUN_DNN_MACRO(...)                                       \
    {                                                            \
        dnn_proxy.exec(opr.get(), tensor_array_dnn);             \
        check_tensors(tensor_array_dnn, tensor_array, m_epsilon, \
                      m_max_avg_error, m_max_avg_biased_error);  \
    }
#else
#define INIT_TENSOR_MACRO(...) \
    {}
#define RUN_DNN_MACRO(...) \
    {}
#endif

#define DEF_CV_OPR(_Opr)                                                      \
    template <>                                                               \
    void Checker<_Opr>::exec(TensorLayoutArray all_layouts) {                 \
        using CvOpr = _Opr;                                                   \
        using CCProxy = CCOprProxy<CvOpr>;                                    \
        using DnnOpr = CvOpr::DnnOpr;                                         \
        using DnnProxy = megdnn::test::DnnOprProxy<DnnOpr>;                   \
        Runner<DnnOpr> runner;                                                \
        auto dnn_handle = runner.get_dnn_handle();                            \
        auto opr = dnn_handle->template create_operator<DnnOpr>();            \
        CvOpr cv_opr;                                                         \
        cv_opr.param() = m_param;                                             \
        cv_opr.reformat_layout(&cv_opr, all_layouts);                         \
        opr->param() = cv_opr.dnn_param(m_param);                             \
        DnnProxy dnn_proxy;                                                   \
        if (!m_run_cc_only)                                                   \
            dnn_proxy.deduce_layout(opr.get(), all_layouts);                  \
                                                                              \
        auto tensor_array_storage =                                           \
                dnn_alloc_tensors(dnn_handle, all_layouts, 0);                \
        auto tensor_array_naive_storage =                                     \
                dnn_alloc_tensors(dnn_handle, all_layouts, 0);                \
        auto tensor_array_dnn = *tensor_array_naive_storage;                  \
        auto tensor_array = *tensor_array_storage;                            \
        INIT_TENSOR_MACRO();                                                  \
        CCProxy cc_proxy;                                                     \
        cc_proxy.exec(&cv_opr, tensor_array, m_arch, {}, m_kernel_symbol, {}, \
                      false);                                                 \
        if (!m_run_cc_only)                                                   \
            RUN_DNN_MACRO();                                                  \
    }                                                                         \
    template class Checker<_Opr>;

}  // namespace test
}  // namespace megcc
namespace megcc {
namespace test {
template class Checker<megdnn::ElemwiseForward>;
template class Checker<megdnn::ElemwiseMultiType>;
template class Checker<megdnn::ConvolutionForward>;
template class Checker<megdnn::ConvBiasForward>;
template class Checker<megdnn::ConvolutionBackwardData>;
template class Checker<megdnn::PoolingForward>;
template class Checker<megdnn::MatrixMulForward>;
template class Checker<megdnn::MatrixInverse>;
template class Checker<megdnn::IndexingMultiAxisVec>;
template class Checker<megdnn::IndexingOneHot>;
template class Checker<megdnn::ReduceForward>;
template class Checker<megdnn::WarpAffineForward>;
template class Checker<megdnn::WarpPerspectiveForward>;
template class Checker<megdnn::BatchedMatrixMulForward>;
template class Checker<megdnn::TypeCvtForward>;
template class Checker<megdnn::TopK>;
template class Checker<megdnn::RelayoutForward>;
template class Checker<megdnn::PowC>;
template class Checker<megdnn::ResizeForward>;
template class Checker<megdnn::ArgsortForward>;
template class Checker<megdnn::ConcatForward>;
template class Checker<megdnn::ArgmaxForward>;

//! CV
DEF_CV_OPR(megdnn::CVtranspose);
DEF_CV_OPR(megdnn::CVflip);
DEF_CV_OPR(megdnn::CVResize);
DEF_CV_OPR(megdnn::CVRotate);
DEF_CV_OPR(megdnn::CVRoicopy);
DEF_CV_OPR(megdnn::CVCvtColor);
DEF_CV_OPR(megdnn::CVWarpAffine);

}  // namespace test
}  // namespace megcc
