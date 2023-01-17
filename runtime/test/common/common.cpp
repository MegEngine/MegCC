/**
 * \file runtime/test/common/common.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "common.h"
#include <math.h>
#include "float.h"

using namespace test;

namespace {

static int equal_layout(const Layout& layout0, const Layout& layout1) {
    if (layout0.nr_dim != layout1.nr_dim) {
        return false;
    }
    for (int32_t i = 0; i < layout0.nr_dim; i++) {
        if (layout0.dims[i] != layout1.dims[i] ||
            layout0.stride[i] != layout1.stride[i]) {
            return false;
        }
    }
    return true;
}

static std::string layout_to_string(const Layout& layout) {
    std::stringstream ss;
    ss << "Layout: dims[";
    for (int32_t i = 0; i < layout.nr_dim; i++) {
        ss << layout.dims[i] << ",";
    }
    ss << "], strides[";
    for (int32_t i = 0; i < layout.nr_dim; i++) {
        ss << layout.stride[i] << ",";
    }
    ss << "]";
    return ss.str();
}

static inline float diff(float x, float y) {
    return x - y;
}
static inline int diff(int x, int y) {
    return x - y;
}

static inline bool good_float(float val) {
    return std::isfinite(val);
}

static inline bool good_float(int) {
    return true;
}

template <typename ctype>
::testing::AssertionResult assert_tensor_eq_with_dtype(
        const char* expr0, const char* expr1, const Tensor& v0,
        const Tensor& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    size_t nr_elem = 1;
    for (int i = 0; i < v0.layout.nr_dim; ++i) {
        nr_elem *= v0.layout.dims[i];
    }
    double error_sum = 0;
    double error_sum_biased = 0;
    const ctype* ptr0 = static_cast<const ctype*>(v0.ptr);
    const ctype* ptr1 = static_cast<const ctype*>(v1.ptr);
    for (size_t i = 0; i < nr_elem; ++i) {
        ctype iv0 = ptr0[i], iv1 = ptr1[i];
        float err = diff(iv0, iv1);
        error_sum += std::abs(err);
        error_sum_biased += err;
        if (!good_float(iv0) || !good_float(iv1) || std::abs(err) > maxerr) {
            return ::testing::AssertionFailure()
                   << "Unequal value\n"
                   << "Value of: " << expr1 << "\n"
                   << "  Actual: " << (iv1 + 0) << "\n"
                   << "Expected: " << expr0 << "\n"
                   << "Which is: " << (iv0 + 0) << "\n"
                   << "At index: " << i << "\n"
                   << "tensor v0 layout : " << layout_to_string(v0.layout)
                   << "\n"
                   << "tensor v1 layout : " << layout_to_string(v1.layout)
                   << "\n";
        }
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

::testing::AssertionResult assert_tensor_eq(
        const char* expr0, const char* expr1, const char* /*expr_maxerr*/,
        const char* /*expr_maxerr_avg*/, const char* /*expr_maxerr_avg*/,
        const Tensor& v0, const Tensor& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    if (!equal_layout(v0.layout, v1.layout)) {
        return ::testing::AssertionFailure()
               << "Layout mismatch\n"
               << "Value of: " << expr1 << "\n"
               << "  Actual: " << layout_to_string(v1.layout) << "\n"
               << "Expected: " << expr0 << "\n"
               << "Which is: " << layout_to_string(v0.layout) << "\n";
    }
    if (v0.dtype.type_enum != v1.dtype.type_enum) {
        return ::testing::AssertionFailure()
               << "Dtype mismatch " << v0.dtype.type_enum << " vs "
               << v1.dtype.type_enum << "\n ";
    }
    switch (v0.dtype.type_enum) {
#define CASE(enum_, ctype_)                                                   \
    case enum_: {                                                             \
        return assert_tensor_eq_with_dtype<ctype_>(                           \
                expr0, expr1, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased); \
    }
        CASE(TinyNN_FLOAT, float)
        CASE(TinyNN_INT, int)
        CASE(TinyNN_INT8, int8_t)
        CASE(TinyNN_UINT8, uint8_t)
        CASE(TinyNN_INT16, int16_t)
        default:
            printf("unsupport dtype in check tensor equal.");
            abort();
#undef CASE
    }
}

}  // namespace

void test::check_tensor(const Tensor& computed, const Tensor& expected,
                        float epsilon, float max_avg_error,
                        float max_avg_biased_error) {
    if (expected.layout.nr_dim == 0)
        return;
    ASSERT_TENSOR_EQ_EPS_AVG(computed, expected, epsilon, max_avg_error,
                             max_avg_biased_error);
}

std::shared_ptr<Tensor> test::create_tensor(std::vector<uint32_t> shape,
                                            TinyNNDType dtype_enum, void* ptr) {
    auto tensor = std::make_shared<Tensor>();
    tensor->is_dynamic = true;
    DType dtype;
    dtype.type_enum = dtype_enum;
    tensor->dtype = dtype;
    Layout layout;
    layout.nr_dim = shape.size();
    for (uint32_t i = 0; i < shape.size(); i++) {
        layout.dims[i] = shape[i];
    }
    layout.stride[layout.nr_dim - 1] = 1;
    for (int index = layout.nr_dim - 2; index >= 0; index--) {
        layout.stride[index] =
                layout.dims[index + 1] * layout.stride[index + 1];
    }
    tensor->layout = layout;
    tensor->ptr = ptr;
    return tensor;
}
class SimpleVM {
    CombineModel* cb_model;

public:
    SimpleVM() {
        cb_model = new CombineModel;
        cb_model->nr_device_model = 1;
        cb_model->device_models = new DeviceModel*;
        cb_model->device_models[0] = new DeviceModel;
        cb_model->active_device_model_idx = 0;
        cb_model->host_dev.device_type = TinyNN_BARE_METAL;
        init_device(&(cb_model->host_dev));
        DeviceModel* model = cb_model->device_models[0];
        model->device.device_type = TinyNN_BARE_METAL;
        init_device(&model->device);
        model->opt = create_runtime_opt(&model->device);
        vm_attach(cb_model);
    }
    VM* getvm() { return (VM*)(cb_model->vm); }
    ~SimpleVM() {
        auto vm = getvm();
        delete vm->model->device_models[0];
        delete vm->model->device_models;
        delete vm->model;
        vm_detach(cb_model);
    }
};

VM* test::create_vm() {
    static SimpleVM svm;
    return svm.getvm();
}

//! fake function to pass the link
void register_op(VM* vm) {
    LOG_DEBUG(
            "Just register a fake op instruction, can't be used to test op.\n");
}

// vim: syntax=cpp.doxygen
