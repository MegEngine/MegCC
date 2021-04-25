/**
 * \file runtime/src/lite/tensor.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <stdbool.h>
#include "data_struct.h"
#include "init.h"
#include "io_tensor.h"
#include "lite-c/tensor_c.h"
#include "tinynn.h"

static inline ComboIOTensor* to_inner_tensor(LiteTensor tensor) {
    return (ComboIOTensor*)tensor;
}
static inline TinyNNStatus tinynn_to_lite_dtype(TinyNNDType dtype_enum,
                                                LiteCDataType* target_dtype) {
    if (!target_dtype) {
        return TinyNN_ERROR_NULL_PTR;
    }
    switch (dtype_enum) {
        case TinyNN_FLOAT:
            *target_dtype = LITE_FLOAT;
            break;
        case TinyNN_FLOAT16:
            *target_dtype = LITE_HALF;
            break;
        case TinyNN_INT:
            *target_dtype = LITE_INT;
            break;
        case TinyNN_INT8:
            *target_dtype = LITE_INT8;
            break;
        case TinyNN_INT16:
            *target_dtype = LITE_INT16;
            break;
        case TinyNN_UINT8:
            *target_dtype = LITE_UINT8;
            break;
        default:
            LOG_ERROR("failed convert tinynn dtype to lite\n");
            return TinyNN_ERROR_OUT_OF_RANGE;
    }
    return TinyNN_SUCCESS;
}

static inline bool compare_layout(Tensor* tensor, LiteLayout layout) {
    if (tensor->layout.nr_dim != layout.ndim) {
        return false;
    }
    LiteCDataType lite_dtype;
    tinynn_to_lite_dtype(tensor->dtype.type_enum, &lite_dtype);
    if (lite_dtype != layout.data_type) {
        return false;
    }
    for (int i = 0; i < layout.ndim; ++i) {
        if (tensor->layout.dims[i] != layout.shapes[i]) {
            return false;
        }
    }
    return true;
}

int LITE_get_tensor_memory(const LiteTensor tensor_, void** data) {
    if (!tensor_ || !data) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    Tensor* tensor = get_active_tensor((ComboIOTensor*)tensor_);
    *data = tensor->ptr;
    return TinyNN_SUCCESS;
}

int LITE_set_tensor_layout(LiteTensor tensor_, const LiteLayout layout) {
    if (!tensor_) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    ComboIOTensor* tensor_pack = to_inner_tensor(tensor_);
    for (int i = 0; i < tensor_pack->model->nr_device_model; ++i) {
        if (compare_layout(tensor_pack->tensors[i], layout)) {
            tensor_pack->model->active_device_model_idx = i;
            return TinyNN_SUCCESS;
        }
    }
    return TinyNN_ERROR_INVALID_LAYOUT;
}

int LITE_reset_tensor_memory(LiteTensor tensor_, void* prepared_data,
                             size_t data_length_in_byte) {
    if (!tensor_ || !prepared_data) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    Tensor* tensor = get_active_tensor(to_inner_tensor(tensor_));
    size_t tensor_len = tensor_length_in_byte(tensor);
    LOG_DEBUG("active %d model\n",
              to_inner_tensor(tensor_)->model->active_device_model_idx);
    if (data_length_in_byte != tensor_len) {
        LOG_ERROR("reset tensor memory with not equal size %zu != %zu\n",
                  data_length_in_byte, tensor_len);
        return TinyNN_ERROR;
    }
    tensor->ptr = prepared_data;
    return TinyNN_SUCCESS;
}

int LITE_get_tensor_layout(const LiteTensor tensor_, LiteLayout* layout) {
    if (!tensor_) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    if (!layout) {
        LOG_ERROR("input layout is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    Tensor* tensor = get_active_tensor(to_inner_tensor(tensor_));
    TinyNNStatus status =
            tinynn_to_lite_dtype(tensor->dtype.type_enum, &(layout->data_type));
    if (status != TinyNN_SUCCESS) {
        return status;
    }
    layout->ndim = tensor->layout.nr_dim;
    for (int i = 0; i < layout->ndim; ++i) {
        layout->shapes[i] = tensor->layout.dims[i];
    }
    return TinyNN_SUCCESS;
}

int LITE_destroy_tensor(LiteTensor tensor_) {
    ComboIOTensor* tensor = (ComboIOTensor*)tensor_;
    CombineModel* model = tensor->model;
    model->combo_iotensor = NULL;
    tinynn_free(tensor->tensors);
    tinynn_free(tensor);
    return TinyNN_SUCCESS;
}
