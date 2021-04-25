/**
 * \file runtime/src/lite/tensor.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "data_struct.h"

static inline ComboIOTensor* get_empty_io_tensor(CombineModel* model) {
    if (model->combo_iotensor) {
        return model->combo_iotensor;
    } else {
        ComboIOTensor* res = tinynn_malloc(sizeof(ComboIOTensor));
        res->model = model;
        res->tensors = tinynn_malloc(sizeof(Tensor*) * model->nr_device_model);
        model->combo_iotensor = res;
        return res;
    }
}

static inline Tensor* get_active_tensor(ComboIOTensor* tensor) {
    return tensor->tensors[tensor->model->active_device_model_idx];
}
