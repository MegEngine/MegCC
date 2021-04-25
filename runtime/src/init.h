/**
 * \file runtime/src/init.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#ifndef CORE_H
#define CORE_H

#include "data_struct.h"
#include "device.h"
#include "tinynn.h"

#define ERROR_ASSERT(_condition, _error_no, _stm) \
    if (!(_condition)) {                          \
        *error = _error_no;                       \
        _stm;                                     \
    }

#define FREE(x)                \
    if (x) {                   \
        tinynn_free((void*)x); \
    }

#define DEV_FREE(x, dev_vm)      \
    if (x) {                     \
        free_dev_ptr(x, dev_vm); \
    }

TinyNNStatus init_model_weights(CombineModel* model);

TinyNNStatus init_model_memory(CombineModel* model);



static inline int dtype_length(TinyNNDType dtype, TinyNNStatus* error) {
    switch (dtype) {
        case TinyNN_FLOAT:
        case TinyNN_INT:
        case TinyNN_QINT32:
            return 4;
        case TinyNN_FLOAT16:
        case TinyNN_INT16:
            return 2;
        case TinyNN_INT8:
        case TinyNN_UINT8:
        case TinyNN_QINT8:
            return 1;
        default: {
            LOG_ERROR("no support data type. enum value is %d\n", (int)dtype);
            if (error)
                *error = TinyNN_ERROR_NO_IMPLEMENT;
        }
    }
    tinynn_trap();
}

static inline size_t tensor_length_in_byte(const Tensor* tensor) {
    Layout layout = tensor->layout;
    size_t length = layout.nr_dim > 0 ? 1 : 0;
    for (int i = 0; i < layout.nr_dim; i++) {
        int stride =
                layout.stride[i] > 0 ? layout.stride[i] : -layout.stride[i];
        length += stride * (layout.dims[i] - 1);
    }
    length *= dtype_length((tensor)->dtype.type_enum, NULL);
    return length;
}
#endif
