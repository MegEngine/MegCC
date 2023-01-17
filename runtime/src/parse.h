/**
 * \file runtime/src/parse.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#ifndef PARSE_H
#define PARSE_H

#include "data_struct.h"
#include "stdio.h"
#include "tinynn.h"

#include "model_reader.h"

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(MegCC, x)

static inline char* get_string(const char* in) {
    if (!in)
        return NULL;
    int len = strlen(in) + 1;
    char* ret = (char*)tinynn_malloc(len + 1);
    strcpy(ret, in);
    return ret;
}

TinyNNStatus parse_tensor(Tensor* tensor, ns(Tensor_table_t) fbs_tensor,
                          int tensor_id);

TinyNNStatus parse_weight(Tensor* weight, ns(Weight_table_t) fbs_weight,
                          Device* host_dev);

TinyNNStatus parse_device_model(DeviceModel* model, CombineModel* c_model,
                                ns(DeviceModel_table_t) device_model);

//! all resource are allocate here
TinyNNStatus parse_model(void* buffer, size_t size, CombineModel* model,
                         int share_weights);

#endif

// vim: syntax=cpp.doxygen
