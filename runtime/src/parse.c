/**
 * \file runtime/src/parse.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "parse.h"
#include <stdbool.h>
#include <stdlib.h>
#include "device.h"
#include "init.h"
#include "vm.h"
#include "kernels.h"

static uint as_uint(const float x) {
    return *(uint*)&x;
}
static float as_float(const uint x) {
    return *(float*)&x;
}

//! From:
//! https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
static float half_to_float(const uint16_t x) {
    const uint e = (x & 0x7C00) >> 10;  // exponent
    const uint m = (x & 0x03FF) << 13;  // mantissa
    const uint v =
            as_uint((float)m) >> 23;  // evil log2 bit hack to count leading
                                      // zeros in denormalized format
    return as_float(
            (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
            ((e == 0) & (m != 0)) *
                    ((v - 37) << 23 |
                     ((m << (150 - v)) &
                      0x007FE000)));  // sign : normalized : denormalized
}

static TinyNNDType dtype_from_fbs(ns(DTypeEnum_enum_t) fbs_dtype) {
    switch (fbs_dtype) {
        case ns(DTypeEnum_Float32):
            return TinyNN_FLOAT;
        case ns(DTypeEnum_Float16):
            return TinyNN_FLOAT16;
        case ns(DTypeEnum_Int32):
            return TinyNN_INT;
        case ns(DTypeEnum_Int8):
            return TinyNN_INT8;
        case ns(DTypeEnum_Int16):
            return TinyNN_INT16;
        case ns(DTypeEnum_Uint8):
            return TinyNN_UINT8;
        case ns(DTypeEnum_QInt8):
            return TinyNN_QINT8;
        case ns(DTypeEnum_QInt32):
            return TinyNN_QINT32;
        case ns(DTypeEnum_QUint8):
            return TinyNN_QUINT8;
        default: {
            LOG_ERROR("no support data type from fbs.\n");
        }
    }
    tinynn_trap();
}

static TinyNNFormat format_from_fbs(ns(Format_enum_t) fbs_format) {
    switch (fbs_format) {
        case ns(Format_NCHW):
            return TinyNN_NCHW;
        case ns(Format_NHWC):
            return TinyNN_NHWC;
        case ns(Format_NCHW4):
            return TinyNN_NCHW4;
        case ns(Format_NCHW8):
            return TinyNN_NCHW8;
        case ns(Format_OIHW):
            return TinyNN_OIHW;
        default: {
            LOG_ERROR("no support format from fbs.\n");
        }
    }
    tinynn_trap();
}

static TinyNNDevice device_from_fbs(ns(Device_enum_t) fbs_device) {
    switch (fbs_device) {
        case ns(Device_BARE_METAL):
            return TinyNN_BARE_METAL;
        case ns(Device_ARM64):
            return TinyNN_ARM64;
        case ns(Device_ARM32):
            return TinyNN_ARM32;
        case ns(Device_ARM64_V82):
            return TinyNN_ARM64_V82;
        case ns(Device_ARM32_V82):
            return TinyNN_ARM32_V82;
        case ns(Device_OPENCL_MALI):
            return TinyNN_OPENCL_MALI;
        default: {
            LOG_ERROR("no support device from fbs.\n");
        }
    }
    tinynn_trap();
}

static DTypeParam dtype_param_from_fbs(ns(DType_table_t) fbs_dtype) {
    DTypeParam param = {0.0f, 0};
    if (ns(DType_param_is_present(fbs_dtype))) {
        ns(DTypeParam_table_t) fbs_param = ns(DType_param(fbs_dtype));
        param.scale = ns(DTypeParam_scale(fbs_param));
        param.zero_point = ns(DTypeParam_zero_point(fbs_param));
    }
    return param;
}

TinyNNStatus parse_tensor(Tensor* tensor, ns(Tensor_table_t) fbs_tensor,
                          int tensor_id) {
    //! dtype
    ns(DType_table_t) fbs_dtype = ns(Tensor_dtype(fbs_tensor));
    tensor->dtype.type_enum = dtype_from_fbs(ns(DType_type(fbs_dtype)));
    tensor->dtype.param = dtype_param_from_fbs(fbs_dtype);

    //! name
    const char* name = ns(Tensor_name(fbs_tensor));
    if (strlen(name) > 0) {
        tensor->name = get_string(name);
    } else {
        char name[20] = "tensor:";
        sprintf(name, "tensor:%d", tensor_id);
        tensor->name = get_string(name);
    }
    //! use_count
    tensor->use_count = ns(Tensor_use_count(fbs_tensor));
    tensor->is_dynamic = ns(Tensor_dynamic(fbs_tensor));
    //! offset
    tensor->offset = ns(Tensor_offset(fbs_tensor));
    LOG_DEBUG("Tensor info: name=%s, use_count=%d, offset=%zu, is_dynamic=%d\n",
              tensor->name, tensor->use_count, tensor->offset,
              tensor->is_dynamic);
    //! layout
    ns(Layout_table_t) fbs_layout = ns(Tensor_layout(fbs_tensor));
    flatbuffers_int32_vec_t fbs_dims = ns(Layout_dims(fbs_layout));
    flatbuffers_int32_vec_t fbs_stride = ns(Layout_stride(fbs_layout));
    tensor->layout.nr_dim = flatbuffers_int32_vec_len(fbs_dims);
    LOG_DEBUG("Tensor layout: [");
    for (int i = 0; i < tensor->layout.nr_dim; i++) {
        tensor->layout.dims[i] = flatbuffers_int32_vec_at(fbs_dims, i);
        LOG_DEBUG("%d,", tensor->layout.dims[i]);
        tensor->layout.stride[i] = flatbuffers_int32_vec_at(fbs_stride, i);
    }
    //! format
    tensor->layout.format = format_from_fbs(ns(Layout_format(fbs_layout)));
    LOG_DEBUG("], format=%d\n", tensor->layout.format);

    tensor->is_weight = 0;

    tensor->size = tensor_length_in_byte(tensor);
    return TinyNN_SUCCESS;
}

TinyNNStatus parse_weight(Tensor* weight, ns(Weight_table_t) fbs_weight,
                          Device* host_dev) {
    //! dtype
    ns(DType_table_t) fbs_dtype = ns(Weight_dtype(fbs_weight));
    weight->dtype.type_enum = dtype_from_fbs(ns(DType_type(fbs_dtype)));
    weight->dtype.param = dtype_param_from_fbs(fbs_dtype);

    //! name
    weight->name = get_string(ns(Weight_name(fbs_weight)));
    //! use_count
    weight->use_count = ns(Weight_use_count(fbs_weight));
    LOG_DEBUG("weight info: name=%s, use_count=%d \n", weight->name,
              weight->use_count);

    ns(Layout_table_t) fbs_layout = ns(Weight_layout(fbs_weight));
    flatbuffers_int32_vec_t fbs_dims = ns(Layout_dims(fbs_layout));
    flatbuffers_int32_vec_t fbs_stride = ns(Layout_stride(fbs_layout));
    weight->layout.nr_dim = flatbuffers_int32_vec_len(fbs_dims);
    LOG_DEBUG("weight layout: [");
    for (int i = 0; i < weight->layout.nr_dim; i++) {
        weight->layout.dims[i] = flatbuffers_int32_vec_at(fbs_dims, i);
        LOG_DEBUG("%d,", weight->layout.dims[i]);
        weight->layout.stride[i] = flatbuffers_int32_vec_at(fbs_stride, i);
    }
    //! format
    weight->layout.format = format_from_fbs(ns(Layout_format(fbs_layout)));

    weight->checksum = ns(Weight_checksum(fbs_weight));

    weight->is_weight = 1;

    flatbuffers_int8_vec_t data = ns(Weight_data(fbs_weight));
    size_t weight_length_in_byte = flatbuffers_int8_vec_len(data);
    LOG_DEBUG("], format=%d, weight length=%zu\n", weight->layout.format,
              weight_length_in_byte);

    //! check data length
    size_t length = tensor_length_in_byte(weight);
    int compressed = ns(Weight_compressed(fbs_weight));
    if (!compressed) {
        TINYNN_ASSERT_MSG(length == weight_length_in_byte,
                          "weight length error when parse.\n");
    } else {
        LOG_DEBUG("Model weights is compressed.\n");
    }
    weight->size = weight_length_in_byte;

    //! don't allocate new memory directly, just share the model memory, and
    //! align them after all data have parsed
    weight->ptr = (void*)data;
    weight->is_shared = 1;

    return TinyNN_SUCCESS;
}

static int weight_ptr_compare(const void* item0, const void* item1) {
    void* ptr0 = (*((const Tensor**)item0))->ptr;
    void* ptr1 = (*((const Tensor**)item1))->ptr;
    if ((uintptr_t)ptr0 < (uintptr_t)ptr1) {
        return -1;
    } else if ((uintptr_t)ptr0 > (uintptr_t)ptr1) {
        return 1;
    } else {
        return 0;
    }
}

static TinyNNStatus alignment_or_alloc_weights(CombineModel* model,
                                               void* buffer,
                                               int share_weights) {
    int nr_weight = model->nr_origin_weight;
    uintptr_t current_addr = (uintptr_t)buffer;
    Device* host_dev = &model->host_dev;
    unsigned int alignment = host_dev->alignment;
    //! sort all the weights by their ptr
    Tensor** sorted_weights = tinynn_malloc(sizeof(Tensor*) * nr_weight);
    for (int i = 0; i < nr_weight; i++) {
        sorted_weights[i] = model->weights + i;
    }
    qsort(sorted_weights, nr_weight, sizeof(Tensor*), weight_ptr_compare);
    //! align or alloc them
    for (int i = 0; i < nr_weight; i++) {
        Tensor* weight = sorted_weights[i];
        size_t length = weight->size;
        size_t layout_size = tensor_length_in_byte(weight);
        int compressed = 0;
        if (length > 0 && weight->dtype.type_enum == TinyNN_FLOAT &&
            length == layout_size / 2) {
            compressed = 1;
            LOG_DEBUG(
                    "The weight data is compressed, decompressed it to "
                    "fp32.\n");
        }

        void* origin_ptr = weight->ptr;
        TINYNN_ASSERT_MSG(current_addr < (uintptr_t)origin_ptr,
                          "weight pointer addr error.\n");
        uintptr_t alignment_addr =
                (uintptr_t)origin_ptr & (~((uintptr_t)alignment - 1));
        if (!share_weights || alignment_addr < current_addr || compressed) {
            if (!(weight->ptr = host_dev->malloc(layout_size))) {
                LOG_ERROR("malloc weight memory fail.\n");
            }
            weight->is_shared = 0;
            if (!compressed) {
                memcpy(weight->ptr, origin_ptr, length);
            } else {
                weight->size = layout_size;
                uint16_t* src = (uint16_t*)origin_ptr;
                float* dst = (float*)(weight->ptr);
                for (int id = 0; id < layout_size / sizeof(float); id++) {
                    *dst = half_to_float(*src);
                    dst++;
                    src++;
                }
            }
        } else {
            uintptr_t more_fit_addr = (current_addr + alignment - 1) &
                                      (~((uintptr_t)alignment - 1));
            alignment_addr = alignment_addr > more_fit_addr ? more_fit_addr
                                                            : alignment_addr;
            if (alignment_addr != (uintptr_t)origin_ptr) {
                LOG_DEBUG("align weight: %d from %p to %p\n", i, origin_ptr,
                          (void*)alignment_addr);
                TINYNN_ASSERT_MSG(alignment_addr < (uintptr_t)origin_ptr,
                                  "Align Ptr must be ahead of origin ptr.");
                memmove((void*)alignment_addr, origin_ptr, length);
                weight->ptr = (void*)alignment_addr;
                weight->is_shared = 1;
                current_addr = alignment_addr + length;
            } else {
                weight->is_shared = 1;
                current_addr = (uintptr_t)origin_ptr + length;
            }
        }
    }
    tinynn_free(sorted_weights);
    return TinyNN_SUCCESS;
}

TinyNNStatus parse_device_model(DeviceModel* model,
                                ns(DeviceModel_table_t) fbs_device_model) {
    //! parse tensor
    ns(Tensor_vec_t) fbs_tensors =
            ns(DeviceModel_tensor_pool(fbs_device_model));
    int nr_tensor = ns(Tensor_vec_len(fbs_tensors));
    LOG_DEBUG("device model tensor number: %d\n", nr_tensor);
    model->nr_tensor = nr_tensor;
    model->tensors = tinynn_malloc(sizeof(Tensor) * nr_tensor);
    memset(model->tensors, 0, sizeof(Tensor) * nr_tensor);
    for (int i = 0; i < nr_tensor; i++) {
        LOG_DEBUG("parse tensor id: %d\n", i);
        ns(Tensor_table_t) fbs_tensor = ns(Tensor_vec_at(fbs_tensors, i));
        Tensor* tensor = model->tensors + i;
        if (parse_tensor(tensor, fbs_tensor, i) != TinyNN_SUCCESS) {
            LOG_ERROR("parse tensor error!\n");
            goto exit;
        }
    }

    //! parse instructions
    ns(Instruction_union_vec_t) fbs_instructions_union =
            ns(DeviceModel_instructions_union(fbs_device_model));
    int nr_instruction = ns(Instruction_union_vec_len(fbs_instructions_union));
    LOG_DEBUG("device model instruction number: %d\n", nr_instruction);
    model->nr_instruction = nr_instruction;
    model->instructions = tinynn_malloc(sizeof(Instruction) * nr_instruction);
    memset(model->instructions, 0, sizeof(Instruction) * nr_instruction);
    for (int i = 0; i < nr_instruction; i++) {
        LOG_DEBUG("parse instruction id: %d\n", i);
        ns(Instruction_union_t) fbs_instruction_union =
                ns(Instruction_union_vec_at(fbs_instructions_union, i));
        Instruction* inst = model->instructions + i;
        if (vm_instruction_load(vm_global_inst(), fbs_instruction_union,
                                inst) != TinyNN_SUCCESS) {
            goto exit;
        }
    }

    //! inputs
    flatbuffers_int32_vec_t fbs_inputs =
            ns(DeviceModel_inputs(fbs_device_model));
    model->nr_input = flatbuffers_int32_vec_len(fbs_inputs);
    model->inputs = tinynn_malloc(model->nr_input * sizeof(Tensor*));
    memset(model->inputs, 0, model->nr_input * sizeof(Tensor*));
    for (int i = 0; i < model->nr_input; i++) {
        int index = flatbuffers_int32_vec_at(fbs_inputs, i);
        *(model->inputs + i) = model->tensors + index;
    }
    //! outputs
    flatbuffers_int32_vec_t fbs_outputs =
            ns(DeviceModel_outputs(fbs_device_model));
    model->nr_output = flatbuffers_int32_vec_len(fbs_outputs);
    model->outputs = tinynn_malloc(model->nr_output * sizeof(Tensor*));
    memset(model->outputs, 0, model->nr_output * sizeof(Tensor*));
    for (int i = 0; i < model->nr_output; i++) {
        int index = flatbuffers_int32_vec_at(fbs_outputs, i);
        *(model->outputs + i) = model->tensors + index;
    }
    return TinyNN_SUCCESS;
exit:
    if (model->tensors) {
        for (int i = 0; i < model->nr_tensor; i++) {
            Tensor* tensor = model->tensors + i;
            FREE(tensor->name);
        }
        tinynn_free(model->tensors);
    }
    if (model->instructions) {
        for (int i = 0; i < model->nr_instruction; i++) {
            Instruction* inst = model->instructions + i;
            if (inst->tag == TinyNN_INST_OPR) {
                Opr* opr = &inst->workload.opr;
                FREE(opr->inputs);
                FREE(opr->outputs);
                FREE(opr->name);
                FREE(opr->type);
            }
        }
        tinynn_free(model->instructions);
    }
    return TinyNN_ERROR_MODEL_PARSE;
}

static inline bool valid_device_check(TinyNNDevice device) {
    if (device == TinyNN_BARE_METAL) {
        return true;
    }
#if defined(__aarch64__)
    else if (device == TinyNN_ARM64) {
        return true;
    }
#elif defined(__arm__)
    else if (device == TinyNN_ARM32) {
        return true;
    }
#endif
    return false;
}

//! all resource are allocate here
TinyNNStatus parse_model(void* buffer, size_t size, CombineModel* model,
                         int share_weights) {
    load_kernel_init_function();
    ns(Model_table_t) fbs_model;
    if (!(fbs_model = ns(Model_as_root(buffer)))) {
        LOG_ERROR("Model not available\n");
        return -1;
    }
    if (share_weights) {
        model->model_ptr = buffer;
        model->model_len = size;
    } else {
        model->model_ptr = NULL;
        model->model_len = 0;
    }
    //! name, model_id, device
    model->name = get_string(MegCC_Model_name(fbs_model));
    model->model_id = ns(Model_model_id(fbs_model));
    model->const_shape = ns(Model_const_shape(fbs_model));
    LOG_DEBUG("log model: %s, model id: %zu, const shape: %d\n", model->name,
              model->model_id, model->const_shape);
    //! host device used to alloc weight
    memset(&(model->host_dev), 0, sizeof(Device));
    model->host_dev.device_type = TinyNN_BARE_METAL;
    TINYNN_ASSERT(init_device(&(model->host_dev)) == TinyNN_SUCCESS);

    //! parse weight
    ns(Weight_vec_t) fbs_weights = ns(Model_weight_pool(fbs_model));
    int nr_weight = ns(Weight_vec_len(fbs_weights));
    model->nr_origin_weight = nr_weight;
    model->weights = tinynn_malloc(sizeof(Tensor) * nr_weight);
    LOG_DEBUG("model weights number: %d\n", nr_weight);
    memset(model->weights, 0, sizeof(Tensor) * nr_weight);
    for (int i = 0; i < nr_weight; i++) {
        LOG_DEBUG("parse weight id: %d\n", i);
        ns(Weight_table_t) fbs_weight = ns(Weight_vec_at(fbs_weights, i));
        Tensor* weight = model->weights + i;
        if (parse_weight(weight, fbs_weight, &(model->host_dev)) !=
            TinyNN_SUCCESS) {
            LOG_ERROR("parse weight error!\n");
            goto exit;
        }
    }

    //! parse device model
    ns(DeviceModel_vec_t) fbs_device_models =
            ns(Model_device_models(fbs_model));
    int nr_model = ns(DeviceModel_vec_len(fbs_device_models));
    int nr_valid_device_model = 0;
    {
        //! cal max memory for tensor and valid device model number
        size_t max_tensor_size = 0;
        for (int i = 0; i < nr_model; i++) {
            ns(DeviceModel_table_t) fbs_device_model =
                    ns(DeviceModel_vec_at(fbs_device_models, i));
            ns(Device_enum_t) fbs_device =
                    ns(DeviceModel_device(fbs_device_model));
            TinyNNDevice device_enum = device_from_fbs(fbs_device);
            if (!valid_device_check(device_enum)) {
                continue;
            }
            nr_valid_device_model++;
            size_t tensor_size =
                    ns(DeviceModel_tensor_memory(fbs_device_model));
            max_tensor_size = max_tensor_size < tensor_size ? tensor_size
                                                            : max_tensor_size;
        }
        model->max_tensor_memroy = tinynn_malloc(sizeof(Memory));
        model->max_tensor_memroy->length_in_byte = max_tensor_size;
        model->max_tensor_memroy->ptr = NULL;
        model->is_own_tensor_memory = 1;
        LOG_DEBUG("max_tensor_memroy number %zu.\n",
                  model->max_tensor_memroy->length_in_byte);
    }
    LOG_DEBUG("device model number: %d, valid device model %d\n", nr_model,
              nr_valid_device_model);
    model->device_models =
            tinynn_malloc(sizeof(DeviceModel*) * nr_valid_device_model);
    model->nr_device_model = nr_valid_device_model;
    memset(model->device_models, 0,
           sizeof(DeviceModel*) * nr_valid_device_model);

    for (int i = 0, valid_model_idx = 0; i < nr_model; i++) {
        ns(DeviceModel_table_t) fbs_device_model =
                ns(DeviceModel_vec_at(fbs_device_models, i));
        ns(Device_enum_t) fbs_device = ns(DeviceModel_device(fbs_device_model));
        TinyNNDevice device_enum = device_from_fbs(fbs_device);
        if (!valid_device_check(device_enum)) {
            continue;
        }
        DeviceModel* dev_model = tinynn_malloc(sizeof(DeviceModel));
        LOG_DEBUG("parse device model %d device is %d with %d\n", i,
                  device_enum, dev_model->device.device_type);
        memset(dev_model, 0, sizeof(DeviceModel));
        model->device_models[valid_model_idx] = dev_model;
        model->active_device_model_idx = valid_model_idx;
        valid_model_idx++;
        dev_model->device.device_type = device_enum;
        if (init_device(&dev_model->device) != TinyNN_SUCCESS) {
            goto exit;
        }
        dev_model->opt = create_runtime_opt(&dev_model->device);
        if (parse_device_model(dev_model, fbs_device_model) != TinyNN_SUCCESS) {
            LOG_ERROR("parse device model error!\n");
            goto exit;
        }
    }
    model->active_device_model_idx = 0;
    alignment_or_alloc_weights(model, buffer, share_weights);
    return TinyNN_SUCCESS;
exit:
    if (model->name) {
        tinynn_free(model->name);
    }
    if (model->model_ptr) {
        tinynn_free(model->model_ptr);
    }
    if (model->is_own_tensor_memory && model->max_tensor_memroy) {
        tinynn_free(model->max_tensor_memroy);
    }
    if (model->weights) {
        for (int i = 0; i < model->nr_origin_weight; i++) {
            Tensor* weight = model->weights + i;
            if (weight->ptr && !weight->is_shared) {
                model->host_dev.free(weight->ptr);
            }
            if (weight->name)
                tinynn_free(weight->name);
        }
        tinynn_free(model->weights);
    }
    if (model->device_models) {
        for (int i = 0; i < model->nr_device_model; ++i) {
            tinynn_free(model->device_models[i]);
        }
        tinynn_free(model->device_models);
    }
    return TinyNN_ERROR_MODEL_PARSE;
}

// vim: syntax=cpp.doxygen
