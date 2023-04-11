#ifndef VM_COMMON_H
#define VM_COMMON_H

#include <stdbool.h>
#include "../init.h"
#include "data_struct.h"
#include "parse.h"
#include "tensor_util.h"
#include "vm.h"

static inline TinyNNStatus alloc_tensor_opt(Tensor* tensor, const RuntimeOpt* opt) {
    if (tensor->is_dynamic) {
        size_t length_in_byte = tensor_length_in_byte(tensor);
        if (!tensor->ptr || tensor->size < length_in_byte) {
            if (tensor->ptr)
                opt->device->free(tensor->ptr);
            tensor->ptr = opt->device->malloc(length_in_byte);
            tensor->size = length_in_byte;
            tensor->offset = 0;
        }
    } else {
        LOG_DEBUG("tensor is static, no memory allocated.\n");
    }
    return TinyNN_SUCCESS;
}

static inline DeviceModel* get_active_device_model(VM* vm) {
    return vm->model->device_models[vm->model->active_device_model_idx];
}

static inline TinyNNStatus alloc_tensor(Tensor* tensor, VM* vm) {
    DeviceModel* model = get_active_device_model(vm);
    return alloc_tensor_opt(tensor, &(model->opt));
}

static inline void free_dev_ptr(void* dev_ptr, VM* vm) {
    if (dev_ptr) {
        DeviceModel* model = get_active_device_model(vm);
        model->opt.device->free(dev_ptr);
    }
}

static TinyNNStatus parase_inputs(
        Tensor** inputs, int nr_input, DeviceModel* model, CombineModel* com_model,
        flatbuffers_int32_vec_t fbs_inputs, flatbuffers_int8_vec_t fbs_input_types) {
    for (int i = 0; i < nr_input; i++) {
        int index = flatbuffers_int32_vec_at(fbs_inputs, i);
        int input_type = flatbuffers_int8_vec_at(fbs_input_types, i);
        if (input_type == ns(TensorType_TENSOR)) {
            *(inputs + i) = model->tensors + index;
            LOG_DEBUG(
                    "\t\tinput %d tensor index=%d offset %zu size %zu\n", i, index,
                    (*(inputs + i))->offset, (*(inputs + i))->size);
        } else {
            *(inputs + i) = com_model->weights + index;
            LOG_DEBUG("\t\tinput %d weight index=%d \n", i, index);
        }
    }
    return TinyNN_SUCCESS;
}

static inline void print_tensor_layout(Layout layout) {
    LOG_DEBUG_NO_PREFIX("layout is [");
    for (int i = 0; i < layout.nr_dim; ++i) {
        LOG_DEBUG_NO_PREFIX("%d(%d), ", layout.dims[i], layout.stride[i]);
    }
    LOG_DEBUG_NO_PREFIX("]\n");
}

#if TINYNN_DUMP_TENSOR
static inline void print_tensor(Tensor* output) {
    TINYNN_ASSERT(output);
    float avg = 0;
    int nr_elem = 0;
    int print_num = 5;
    size_t total_length = tensor_length_in_byte(output);
    Layout layout = output->layout;
    LOG_DEBUG("  %s [", output->name);
    if (output->dtype.type_enum == TinyNN_FLOAT) {
        nr_elem = total_length / sizeof(float);
        float sum = 0;
        int offset = 0;
        if (layout.stride[layout.nr_dim - 1] < 0) {
            offset = layout.dims[layout.nr_dim - 1] - 1;
        }
        float* output_ptr = (float*)(output->ptr) - offset;
        for (int i = 0; i < nr_elem; ++i) {
            sum += output_ptr[i];
            if (i < print_num) {
                LOG_DEBUG_NO_PREFIX("%f, ", output_ptr[i]);
            }
        }
        avg = sum / nr_elem;
    } else if (
            output->dtype.type_enum == TinyNN_INT8 ||
            output->dtype.type_enum == TinyNN_QINT8) {
        nr_elem = total_length / sizeof(int8_t);
        float sum = 0;
        for (int i = 0; i < nr_elem; ++i) {
            sum += ((int8_t*)(output->ptr))[i];
            if (i < print_num) {
                LOG_DEBUG_NO_PREFIX("%d, ", ((int8_t*)(output->ptr))[i]);
            }
        }
        avg = sum / nr_elem;
    } else if (
            output->dtype.type_enum == TinyNN_UINT8 ||
            output->dtype.type_enum == TinyNN_QUINT8) {
        nr_elem = total_length / sizeof(int8_t);
        float sum = 0;
        for (int i = 0; i < nr_elem; ++i) {
            sum += ((uint8_t*)(output->ptr))[i];
            if (i < print_num) {
                LOG_DEBUG_NO_PREFIX("%d, ", ((uint8_t*)(output->ptr))[i]);
            }
        }
        avg = sum / nr_elem;
    } else if (
            output->dtype.type_enum == TinyNN_INT ||
            output->dtype.type_enum == TinyNN_QINT32) {
        nr_elem = total_length / sizeof(int32_t);
        float sum = 0;
        for (int i = 0; i < nr_elem; ++i) {
            sum += ((int32_t*)(output->ptr))[i];
            if (i < print_num) {
                LOG_DEBUG_NO_PREFIX("%d, ", ((int32_t*)(output->ptr))[i]);
            }
        }
        avg = sum / nr_elem;
    } else {
        TINYNN_ASSERT_MSG(0, "not supported dtype.");
    }
    LOG_DEBUG_NO_PREFIX("]");
    LOG_DEBUG_NO_PREFIX(
            "avg %f  [%d(%d) %d(%d) %d(%d) %d(%d) %d(%d)] %d dim used\n", avg,
            output->layout.dims[0], output->layout.stride[0], output->layout.dims[1],
            output->layout.stride[1], output->layout.dims[2], output->layout.stride[2],
            output->layout.dims[3], output->layout.stride[3], output->layout.dims[4],
            output->layout.stride[4], output->layout.nr_dim);
}

static inline void log_tensor(Tensor* output, char* post_fix, Tensor* input) {
    TINYNN_ASSERT(output);
    print_tensor(output);
    static int dump_cnt = 0;
    size_t total_length = tensor_length_in_byte(output);
    FILE* write_ptr;
    //! may overflow
    char name[1024];
    int dump_name_len = 0;
    dump_name_len = sprintf(name, "dump/%d__%s_%s", dump_cnt++, output->name, post_fix);
    for (int i = 0; i < output->layout.nr_dim; ++i) {
        dump_name_len += sprintf(name + dump_name_len, "_%d", output->layout.dims[i]);
    }
    write_ptr = tinynn_fopen(name, "wb");  // w for write, b for binary
    tinynn_fwrite(output->ptr, 1, total_length, write_ptr);
    tinynn_fclose(write_ptr);

    Layout src_layout = input->layout;
    Layout dst_layout = output->layout;
    void* src_data = input->ptr;
    void* dst_data = output->ptr;
    LOG_DEBUG(
            "dump %s %p to %p dim %d [%u,%u,%u,%u,%u](%d,%d,%d,%d,%d) "
            "to "
            "dim %d [%u,%u,%u,%u,%u](%d,%d,%d,%d,%d)\n",
            name, src_data, dst_data, src_layout.nr_dim, src_layout.dims[0],
            src_layout.dims[1], src_layout.dims[2], src_layout.dims[3],
            src_layout.dims[4], src_layout.stride[0], src_layout.stride[1],
            src_layout.stride[2], src_layout.stride[3], src_layout.stride[4],
            dst_layout.nr_dim, dst_layout.dims[0], dst_layout.dims[1],
            dst_layout.dims[2], dst_layout.dims[3], dst_layout.dims[4],
            dst_layout.stride[0], dst_layout.stride[1], dst_layout.stride[2],
            dst_layout.stride[3], dst_layout.stride[4]);

    dump_name_len = sprintf(name + dump_name_len, "_input0");
    write_ptr = tinynn_fopen(name, "wb");  // w for write, b for binary
    tinynn_fwrite(input->ptr, 1, tensor_length_in_byte(input), write_ptr);
    tinynn_fclose(write_ptr);
    LOG_DEBUG("\n");
}
#endif

#endif  // VM_COMMON_H

// vim: syntax=cpp.doxygen
