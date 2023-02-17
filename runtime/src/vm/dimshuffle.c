/**
 * \file runtime/src/vm/dimshuffle.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "common.h"
#include "init.h"
#include "parse.h"
#include "vm.h"
#include "vm/registry.h"
#if ENABLE_INST_DIMSHUFFLE
static TinyNNStatus load(flatbuffers_generic_t fbs_inst, Instruction* inst, VM* vm) {
    Dimshuffle* dimshuffle = &inst->workload.dimshuffle;
    ns(Dimshuffle_table_t) fbs_dimshuffle = (ns(Dimshuffle_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_DIMSHUFFLE;
    int32_t input_idx = ns(Dimshuffle_input(fbs_dimshuffle));
    int32_t input_type = ns(Dimshuffle_input_type(fbs_dimshuffle));
    DeviceModel* model = get_active_device_model(vm);
    if (input_type == ns(TensorType_TENSOR)) {
        dimshuffle->input = model->tensors + input_idx;
    } else {
        dimshuffle->input = vm->model->weights + input_idx;
    }
    int32_t output_idx = ns(Dimshuffle_output(fbs_dimshuffle));
    dimshuffle->output = model->tensors + output_idx;
    LOG_DEBUG(
            "\t dimshuffle inputs tensor id:%d, output tensor id:%d\n", input_idx,
            output_idx);

    flatbuffers_int32_vec_t fbs_pattern = ns(Dimshuffle_pattern(fbs_dimshuffle));
    dimshuffle->pattern_dim = flatbuffers_int32_vec_len(fbs_pattern);
    for (int32_t idx = 0; idx < dimshuffle->pattern_dim; idx++) {
        dimshuffle->pattern[idx] = flatbuffers_int32_vec_at(fbs_pattern, idx);
    }
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute(Instruction* inst, VM* vm) {
    Tensor* output = inst->workload.dimshuffle.output;
    Dimshuffle* dimshuffle = &inst->workload.dimshuffle;
    int32_t nr_dim = dimshuffle->pattern_dim;
    Tensor input = *dimshuffle->input;
    Layout origin_layout = input.layout;
    TINYNN_ASSERT(nr_dim == origin_layout.nr_dim);
    for (int32_t i = 0; i < nr_dim; i++) {
        int32_t axis = dimshuffle->pattern[i];
        input.layout.dims[i] = origin_layout.dims[axis];
        input.layout.stride[i] = origin_layout.stride[axis];
    }
    output->dtype = input.dtype;
    output->layout = input.layout;
    //! init output stride
    output->layout.stride[output->layout.nr_dim - 1] = 1;
    for (int index = output->layout.nr_dim - 2; index >= 0; index--) {
        output->layout.stride[index] =
                output->layout.dims[index + 1] * output->layout.stride[index + 1];
    }
    alloc_tensor(output, vm);
    //! do dimshuffle naive
    size_t nr_elem = 1;
    for (int i = 0; i < output->layout.nr_dim; ++i) {
        nr_elem *= output->layout.dims[i];
    }
    NoconIter src_iter = init_iter(input.layout);
    NoconIter dst_iter = init_iter(output->layout);
    if (dtype_length((input).dtype.type_enum, NULL) == 1) {
        char* dst_data = output->ptr;
        char* src_data = input.ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(input.layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else if (dtype_length((input).dtype.type_enum, NULL) == 2) {
        int16_t* dst_data = output->ptr;
        int16_t* src_data = input.ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(input.layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else if (dtype_length(input.dtype.type_enum, NULL) == 4) {
        int32_t* dst_data = output->ptr;
        int32_t* src_data = input.ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(input.layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else {
        LOG_ERROR("unsupport dtype in dimshuffle.\n");
        return TinyNN_ERROR_UNSUPPORTED_DTYPE_TYPE;
    }
#if TINYNN_DUMP_TENSOR
    log_tensor(dimshuffle->output, "dimshuffle", dimshuffle->input);
#endif
    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct(VM* vm, Instruction* inst) {
    if (inst->workload.dimshuffle.output->is_dynamic) {
        DEV_FREE(inst->workload.dimshuffle.output->ptr, vm);
    }
    return TinyNN_SUCCESS;
}

void register_dimshuffle(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_Dimshuffle), &load);
    vm_register_instruction_destruct(vm, TinyNN_INST_DIMSHUFFLE, &destruct);
    vm_register_instruction_call(vm, TinyNN_INST_DIMSHUFFLE, &execute);
}
#else
void register_dimshuffle(VM* vm) {}
#endif
// vim: syntax=cpp.doxygen
