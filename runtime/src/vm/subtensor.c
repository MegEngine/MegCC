/**
 * \file runtime/src/vm/subtensor.c
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
#if ENABLE_INST_SUBTENSOR || ENABLE_INST_SETSUBTENSOR
static TinyNNStatus sort_descs(IndexDesc* descs, int32_t nr_descs,
                               IndexDesc* flags) {
    //! sort the desc with decrease order
    for (int32_t i = 0; i < nr_descs; i++) {
        for (int32_t j = i + 1; j < nr_descs; j++) {
            if (descs[i].axis < descs[j].axis) {
                IndexDesc tmp = descs[j];
                descs[j] = descs[i];
                descs[i] = tmp;
                tmp = flags[j];
                flags[j] = flags[i];
                flags[i] = tmp;
            }
        }
    }
    return TinyNN_SUCCESS;
}

//! compute output shape and update the copied input layout
static uint32_t update_layout(Tensor** inputs, Tensor* input_copy,
                              Tensor* output, IndexDesc* descs,
                              IndexDesc* flags, uint32_t nr_desc) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < nr_desc; i++) {
        IndexDesc desc = descs[i];
        IndexDesc flag = flags[i];
        TINYNN_ASSERT_MSG(flag.axis == 0, "index describe axis must static.");
        int32_t axis = desc.axis;
        //! if index is valid
        if (flag.index != -1) {
            int32_t index = flag.index == 0
                                    ? desc.index
                                    : get_tensor_value(inputs[desc.index], 0);
            output->layout.dims[axis] = 1;
            //! remove this axis
            if (output->layout.nr_dim > 1) {
                for (int index = axis; index < output->layout.nr_dim - 1;
                     index++) {
                    output->layout.dims[index] = output->layout.dims[index + 1];
                }
                output->layout.nr_dim--;
            }
            input_copy->layout.dims[axis] = 1;
            offset += input_copy->layout.stride[axis] * index;
        } else {
            int32_t start = flag.start != 1
                                    ? desc.start
                                    : get_tensor_value(inputs[desc.start], 0);
            int32_t end = flag.end != 1 ? desc.end
                                        : get_tensor_value(inputs[desc.end], 0);
            end = (end < 0 ? end + input_copy->layout.dims[axis] + 1 : end);
            int32_t step = flag.step != 1
                                   ? desc.step
                                   : get_tensor_value(inputs[desc.step], 0);
            int32_t step_abs = (step < 0 ? -step : step);
            //! if step < 0 and start is not valid, default set to the max shape
            //! of the axis
            if (step < 0 && flag.start == -1) {
                start = inputs[0]->layout.dims[axis];
            }

            output->layout.dims[axis] = (end - start + step_abs - 1) / step_abs;
            input_copy->layout.dims[axis] = output->layout.dims[axis];
            offset += input_copy->layout.stride[axis] * start;
            input_copy->layout.stride[axis] *= step;
        }
    }
    //! init output stride
    output->layout.stride[output->layout.nr_dim - 1] = 1;
    for (int index = output->layout.nr_dim - 2; index >= 0; index--) {
        output->layout.stride[index] = output->layout.dims[index + 1] *
                                       output->layout.stride[index + 1];
    }
    //! offset the source ptr
    input_copy->ptr =
            (char*)(input_copy->ptr) +
            offset * dtype_length((input_copy)->dtype.type_enum, NULL);
    return offset;
}

#endif

#if ENABLE_INST_SUBTENSOR
static TinyNNStatus parse_subtensor(IndexDesc** descs,
                                    flatbuffers_generic_t fbs_desc) {
    int nr_desc = ns(IndexDesc_vec_len(fbs_desc));
    IndexDesc* ptr = tinynn_malloc(sizeof(IndexDesc) * nr_desc);
    *descs = ptr;
    for (int i = 0; i < nr_desc; i++) {
        ns(IndexDesc_table_t) fbs_index_desc =
                ns(IndexDesc_vec_at(fbs_desc, i));
        ptr[i].axis = ns(IndexDesc_axis(fbs_index_desc));
        ptr[i].start = ns(IndexDesc_start(fbs_index_desc));
        ptr[i].end = ns(IndexDesc_end(fbs_index_desc));
        ptr[i].step = ns(IndexDesc_step(fbs_index_desc));
        ptr[i].index = ns(IndexDesc_index(fbs_index_desc));
    }
    return TinyNN_SUCCESS;
}

static TinyNNStatus load_subtensor(flatbuffers_generic_t fbs_inst,
                                   Instruction* inst, VM* vm) {
    SubTensor* subtensor = &inst->workload.subtensor;
    ns(SubTensor_table_t) fbs_subtensor = (ns(SubTensor_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_SUBTENSOR;
    flatbuffers_int32_vec_t fbs_inputs = ns(SubTensor_inputs(fbs_subtensor));
    flatbuffers_int8_vec_t fbs_input_types =
            ns(SubTensor_input_types(fbs_subtensor));
    subtensor->nr_input = flatbuffers_int32_vec_len(fbs_inputs);

    int total_input = subtensor->nr_input;
    subtensor->inputs = tinynn_malloc(total_input * sizeof(Tensor*));
    DeviceModel* model = get_active_device_model(vm);
    LOG_DEBUG("\t subtensor inputs tensor number:%d\n", subtensor->nr_input);
    //! parse the input
    parase_inputs(subtensor->inputs, total_input, model, vm->model, fbs_inputs,
                  fbs_input_types);

    int32_t output_idx = ns(SubTensor_output(fbs_subtensor));
    subtensor->output = model->tensors + output_idx;

    ns(IndexDesc_vec_t) fbs_descs = ns(SubTensor_descs(fbs_subtensor));
    ns(IndexDesc_vec_t) fbs_flags = ns(SubTensor_flags(fbs_subtensor));
    parse_subtensor(&subtensor->descs, fbs_descs);
    parse_subtensor(&subtensor->flags, fbs_flags);
    subtensor->nr_descs = ns(IndexDesc_vec_len(fbs_descs));
    TINYNN_ASSERT_MSG(ns(IndexDesc_vec_len(fbs_descs)) ==
                          ns(IndexDesc_vec_len(fbs_flags)),
                  "The size of subtensor descs and flags is not equal.");
    sort_descs(subtensor->descs, subtensor->nr_descs, subtensor->flags);

    return TinyNN_SUCCESS;
}

static TinyNNStatus execute_subtensor(Instruction* inst, VM* vm) {
    Tensor **inputs = inst->workload.subtensor.inputs,
           *output = inst->workload.subtensor.output;
    SubTensor* subtensor = &inst->workload.subtensor;
    Tensor input_copy = *(inputs[0]);
    //! deduce output shape, and modify the input stride
    output->layout = input_copy.layout;
    output->dtype = inputs[0]->dtype;
    update_layout(inputs, &input_copy, output, subtensor->descs,
                  subtensor->flags, subtensor->nr_descs);
    //! alloc output
    TINYNN_ASSERT_MSG(output->is_dynamic,
                  "Subtensor output tensor should be dynamic.");
    alloc_tensor(output, vm);
    //! do subtensor
    size_t nr_elem = 1;
    for (int i = 0; i < output->layout.nr_dim; ++i) {
        nr_elem *= output->layout.dims[i];
    }

    NoconIter src_iter = init_iter(input_copy.layout);
    NoconIter dst_iter = init_iter(output->layout);
    if (dtype_length((input_copy).dtype.type_enum, NULL) == 1) {
        char* dst_data = output->ptr;
        char* src_data = input_copy.ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(input_copy.layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else if (dtype_length((input_copy).dtype.type_enum, NULL) == 2) {
        int16_t* dst_data = output->ptr;
        int16_t* src_data = input_copy.ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(input_copy.layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else if (dtype_length((input_copy).dtype.type_enum, NULL) == 4) {
        int32_t* dst_data = output->ptr;
        int32_t* src_data = input_copy.ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(input_copy.layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else {
        LOG_ERROR("unsupport dtype in subtensor.\n");
        return TinyNN_ERROR_UNSUPPORTED_DTYPE_TYPE;
    }
#if TINYNN_DUMP_TENSOR
    log_tensor(subtensor->output, "subtensor", subtensor->inputs[0]);
#endif
    return TinyNN_SUCCESS;
}
static TinyNNStatus destruct_subtensor(VM* vm, Instruction* inst) {
    FREE(inst->workload.subtensor.inputs);
    FREE(inst->workload.subtensor.descs);
    FREE(inst->workload.subtensor.flags);
    if (inst->workload.subtensor.output->is_dynamic) {
        DEV_FREE(inst->workload.subtensor.output->ptr, vm);
    }
    return TinyNN_SUCCESS;
}
void register_subtensor(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_SubTensor),
                                 &load_subtensor);
    vm_register_instruction_destruct(vm, TinyNN_INST_SUBTENSOR,
                                     &destruct_subtensor);
    vm_register_instruction_call(vm, TinyNN_INST_SUBTENSOR, &execute_subtensor);
}
#else
void register_subtensor(VM* vm) {}
#endif

#if ENABLE_INST_SETSUBTENSOR
static TinyNNStatus load_setsubtensor(flatbuffers_generic_t fbs_inst,
                                      Instruction* inst, VM* vm) {
    SetSubTensor* set_subtensor = &inst->workload.set_subtensor;
    ns(SetSubTensor_table_t) fbs_set_subtensor =
            (ns(SetSubTensor_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_SETSUBTENSOR;
    flatbuffers_int32_vec_t fbs_inputs =
            ns(SetSubTensor_inputs(fbs_set_subtensor));
    flatbuffers_int8_vec_t fbs_input_types =
            ns(SetSubTensor_input_types(fbs_set_subtensor));
    set_subtensor->nr_input = flatbuffers_int32_vec_len(fbs_inputs);

    int total_input = set_subtensor->nr_input;
    set_subtensor->inputs = tinynn_malloc(total_input * sizeof(Tensor*));
    DeviceModel* model = get_active_device_model(vm);
    LOG_DEBUG("\t setsubtensor inputs tensor number:%d\n",
              set_subtensor->nr_input);
    //! parse the input
    parase_inputs(set_subtensor->inputs, total_input, model, vm->model,
                  fbs_inputs, fbs_input_types);

    int32_t output_idx = ns(SetSubTensor_output(fbs_set_subtensor));
    set_subtensor->output = model->tensors + output_idx;

    ns(IndexDesc_vec_t) fbs_descs = ns(SetSubTensor_descs(fbs_set_subtensor));
    ns(IndexDesc_vec_t) fbs_flags = ns(SetSubTensor_flags(fbs_set_subtensor));
    parse_subtensor(&set_subtensor->descs, fbs_descs);
    parse_subtensor(&set_subtensor->flags, fbs_flags);
    TINYNN_ASSERT_MSG(ns(IndexDesc_vec_len(fbs_descs)) ==
                          ns(IndexDesc_vec_len(fbs_flags)),
                  "The size of setsubtensor descs and flags is not equal.");
    set_subtensor->nr_descs = ns(IndexDesc_vec_len(fbs_descs));
    sort_descs(set_subtensor->descs, set_subtensor->nr_descs,
               set_subtensor->flags);

    return TinyNN_SUCCESS;
}
static TinyNNStatus execute_setsubtensor(Instruction* inst, VM* vm) {
    Tensor **inputs = inst->workload.set_subtensor.inputs,
           *output = inst->workload.set_subtensor.output;
    SetSubTensor* set_subtensor = &inst->workload.set_subtensor;
    Tensor dymmy_src = *(inputs[0]);
    Tensor* src = inputs[0];
    Tensor* value = inputs[1];
    const int elem_bytes = dtype_length(value->dtype.type_enum, NULL);
    output->layout = inputs[0]->layout;
    //! alloc output
    size_t length_in_byte = tensor_length_in_byte(src);
    alloc_tensor(output, vm);
    //! if src_len == value_len and contig, fast forward
    bool is_src_contig = is_contiguous(src->layout);
    bool is_value_contig = is_contiguous(value->layout);
    size_t length_of_value = tensor_length_in_byte(value);
    if (is_src_contig && is_value_contig && length_of_value == length_in_byte) {
        memcpy(output->ptr, value->ptr, length_of_value);
        return TinyNN_SUCCESS;
    }
    //! copy all memory to dst
    memcpy(output->ptr, src->ptr, length_in_byte);
    //! deduce output shape, and modify the input stride
    uint32_t offset =
            update_layout(inputs, output, &dymmy_src, set_subtensor->descs,
                          set_subtensor->flags, set_subtensor->nr_descs);
    //! do set_subtensor
    size_t nr_elem = 1;
    for (int i = 0; i < output->layout.nr_dim; ++i) {
        nr_elem *= output->layout.dims[i];
    }

    NoconIter src_iter = init_iter(value->layout);
    NoconIter dst_iter = init_iter(output->layout);
    if (elem_bytes == 1) {
        char* dst_data = output->ptr;
        char* src_data = value->ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(value->layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else if (elem_bytes == 2) {
        int16_t* dst_data = output->ptr;
        int16_t* src_data = value->ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(value->layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else if (elem_bytes == 4) {
        int32_t* dst_data = output->ptr;
        int32_t* src_data = value->ptr;
        for (size_t i = 0; i < nr_elem; ++i) {
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(value->layout, &src_iter);
            inc_iter(output->layout, &dst_iter);
        }
    } else {
        LOG_ERROR("unsupport dtype in set_subtensor.\n");
        return TinyNN_ERROR_UNSUPPORTED_DTYPE_TYPE;
    }
    output->ptr = (char*)(output->ptr) -
                  offset * dtype_length(output->dtype.type_enum, NULL);
    output->layout = src->layout;
#if TINYNN_DUMP_TENSOR
    log_tensor(set_subtensor->output, "set_subtensor",
               set_subtensor->inputs[0]);
#endif

    return TinyNN_SUCCESS;
}
static TinyNNStatus destruct_setsubtensor(VM* vm, Instruction* inst) {
    FREE(inst->workload.set_subtensor.inputs);
    FREE(inst->workload.set_subtensor.descs);
    FREE(inst->workload.set_subtensor.flags);
    if (inst->workload.set_subtensor.output->is_dynamic) {
        DEV_FREE(inst->workload.set_subtensor.output->ptr, vm);
    }
    return TinyNN_SUCCESS;
}
void register_setsubtensor(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_SetSubTensor),
                                 &load_setsubtensor);
    vm_register_instruction_destruct(vm, TinyNN_INST_SETSUBTENSOR,
                                     &destruct_setsubtensor);
    vm_register_instruction_call(vm, TinyNN_INST_SETSUBTENSOR,
                                 &execute_setsubtensor);
}
#else
void register_setsubtensor(VM* vm) {}
#endif

// vim: syntax=cpp.doxygen
