/**
 * \file runtime/src/vm/broadcast.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "common.h"
#include "init.h"
#include "math.h"
#include "parse.h"
#include "vm.h"
#include "vm/registry.h"
#if ENABLE_INST_BROADCAST
static TinyNNStatus load_broadcast(flatbuffers_generic_t fbs_inst,
                                   Instruction* inst, VM* vm) {
    BroadCast* broadcast = &inst->workload.broadcast;
    ns(BroadCast_table_t) fbs_broadcast = (ns(BroadCast_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_BROADCAST;
    flatbuffers_int32_vec_t fbs_inputs = ns(BroadCast_inputs(fbs_broadcast));
    flatbuffers_int8_vec_t fbs_input_types =
            ns(BroadCast_input_types(fbs_broadcast));
    int32_t nr_input = flatbuffers_int32_vec_len(fbs_inputs);
    TINYNN_ASSERT_MSG(nr_input == 2, "BroadCast input is not 2.");

    DeviceModel* model = get_active_device_model(vm);
    LOG_DEBUG("\t BroadCast inputs tensor number:%d\n", nr_input);
    //! parse the input
    parase_inputs(broadcast->inputs, nr_input, model, vm->model, fbs_inputs,
                  fbs_input_types);
    int32_t output_idx = ns(BroadCast_output(fbs_broadcast));
    broadcast->output = model->tensors + output_idx;
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute_broadcast(Instruction* inst, VM* vm) {
    Tensor* output = inst->workload.broadcast.output;
    BroadCast* broadcast = &inst->workload.broadcast;
    Tensor input = *broadcast->inputs[0];
    Tensor tshape = *broadcast->inputs[1];
    Layout origin_layout = input.layout;
    uint32_t* shape_ptr = tshape.ptr;
    uint32_t nr_dim = tshape.layout.dims[0];
    output->layout.nr_dim = nr_dim;
    for (uint32_t i = 0; i < nr_dim; i++) {
        output->layout.dims[i] = shape_ptr[i];
    }
    output->dtype = input.dtype;
    //! init output stride
    output->layout.stride[output->layout.nr_dim - 1] = 1;
    for (int index = output->layout.nr_dim - 2; index >= 0; index--) {
        output->layout.stride[index] = output->layout.dims[index + 1] *
                                       output->layout.stride[index + 1];
    }
    //! alloc output
    alloc_tensor(output, vm);
    //! broadcast input
    TINYNN_ASSERT_MSG(output->layout.nr_dim >= input.layout.nr_dim,
                  "Broadcast output dim should large than input.");
    int32_t dim_diff = output->layout.nr_dim - input.layout.nr_dim;
    Layout in_layout;
    in_layout.nr_dim = output->layout.nr_dim;
    int32_t i = 0;
    for (; i < dim_diff; i++) {
        in_layout.dims[i] = output->layout.dims[i];
        in_layout.stride[i] = 0;
    }
    for (; i < output->layout.nr_dim; i++) {
        if (input.layout.dims[i] == 1 && output->layout.dims[i] != 1) {
            in_layout.stride[i] = 0;
        } else {
            in_layout.stride[i] = input.layout.stride[i - dim_diff];
        }
        in_layout.dims[i] = output->layout.dims[i];
    }
    input.layout = in_layout;

    //! do broadcast copy
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
        LOG_ERROR("unsupport dtype in broadcast.\n");
        return TinyNN_ERROR_UNSUPPORTED_DTYPE_TYPE;
    }
#if TINYNN_DUMP_TENSOR
    log_tensor(broadcast->output, "broadcast", broadcast->inputs[0]);
#endif
    return TinyNN_SUCCESS;
}

static TinyNNStatus load_shape_of(flatbuffers_generic_t fbs_inst, Instruction* inst,
                         VM* vm) {
    ShapeOf* shape_of= &inst->workload.shape_of;
    ns(ShapeOf_table_t) fbs_shapeof = (ns(ShapeOf_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_SHAPEOF;
    int32_t input_idx = ns(ShapeOf_input(fbs_shapeof));
    int32_t input_type = ns(ShapeOf_input_type(fbs_shapeof));
    DeviceModel* model = get_active_device_model(vm);
    if (input_type == ns(TensorType_TENSOR)) {
        shape_of->input = model->tensors + input_idx;
    } else {
        shape_of->input = vm->model->weights + input_idx;
    }
    int32_t output_idx = ns(ShapeOf_output(fbs_shapeof));
    shape_of->output = model->tensors + output_idx;
    LOG_DEBUG("\t Load ShapeOf inputs tensor id:%d, output tensor id:%d\n",
              input_idx, output_idx);
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute_shape_of(Instruction* inst, VM* vm) {
    Tensor* output = inst->workload.shape_of.output;
    ShapeOf* shape_of= &inst->workload.shape_of;
    Layout layout = shape_of->input->layout;
    output->dtype.type_enum = TinyNN_INT;
    //! init output stride
    output->layout.nr_dim = 1;
    output->layout.dims[0] = layout.nr_dim;
    output->layout.stride[0] = 1;
    //! alloc output
    TINYNN_ASSERT_MSG(output->is_dynamic,
		    "dimshuffle output tensor should be dynamic.");
    alloc_tensor(output, vm);
    //! broadcast input
    int32_t* ptr = output->ptr;
    for (int i = 0; i < layout.nr_dim; i++) {
        ptr[i] = layout.dims[i];
    }
#if TINYNN_DUMP_TENSOR
    log_tensor(shape_of->output, "shape_of", shape_of->input);
#endif
    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct_broadcast(VM* vm, Instruction* inst) {
    if (inst->workload.broadcast.output->is_dynamic) {
        DEV_FREE(inst->workload.broadcast.output->ptr, vm);
    }
    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct_shape_of(VM* vm, Instruction* inst) {
    if (inst->workload.shape_of.output->is_dynamic) {
        DEV_FREE(inst->workload.shape_of.output->ptr, vm);
    }
    return TinyNN_SUCCESS;
}

void register_broadcast_shape_of(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_BroadCast),
                                 &load_broadcast);
    vm_register_instruction_call(vm, TinyNN_INST_BROADCAST, &execute_broadcast);
    vm_register_instruction_destruct(vm, TinyNN_INST_BROADCAST,
                                     &destruct_broadcast);

    vm_register_instruction_load(vm, ns(Instruction_ShapeOf), &load_shape_of);
    vm_register_instruction_call(vm, TinyNN_INST_SHAPEOF, &execute_shape_of);
    vm_register_instruction_destruct(vm, TinyNN_INST_SHAPEOF,
                                     &destruct_shape_of);
}
#else
void register_broadcast_shape_of(VM* vm) {}
#endif
// vim: syntax=cpp.doxygen
