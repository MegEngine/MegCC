/**
 * \file runtime/src/vm/reshape.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "init.h"
#include "parse.h"
#include "vm.h"
#include "vm/common.h"
#include "vm/instruction.h"
#include "vm/registry.h"
#if ENABLE_INST_RESHAPE
static TinyNNStatus load(flatbuffers_generic_t fbs_inst, Instruction* inst,
                         VM* vm) {
    Reshape* reshape = &inst->workload.reshape;
    ns(Reshape_table_t) fbs_reshape = (ns(Reshape_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_RESHAPE;
    flatbuffers_int32_vec_t fbs_inputs = ns(Reshape_inputs(fbs_reshape));
    flatbuffers_int8_vec_t fbs_input_types =
            ns(Reshape_input_types(fbs_reshape));
    int32_t nr_input = flatbuffers_int32_vec_len(fbs_inputs);
    TINYNN_ASSERT_MSG(nr_input == 2, "Reshape input is not 2.");
    reshape->inputs = tinynn_malloc(nr_input * sizeof(Tensor*));
    DeviceModel* model = get_active_device_model(vm);
    parase_inputs(reshape->inputs, nr_input, model, vm->model, fbs_inputs,
                  fbs_input_types);

    int32_t output_idx = ns(Reshape_output(fbs_reshape));
    reshape->output = model->tensors + output_idx;
    LOG_DEBUG("parse reshape, input idx\n");
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute(Instruction* inst, VM* vm) {
    Tensor *input = inst->workload.reshape.inputs[0],
           *shape_tensor = inst->workload.reshape.inputs[1],
           *output = inst->workload.reshape.output;
    output->layout = input->layout;
    output->layout.nr_dim = shape_tensor->layout.dims[0];
    uint32_t nr_elem = 1;
    for (int i = 0; i < input->layout.nr_dim; ++i) {
        nr_elem *= input->layout.dims[i];
    }
    int* tshape = shape_tensor->ptr;
    int neg_axis = -1;
    for (int i = 0; i < shape_tensor->layout.dims[0]; ++i) {
        if (tshape[i] > 0) {
            output->layout.dims[i] = tshape[i];
            nr_elem = nr_elem / output->layout.dims[i];
        } else {
            TINYNN_ASSERT(tshape[i] == -1 && neg_axis == -1);
            neg_axis = i;
        }
    }
    if (neg_axis >= 0) {
        output->layout.dims[neg_axis] = nr_elem;
    }
    force_layout_contiguous(&(output->layout));
    output->ptr = input->ptr;
    output->size = tensor_length_in_byte(output);
    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct(VM* vm, Instruction* inst) {
    FREE(inst->workload.reshape.inputs);
    return TinyNN_SUCCESS;
}

void register_reshape(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_Reshape), &load);
    vm_register_instruction_call(vm, TinyNN_INST_RESHAPE, &execute);
    vm_register_instruction_destruct(vm, TinyNN_INST_RESHAPE, &destruct);
}
#else
void register_reshape(VM* vm) {}
#endif
// vim: syntax=cpp.doxygen
