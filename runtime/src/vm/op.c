/**
 * \file runtime/src/vm/op.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "common.h"
#include "init.h"
#include "kernels.h"
#include "parse.h"
#include "vm.h"
#include "vm/registry.h"

#if TINYNN_DUMP_TENSOR
#include <stdlib.h>
#endif
static TinyNNStatus parse_opr(DeviceModel* model, Opr* opr,
                              CombineModel* combine_model,
                              ns(Opr_table_t) fbs_opr) {
    //! name and opr type
    opr->name = get_string(ns(Opr_name(fbs_opr)));
    opr->type = get_string(ns(Opr_type(fbs_opr)));

    //! init function and kernel function
    opr->init_func = ns(Opr_init_id(fbs_opr));
    opr->kernel_func = ns(Opr_kernel_id(fbs_opr));
    opr->deduce_shape_func = ns(Opr_deduce_id(fbs_opr));
    opr->workspace_func = ns(Opr_workspace_id(fbs_opr));

    LOG_DEBUG("Operator info: symbol=%s, kernel id=%d, init_id=%d\n", opr->type,
              opr->kernel_func, opr->init_func);
    //! inputs
    flatbuffers_int32_vec_t fbs_inputs = ns(Opr_inputs(fbs_opr));
    flatbuffers_int8_vec_t fbs_input_types = ns(Opr_input_types(fbs_opr));
    opr->nr_input = flatbuffers_int32_vec_len(fbs_inputs);
    //! weights
    int total_input = opr->nr_input;
    opr->inputs = tinynn_malloc(total_input * sizeof(Tensor*));
    memset(opr->inputs, 0, total_input * sizeof(Tensor*));
    LOG_DEBUG("\t Opr inputs tensor number:%d\n", opr->nr_input);
    //! parse the input
    parase_inputs(opr->inputs, total_input, model, combine_model, fbs_inputs,
                  fbs_input_types);
    //! outputs
    flatbuffers_int32_vec_t fbs_outputs = ns(Opr_outputs(fbs_opr));
    opr->nr_output = flatbuffers_int32_vec_len(fbs_outputs);
    opr->outputs = tinynn_malloc(opr->nr_output * sizeof(Tensor*));
    memset(opr->outputs, 0, opr->nr_output * sizeof(Tensor*));
    LOG_DEBUG("\t Opr outputs tensor number:%d\n", opr->nr_output);
    for (int i = 0; i < opr->nr_output; i++) {
        int index = flatbuffers_int32_vec_at(fbs_outputs, i);
        *(opr->outputs + i) = model->tensors + index;
        LOG_DEBUG(
                "\t\toutput %d is tensor, tensor index=%d offset %zu size "
                "%zu\n",
                i, index, (*(opr->outputs + i))->offset,
                (*(opr->outputs + i))->size);
    }

    //! workspace
    ns(Workspace_table_t) fbs_workspace = ns(Opr_workspace(fbs_opr));
    opr->workspace.size = ns(Workspace_size(fbs_workspace));
    opr->workspace.offset = ns(Workspace_offset(fbs_workspace));
    //! workspace ptr will be set in init_model_memory
    opr->workspace.ptr = NULL;
    LOG_DEBUG("\tworkspace size is %zu offset %zu\n", opr->workspace.size,
              ns(Workspace_offset(fbs_workspace)));
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute_single_opr(const Opr* opr, const RuntimeOpt* opt) {
    LOG_DEBUG("execute kernel name:%s\n", opr->type);
    if (opr->deduce_shape_func >= 0) {
        LOG_DEBUG("\tTry deduce shape, used only in dynamic shape op\n");
        DeduceFunc deduce = deduce_func[opr->deduce_shape_func];
        deduce(opr->inputs, opr->nr_input, opr->outputs, opr->nr_output);
        for (int i = 0; i < opr->nr_output; ++i) {
            alloc_tensor_opt(opr->outputs[i], opt);
        }
    }
    LOG_DEBUG("\tWith inputs tensor number:%d\n", opr->nr_input);
#if TINYNN_DUMP_TENSOR
    for (int i = 0; i < opr->nr_input; i++) {
        LOG_DEBUG("\t\tid %d:%s layout:", i, opr->inputs[i]->name);
        Layout layout = opr->inputs[i]->layout;
        for (int dim = 0; dim < layout.nr_dim; ++dim) {
            LOG_DEBUG_NO_PREFIX("%d(%d), ", layout.dims[dim],
                                layout.stride[dim]);
        }
        LOG_DEBUG_NO_PREFIX("dtype = %d, scale = %f, ptr = %p, offset %zu\n",
                            opr->inputs[i]->dtype.type_enum,
                            opr->inputs[i]->dtype.param.scale,
                            opr->inputs[i]->ptr, opr->inputs[i]->offset);
        print_tensor(opr->inputs[i]);
    }
    LOG_DEBUG("\tWith outputs tensor number:%d\n", opr->nr_output);
    for (int i = 0; i < opr->nr_output; i++) {
        LOG_DEBUG("\t\tid %d:%s layout:", i, (*(opr->outputs + i))->name);
        Layout layout = opr->outputs[i]->layout;
        for (int dim = 0; dim < layout.nr_dim; ++dim) {
            LOG_DEBUG_NO_PREFIX("%d(%d), ", layout.dims[dim],
                                layout.stride[dim]);
        }
        LOG_DEBUG_NO_PREFIX("dtype = %d, scale = %f, ptr = %p, offset %zu\n",
                            opr->outputs[i]->dtype.type_enum,
                            opr->outputs[i]->dtype.param.scale,
                            opr->outputs[i]->ptr, opr->outputs[i]->offset);
    }
#endif
    int kernel_index = opr->kernel_func;
    if (kernel_index >= NR_KERNELS) {
        LOG_ERROR(
                "Kernel function index %d is out of range, max is "
                "%d.\n",
                kernel_index, NR_KERNELS);
        return TinyNN_ERROR_OUT_OF_RANGE;
    }
    KernelFunc kernel = kernels[kernel_index];
#if TINYNN_SANITY_ALLOC
    Tensor inputs_alloc[opr->nr_input];
    Tensor* inputs_alloc_ptr[opr->nr_input];
    int input_offset_fix[opr->nr_input];
    for (int i = 0; i < opr->nr_input; i++) {
        inputs_alloc[i] = *(opr->inputs[i]);
        inputs_alloc[i].ptr = tinynn_malloc(inputs_alloc[i].size);
        int offset_fix = 0;
        Layout layout = inputs_alloc[i].layout;
        int dtype_len = dtype_length(inputs_alloc[i].dtype.type_enum, NULL);
        for (int dim = 0; dim < layout.nr_dim; ++dim) {
            if (layout.stride[dim] < 0) {
                offset_fix += -layout.stride[dim] * (layout.dims[dim] - 1) *
                              dtype_len;
            }
        }
        input_offset_fix[i] = offset_fix;
        memcpy(inputs_alloc[i].ptr, opr->inputs[i]->ptr - offset_fix,
               inputs_alloc[i].size);
        inputs_alloc[i].ptr += offset_fix;
        inputs_alloc_ptr[i] = &inputs_alloc[i];
    }
    Tensor outputs_alloc[opr->nr_output];
    Tensor* outputs_alloc_ptr[opr->nr_output];
    for (int i = 0; i < opr->nr_output; i++) {
        outputs_alloc[i] = *opr->outputs[i];
        outputs_alloc[i].ptr = tinynn_malloc(outputs_alloc[i].size);
        memcpy(outputs_alloc[i].ptr, opr->outputs[i]->ptr,
               outputs_alloc[i].size);
        outputs_alloc_ptr[i] = &outputs_alloc[i];
    }
    Workspace workspace;
    workspace.size = opr->workspace.size;
    workspace.ptr = tinynn_malloc(opr->workspace.size);
    TinyNNStatus error =
            kernel(inputs_alloc_ptr, opr->nr_input, outputs_alloc_ptr,
                   opr->nr_output, &workspace, opt);
    for (int i = 0; i < opr->nr_input; i++) {
        int fix_offset = input_offset_fix[i];
        memcpy(opr->inputs[i]->ptr - fix_offset,
               inputs_alloc[i].ptr - fix_offset, inputs_alloc[i].size);
        FREE(inputs_alloc[i].ptr - fix_offset);
    }
    for (int i = 0; i < opr->nr_output; i++) {
        memcpy(opr->outputs[i]->ptr, outputs_alloc[i].ptr,
               outputs_alloc[i].size);
        FREE(outputs_alloc[i].ptr);
    }
    FREE(workspace.ptr);

#else
    TinyNNStatus error = kernel(opr->inputs, opr->nr_input, opr->outputs,
                                opr->nr_output, &opr->workspace, opt);
#endif
#if TINYNN_DUMP_TENSOR
    log_tensor(opr->outputs[0], opr->type, opr->inputs[0]);
#endif
    if (error != TinyNN_SUCCESS) {
        LOG_ERROR("kernel compute error in opr: %s\n", opr->name);
    }
    return error;
}

static TinyNNStatus load(flatbuffers_generic_t fbs_inst, Instruction* inst,
                         VM* vm) {
    Opr* opr = &inst->workload.opr;
    ns(Opr_table_t) fbs_opr = (ns(Opr_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_OPR;
    DeviceModel* model = get_active_device_model(vm);
    if (parse_opr(model, opr, vm->model, fbs_opr) != TinyNN_SUCCESS) {
        LOG_ERROR("parse opr error!\n");
        return TinyNN_ERROR_MODEL_PARSE;
    }
    return TinyNN_SUCCESS;
}

static TinyNNStatus execute(Instruction* inst, VM* vm) {
    DeviceModel* model = get_active_device_model(vm);
    return execute_single_opr(&inst->workload.opr, &model->opt);
}

static TinyNNStatus destruct(VM* vm, Instruction* inst) {
    Opr* opr = &inst->workload.opr;
    FREE(opr->inputs);
    DeviceModel* model = get_active_device_model(vm);
    for (int i = 0; i < opr->nr_output; ++i) {
        if (opr->outputs[i]->is_dynamic) {
            model->opt.device->free(opr->outputs[i]->ptr);
        }
    }
    FREE(opr->outputs);
    FREE(opr->name);
    FREE(opr->type);
    return TinyNN_SUCCESS;
}

void register_op(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_Opr), &load);
    vm_register_instruction_destruct(vm, TinyNN_INST_OPR, &destruct);
    vm_register_instruction_call(vm, TinyNN_INST_OPR, &execute);
}

// vim: syntax=cpp.doxygen
