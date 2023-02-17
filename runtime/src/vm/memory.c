/**
 * \file runtime/src/vm/memory.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "init.h"
#include "parse.h"
#include "vm.h"
#include "vm/registry.h"

#if ENABLE_MEMORY_MANAGEMENT
static TinyNNStatus load_alloc_device(
        flatbuffers_generic_t fbs_inst, Instruction* inst, VM* vm) {
    LOG_DEBUG("\t Load instruction DevMemAlloc.\n");
    DevMemAlloc* alloc = &inst->workload.dev_mem_alloc;
    ns(DevMemAlloc_table_t) fbs_alloc = (ns(DevMemAlloc_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_DEV_MEM_ALLOC;
    int32_t index = ns(DevMemAlloc_tensor(fbs_alloc));
    DeviceModel* model = get_active_device_model(vm);
    alloc->tensor = model->tensors + index;
    return TinyNN_SUCCESS;
}

static TinyNNStatus load_free_device(
        flatbuffers_generic_t fbs_inst, Instruction* inst, VM* vm) {
    LOG_DEBUG("\t Load instruction FreeMemAlloc.\n");
    DevMemFree* free = &inst->workload.dev_mem_free;
    ns(DevMemFree_table_t) fbs_free = (ns(DevMemFree_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_DEV_MEM_FREE;
    int32_t index = ns(DevMemFree_tensor(fbs_free));
    DeviceModel* model = get_active_device_model(vm);
    free->tensor = model->tensors + index;
    return TinyNN_SUCCESS;
}

static TinyNNStatus alloc_device_tensor(Instruction* inst, VM* vm) {
    Tensor* tensor = inst->workload.dev_mem_alloc.tensor;
    DeviceModel* model = get_active_device_model(vm);
    RuntimeOpt* opt = &model->opt;
    if (!tensor || !opt)
        return TinyNN_ERROR_NULL_PTR;
    size_t length_in_bytes = tensor_length_in_byte(tensor);
    tensor->ptr = opt->device->malloc(length_in_bytes);
    return TinyNN_SUCCESS;
}

static TinyNNStatus free_device_tensor(Instruction* inst, VM* vm) {
    Tensor* tensor = inst->workload.dev_mem_alloc.tensor;
    DeviceModel* model = get_active_device_model(vm);
    RuntimeOpt* opt = &model->opt;
    if (!tensor || !opt)
        return TinyNN_ERROR_NULL_PTR;
    opt->device->free(tensor->ptr);
    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct(VM* vm, Instruction* inst) {
    return TinyNN_SUCCESS;
}

void register_memory_management(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_DevMemAlloc), &load_alloc_device);
    vm_register_instruction_call(vm, TinyNN_INST_DEV_MEM_ALLOC, &alloc_device_tensor);
    vm_register_instruction_destruct(vm, TinyNN_INST_DEV_MEM_ALLOC, &destruct);
    vm_register_instruction_load(vm, ns(Instruction_DevMemFree), &load_free_device);
    vm_register_instruction_call(vm, TinyNN_INST_DEV_MEM_FREE, &free_device_tensor);
    vm_register_instruction_destruct(vm, TinyNN_INST_DEV_MEM_FREE, &destruct);
}
#else
void register_memory_management(VM* vm) {}
#endif
// vim: syntax=cpp.doxygen
