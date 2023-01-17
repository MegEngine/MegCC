/**
 * \file runtime/src/vm.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "vm.h"
#include "init.h"
#include "vm/registry.h"

void register_all(VM* vm) {
    register_op(vm);
    register_memory_management(vm);
    register_memforward(vm);
    register_subtensor(vm);
    register_setsubtensor(vm);
    register_dimshuffle(vm);
    register_broadcast_shape_of(vm);
    register_reshape(vm);
    register_extern_opr(vm);
}

TinyNNStatus vm_reset(VM* vm) {
    memset(vm, 0, sizeof(VM));
    register_all(vm);
    return TinyNN_SUCCESS;
}

TinyNNStatus vm_attach(CombineModel* model) {
    void* vm = tinynn_malloc(sizeof(VM));
    vm_reset((VM*)vm);
    model->vm = vm;
    VM* vm_r = (VM*)vm;
    vm_r->model = model;
    return TinyNN_SUCCESS;
}
TinyNNStatus vm_detach(CombineModel* model) {
    FREE(model->vm);
    return TinyNN_SUCCESS;
}

TinyNNStatus vm_instruction_call(VM* vm, Instruction* inst) {
    if (!vm->model) {
        LOG_ERROR("VM hasn't been attached yet\n");
        return TinyNN_ERROR;
    }
    if (inst->tag >= TinyNN_INSTRUCTION_TYPE_END)
        return TinyNN_ERROR_OUT_OF_RANGE;
    InstructionCall caller = vm->inst_call[inst->tag];
    if (!caller) {
        LOG_ERROR("unsupported instruction %s\n",
                  instruction_type_name(inst->tag));
        return TinyNN_ERROR_UNSUPPORTED_INSTRUCTION_TYPE;
    }
    return caller(inst, vm);
}

TinyNNStatus vm_instruction_load(VM* vm, flatbuffers_union_t fbs_union,
                                 Instruction* inst) {
    if (!vm->model) {
        LOG_ERROR("VM hasn't been attached yet\n");
        return TinyNN_ERROR;
    }
    if (fbs_union.type >= MegCC_Instruction_INSTRUCTION_TABLE_END)
        return TinyNN_ERROR_OUT_OF_RANGE;
    InstructionLoad loader = vm->inst_load[fbs_union.type];
    if (!loader) {
        LOG_ERROR("unsupported instruction %s\n",
                  MegCC_Instruction_type_name(fbs_union.type));
        return TinyNN_ERROR_UNSUPPORTED_INSTRUCTION_TYPE;
    }
    return loader(fbs_union.value, inst, vm);
}

TinyNNStatus vm_instruction_destruct(VM* vm, Instruction* inst) {
    if (!vm->model) {
        LOG_ERROR("VM hasn't been attached yet\n");
        return TinyNN_ERROR;
    }
    if (inst->tag >= MegCC_Instruction_INSTRUCTION_TABLE_END)
        return TinyNN_ERROR_OUT_OF_RANGE;
    InstructionDestruct destructor = vm->inst_destruct[inst->tag];
    if (!destructor) {
        LOG_ERROR("unsupported instruction %s\n",
                  instruction_type_name(inst->tag));
        return TinyNN_ERROR_UNSUPPORTED_INSTRUCTION_TYPE;
    }
    return destructor(vm, inst);
}

TinyNNStatus vm_register_instruction_call(VM* vm, InstructionType type,
                                          InstructionCall func) {
    if (type >= TinyNN_INSTRUCTION_TYPE_END)
        return TinyNN_ERROR_OUT_OF_RANGE;
    if (vm->inst_call[type]) {
        LOG_ERROR("duplicated instruction caller for type %s\n",
                  instruction_type_name(type));
        return TinyNN_ERROR;
    }
    vm->inst_call[type] = func;
    return TinyNN_SUCCESS;
}

TinyNNStatus vm_register_instruction_load(VM* vm, flatbuffers_union_type_t type,
                                          InstructionLoad func) {
    if (type >= MegCC_Instruction_INSTRUCTION_TABLE_END)
        return TinyNN_ERROR_OUT_OF_RANGE;
    if (vm->inst_load[type]) {
        LOG_ERROR("duplicated instruction loader for type %s\n",
                  MegCC_Instruction_type_name(type));
        return TinyNN_ERROR;
    }
    vm->inst_load[type] = func;
    return TinyNN_SUCCESS;
}

TinyNNStatus vm_register_instruction_destruct(VM* vm, InstructionType type,
                                              InstructionDestruct func) {
    if (type >= MegCC_Instruction_INSTRUCTION_TABLE_END)
        return TinyNN_ERROR_OUT_OF_RANGE;
    if (vm->inst_destruct[type]) {
        LOG_ERROR("duplicated instruction destructor for type %s\n",
                  instruction_type_name(type));
        return TinyNN_ERROR;
    }
    vm->inst_destruct[type] = func;
    return TinyNN_SUCCESS;
}

// vim: syntax=cpp.doxygen
