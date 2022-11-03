/**
 * \file runtime/src/vm.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#ifndef VM_H
#define VM_H

#include "data_struct.h"
#include "tinynn.h"
#include "vm/instruction.h"

#include "model_reader.h"

struct VM;

typedef TinyNNStatus (*InstructionCall)(Instruction*, struct VM*);

typedef TinyNNStatus (*InstructionLoad)(flatbuffers_generic_t, Instruction*,
                                        struct VM*);

typedef TinyNNStatus (*InstructionDestruct)(struct VM*, Instruction*);

typedef struct VM {
    //! VM internal status
    enum Stage {
        TinyNN_VM_STAGE_NONE = 0,
        TinyNN_VM_STAGE_LOAD = 1,
        TinyNN_VM_STAGE_INIT = 2,
        TinyNN_VM_STAGE_RUN = 3,
        TinyNN_VM_STAGE_TERMINATE = 4
    } stage;

    //! instruction callers
    InstructionCall inst_call[TinyNN_INSTRUCTION_TYPE_END];

    //! instruction loaders
    InstructionLoad inst_load[MegCC_Instruction_INSTRUCTION_TABLE_END];

    InstructionDestruct inst_destruct[MegCC_Instruction_INSTRUCTION_TABLE_END];

    //! attached model
    CombineModel* model;

    //! flag init
    int init;
} VM;

//! get virtual machine singleton instance
VM* vm_global_inst();

//! reset virtual machine state and register all instructions handler
TinyNNStatus vm_reset(VM* vm);

//! attach model onto vm
TinyNNStatus vm_attach(VM* vm, CombineModel* model);

//! detach model from vm
TinyNNStatus vm_detach(VM* vm, CombineModel* model);

//! call instruction inst in vm
TinyNNStatus vm_instruction_call(VM* vm, Instruction* inst);

//! load instruction from flatbuffers model
TinyNNStatus vm_instruction_load(VM* vm, flatbuffers_union_t fbs_union,
                                 Instruction* inst);

//! load instruction from flatbuffers model
TinyNNStatus vm_instruction_destruct(VM* vm, Instruction* inst);

//! register instruction caller corresponding to given instruction type
TinyNNStatus vm_register_instruction_call(VM* vm, InstructionType type,
                                          InstructionCall func);

//! register instruction loader corresponding to given flatbuffers type
TinyNNStatus vm_register_instruction_load(VM* vm, flatbuffers_union_type_t type,
                                          InstructionLoad func);

//! register instruction destructor corresponding to given flatbuffers type
TinyNNStatus vm_register_instruction_destruct(VM* vm, InstructionType type,
                                              InstructionDestruct func);

#endif  // VM_H

// vim: syntax=cpp.doxygen
