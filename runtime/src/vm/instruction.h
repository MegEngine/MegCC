/**
 * \file runtime/src/vm/instruction.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#ifndef VM_INSTRUCTION_H
#define VM_INSTRUCTION_H

#include "data_struct.h"
#include "model_reader.h"
#include "runtime_inst_switch.h"
#include "extern_c_opr.h"

// clang-format off
#define FOR_EACH_INSTRUCTION_TYPE(cb) \
    cb(TinyNN_INST_OPR) \
    cb(TinyNN_INST_DEV_MEM_ALLOC) \
    cb(TinyNN_INST_DEV_MEM_FREE) \
    cb(TinyNN_INST_MEM_FORWARD)\
    cb(TinyNN_INST_DIMSHUFFLE)\
    cb(TinyNN_INST_BROADCAST)\
    cb(TinyNN_INST_CONCAT)\
    cb(TinyNN_INST_SUBTENSOR)\
    cb(TinyNN_INST_SETSUBTENSOR)\
    cb(TINYNN_INST_ARITHMETIC)\
    cb(TinyNN_INST_SHAPEOF)\
    cb(TinyNN_INST_WARPPERSPECTIVE)\
    cb(TinyNN_INST_TYPECVT)\
    cb(TinyNN_INST_INDEXING_MULTI_AXIS)\
    cb(TinyNN_INST_ARGSORT)\
    cb(TinyNN_INST_RESHAPE)\
    cb(TinyNN_INST_EXTERN_OPR)

typedef enum {
    TinyNN_INST_NONE = 0,

#define D(x) x,
    FOR_EACH_INSTRUCTION_TYPE(D)
#undef D

    TinyNN_INSTRUCTION_TYPE_END
} InstructionType;

static inline const char* instruction_type_name(InstructionType type) {
    switch (type) {
#define V(x) \
    case x:  \
        return #x;

        FOR_EACH_INSTRUCTION_TYPE(V)
#undef V

        default:
            return "UNKNOWN_INSTRUCTION_TYPE";
    }
}
// clang-format on

typedef struct {
   int32_t axis, start, end, step, index;
} IndexDesc;

//! all opr is of the same behave, use the assigned input, weights to compute
//! the assigned output by the kernel function, when some operator or kernel
//! preprocess weights by init_func
typedef struct {
    Tensor** inputs;
    int nr_input;
    Tensor** outputs;
    int nr_output;

    Workspace workspace;

    int init_func;
    int kernel_func;
    int deduce_shape_func;
    int workspace_func;

    char* name;
    char* type;
} Opr;

typedef struct {
    Tensor* tensor;
} DevMemAlloc;

typedef struct {
    Tensor* tensor;
} DevMemFree;

typedef enum {
    TinyNN_MemForward_Reshape=0,
    TinyNN_MemForward_Subtensor=1,
} MemForwardType;

typedef struct {
    Tensor* input;
    Tensor* output;
    int offset;
    MemForwardType type;
} MemForward;

typedef struct {
    int32_t pattern_dim;
    int32_t pattern[MAX_DIM];
    Tensor* input;
    Tensor* output;
} Dimshuffle;

typedef struct {
    Tensor* inputs[2];
    Tensor* output;
} BroadCast;

typedef struct {
    MegCC_ArithMode_enum_t mode;
    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} Arithmetic;

typedef struct {
    int32_t axis;
    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} Concat;

typedef struct {
    int8_t nr_descs;
    //! iterms is the arrary of subtensor param axis, start, end, step, index
    IndexDesc* descs;
    //! flags the corresponding of subtensor param start, end, step, index
    //! if -1, mean it is not exist
    //! if 0, mean the corresponding descs is static value
    //! if 1, mean the corresponding descs is dynamic, and descs store the index
    //! of input
    IndexDesc* flags;

    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} SubTensor;

typedef struct {
    int8_t nr_descs;
    //! iterms is the arrary of subtensor param axis, start, end, step, index
    IndexDesc* descs;
    //! flags the corresponding of subtensor param start, end, step, index
    //! if -1, mean it is not exist
    //! if 0, mean the corresponding descs is static value
    //! if 1, mean the corresponding descs is dynamic, and descs store the index
    //! of input
    IndexDesc* flags;

    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} SetSubTensor;

typedef struct {
    Tensor* input;
    Tensor* output;
} ShapeOf;

typedef struct {
    MegCC_InterpolationMode_enum_t imode;
    MegCC_BorderModeMode_enum_t bmode;
    MegCC_Format_enum_t format;
    int32_t nr_mat_id;
    int32_t* mat_id;
    float bval;
    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} WarpPerspective;

typedef struct {
    const char* idtype;
    const char* odtype;
    Tensor* input;
    Tensor* output;
} TypeCvt;

typedef struct {
    const char* order;
    Tensor* input;
    Tensor* output;
} ArgSort;

typedef struct {
    int32_t axis[MAX_DIM];
    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} IndexingMultiAxis;

typedef struct {
    int32_t nr_input;
    Tensor** inputs;
    Tensor* output;
} Reshape;

typedef struct {
    int32_t nr_input;
    int32_t nr_output;
    Tensor** inputs;
    MGBTensor* mgb_inputs;
    MGBTensor* mgb_outputs;
    Tensor** outputs;
    MGBOprDesc* desc;
} ExternOpr;

typedef struct Instruction {
    InstructionType tag;
    union {
        Opr opr;
        DevMemAlloc dev_mem_alloc;
        DevMemFree dev_mem_free;
        MemForward mem_forward;
        Dimshuffle dimshuffle;
        BroadCast broadcast;
        Arithmetic arithmetic;
        Concat concat;
        SubTensor subtensor;
        SetSubTensor set_subtensor;
        ShapeOf shape_of;
        WarpPerspective warpperspective;
        TypeCvt type_cvt;
        IndexingMultiAxis indexing_multi_axis;
        ArgSort argsort;
        Reshape reshape;
        ExternOpr extern_opr;
    } workload;
#if TINYNN_PROFILE_KERNEL
    float time_ms;
    int time_count;
#endif
} Instruction;

#endif  // VM_INSTRUCTION_H

// vim: syntax=cpp.doxygen
