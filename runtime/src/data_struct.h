/**
 * \file runtime/src/data_struct.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H

#include "device.h"
#include "stdio.h"
#include "stdlib.h"
#include "tinynn.h"
#include "utils.h"

struct Instruction;

typedef struct {
    float scale;
    uint8_t zero_point;
} DTypeParam;

typedef struct {
    TinyNNDType type_enum;
    DTypeParam param;
} DType;

//! the tensor layout which support reshape layout from input tensor
typedef struct {
    int nr_dim;
    uint32_t dims[MAX_DIM];
    int stride[MAX_DIM];
    TinyNNFormat format;
} Layout;

typedef struct {
    void* ptr;
    size_t length_in_byte;
} Memory;

typedef struct Tensor {
    char* name;
    DType dtype;
    Layout layout;
    void* ptr;
    size_t offset;
    //! used for memory runtime memory plan
    int use_count;

    //!flag tensor type, weights or tensor
    int is_weight;
    int is_dynamic;
    uint32_t checksum;
    size_t size;
    //! flag whether the memory is shared from model memory
    int is_shared;
} Tensor;

typedef struct {
    void* ptr;
    size_t size;
    size_t offset;
} Workspace;

typedef struct {
    //! model member
    //! all tensors store together
    Tensor* tensors;
    int nr_tensor;

    Tensor* processed_weights;
    int nr_processed_weight;

    //! all instructions store together
    struct Instruction* instructions;
    int nr_instruction;

    //! model info
    Tensor** inputs;
    int nr_input;
    Tensor** outputs;
    int nr_output;

    int nr_threads;
    int have_init;
    Device device;
    RuntimeOpt opt;
} DeviceModel;

struct ComboIOTensorS;

typedef struct {
    //! device model share max_tensor_memroy
    Memory* max_tensor_memroy;
    //! when multi CombineMode share the max_tensor_memory, this flag whether
    //! the memory allocated by itself
    int is_own_tensor_memory;
    //! all weights store together, nr_origin_weight is the weights number
    //! read from the model. as opr init function will product processed
    //! weights, processed weights store in processed_weights, number is
    //! processed_weights
    Tensor* weights;
    int nr_origin_weight;

    DeviceModel** device_models;
    int nr_device_model;
    int active_device_model_idx;

    int have_init;
    char* name;
    int const_shape;
    size_t model_id;
    Device host_dev;

    void* model_ptr;
    size_t model_len;
    struct ComboIOTensorS* combo_iotensor;
} CombineModel;

typedef struct ComboIOTensorS {
    //! len is nr_device_model
    Tensor** tensors;

    //! ref to model
    CombineModel* model;
} ComboIOTensor;

//! the uniform kernel function for all kernels of all operators
typedef TinyNNStatus (*KernelFunc)(Tensor** inputs, int nr_input,
                                   Tensor** outputs, int nr_output,
                                   const Workspace* workspace,
                                   const RuntimeOpt* opt);

//! the uniform init function for all operators
//! 1. if out_weights is NULL and nr_out_weights is not NULL, this kernel just
//!     get the out_weights size
//! 2. if out_weights is not NULL and nr_out_weight is NULL, this kernel just
//!     fill the weights information
//! 3. if out_weights and nr_out_weight are both not NULL, it works for weights
//!     init, and fill data in the out_weights, if the in_weights is not used by
//!     the opr, it's use_count will minus one.
typedef TinyNNStatus (*InitFunc)(Tensor** inputs, int nr_input,
                                 Tensor* out_weights, int* nr_out_weight,
                                 const RuntimeOpt* opt);

//! the uniform workspace function for all operators
typedef TinyNNStatus (*WorkspaceFunc)(Tensor** inputs, int nr_input,
                                      int nr_thread, size_t* workspace);

//! the uniform kernel function for all kernels of all operators
typedef TinyNNStatus (*DeduceFunc)(Tensor** inputs, int nr_input,
                                   Tensor** outputs, int nr_output);

#endif

// vim: syntax=cpp.doxygen
