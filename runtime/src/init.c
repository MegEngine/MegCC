/**
 * \file runtime/src/init.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <stdbool.h>
#include <string.h>
#include "stdio.h"
#include "stdlib.h"

#include "data_struct.h"
#include "init.h"
#include "kernels.h"
#include "tinynn.h"
#include "vm/instruction.h"

static int is_equal_tensor_layout(Tensor* tensor0, Tensor* tensor1) {
    if (tensor0->dtype.type_enum != tensor1->dtype.type_enum) {
        return 0;
    }
    Layout layout0 = tensor0->layout;
    Layout layout1 = tensor1->layout;
    if (layout0.nr_dim != layout1.nr_dim) {
        return 0;
    }
    for (int i = 0; i < layout0.nr_dim; ++i) {
        if (layout0.dims[i] != layout1.dims[i]) {
            return 0;
        }
    }
    return 1;
}

static int get_ins_nr_processed_weights(Opr* opr, DeviceModel* model) {
    int init_index = opr->init_func;
    if (init_index >= NR_INIT) {
        LOG_ERROR(
                "Init function index %d is out of range, max is "
                "%d.\n",
                init_index, NR_INIT);
        return TinyNN_ERROR_OUT_OF_RANGE;
    }
    if (init_index >= 0) {
        InitFunc init = init_kernels[init_index];
        //! get the new weights number
        int size = 0;
        LOG_DEBUG("init_ptr = %p init_index = %d\n", init, init_index);
        TINYNN_ASSERT(init);
        init(opr->inputs, opr->nr_input, NULL, &size, &model->opt);
        LOG_DEBUG(
                "opr symbol %s need preprocess weights number "
                "is:%d\n",
                opr->type, size);
        TINYNN_ASSERT_MSG(
                size <= 1,
                "new only support only one processed weights, but get %d.\n");
        return size;
    }
    return 0;
}

//! update the opr weights index with the init weight, if
//! origin weights is no need, share it or free it
static void postprocess_weight_memory(
        Opr* opr, DeviceModel* model, Tensor* new_weight) {
    for (int j = 0; j < opr->nr_input; j++) {
        Tensor* old_weight = (Tensor*)(*(opr->inputs + j));
        //! FIXME: maybe name is not suitable
        if (new_weight->name == old_weight->name) {
            TINYNN_ASSERT(new_weight->name != NULL);
            LOG_DEBUG(
                    "opr symbol %s preprocessed weights "
                    ":%s.\n",
                    opr->type, old_weight->name);
            old_weight->use_count -= 1;
            if (old_weight->use_count <= 0) {
                if (!old_weight->is_shared) {
                    model->device.free(old_weight->ptr);
                    old_weight->ptr = NULL;
                    LOG_DEBUG(
                            "opr symbol %s free preprocessed weights: % s.\n ",
                            opr->type, old_weight->name);
                } else if (old_weight->size >= new_weight->size) {
                    memcpy(old_weight->ptr, new_weight->ptr, new_weight->size);
                    model->device.free(new_weight->ptr);
                    LOG_DEBUG(
                            "processed weights share memory with old "
                            "weights\n");
                    new_weight->ptr = old_weight->ptr;
                    new_weight->is_shared = 1;
                }
            }
            *(opr->inputs + j) = new_weight;
            break;
        }
    }
}

static size_t get_opr_weights_process_size(
        Opr* opr, DeviceModel* model, Tensor* weight) {
    //! get the information and allocate memory
    int init_index = opr->init_func;
    if (init_index >= NR_INIT) {
        LOG_ERROR(
                "Init function index %d is out of range, max is "
                "%d.\n",
                init_index, NR_INIT);
        return TinyNN_ERROR_OUT_OF_RANGE;
    }
    if (init_index >= 0) {
        InitFunc init = init_kernels[init_index];
        init(opr->inputs, opr->nr_input, weight, NULL, &model->opt);
        return tensor_length_in_byte(weight);
    }
    return 0;
}

static void init_ins_weights(Opr* opr, DeviceModel* model, Tensor* weight) {
    //! get the information and allocate memory
    int init_index = opr->init_func;
    size_t weight_length = get_opr_weights_process_size(opr, model, weight);
    InitFunc init = init_kernels[init_index];
    if (weight_length > 0 && init_index >= 0) {
        LOG_DEBUG(
                "opr symbol %s preprocess weight need memory:%zu\n", opr->type,
                weight_length);
        weight->ptr = model->device.malloc(weight_length);
        weight->size = weight_length;
        weight->is_weight = 1;
        weight->is_shared = 0;
        //! call the init function
        LOG_DEBUG("opr symbol %s preprocess weights.\n", opr->type);
        int nr_processed_weight = 1;
        init(opr->inputs, opr->nr_input, weight, &nr_processed_weight, &model->opt);
    }
}

static int count_total_processed_weights_number(
        CombineModel* combo_model, int model_id) {
    DeviceModel* model = combo_model->device_models[model_id];
    //! run opr init function
    int nr_instruction = model->nr_instruction;
    int total_processed_weights = 0;
    LOG_DEBUG("execute weight preprocess\n");
    for (int i = 0; i < nr_instruction; i++) {
        Instruction* inst = model->instructions + i;
        if (inst->tag == TinyNN_INST_OPR) {
            Opr* opr = &inst->workload.opr;
            TINYNN_ASSERT(opr);
            int size = get_ins_nr_processed_weights(opr, model);
            total_processed_weights += size;
        }
    }
    return total_processed_weights;
}

static TinyNNStatus init_signle_device_model(CombineModel* combo_model) {
    DeviceModel* model = combo_model->device_models[0];
    int nr_instruction = model->nr_instruction;
    int total_processed_weights = count_total_processed_weights_number(combo_model, 0);
    LOG_DEBUG("calc total_processed_weights done\n");
    if (total_processed_weights < 1) {
        return TinyNN_SUCCESS;
    }
    //! allocate the processed memory
    model->nr_processed_weight = total_processed_weights;
    model->processed_weights = tinynn_malloc(total_processed_weights * sizeof(Tensor));
    memset(model->processed_weights, 0, total_processed_weights * sizeof(Tensor));

    int processed_weights_index = 0;
    for (int i = 0; i < nr_instruction; i++) {
        Instruction* inst = model->instructions + i;
        if (inst->tag == TinyNN_INST_OPR) {
            Opr* opr = &inst->workload.opr;
            int size = get_ins_nr_processed_weights(opr, model);
            if (size == 1) {
                Tensor* weight = model->processed_weights + processed_weights_index;
                init_ins_weights(opr, model, weight);
                postprocess_weight_memory(opr, model, weight);
                processed_weights_index += size;
            } else {
                TINYNN_ASSERT_MSG(size == 0, "Now only support one processed weigh.\n");
            }
        }
    }
    return TinyNN_SUCCESS;
}

static void broadcast_postprocess_weight_memory(
        int opr_id, CombineModel* combo_model, Tensor* weight) {
    DeviceModel* model = combo_model->device_models[0];
    Instruction* ins = model->instructions + opr_id;
    Opr* opr = &(ins->workload.opr);
    for (int j = 0; j < opr->nr_input; j++) {
        Tensor* old_weight = (Tensor*)(*(opr->inputs + j));
        //! FIXME: maybe name is not suitable
        if (weight->name == old_weight->name) {
            TINYNN_ASSERT(weight->name != NULL);
            LOG_DEBUG(
                    "opr symbol %s preprocessed weights "
                    ":%s.\n",
                    opr->type, old_weight->name);
            old_weight->use_count -= combo_model->nr_device_model;
            if (old_weight->use_count <= 0) {
                if (!old_weight->is_shared) {
                    model->device.free(old_weight->ptr);
                    old_weight->ptr = NULL;
                    LOG_DEBUG(
                            "opr symbol %s free preprocessed weights: % s.\n ",
                            opr->type, old_weight->name);
                } else if (old_weight->size >= weight->size) {
                    memcpy(old_weight->ptr, weight->ptr, weight->size);
                    model->device.free(weight->ptr);
                    LOG_DEBUG(
                            "processed weights share memory with old "
                            "weights\n");
                    weight->ptr = old_weight->ptr;
                    weight->is_shared = 1;
                }
            }
            *(opr->inputs + j) = weight;
            break;
        }
    }
    //! broad cast the new weights to other device model
    for (int model_idx = 1; model_idx < combo_model->nr_device_model; ++model_idx) {
        DeviceModel* model = combo_model->device_models[model_idx];
        Instruction* ins = model->instructions + opr_id;
        Opr* opr = &(ins->workload.opr);
        for (int j = 0; j < opr->nr_input; j++) {
            Tensor* old_weight = (Tensor*)(*(opr->inputs + j));
            //! FIXME: maybe name is not suitable
            if (weight->name == old_weight->name) {
                TINYNN_ASSERT(weight->name != NULL);

                *(opr->inputs + j) = weight;
            }
        }
    }
}

/*! likely device model condition:
 * 1. same number of instruction
 * 2. same instruction type of every instruction
 * 3. same init function id
 * 4. same weights id
 * 5. same processed weights layout
 */
static int is_likely_multi_device_model(CombineModel* combo_model) {
    //! compute the tensor memory ptr
    for (int model_idx = 1; model_idx < combo_model->nr_device_model; ++model_idx) {
        DeviceModel* model = combo_model->device_models[model_idx];
        DeviceModel* model0 = combo_model->device_models[0];
        //! if instruction is not equal, return 0
        if (model->nr_instruction != model0->nr_instruction) {
            return 0;
        }
        int nr_instruction = model->nr_instruction;
        for (int ins_idx = 0; ins_idx < nr_instruction; ins_idx++) {
            Instruction* ins0 = model0->instructions + ins_idx;
            Instruction* ins = model->instructions + ins_idx;
            //! if each instruction type is not equal, return 0
            if (ins0->tag != ins->tag) {
                return 0;
            }
            if (ins0->tag == TinyNN_INST_OPR) {
                Opr* opr0 = &ins0->workload.opr;
                Opr* opr = &ins->workload.opr;
                if (opr0->init_func >= 0 && opr0->init_func < NR_INIT) {
                    //! if each instruction init function is not equal,
                    //! return 0
                    if (opr0->init_func != opr->init_func) {
                        return 0;
                    }
                    InitFunc init = init_kernels[opr0->init_func];

                    Tensor tmp_weights0, tmp_weights;
                    init(opr0->inputs, opr0->nr_input, &tmp_weights0, NULL,
                         &model->opt);
                    init(opr->inputs, opr->nr_input, &tmp_weights, NULL, &model->opt);
                    if (!is_equal_tensor_layout(&tmp_weights, &tmp_weights0)) {
                        return 0;
                    }
                }
                if (opr0->nr_input != opr->nr_input) {
                    return 0;
                }
                for (int input_idx = 0; input_idx < opr0->nr_input; input_idx++) {
                    if (!opr0->inputs[input_idx]->is_weight) {
                        continue;
                    } else {
                        //! each opr weight is the same
                        if (opr->inputs[input_idx] != opr0->inputs[input_idx]) {
                            return 0;
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static TinyNNStatus init_multi_likely_device_model(CombineModel* combo_model) {
    DeviceModel* model = combo_model->device_models[0];
    int nr_instruction = model->nr_instruction;
    int total_processed_weights = count_total_processed_weights_number(combo_model, 0);
    LOG_DEBUG("calc total_processed_weights done\n");
    if (total_processed_weights < 1) {
        return TinyNN_SUCCESS;
    }
    //! allocate the processed memory

    for (int model_idx = 0; model_idx < combo_model->nr_device_model; ++model_idx) {
        if (model_idx == 0) {
            DeviceModel* model = combo_model->device_models[model_idx];
            model->nr_processed_weight = total_processed_weights;
            model->processed_weights =
                    tinynn_malloc(total_processed_weights * sizeof(Tensor));
            memset(model->processed_weights, 0,
                   total_processed_weights * sizeof(Tensor));
        } else {
            DeviceModel* model = combo_model->device_models[model_idx];
            //! this model  will share weights from model0
            model->nr_processed_weight = 0;
            model->processed_weights = NULL;
        }
    }
    int processed_weights_index = 0;
    for (int i = 0; i < nr_instruction; i++) {
        Instruction* inst = model->instructions + i;
        if (inst->tag == TinyNN_INST_OPR) {
            Opr* opr = &inst->workload.opr;
            int size = get_ins_nr_processed_weights(opr, model);
            if (size == 1) {
                Tensor* weight = model->processed_weights + processed_weights_index;
                init_ins_weights(opr, model, weight);
                broadcast_postprocess_weight_memory(i, combo_model, weight);
                processed_weights_index += size;
            }
        }
    }
    return TinyNN_SUCCESS;
}

TinyNNStatus init_model_weights(CombineModel* combo_model) {
    if (!combo_model) {
        return TinyNN_ERROR_NULL_PTR;
    }
    //! init weights with the init function
    if (combo_model->nr_device_model == 1) {
        init_signle_device_model(combo_model);
    } else if (is_likely_multi_device_model(combo_model)) {
        init_multi_likely_device_model(combo_model);
    } else {
        //! compute the tensor memory ptr
        for (int model_idx = 0; model_idx < combo_model->nr_device_model; ++model_idx) {
            DeviceModel* model = combo_model->device_models[model_idx];
            //! run opr init function
            int nr_instruction = model->nr_instruction;
            LOG_DEBUG("execute weight preprocess\n");
            int total_processed_weights =
                    count_total_processed_weights_number(combo_model, model_idx);
            LOG_DEBUG("calc total_processed_weights done\n");
            if (total_processed_weights < 1) {
                continue;
            }
            //! allocate the processed memory
            model->nr_processed_weight = total_processed_weights;
            model->processed_weights =
                    tinynn_malloc(total_processed_weights * sizeof(Tensor));
            memset(model->processed_weights, 0,
                   total_processed_weights * sizeof(Tensor));

            int processed_weights_index = 0;
            for (int i = 0; i < nr_instruction; i++) {
                Instruction* inst = model->instructions + i;
                if (inst->tag == TinyNN_INST_OPR) {
                    Opr* opr = &inst->workload.opr;
                    int nr_weights = get_ins_nr_processed_weights(opr, model);

                    if (nr_weights > 0) {
                        Tensor* new_weight =
                                model->processed_weights + processed_weights_index;
                        size_t weight_length =
                                get_opr_weights_process_size(opr, model, new_weight);
                        int size = 0;
                        LOG_DEBUG(
                                "opr symbol %s preprocess weight id %d "
                                "need "
                                "memory:%zu\n",
                                opr->type, i, weight_length);
                        new_weight->ptr = model->device.malloc(weight_length);
                        new_weight->size = weight_length;
                        new_weight->is_weight = 1;
                        new_weight->is_shared = 0;
                        processed_weights_index++;

                        //! call the init function
                        LOG_DEBUG("opr symbol %s preprocess weights.\n", opr->type);

                        int init_index = opr->init_func;
                        InitFunc init = init_kernels[init_index];
                        init(opr->inputs, opr->nr_input, new_weight, &size,
                             &model->opt);

                        //! update the opr weights index with the init
                        //! weight, if origin weights is no need, free
                        //! it
                        for (int j = 0; j < opr->nr_input; j++) {
                            Tensor* old_weight = (Tensor*)(*(opr->inputs + j));
                            //! FIXME: maybe name is not suitable
                            if (new_weight->name == old_weight->name) {
                                TINYNN_ASSERT(new_weight->name != NULL);
                                LOG_DEBUG(
                                        "opr symbol %s preprocessed "
                                        "weights "
                                        ":%s.\n",
                                        opr->type, old_weight->name);
                                old_weight->use_count -= 1;
                                if (old_weight->use_count <= 0 &&
                                    !old_weight->is_shared) {
                                    model->device.free(old_weight->ptr);
                                    LOG_DEBUG(
                                            "opr symbol %s free "
                                            "preprocessed "
                                            "weights "
                                            ":%s.\n",
                                            opr->type, old_weight->name);
                                }
                                *(opr->inputs + j) = new_weight;
                            }
                        }
                    }
                }
            }
        }
    }
    return TinyNN_SUCCESS;
}

TinyNNStatus init_model_memory(CombineModel* combo_model) {
    if (!combo_model) {
        return TinyNN_ERROR_NULL_PTR;
    }
    if (combo_model->have_init) {
        return TinyNN_SUCCESS;
    }
    if (!combo_model->max_tensor_memroy->ptr) {
        DeviceModel* model =
                combo_model->device_models[combo_model->active_device_model_idx];
        combo_model->max_tensor_memroy->ptr =
                model->device.malloc(combo_model->max_tensor_memroy->length_in_byte);
    }
    //! compute the tensor memory ptr
    for (int model_idx = 0; model_idx < combo_model->nr_device_model; ++model_idx) {
        DeviceModel* model = combo_model->device_models[model_idx];
        int8_t* tensor_memory = combo_model->max_tensor_memroy->ptr;
        TINYNN_ASSERT(tensor_memory);
        LOG_DEBUG("Init tensor by offset\n");
        for (int i = 0; i < model->nr_tensor; i++) {
            Tensor* tensor = model->tensors + i;
            TINYNN_ASSERT(tensor);
            if (!tensor->is_dynamic) {
                tensor->ptr = tensor_memory + tensor->offset;

                LOG_DEBUG(
                        "Init model %d tensor %s by offset %zu to "
                        "%p\n",
                        model_idx, tensor->name, tensor->offset, tensor->ptr);
            } else {
                tensor->ptr = NULL;
                tensor->offset = 0;
            }
        }
        int nr_instruction = model->nr_instruction;
        for (int i = 0; i < nr_instruction; i++) {
            Instruction* inst = model->instructions + i;
            if (inst->tag == TinyNN_INST_OPR) {
                Opr* opr = &inst->workload.opr;
                opr->workspace.ptr = tensor_memory + opr->workspace.offset;
            }
        }
    }
    combo_model->have_init = 1;
    return TinyNN_SUCCESS;
}

// vim: syntax=cpp.doxygen
