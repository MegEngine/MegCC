/**
 * \file runtime/src/lite/network.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <stdbool.h>
#include "data_struct.h"
#include "init.h"
#include "io_tensor.h"
#include "lite-c/network_c.h"
#include "parse.h"
#include "tinynn.h"
#include "vm.h"

#define STD_OPT_INIT                                                       \
    {                                                                      \
        .weight_preprocess = false, .fuse_preprocess = false,              \
        .fake_next_exec = false, .var_sanity_check_first_run = true,       \
        .const_shape = false, .force_dynamic_alloc = false,                \
        .force_output_dynamic_alloc = false,                               \
        .no_profiling_on_shape_change = false, .jit_level = 0,             \
        .comp_node_seq_record_level = 0, .graph_opt_level = 2,             \
        .async_exec_level = 1, .enable_nchw44 = 0, .enable_nchw44_dot = 0, \
        .enable_nchw88 = 0, .enable_nhwcd4 = 0, .enable_nchw4 = 0,         \
        .enable_nchw32 = 0, .enable_nchw64 = 0,                            \
    }
//! define a default Options
const LiteOptions default_option = STD_OPT_INIT;

//! define a default config
LiteConfig default_config_t = {.has_compression = false,
                               .device_id = -1,
                               .device_type = LITE_CPU,
                               .backend = LITE_DEFAULT,
                               .bare_model_cryption_name = NULL,
                               .options = STD_OPT_INIT};
LiteConfig* default_config() {
    return &default_config_t;
}

//! define a default NetworkIO
LiteNetworkIO default_network_io_t = {
        .inputs = NULL, .outputs = NULL, .input_size = 0, .output_size = 0};

LiteNetworkIO* default_network_io() {
    return &default_network_io_t;
}

static inline DeviceModel* get_active_device_model(CombineModel* combo_model) {
    return combo_model->device_models[combo_model->active_device_model_idx];
}

int LITE_make_network(LiteNetwork* network, const LiteConfig config,
                      const LiteNetworkIO network_io) {
    CombineModel* model = tinynn_malloc(sizeof(CombineModel));
    memset(model, 0, sizeof(CombineModel));
    vm_attach(vm_global_inst(), model);
    *network = model;
    LOG_DEBUG("create model and ignore all config\n");
    return TinyNN_SUCCESS;
}

int LITE_load_model_from_mem(LiteNetwork network, void* model_mem,
                             size_t size) {
    TinyNNStatus parse_status = parse_model(model_mem, size, network, 0);
    if (parse_status != TinyNN_SUCCESS) {
        LOG_DEBUG("load model from memory failed\n");
        return parse_status;
    }
    return init_model_weights(network);
}

int LITE_load_model_with_shared_mem(LiteNetwork network, void* model_mem,
                                    size_t size) {
    TinyNNStatus parse_status = parse_model(model_mem, size, network, 1);
    if (parse_status != TinyNN_SUCCESS) {
        LOG_DEBUG("load model from memory failed\n");
        return parse_status;
    }
    return init_model_weights(network);
}

int LITE_load_model_from_path(LiteNetwork network, const char* model_path) {
    LOG_DEBUG("load model from %s\n", model_path);

    FILE* fin = tinynn_fopen(model_path, "rb");
    if (!fin) {
        LOG_ERROR("Open file error!!\n");
        return -1;
    }
    tinynn_fseek(fin, 0, SEEK_END);
    size_t size = tinynn_ftell(fin);
    tinynn_fseek(fin, 0, SEEK_SET);
    void* ptr = tinynn_malloc(size);
    size_t read_bytes = tinynn_fread(ptr, 1, size, fin);
    TINYNN_ASSERT(size == read_bytes);
    tinynn_fclose(fin);
    TinyNNStatus parse_status = parse_model(ptr, size, network, 1);
    if (parse_status != TinyNN_SUCCESS) {
        LOG_DEBUG("load model from memory failed\n");
        return parse_status;
    } else {
        return init_model_weights(network);
    }
}

int LITE_get_io_tensor(LiteNetwork network, const char* io_name,
                       LiteTensorPhase phase, LiteTensor* res_tensor) {
    LOG_DEBUG("get tensor by name:%s\n", io_name);
    if (!network || !io_name || !res_tensor) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    CombineModel* cb_model = (CombineModel*)network;
    if (!cb_model->have_init) {
        init_model_memory(cb_model);
    }
    DeviceModel* model = get_active_device_model(cb_model);

    ComboIOTensor* tensor_pack = get_empty_io_tensor(cb_model);
    if (phase == LITE_INPUT || phase == LITE_IO) {
        int nr_input = model->nr_input;
        for (int i = 0; i < nr_input; i++) {
            Tensor* tensor = *(model->inputs + i);
            if (!strcmp(io_name, tensor->name)) {
                for (int model_idx = 0; model_idx < cb_model->nr_device_model;
                     ++model_idx) {
                    Tensor* tensor_ptr =
                            *(cb_model->device_models[model_idx]->inputs + i);

                    tensor_pack->tensors[model_idx] = tensor_ptr;
                }
                *res_tensor = tensor_pack;
                return TinyNN_SUCCESS;
            }
        }
    }

    if (phase == LITE_OUTPUT || phase == LITE_IO) {
        int nr_output = model->nr_output;
        for (int i = 0; i < nr_output; i++) {
            Tensor* tensor = *(model->outputs + i);
            if (!strcmp(io_name, tensor->name)) {
                for (int model_idx = 0; model_idx < cb_model->nr_device_model;
                     ++model_idx) {
                    tensor_pack->tensors[model_idx] =
                            *(cb_model->device_models[model_idx]->outputs + i);
                }
                *res_tensor = tensor_pack;
                return TinyNN_SUCCESS;
            }
        }
    }
    return TinyNN_ERROR_NO_FOUND;
}

int LITE_get_tensor_total_size_in_byte(const LiteTensor tensor, size_t* size) {
    if (!tensor || !size) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    Tensor* inner_tensor = get_active_tensor((ComboIOTensor*)tensor);
    *size = tensor_length_in_byte(inner_tensor);
    return TinyNN_SUCCESS;
}

int LITE_forward(const LiteNetwork network) {
    LOG_DEBUG("execute model\n");
    if (!network) {
        return TinyNN_ERROR_NULL_PTR;
    }
    CombineModel* cb_model = (CombineModel*)network;
    if (!cb_model->have_init) {
        init_model_memory(cb_model);
    }
    DeviceModel* model = get_active_device_model(cb_model);
    int nr_instruction = model->nr_instruction;
    for (int inst_idx = 0; inst_idx < nr_instruction; inst_idx++) {
        Instruction* inst = model->instructions + inst_idx;
        LOG_DEBUG("execute instruction id: %d, instruction type %s\n", inst_idx,
                  instruction_type_name(inst->tag));
#if TINYNN_PROFILE_KERNEL
        int32_t start_tv_sec, start_tv_usec, end_tv_sec, end_tv_usec;
        tinynn_gettime(&start_tv_sec, &start_tv_usec);
#endif
        TinyNNStatus error = vm_instruction_call(vm_global_inst(), inst);
        if (error != TinyNN_SUCCESS) {
            return error;
        }
#if TINYNN_PROFILE_KERNEL
        tinynn_gettime(&end_tv_sec, &end_tv_usec);
        float time_ms = (end_tv_sec - start_tv_sec) * 1000.f +
                        (end_tv_usec - start_tv_usec) / 1000.f;
        inst->time_ms += time_ms;
        inst->time_count++;
        if (inst->tag == TinyNN_INST_OPR) {
            Opr* opr = &inst->workload.opr;

            Layout in_layout = opr->inputs[0]->layout;
            Layout out_layout = opr->outputs[0]->layout;
            LOG_INFO(
                    " instruction: %s \nuse %fms \t"
                    "[%d(%d), %d(%d), %d(%d), %d(%d), %d(%d)] \t"
                    "[%d(%d), %d(%d), %d(%d), %d(%d), %d(%d)]\n",
                    opr->type, inst->time_ms / inst->time_count,
                    in_layout.dims[0], in_layout.stride[0], in_layout.dims[1],
                    in_layout.stride[1], in_layout.dims[2], in_layout.stride[2],
                    in_layout.dims[3], in_layout.stride[3], in_layout.dims[4],
                    in_layout.stride[4], out_layout.dims[0],
                    out_layout.stride[0], out_layout.dims[1],
                    out_layout.stride[1], out_layout.dims[2],
                    out_layout.stride[2], out_layout.dims[3],
                    out_layout.stride[3], out_layout.dims[4],
                    out_layout.stride[4]);

        } else {
            LOG_INFO("execute used time %f ms of instruction %s.\n",
                     inst->time_ms / inst->time_count,
                     instruction_type_name(inst->tag));
        }
#endif
    }
    return TinyNN_SUCCESS;
}

int LITE_wait(const LiteNetwork network) {
    LOG_DEBUG("wait not impl\n");
    return TinyNN_SUCCESS;
}

int LITE_get_output_name(const LiteNetwork network, size_t index,
                         const char** name) {
    if (!network) {
        return TinyNN_ERROR_NULL_PTR;
    }
    CombineModel* cb_model = (CombineModel*)network;
    DeviceModel* model = get_active_device_model(cb_model);
    Tensor* tensor = *(model->outputs + index);
    *name = tensor->name;
    return TinyNN_SUCCESS;
}

int LITE_get_all_input_name(const LiteNetwork network, size_t* size,
                            const char** name) {
    if (!network) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    CombineModel* cb_model = (CombineModel*)network;
    DeviceModel* model = get_active_device_model(cb_model);
    if (size) {
        *size = model->nr_input;
    }
    if (name) {
        for (int i = 0; i < model->nr_input; i++) {
            Tensor* tensor = *(model->inputs + i);
            *(name + i) = tensor->name;
        }
    }
    return TinyNN_SUCCESS;
}

int LITE_get_all_output_name(const LiteNetwork network, size_t* size,
                             const char** name) {
    if (!network) {
        LOG_ERROR("input pointer is NULL\n");
        return TinyNN_ERROR_NULL_PTR;
    }
    CombineModel* cb_model = (CombineModel*)network;
    DeviceModel* model = get_active_device_model(cb_model);
    if (size) {
        *size = model->nr_output;
    }
    if (name) {
        for (int i = 0; i < model->nr_output; i++) {
            Tensor* tensor = *(model->outputs + i);
            *(name + i) = tensor->name;
        }
    }
    return TinyNN_SUCCESS;
}

int LITE_share_runtime_memroy(LiteNetwork src_network,
                              LiteNetwork dst_network) {
    CombineModel* src_model = (CombineModel*)src_network;
    CombineModel* dst_model = (CombineModel*)dst_network;
    TINYNN_ASSERT_MSG(
            !src_model->have_init && !dst_model->have_init,
            "shared runtime memory should be called before model run.\n");
    size_t src_memory_length = src_model->max_tensor_memroy->length_in_byte;
    size_t dst_memory_length = dst_model->max_tensor_memroy->length_in_byte;

    if (dst_model->is_own_tensor_memory) {
        tinynn_free(dst_model->max_tensor_memroy);
    }
    dst_model->max_tensor_memroy = src_model->max_tensor_memroy;
    dst_model->is_own_tensor_memory = 0;
    dst_model->max_tensor_memroy->length_in_byte =
            src_memory_length > dst_memory_length ? src_memory_length
                                                  : dst_memory_length;
    return TinyNN_SUCCESS;
}

int LITE_destroy_network(LiteNetwork network) {
    LOG_DEBUG("delete model\n");
    if (!network) {
        return TinyNN_ERROR_NULL_PTR;
    }
    CombineModel* cb_model = (CombineModel*)network;
    FREE(cb_model->model_ptr);
    //! origin weight
    for (int i = 0; i < cb_model->nr_origin_weight; i++) {
        Tensor* weight = cb_model->weights + i;
        FREE(weight->name);
        //! only the use count>0, the memory is not free
        if (weight->use_count > 0 && !weight->is_shared) {
            cb_model->host_dev.free(weight->ptr);
        }
    }
    FREE(cb_model->weights);
    //! free shared tensor memory
    if (cb_model->is_own_tensor_memory && cb_model->max_tensor_memroy) {
        if (cb_model->max_tensor_memroy->ptr) {
            DeviceModel* model = get_active_device_model(cb_model);
            model->device.free(cb_model->max_tensor_memroy->ptr);
        }
        tinynn_free(cb_model->max_tensor_memroy);
    }
    //! free device model
    for (int model_idx = 0; model_idx < cb_model->nr_device_model;
         ++model_idx) {
        DeviceModel* model = cb_model->device_models[model_idx];
        //! preprocessed weight
        for (int i = 0; i < model->nr_processed_weight; i++) {
            Tensor* weight = model->processed_weights + i;
            if (!weight->is_shared)
                model->device.free(weight->ptr);
        }
        FREE(model->processed_weights);

        //! instruction
        for (int i = 0; i < model->nr_instruction; i++) {
            Instruction* inst = model->instructions + i;
            vm_instruction_destruct(vm_global_inst(), inst);
        }
        FREE(model->instructions);

        //! tensor
        for (int i = 0; i < model->nr_tensor; i++) {
            Tensor* tensor = model->tensors + i;
            FREE(tensor->name);
        }
        FREE(model->tensors);

        //! model member
        FREE(model->inputs);
        FREE(model->outputs);
        FREE(model);
    }
    FREE(cb_model->device_models);

    //! free combine model struct
    FREE(cb_model);
    return TinyNN_SUCCESS;
}
