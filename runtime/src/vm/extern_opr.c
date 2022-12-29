/**
 * \file runtime/src/vm/extern_opr.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "extern_c_opr.h"
#include "utils.h"
#include "vm.h"

#if ENABLE_INST_EXTERN_OPR

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "init.h"
#include "parse.h"
#include "vm/common.h"
#include "vm/instruction.h"
#include "vm/registry.h"

typedef struct LoaderMap {
    MGBOprLoader loader;
} LoaderMap;

typedef struct LoaderMapVec {
    LoaderMap* loader_map;
    size_t size;
    size_t capacity;
} LoaderMapVec;

static LoaderMapVec loader_maps;

static int insert_loader(LoaderMapVec* lm, const MGBOprLoader* loader) {
    for (int i = 0; i < lm->size; ++i) {
        if (strcmp(lm->loader_map[i].loader.name, loader->name) == 0) {
            return 0;
        }
    }

    if (lm->capacity == 0) {
        lm->capacity = 2;
        lm->loader_map = tinynn_malloc(sizeof(LoaderMap) * lm->capacity);
    }
    if (lm->size >= lm->capacity) {
        lm->capacity *= 2;
        LoaderMap* tmp = tinynn_malloc(sizeof(LoaderMap) * lm->capacity);
        memcpy(tmp, lm->loader_map, sizeof(LoaderMap) * lm->size);
        tinynn_free(lm->loader_map);
        lm->loader_map = tmp;
    }

    lm->loader_map[lm->size].loader = *loader;
    ++lm->size;
    return 1;
}

static int register_loader(const MGBOprLoader* loader) {
    return insert_loader(&loader_maps, loader);
}

static int delete_loader(LoaderMapVec* lm, const char* name) {
    for (int i = 0; i < lm->size; ++i) {
        if (strcmp(lm->loader_map[i].loader.name, name) == 0) {
            memmove(lm->loader_map + i, lm->loader_map + i + 1,
                    lm->size - i - 1);
            --lm->size;
            return 1;
        }
    }
    return 0;
}

static int unregister_loader(const char* name) {
    return delete_loader(&loader_maps, name);
}

static LoaderMap* find_loader_by_name(const LoaderMapVec* lm,
                                      const char* name) {
    for (int i = 0; i < lm->size; ++i) {
        if (strcmp(lm->loader_map[i].loader.name, name) == 0) {
            return lm->loader_map + i;
        }
    }
    return NULL;
}

static void free_loader_maps(LoaderMapVec* lm) {
    if (lm->loader_map) {
        tinynn_free(lm->loader_map);
        lm->loader_map = NULL;
    }
}

//! get API ptr for specific version; return nullptr if version mismatch
const MGBExternCOprApi* megcc_get_extern_c_opr_api_versioned(int version) {
    static MGBExternCOprApi api;
    api.unregister_loader = unregister_loader;
    TINYNN_ASSERT_MSG(version >= 0x24,
                      "Extern opr loader version must greater than 0x24.\n");

    if (version != MGB_EXTERN_C_OPR_VERSION) {
        return NULL;
    }

    api.register_loader = register_loader;
    return &api;
}

// Convert Tensor to MGBTensor, except MGBTensor.data.
static void Tensor2MGBTensor(const Tensor* tensor, MGBTensor* mgb_tensor) {
    mgb_tensor->layout.shape.ndim = tensor->layout.nr_dim;
    for (int i = 0; i < tensor->layout.nr_dim; ++i) {
        mgb_tensor->layout.shape.shape[i] = tensor->layout.dims[i];
    }
    switch (tensor->dtype.type_enum) {
        case TinyNN_FLOAT:
            mgb_tensor->layout.dtype = MGB_DTYPE_FLOAT32;
            break;
        case TinyNN_FLOAT16:
            mgb_tensor->layout.dtype = MGB_DTYPE_FLOAT16;
            break;
        case TinyNN_INT:
            mgb_tensor->layout.dtype = MGB_DTYPE_INT32;
            break;
        case TinyNN_INT16:
            mgb_tensor->layout.dtype = MGB_DTYPE_INT16;
            break;
        case TinyNN_UINT8:
            mgb_tensor->layout.dtype = MGB_DTYPE_UINT8;
            break;
        default:
            TINYNN_ASSERT_MSG(0, "Unsupport data type\n");
    }
}

static void MGBTensor2Tensor(const MGBTensor* mgb_tensor, Tensor* tensor) {
    tensor->layout.nr_dim = mgb_tensor->layout.shape.ndim;
    for (int i = 0; i < mgb_tensor->layout.shape.ndim; ++i) {
        tensor->layout.dims[i] = mgb_tensor->layout.shape.shape[i];
    }

    switch (mgb_tensor->layout.dtype) {
        case MGB_DTYPE_FLOAT32:
            tensor->dtype.type_enum = TinyNN_FLOAT;
            break;
        case MGB_DTYPE_FLOAT16:
            tensor->dtype.type_enum = TinyNN_FLOAT16;
            break;
        case MGB_DTYPE_INT32:
            tensor->dtype.type_enum = TinyNN_INT;
            break;
        case MGB_DTYPE_INT16:
            tensor->dtype.type_enum = TinyNN_INT16;
            break;
        case MGB_DTYPE_UINT8:
            tensor->dtype.type_enum = TinyNN_UINT8;
            break;
        default:
            TINYNN_ASSERT_MSG(0, "Unsupport data type\n");
    }
}

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define RTLD_LAZY 0

static void* dlopen(const char* file, int) {
    return (void*)(LoadLibrary(file));
}

static void* dlsym(void* handle, const char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return (void*)symbol;
}

#else
#include <dlfcn.h>
#endif

static int has_set_env_and_loader = 0;

static TinyNNStatus load(flatbuffers_generic_t fbs_inst, Instruction* inst,
                         VM* vm) {
    ExternOpr* extern_opr = &inst->workload.extern_opr;
    DeviceModel* model = get_active_device_model(vm);
    ns(ExternOpr_table_t) fbs_extern_opr = (ns(ExternOpr_table_t))(fbs_inst);
    inst->tag = TinyNN_INST_EXTERN_OPR;

    flatbuffers_int32_vec_t fbs_inputs = ns(ExternOpr_input(fbs_extern_opr));
    extern_opr->nr_input = flatbuffers_int32_vec_len(fbs_inputs);
    extern_opr->inputs = tinynn_malloc(sizeof(Tensor*) * extern_opr->nr_input);
    for (int i = 0; i < extern_opr->nr_input; ++i) {
        extern_opr->inputs[i] = model->tensors + fbs_inputs[i];
    }

    flatbuffers_int32_vec_t fbs_outputs = ns(ExternOpr_output(fbs_extern_opr));
    extern_opr->nr_output = flatbuffers_int32_vec_len(fbs_outputs);
    extern_opr->outputs =
            tinynn_malloc(sizeof(Tensor*) * extern_opr->nr_output);
    for (int i = 0; i < extern_opr->nr_output; ++i) {
        extern_opr->outputs[i] = model->tensors + fbs_outputs[i];
    }

    char* name = ns(ExternOpr_name(fbs_extern_opr));
    const void* data = ns(ExternOpr_data(fbs_extern_opr));
    size_t data_len = ns(ExternOpr_data_len(fbs_extern_opr));
    int idx = 0;
    while (name[idx] != '\0' && name[idx] != ':')
        ++idx;
    name[idx] = '\0';

    if (!has_set_env_and_loader) {
        const void* extra_data = data + data_len;
        // parse and set ENV
        size_t nr_env = *(size_t*)extra_data;
        extra_data += sizeof(size_t);
        for (int i = 0; i < nr_env; ++i) {
            size_t env_len = *(size_t*)extra_data;
            extra_data += sizeof(size_t);
            char* env = (char*)tinynn_malloc(env_len + 1);
            memcpy(env, extra_data, env_len);
            env[env_len] = '\0';
            extra_data += env_len;

            size_t value_len = *(size_t*)extra_data;
            extra_data += sizeof(size_t);
            char* value = (char*)tinynn_malloc(value_len + 1);
            memcpy(value, extra_data, value_len);
            value[value_len] = '\0';
            extra_data += value_len;

            TINYNN_ASSERT_MSG((!setenv(env, value, 1)),
                              "setenv failed.\n");  // 1 means overwrite when
                                                    // 'env' does exist.
            LOG_DEBUG("Set ENV: %s=%s\n", env, value);

            tinynn_free(env);
            tinynn_free(value);
        }

        // load loader
        size_t loader_path_len = *(size_t*)extra_data;
        extra_data += sizeof(size_t);
        if (loader_path_len) {
            char* loader_path = tinynn_malloc(loader_path_len + 1);
            memcpy(loader_path, extra_data, loader_path_len);
            extra_data += loader_path_len;
            loader_path[loader_path_len] = '\0';
            LOG_DEBUG("Try to load loader in path %s.\n", loader_path);
            void* handle = dlopen(loader_path, RTLD_LAZY);
            // if dlopen failed, but loader path is NOT absolute path.
            if (!handle && loader_path[0] != '/') {
                // try current path
                char* extend_loader_path = tinynn_malloc(loader_path_len + 3);
                extend_loader_path[0] = '.';
                extend_loader_path[1] = '/';
                memcpy(extend_loader_path + 2, loader_path,
                       loader_path_len + 1);
                LOG_DEBUG(
                        "Load loader in path %s failed. Now try to load loader "
                        "in path %s.\n",
                        loader_path, extend_loader_path);
                handle = dlopen(extend_loader_path, RTLD_LAZY);
                tinynn_free(extend_loader_path);
            }
            tinynn_free(loader_path);
            TINYNN_ASSERT_MSG(handle,
                              "Load loader failed. Can NOT find loader file in "
                              "given path.\n");

            size_t interface_len = *(size_t*)extra_data;
            extra_data += sizeof(size_t);
            char* c_opr_lib_interface = tinynn_malloc(interface_len + 1);
            memcpy(c_opr_lib_interface, extra_data, interface_len);
            c_opr_lib_interface[interface_len] = '\0';
            void (*func)(const MGBExternCOprApi* (*)(int)) = NULL;
            *(void**)&func = dlsym(handle, c_opr_lib_interface);
            tinynn_free(c_opr_lib_interface);
            TINYNN_ASSERT_MSG(func, "load init interface of loader failed.\n");
            func(megcc_get_extern_c_opr_api_versioned);
        }
        has_set_env_and_loader = 1;
    }

    LoaderMap* loader_map = find_loader_by_name(&loader_maps, name);
    TINYNN_ASSERT_MSG(loader_map, "Wrong loader.\n");
    extern_opr->desc = loader_map->loader.create_desc(extern_opr->nr_input,
                                                      data, data_len);

    extern_opr->mgb_inputs =
            tinynn_malloc(sizeof(MGBTensor) * extern_opr->nr_input);
    MGBTensorShape* inputs_shape =
            tinynn_malloc(sizeof(MGBTensorShape) * extern_opr->nr_input);
    MGBDType* inputs_type =
            tinynn_malloc(sizeof(MGBDType) * extern_opr->nr_input);
    for (int i = 0; i < extern_opr->nr_input; ++i) {
        Tensor2MGBTensor(extern_opr->inputs[i], extern_opr->mgb_inputs + i);
        inputs_shape[i] = extern_opr->mgb_inputs[i].layout.shape;
        inputs_type[i] = extern_opr->mgb_inputs[i].layout.dtype;
    }

    extern_opr->mgb_outputs =
            tinynn_malloc(sizeof(MGBTensor) * extern_opr->nr_output);
    MGBTensorShape* outputs_shape =
            tinynn_malloc(sizeof(MGBTensorShape) * extern_opr->nr_output);
    MGBDType* outputs_type =
            tinynn_malloc(sizeof(MGBDType) * extern_opr->nr_output);

    extern_opr->desc->infer_shape(extern_opr->desc, inputs_shape,
                                  outputs_shape);
    if (extern_opr->desc->infer_dtype) {
        extern_opr->desc->infer_dtype(extern_opr->desc, inputs_type,
                                      outputs_type);
    } else {
        for (int i = 0; i < extern_opr->nr_output; ++i) {
            outputs_type[i] = inputs_type[0];
        }
    }

    for (int i = 0; i < extern_opr->nr_output; ++i) {
        extern_opr->mgb_outputs[i].layout.dtype = outputs_type[i];
        extern_opr->mgb_outputs[i].layout.shape.ndim = outputs_shape[i].ndim;
        for (int j = 0; j < extern_opr->mgb_outputs[i].layout.shape.ndim; ++j) {
            extern_opr->mgb_outputs[i].layout.shape.shape[j] =
                    outputs_shape[i].shape[j];
        }
    }

    tinynn_free(outputs_shape);
    tinynn_free(outputs_type);

    tinynn_free(inputs_shape);
    tinynn_free(inputs_type);

    return TinyNN_SUCCESS;
}

static TinyNNStatus execute(Instruction* inst, VM* vm) {
    ExternOpr* extern_opr = &inst->workload.extern_opr;

    for (int i = 0; i < extern_opr->nr_input; ++i) {
        extern_opr->mgb_inputs[i].data = extern_opr->inputs[i]->ptr;
    }
    for (int i = 0; i < extern_opr->nr_output; ++i) {
        extern_opr->mgb_outputs[i].data = extern_opr->outputs[i]->ptr;
    }
    extern_opr->desc->execute(extern_opr->desc, extern_opr->mgb_inputs,
                              extern_opr->mgb_outputs);
    for (int i = 0; i < extern_opr->nr_output; ++i) {
        MGBTensor2Tensor(extern_opr->mgb_outputs + i, extern_opr->outputs[i]);
    }

    return TinyNN_SUCCESS;
}

static TinyNNStatus destruct(VM* vm, Instruction* inst) {
    ExternOpr* extern_opr = &inst->workload.extern_opr;

    FREE(extern_opr->inputs);
    FREE(extern_opr->outputs);
    FREE(extern_opr->mgb_outputs);
    FREE(extern_opr->mgb_inputs);

    free_loader_maps(&loader_maps);

    return TinyNN_SUCCESS;
}

void register_extern_opr(VM* vm) {
    vm_register_instruction_load(vm, ns(Instruction_ExternOpr), &load);
    vm_register_instruction_call(vm, TinyNN_INST_EXTERN_OPR, &execute);
    vm_register_instruction_destruct(vm, TinyNN_INST_EXTERN_OPR, &destruct);
}
#else
void register_extern_opr(VM* vm) {}

const MGBExternCOprApi* megcc_get_extern_c_opr_api_versioned(int i) {
    TINYNN_ASSERT_MSG(
            0,
            "Should NOT execute here!!!\n"
            "Maybe there is no extern opr in model, "
            "but command line argument --c-opr-lib/-c is provided.\n");
    return NULL;
}
#endif
// vim: syntax=cpp.doxygen
