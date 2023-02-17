/**
 * \file runtime/example/Nonstandard_OS/tee/ta/megcc_inference_ta.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include <megcc_inference_ta.h>

#define MEGCC_TA_DBG(fmt, v...)          \
    do {                                 \
        DMSG("MEGCC TA:" fmt "\n", ##v); \
    } while (0)

extern float ta_run_megcc_deamo(int iter);
extern int ta_init_megcc_deamo();
extern int ta_free_megcc_deamo();

TEE_Result TA_CreateEntryPoint(void) {
    MEGCC_TA_DBG("");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void) {
    MEGCC_TA_DBG();
}

TEE_Result TA_OpenSessionEntryPoint(
        uint32_t param_types, TEE_Param __maybe_unused params[4],
        void __maybe_unused** sess_ctx) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
            TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
            TEE_PARAM_TYPE_NONE);

    MEGCC_TA_DBG();

    if (param_types != exp_param_types) {
        MEGCC_TA_DBG("error: mismatching param!");
        return TEE_ERROR_BAD_PARAMETERS;
    }

    (void)&params;
    (void)&sess_ctx;

    MEGCC_TA_DBG("sunccess!");

    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void __maybe_unused* sess_ctx) {
    (void)&sess_ctx; /* Unused parameter */
    MEGCC_TA_DBG("Goodbye!");
}

static TEE_Result init_model(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
            TEE_PARAM_TYPE_VALUE_INOUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
            TEE_PARAM_TYPE_NONE);

    (void)&params;
    MEGCC_TA_DBG("");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    int ret = ta_init_megcc_deamo();
    if (ret)
        return TEE_ERROR_GENERIC;

    return TEE_SUCCESS;
}

static TEE_Result run_model(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
            TEE_PARAM_TYPE_VALUE_INOUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
            TEE_PARAM_TYPE_NONE);

    MEGCC_TA_DBG("");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float sum = ta_run_megcc_deamo(params[0].value.a);
    params[0].value.a = sum;
    MEGCC_TA_DBG("set result: %u to ca param", params[0].value.a);

    return TEE_SUCCESS;
}

static TEE_Result free_model(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
            TEE_PARAM_TYPE_VALUE_INOUT, TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE,
            TEE_PARAM_TYPE_NONE);

    (void)&params;
    MEGCC_TA_DBG("");

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    int ret = ta_free_megcc_deamo();
    if (ret)
        return TEE_ERROR_GENERIC;

    return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(
        void __maybe_unused* sess_ctx, uint32_t cmd_id, uint32_t param_types,
        TEE_Param params[4]) {
    (void)&sess_ctx;
    TEE_Time t;
    TEE_GetSystemTime(&t);
    MEGCC_TA_DBG("time: seconds:%d, millis: %d\n", t.seconds, t.millis);

    switch (cmd_id) {
        case TA_MEGCC_INFERENCE_CMD_INIT_MODEL:
            return init_model(param_types, params);
        case TA_MEGCC_INFERENCE_CMD_RUN_MODEL:
            return run_model(param_types, params);
        case TA_MEGCC_INFERENCE_CMD_FREE_MODEL:
            return free_model(param_types, params);
        default:
            return TEE_ERROR_BAD_PARAMETERS;
    }
}
