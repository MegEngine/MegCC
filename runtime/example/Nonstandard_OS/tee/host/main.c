/**
 * \file runtime/example/Nonstandard_OS/tee/host/main.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <megcc_inference_ta.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <tee_client_api.h>
#include "api.h"

static TinyNnCallBack g_cb = {
        .tinynn_log_cb = printf,
        .tinynn_gettime_cb = gettimeofday,
        .tinynn_malloc_cb = malloc,
        .tinynn_free_cb = free,
};

int ca_init_megcc_deamo() {
    // LITE_set_log_level(DEBUG);
    register_tinynn_cb(TINYNN_CB_VERSION, g_cb);
    return init_megcc_test("data", &g_cb);
}

int ca_free_megcc_deamo() {
    return free_megcc_test();
}

float ca_run_megcc_deamo(int iter) {
    return run_megcc_test(iter);
}

#define megcc_ca_unlikely(v) __builtin_expect(!!(v), 0)
#define MEGCC_CA_ERR(fmt, v...)              \
    do {                                     \
        printf("MEGCC CA issue happened: "); \
        printf(fmt "\n", ##v);               \
        __builtin_trap();                    \
    } while (0)

#define MEGCC_CA_INFO(fmt, v...) \
    do {                         \
        printf("MEGCC CA: ");    \
        printf(fmt "\n", ##v);   \
    } while (0)

#define megcc_ca_assert(expr, ...)        \
    do {                                  \
        if (megcc_ca_unlikely(!(expr))) { \
            MEGCC_CA_ERR(__VA_ARGS__);    \
        }                                 \
    } while (0)

int run_ta_model(int iter) {
    TEEC_Result res;
    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_Operation op;
    TEEC_UUID uuid = TA_MEGCC_INFERENCE_UUID;
    uint32_t err_origin;

    res = TEEC_InitializeContext(NULL, &ctx);
    megcc_ca_assert(
            res == TEEC_SUCCESS, "TEEC_InitializeContext failed with code 0x%x", res);

    res = TEEC_OpenSession(
            &ctx, &sess, &uuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    megcc_ca_assert(
            res == TEEC_SUCCESS, "TEEC_Opensession failed with code 0x%x origin 0x%x",
            res, err_origin);

    memset(&op, 0, sizeof(op));

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INOUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
    op.params[0].value.a = iter;

    //! init model
    struct timeval time, e_time;
    gettimeofday(&time, NULL);
    MEGCC_CA_INFO(
            "invoke init model cmd,seconds:%ld, millis: %ld", time.tv_sec,
            time.tv_usec / 1000);
    res = TEEC_InvokeCommand(
            &sess, TA_MEGCC_INFERENCE_CMD_INIT_MODEL, &op, &err_origin);
    megcc_ca_assert(
            res == TEEC_SUCCESS, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
            res, err_origin);

    //! run model
    MEGCC_CA_INFO("begin run model at TA side");
    gettimeofday(&time, NULL);
    res = TEEC_InvokeCommand(&sess, TA_MEGCC_INFERENCE_CMD_RUN_MODEL, &op, &err_origin);
    gettimeofday(&e_time, NULL);
    MEGCC_CA_INFO(
            "run model at TA side take: %ld ms (iter: %d)",
            (e_time.tv_sec - time.tv_sec) * 1000 +
                    (e_time.tv_usec - time.tv_usec) / 1000,
            iter);
    MEGCC_CA_INFO("success run model at TA side");
    megcc_ca_assert(
            res == TEEC_SUCCESS, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
            res, err_origin);

    //! free model
    gettimeofday(&time, NULL);
    MEGCC_CA_INFO(
            "invoke free model cmd,seconds:%ld, millis: %ld", time.tv_sec,
            time.tv_usec / 1000);
    res = TEEC_InvokeCommand(
            &sess, TA_MEGCC_INFERENCE_CMD_FREE_MODEL, &op, &err_origin);
    megcc_ca_assert(
            res == TEEC_SUCCESS, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
            res, err_origin);

    TEEC_CloseSession(&sess);

    TEEC_FinalizeContext(&ctx);

    return op.params[0].value.a;
}

float run_ca_model(int iter) {
    struct timeval time, e_time;
    ca_init_megcc_deamo();

    MEGCC_CA_INFO("begin run model at CA side");
    gettimeofday(&time, NULL);
    float ret = ca_run_megcc_deamo(iter);
    gettimeofday(&e_time, NULL);
    MEGCC_CA_INFO(
            "run model at CA side take: %ld ms (iter: %d)",
            (e_time.tv_sec - time.tv_sec) * 1000 +
                    (e_time.tv_usec - time.tv_usec) / 1000,
            iter);
    MEGCC_CA_INFO("success run model at CA side");

    ca_free_megcc_deamo();

    return ret;
}

int main(void) {
    //! run total iteration number
    int iter = 50;
    float ta_result, ca_result;

    ca_result = run_ca_model(iter);
    ta_result = run_ta_model(iter);
    MEGCC_CA_INFO("TA result is: %f, CA result is: %f", ta_result, ca_result);

    double diff = fabs(ta_result - ca_result);
    if (diff > 1) {
        MEGCC_CA_ERR("result is different in ca and ta\n");
    }

    MEGCC_CA_INFO("run megcc tee example success\n");

    return 0;
}
