/**
 * \file runtime/example/Nonstandard_OS/tee/ta/api.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "api.h"
#include <tee_internal_api.h>

// define libm global error val
int __errno = 0;

static int logger_imp(const char* fmt, ...) {
#define MAX_LOG_SIZE 200
    va_list ap;
    char buff[MAX_LOG_SIZE];
    va_start(ap, fmt);
    vsnprintf(buff, sizeof(buff), fmt, ap);
    va_end(ap);
    //! MAX_LOG_SIZE may overflow
    buff[MAX_LOG_SIZE - 1] = '\n';
    DMSG(buff);
    return 0;
}

static void test_timeimp(int32_t* sec, int32_t* usec) {
    TEE_Time t;
    TEE_GetSystemTime(&t);
    *sec = t.seconds;
    *usec = t.millis;
}

static TinyNnCallBack g_cb = {
        .tinynn_log_cb = logger_imp,
        .tinynn_gettime_cb = test_timeimp,
        .tinynn_malloc_cb = malloc,
        .tinynn_free_cb = free,
};

int ta_init_megcc_deamo() {
    // LITE_set_log_level(LITE_DEBUG);
    register_tinynn_cb(TINYNN_CB_VERSION, g_cb);
    TEE_Time t;
    TEE_GetSystemTime(&t);
    logger_imp(
            "into %s: %d at: (sec: %u millis:%u)\n", __FUNCTION__, __LINE__, t.seconds,
            t.millis);
    init_megcc_test("data", &g_cb);
    return 0;
}

int ta_free_megcc_deamo() {
    TEE_Time t;
    TEE_GetSystemTime(&t);
    logger_imp(
            "into %s: %d at: (sec: %u millis:%u)\n", __FUNCTION__, __LINE__, t.seconds,
            t.millis);
    free_megcc_test();
    return 0;
}

float ta_run_megcc_deamo(int iter) {
    TEE_Time t;
    TEE_GetSystemTime(&t);
    logger_imp(
            "into %s: %d at: (sec: %u millis:%u)\n", __FUNCTION__, __LINE__, t.seconds,
            t.millis);
    return run_megcc_test(iter);
}
