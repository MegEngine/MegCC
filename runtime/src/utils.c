/**
 * \file runtime/src/utils.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "utils.h"
LiteLogLevel g_log_level = WARN;
/*************** callback imp ******************/
#if TINYNN_CALLBACK_ENABLE
static TinyNnCallBack g_cb;
static int has_init_cb = 0;
static int has_fs_cb = 0;
static int has_timer_cb = 0;
// default logger
tinynn_log __tinynn_log__ = NULL;
#else
#include <malloc.h>
#include <stdlib.h>
#include <sys/time.h>
//! default gettime imp, we only use gettimeofday
//! when not define TINYNN_CALLBACK_ENABLE, as some deploy
//! env do not imp gettimeofday api, just trap it
static void default_tinynn_gettime(int32_t* sec, int32_t* usec) {
    struct timeval t;
    gettimeofday(&t, NULL);
    *sec = t.tv_sec;
    *usec = t.tv_usec;
}
//! default callback
static TinyNnCallBack g_cb = {
        .tinynn_log_cb = printf,
        .tinynn_gettime_cb = default_tinynn_gettime,
        .tinynn_malloc_cb = malloc,
        .tinynn_free_cb = free,
        .tinynn_fopen_cb = fopen,
        .tinynn_ftell_cb = ftell,
        .tinynn_fseek_cb = fseek,
        .tinynn_fclose_cb = fclose,
        .tinynn_fwrite_cb = fwrite,
        .tinynn_fread_cb = fread,
};
// default logger
tinynn_log __tinynn_log__ = printf;
#endif
//! use to register global callback, non thread safe
//! TODO: add mutex(NOTICE: some non standard os do not have mutex)
void register_tinynn_cb(int cb_version, const TinyNnCallBack cb) {
#if TINYNN_CALLBACK_ENABLE
    //! cb must have imp tinynn_log_cb
    if (!cb.tinynn_log_cb) {
        //! can not use any logger, just trap
        tinynn_trap();
    }
    //! make __tinynn_log__ can be used ASAP
    __tinynn_log__ = cb.tinynn_log_cb;
    if (cb_version != TINYNN_CB_VERSION) {
        __tinynn_log__(
                "ERROR: mismatch callback version: (%d vs %d), please update "
                "include header files!!\n",
                cb_version, TINYNN_CB_VERSION);
        tinynn_trap();
    }
    if (sizeof(cb) != sizeof(g_cb)) {
        __tinynn_log__(
                "ERROR: mismatch callback struct, please update include header "
                "files!!\n");
        tinynn_trap();
    }
    //! cb have imp mem api
    __tinynn_log__("try register runtime callback to global runtime cb\n");
    if (!cb.tinynn_malloc_cb || !cb.tinynn_free_cb) {
        LOG_ERROR("customer cb do not imp all memory callback, trap now!!\n");
        tinynn_trap();
    }
    memcpy(&g_cb, &cb, sizeof(TinyNnCallBack));
    //! fs api just check at use stage
    if (!g_cb.tinynn_fopen_cb || !g_cb.tinynn_ftell_cb ||
        !g_cb.tinynn_fseek_cb || !g_cb.tinynn_fclose_cb ||
        !g_cb.tinynn_fwrite_cb || !g_cb.tinynn_fread_cb) {
        LOG_WARNING(
                "customer do not imp file api, TINYNN_DUMP_TENSOR and "
                "LITE_load_model_from_path function will trap!!\n")
        has_fs_cb = 0;
    } else {
        has_fs_cb = 1;
    }
    //! timer api just check at use stage
    if (!g_cb.tinynn_gettime_cb) {
        LOG_WARNING(
                "customer do not imp timer api, TINYNN_PROFILE_KERNEL function "
                "will trap!!\n")
        has_timer_cb = 0;
    } else {
        has_timer_cb = 1;
    }

    has_init_cb = 1;
    __tinynn_log__(
            "register runtime callback to global runtime cb success!!\n");
#else
    (void)&cb_version;
    (void)&cb;
    LOG_ERROR(
            "do not support register_tinynn_cb build without "
            "TINYNN_CALLBACK_ENABLE\n");
    tinynn_trap();
#endif
}

static void ensure_already_register() {
#if TINYNN_CALLBACK_ENABLE
    //! may have not available log_cb to print log
    if (0 == has_init_cb) {
        tinynn_trap();
    }
#endif
}

void tinynn_gettime(int32_t* sec, int32_t* usec) {
#if TINYNN_CALLBACK_ENABLE
    ensure_already_register();
    if (!has_timer_cb) {
        LOG_ERROR("customer do not imp tinynn_gettime function, trap now!!\n");
        tinynn_trap();
    }
#endif
    g_cb.tinynn_gettime_cb(sec, usec);
}

void* tinynn_malloc(size_t size) {
    ensure_already_register();
    return g_cb.tinynn_malloc_cb(size);
}

void tinynn_free(void* ptr) {
    ensure_already_register();
    g_cb.tinynn_free_cb(ptr);
}

/*************** file imp ******************/
static void ensure_register_file_api() {
#if TINYNN_CALLBACK_ENABLE
    ensure_already_register();
    if (!has_fs_cb) {
        LOG_ERROR("customer do not imp file api, trap now\n")
        tinynn_trap();
    }
#endif
}
FILE* tinynn_fopen(const char* pathname, const char* mode) {
    ensure_register_file_api();
    return g_cb.tinynn_fopen_cb(pathname, mode);
}
long tinynn_ftell(FILE* stream) {
    ensure_register_file_api();
    return g_cb.tinynn_ftell_cb(stream);
}
int tinynn_fseek(FILE* stream, long offset, int whence) {
    ensure_register_file_api();
    return g_cb.tinynn_fseek_cb(stream, offset, whence);
}
int tinynn_fclose(FILE* stream) {
    ensure_register_file_api();
    return g_cb.tinynn_fclose_cb(stream);
}
size_t tinynn_fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream) {
    ensure_register_file_api();
    return g_cb.tinynn_fwrite_cb(ptr, size, nmemb, stream);
}
size_t tinynn_fread(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    ensure_register_file_api();
    return g_cb.tinynn_fread_cb(ptr, size, nmemb, stream);
}
// vim: syntax=cpp.doxygen
