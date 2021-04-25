/**
 * \file runtime/src/utils.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#ifndef MEGCC_UTILS_H
#define MEGCC_UTILS_H

#include "lite-c/common_enum_c.h"
#include "tinynn_callback.h"

#define tinynn_trap() __builtin_trap()

typedef int (*tinynn_log)(const char* format, ...);
extern tinynn_log __tinynn_log__;
extern LiteLogLevel g_log_level;

#define LOG_ERROR(msg...)                                           \
    if (ERROR >= g_log_level) {                                     \
        __tinynn_log__("TinyNN ERROR:%s@%d: ", __func__, __LINE__); \
        __tinynn_log__(msg);                                        \
    }
#define LOG_WARNING(msg...)                                        \
    if (WARN >= g_log_level) {                                     \
        __tinynn_log__("TinyNN WARN:%s@%d: ", __func__, __LINE__); \
        __tinynn_log__(msg);                                       \
    }
#define LOG_INFO(msg...)                                           \
    if (INFO >= g_log_level) {                                     \
        __tinynn_log__("TinyNN INFO:%s@%d: ", __func__, __LINE__); \
        __tinynn_log__(msg);                                       \
    }
#define LOG_DEBUG(msg...)                                           \
    if (DEBUG >= g_log_level) {                                     \
        __tinynn_log__("TinyNN DEBUG:%s@%d: ", __func__, __LINE__); \
        __tinynn_log__(msg);                                        \
    }
#define LOG_DEBUG_NO_PREFIX(msg...)  \
    if (DEBUG >= g_log_level) {      \
        __tinynn_log__(msg);         \
    }

#define TINYNN_ASSERT(exp)                                                   \
    do {                                                                     \
        if (!(exp)) {                                                        \
            __tinynn_log__("TinyNN ASSERT failed:%s location:%s@%d\n", #exp, \
                           __func__, __LINE__);                              \
            tinynn_trap();                                                   \
        }                                                                    \
    } while (0)

#define TINYNN_ASSERT_MSG(exp, msg...)                                       \
    do {                                                                     \
        if (!(exp)) {                                                        \
            __tinynn_log__("TinyNN ASSERT failed:%s location:%s@%d: ", #exp, \
                           __func__, __LINE__);                              \
            __tinynn_log__("%s\n", msg);                                     \
            tinynn_trap();                                                   \
        }                                                                    \
    } while (0)

/**
 *  public api for internal runtime use
 */
void tinynn_gettime(int32_t* sec, int32_t* usec);
void* tinynn_malloc(size_t size);
void tinynn_free(void*);
FILE* tinynn_fopen(const char* pathname, const char* mode);
long tinynn_ftell(FILE* stream);
int tinynn_fseek(FILE* stream, long offset, int whence);
int tinynn_fclose(FILE* stream);
size_t tinynn_fwrite(const void* ptr, size_t size, size_t nmemb, FILE* stream);
size_t tinynn_fread(void* ptr, size_t size, size_t nmemb, FILE* stream);

#endif
// vim: syntax=cpp.doxygen
