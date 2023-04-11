#ifndef TINYNN_CALLBACK_H_
#define TINYNN_CALLBACK_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "stddef.h"

#define TINYNN_CB_VERSION 0x18

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief libc likely function ops
 */
typedef struct {
    /*!
     * \brief function like printf
     */
    int (*tinynn_log_cb)(const char* format, ...);

    /*!
     * \brief function like malloc/free
     */
    void* (*tinynn_malloc_cb)(size_t size);
    void (*tinynn_free_cb)(void*);

    /*!
     * \brief function like gettime
     * seconds and microseconds
     */
    void (*tinynn_gettime_cb)(int32_t* sec, int32_t* usec);

    /*!
     * \brief function like fopen/ftell/fseek/fclose
     */
    FILE* (*tinynn_fopen_cb)(const char* pathname, const char* mode);
    long (*tinynn_ftell_cb)(FILE* stream);
    int (*tinynn_fseek_cb)(FILE* stream, long offset, int whence);
    int (*tinynn_fclose_cb)(FILE* stream);
    size_t (*tinynn_fwrite_cb)(
            const void* ptr, size_t size, size_t nmemb, FILE* stream);
    size_t (*tinynn_fread_cb)(void* ptr, size_t size, size_t nmemb, FILE* stream);
} TinyNnCallBack;

/**
 *  use to register tinynn callback function
 *  define at TinyNnCallBack
 */
void register_tinynn_cb(int cb_version, const TinyNnCallBack);

#ifdef __cplusplus
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
