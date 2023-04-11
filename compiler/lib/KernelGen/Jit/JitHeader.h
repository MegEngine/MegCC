#pragma once

#include <string>

namespace megcc {
namespace KernelGen {

static std::string get_runtime_define() {
    return R"(

        #define MAX_DIM (7)
        #define TINYNN_ASSERT(exp)
        typedef enum {
            TinyNN_SUCCESS = 0,
            TinyNN_ERROR_NULL_PTR = 1,
            TinyNN_ERROR_STANDER = 2,
            TinyNN_ERROR_RUNTIME = 3,
            TinyNN_ERROR_OUT_OF_RANGE = 4,
            TinyNN_ERROR_NO_FOUND = 5,
            TinyNN_ERROR_NO_IMPLEMENT = 6,
            TinyNN_ERROR_MODEL_PARSE = 7,
            TinyNN_ERROR_OPEN_FILE_ERROR = 8,
            TinyNN_ERROR_MEMORY_MALLOC = 9,
            TinyNN_ERROR_UNSUPPORTED_INSTRUCTION_TYPE = 10,
            TinyNN_ERROR_UNSUPPORTED_DTYPE_TYPE = 11,
            TinyNN_ERROR_INVALID_LAYOUT = 12,
            TinyNN_ERROR = 13,
        } TinyNNStatus;

        typedef enum {
            TinyNN_FLOAT = 0,
            TinyNN_FLOAT16 = 1,
            TinyNN_INT = 2,
            TinyNN_INT8 = 3,
            TinyNN_INT16 = 4,
            TinyNN_UINT8 = 5,
            TinyNN_QINT8 = 100,
            TinyNN_QINT32 = 101,
            TinyNN_QUINT8 = 102,
        } TinyNNDType;

        typedef enum {
            TinyNN_NCHW = 0,
            TinyNN_NHWC,
            TinyNN_NCHW4,
            TinyNN_NCHW8,
            TinyNN_OIHW,
        } TinyNNFormat;

        typedef struct {
            float scale;
            uint8_t zero_point;
        } DTypeParam;

        typedef struct {
            TinyNNDType type_enum;
            DTypeParam param;
        } DType;

        typedef struct {
            int nr_dim;
            uint32_t dims[MAX_DIM];
            int stride[MAX_DIM];
            TinyNNFormat format;
        } Layout;

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
        } Tensor;

    )";
}

static std::string get_stddef_header() {
    return R"(

         #ifndef _STDDEF_H
         #define _STDDEF_H

         typedef __SIZE_TYPE__ size_t;
         typedef __PTRDIFF_TYPE__ ssize_t;
         typedef __WCHAR_TYPE__ wchar_t;
         typedef __PTRDIFF_TYPE__ ptrdiff_t;
         typedef __PTRDIFF_TYPE__ intptr_t;
         typedef __SIZE_TYPE__ uintptr_t;

         #ifndef __int8_t_defined
         #define __int8_t_defined
         typedef signed char int8_t;
         typedef signed short int int16_t;
         typedef signed int int32_t;
         #ifdef __LP64__
         typedef signed long int int64_t;
         #else
         typedef signed long long int int64_t;
         #endif
         typedef unsigned char uint8_t;
         typedef unsigned short int uint16_t;
         typedef unsigned int uint32_t;
         #ifdef __LP64__
         typedef unsigned long int uint64_t;
         #else
         typedef unsigned long long int uint64_t;
         #endif
         #endif

         #ifndef NULL
         #define NULL ((void*)0)
         #endif

         #define offsetof(type, field) ((size_t)&((type *)0)->field)

         void *alloca(size_t size);

         #endif

         /* Older glibc require a wint_t from <stddef.h> (when requested
            by __need_wint_t, as otherwise stddef.h isn't allowed to
            define this type).   Note that this must be outside the normal
            _STDDEF_H guard, so that it works even when we've included the file
            already (without requiring wint_t).  Some other libs define _WINT_T
            if they've already provided that type, so we can use that as guard.
            TCC defines __WINT_TYPE__ for us.  */
         #if defined (__need_wint_t)
         #ifndef _WINT_T
         #define _WINT_T
         typedef __WINT_TYPE__ wint_t;
         #endif
         #undef __need_wint_t
         #endif

    )";
}

static std::string get_stdarg_header() {
    return R"(

        #ifndef _STDARG_H
        #define _STDARG_H

        #ifdef __x86_64__
        #ifndef _WIN64

         //This should be in sync with the declaration on our lib/libtcc1.c
         /* GCC compatible definition of va_list. */
         typedef struct {
             unsigned int gp_offset;
             unsigned int fp_offset;
             union {
                 unsigned int overflow_offset;
                 char *overflow_arg_area;
             };
             char *reg_save_area;
         } __va_list_struct;

         typedef __va_list_struct va_list[1];

         void __va_start(__va_list_struct *ap, void *fp);
         void *__va_arg(__va_list_struct *ap, int arg_type, int size, int align);

         #define va_start(ap, last) __va_start(ap, __builtin_frame_address(0))
         #define va_arg(ap, type)                                                \
             (*(type *)(__va_arg(ap, __builtin_va_arg_types(type), sizeof(type), __alignof__(type))))
         #define va_copy(dest, src) (*(dest) = *(src))
         #define va_end(ap)

         /* avoid conflicting definition for va_list on Macs. */
         #define _VA_LIST_T

        #else /* _WIN64 */
         typedef char *va_list;
         #define va_start(ap,last) __builtin_va_start(ap,last)
         #define va_arg(ap, t) ((sizeof(t) > 8 || (sizeof(t) & (sizeof(t) - 1))) \
         >   ? **(t **)((ap += 8) - 8) : *(t  *)((ap += 8) - 8))
         #define va_copy(dest, src) ((dest) = (src))
         #define va_end(ap)
         #endif

         #elif __arm__
         typedef char *va_list;
         #define _tcc_alignof(type) ((int)&((struct {char c;type x;} *)0)->x)
         #define _tcc_align(addr,type) (((unsigned)addr + _tcc_alignof(type) - 1) \
                                        & ~(_tcc_alignof(type) - 1))
         #define va_start(ap,last) ap = ((char *)&(last)) + ((sizeof(last)+3)&~3)
         #define va_arg(ap,type) (ap = (void *) ((_tcc_align(ap,type)+sizeof(type)+3) \
                                 &~3), *(type *)(ap - ((sizeof(type)+3)&~3)))
         #define va_copy(dest, src) (dest) = (src)
         #define va_end(ap)

         #elif defined(__aarch64__)
         typedef struct {
             void *__stack;
             void *__gr_top;
             void *__vr_top;
             int   __gr_offs;
             int   __vr_offs;
         } va_list;
         #define va_start(ap, last) __va_start(ap, last)
         #define va_arg(ap, type) __va_arg(ap, type)
         #define va_end(ap)
         #define va_copy(dest, src) ((dest) = (src))

         #else /* __i386__ */
         typedef char *va_list;
         /* only correct for i386 */
         #define va_start(ap,last) ap = ((char *)&(last)) + ((sizeof(last)+3)&~3)
         #define va_arg(ap,type) (ap += (sizeof(type)+3)&~3, *(type *)(ap - ((sizeof(type)+3)&~3)))
         #define va_copy(dest, src) (dest) = (src)
         #define va_end(ap)
         #endif

         /* fix a buggy dependency on GCC in libio.h */
         typedef va_list __gnuc_va_list;
         #define _VA_LIST_DEFINED

         #endif /* _STDARG_H */

    )";
}

static std::string get_tcclib_header() {
    std::string headers = get_stddef_header() + get_stdarg_header();
    return headers + R"(

         #ifndef _TCCLIB_H
         #define _TCCLIB_H

         /* stdlib.h */
         void *calloc(size_t nmemb, size_t size);
         void *malloc(size_t size);
         void free(void *ptr);
         void *realloc(void *ptr, size_t size);
         int atoi(const char *nptr);
         long int strtol(const char *nptr, char **endptr, int base);
         unsigned long int strtoul(const char *nptr, char **endptr, int base);
         void exit(int);

         /* stdio.h */
         typedef struct __FILE FILE;
         #define EOF (-1)
         extern FILE *stdin;
         extern FILE *stdout;
         extern FILE *stderr;
         FILE *fopen(const char *path, const char *mode);
         FILE *fdopen(int fildes, const char *mode);
         FILE *freopen(const  char *path, const char *mode, FILE *stream);
         int fclose(FILE *stream);
         size_t  fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
         size_t  fwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);
         int fgetc(FILE *stream);
         char *fgets(char *s, int size, FILE *stream);
         int getc(FILE *stream);
         int getchar(void);
         char *gets(char *s);
         int ungetc(int c, FILE *stream);
         int fflush(FILE *stream);
         int putchar (int c);

         int printf(const char *format, ...);
         int fprintf(FILE *stream, const char *format, ...);
         int sprintf(char *str, const char *format, ...);
         int snprintf(char *str, size_t size, const  char  *format, ...);
         int asprintf(char **strp, const char *format, ...);
         int dprintf(int fd, const char *format, ...);
         int vprintf(const char *format, va_list ap);
         int vfprintf(FILE  *stream,  const  char *format, va_list ap);
         int vsprintf(char *str, const char *format, va_list ap);
         int vsnprintf(char *str, size_t size, const char  *format, va_list ap);
         int vasprintf(char  **strp,  const  char *format, va_list ap);
         int vdprintf(int fd, const char *format, va_list ap);

         void perror(const char *s);

         /* string.h */
         char *strcat(char *dest, const char *src);
         char *strchr(const char *s, int c);
         char *strrchr(const char *s, int c);
         char *strcpy(char *dest, const char *src);
         void *memcpy(void *dest, const void *src, size_t n);
         void *memmove(void *dest, const void *src, size_t n);
         void *memset(void *s, int c, size_t n);
         char *strdup(const char *s);
         size_t strlen(const char *s);

         /* dlfcn.h */
         #define RTLD_LAZY       0x001
         #define RTLD_NOW        0x002
         #define RTLD_GLOBAL     0x100

         void *dlopen(const char *filename, int flag);
         const char *dlerror(void);
         void *dlsym(void *handle, char *symbol);
         int dlclose(void *handle);

         #endif /* _TCCLIB_H */

    )";
}

static std::string get_header_define() {
    return get_tcclib_header() + get_runtime_define();
}

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
