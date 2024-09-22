#ifndef TINYNN_H
#define TINYNN_H

#include "stddef.h"
#include "stdint.h"

#define MAX_DIM (7)

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

typedef enum {
    TinyNN_BARE_METAL = 0,
    TinyNN_ARM64,
    TinyNN_ARM32,
    TinyNN_ARM64_V82,
    TinyNN_ARM32_V82,
    TinyNN_OPENCL_MALI,
    TinyNN_WEB_ASSEMBLY
} TinyNNDevice;

const char* dtype2string(TinyNNDType type);
const char* format2string(TinyNNFormat format);

#endif
// vim: syntax=cpp.doxygen
