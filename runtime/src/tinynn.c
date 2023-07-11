#include "tinynn.h"

#define Enum2String(ENUM) \
    case ENUM:            \
        return #ENUM;

// clang-format off
const char* dtype2string(TinyNNDType type){
    switch(type){
        Enum2String(TinyNN_FLOAT)
        Enum2String(TinyNN_FLOAT16)
        Enum2String(TinyNN_INT)
        Enum2String(TinyNN_INT8)
        Enum2String(TinyNN_INT16)
        Enum2String(TinyNN_UINT8)
        Enum2String(TinyNN_QINT8)
        Enum2String(TinyNN_QINT32)
        Enum2String(TinyNN_QUINT8)
        default: return "UNKNOW";
    }
}

const char* format2string(TinyNNFormat format){
    switch(format){
        Enum2String(TinyNN_NCHW)
        Enum2String(TinyNN_NHWC)
        Enum2String(TinyNN_NCHW4)
        Enum2String(TinyNN_NCHW8)
        Enum2String(TinyNN_OIHW)
        default: return "UNKNOW";
    }
}
// clang-format on

// vim: syntax=cpp.doxygen