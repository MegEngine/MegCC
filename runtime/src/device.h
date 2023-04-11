#ifndef DEVICE_H
#define DEVICE_H

#include "stdio.h"
#include "stdlib.h"
#include "tinynn.h"

typedef struct {
    int alignment;
    TinyNNDevice device_type;
    //! device malloc interface, use void* to represent every memory type
    void* (*malloc)(size_t length);
    //! device free interface
    void (*free)(void* ptr);

} Device;

//! pass runtimeOpt to kernel for expand later
typedef struct RuntimeOpt {
    const Device* device;
} RuntimeOpt;

TinyNNStatus init_device(Device* device);

RuntimeOpt create_runtime_opt(Device* device);

#endif

// vim: syntax=cpp.doxygen
