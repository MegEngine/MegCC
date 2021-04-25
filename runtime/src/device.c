/**
 * \file runtime/src/device.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "stdio.h"
#include "stdlib.h"

#include "data_struct.h"
#include "device.h"

#define CPU_ALIGNMENT (16)

static void* cpu_aligned_malloc(size_t size) {
    if (size == 0) {
        return NULL;
    }
    void* raw_ptr = tinynn_malloc(size + sizeof(void*) + CPU_ALIGNMENT);
    if (!raw_ptr) {
        LOG_ERROR("malloc memory fail.\n");
        return NULL;
    }
    void** ptr =
            (void**)(((size_t)raw_ptr + sizeof(void*) + CPU_ALIGNMENT - 1) &
                     (-CPU_ALIGNMENT));
    ptr[-1] = raw_ptr;
    return ptr;
}

static void cpu_aligned_free(void* ptr) {
    void** raw_ptr = ((void**)(ptr))[-1];
    tinynn_free(raw_ptr);
}

TinyNNStatus init_device(Device* device) {
    if (!device) {
        return TinyNN_ERROR_NULL_PTR;
    }
    if (device->device_type == TinyNN_ARM64 ||
        device->device_type == TinyNN_ARM32 ||
        device->device_type == TinyNN_BARE_METAL) {
        device->alignment = CPU_ALIGNMENT;
        device->malloc = cpu_aligned_malloc;
        device->free = cpu_aligned_free;
    } else {
        LOG_ERROR("not support device.\n");
        return TinyNN_ERROR_NO_IMPLEMENT;
    }
    return TinyNN_SUCCESS;
}

RuntimeOpt create_runtime_opt(Device* device) {
    if (!device) {
        LOG_ERROR("not support device.\n");
    }
    RuntimeOpt opt;
    opt.device = device;
    return opt;
}

// vim: syntax=cpp.doxygen
