/**
 * \file runtime/example/Nonstandard_OS/freeRTOS/tinynn_freeRTOS_example.c
 *
 * This file is part of tinynn, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <stdarg.h>
#include "FreeRTOS.h"
#include "lite-c/common_enum_c.h"
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "tinynn_callback.h"

//! define by mgb-to-tinynn with args: --save-model and xxd -i input.bin
//! >input.c because Bare board has no file system, only support from memory
extern unsigned char model_tiny[];
extern unsigned int model_tiny_len;
extern unsigned char input_bin[];
extern unsigned int input_bin_len;

#define EXAMPLE_CHECK(error_, msg_...) \
    if (error_ != 0) {                 \
        log_func(msg_);                \
        LITE_destroy_network(model);   \
        __builtin_trap();              \
    }

typedef int (*LogFunc)(const char* fmt, ...);
static TinyNnCallBack g_cb = {
        .tinynn_log_cb = printf,
        // .tinynn_gettime_cb = test_timeimp,
        .tinynn_malloc_cb = pvPortMalloc,
        .tinynn_free_cb = vPortFree,
};

void run_tinynn_test(
        unsigned char* model_ptr, int model_len, unsigned char* inptr, int input_len,
        const char* input_name, LogFunc log_func) {
    LiteNetwork model;
    EXAMPLE_CHECK(
            LITE_make_network(&model, *default_config(), *default_network_io()),
            "create model error. \n");

    EXAMPLE_CHECK(
            LITE_load_model_from_mem(model, model_ptr, model_len),
            "load model error. \n");

    LiteTensor input;
    EXAMPLE_CHECK(
            LITE_get_io_tensor(model, input_name, LITE_INPUT, &input),
            "get input tensor failed\n");
    EXAMPLE_CHECK(
            LITE_reset_tensor_memory(input, inptr, input_len),
            "set input ptr failed\n");

    EXAMPLE_CHECK(LITE_forward(model), "run model failed\n");
    EXAMPLE_CHECK(LITE_wait(model), "wait model failed\n");

    size_t nr_output = 0;
    EXAMPLE_CHECK(
            LITE_get_all_output_name(model, &nr_output, NULL),
            "get output number failed\n");
    char* output_name_ptr = (char*)g_cb.tinynn_malloc_cb(nr_output);
    EXAMPLE_CHECK(
            LITE_get_all_output_name(model, NULL, (const char**)&output_name_ptr),
            "get output name failed\n");

    for (size_t i = 0; i < nr_output; ++i) {
        float sum = 0.0f, max = 0.0f;
        LiteTensor output;
        EXAMPLE_CHECK(
                LITE_get_io_tensor(model, &output_name_ptr[i], LITE_OUTPUT, &output),
                "get output tensor failed\n");
        float* output_ptr = NULL;
        EXAMPLE_CHECK(
                LITE_get_tensor_memory(output, (void**)&output_ptr),
                "get output tensor memory failed\n");
        size_t length = 0;
        EXAMPLE_CHECK(
                LITE_get_tensor_total_size_in_byte(output, &length),
                "get output tensor size failed\n");
        for (size_t i = 0; i < length / sizeof(float); i++) {
            sum += output_ptr[i];
            if (max < output_ptr[i]) {
                max = output_ptr[i];
            }
        }
        LiteLayout layout;
        EXAMPLE_CHECK(LITE_get_tensor_layout(output, &layout), "get layout failed\n");
        log_func(
                "output name: %s sum =%f, max=%f, ptr=%p, dim %zu, shape %zu "
                "%zu %zu "
                "%zu\n",
                &output_name_ptr[i], sum, max, output_ptr, layout.ndim,
                layout.shapes[0], layout.shapes[1], layout.shapes[2], layout.shapes[3]);
        EXAMPLE_CHECK(LITE_destroy_tensor(output), "destory output tensor");
    }
    EXAMPLE_CHECK(LITE_destroy_network(model), "free model failed \n");
}

int tinynn_main() {
    LITE_set_log_level(LITE_DEBUG);
    register_tinynn_cb(TINYNN_CB_VERSION, g_cb);
    run_tinynn_test(
            model_tiny, model_tiny_len, input_bin, input_bin_len, "data", printf);
    printf("run tinynn FreeRTOS example success\n");
    return 0;
}
