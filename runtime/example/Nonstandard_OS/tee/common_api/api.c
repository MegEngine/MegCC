/**
 * \file runtime/example/Nonstandard_OS/tee/common_api/api.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "api.h"

extern unsigned char model_tiny[];
extern unsigned int model_tiny_len;
extern unsigned char input_bin[];
extern unsigned int input_bin_len;

static LiteNetwork model;
static TinyNnCallBack* g_cb;

int init_megcc_test(const char* input_name, TinyNnCallBack* cb) {
    g_cb = cb;
    EXAMPLE_CHECK(
            LITE_make_network(&model, *default_config(), *default_network_io()),
            "create model error. \n");

    EXAMPLE_CHECK(
            LITE_load_model_from_mem(model, (char*)model_tiny, model_tiny_len),
            "load model error. \n");

    LiteTensor input;
    EXAMPLE_CHECK(
            LITE_get_io_tensor(model, input_name, LITE_INPUT, &input),
            "get input tensor failed\n");
    EXAMPLE_CHECK(
            LITE_reset_tensor_memory(input, (char*)input_bin, input_bin_len),
            "set input ptr failed\n");

    return 0;
}

float run_megcc_test(int iter) {
    g_cb->tinynn_log_cb("call run_megcc_test with iter: %d\n", iter);

    for (int i = 0; i < iter; i++) {
        g_cb->tinynn_log_cb("run iter: %d/%d\n", i + 1, iter);
        EXAMPLE_CHECK(LITE_forward(model), "run model failed\n");
        EXAMPLE_CHECK(LITE_wait(model), "wait model failed\n");
    }

    size_t nr_output = 0;
    EXAMPLE_CHECK(
            LITE_get_all_output_name(model, &nr_output, NULL),
            "get output number failed\n");
    char* output_name_ptr = (char*)g_cb->tinynn_malloc_cb(nr_output);
    EXAMPLE_CHECK(
            LITE_get_all_output_name(model, NULL, (const char**)&output_name_ptr),
            "get output name failed\n");

    //! return max as result
    float max = 0.0f;
    for (size_t i = 0; i < nr_output; ++i) {
        float sum = 0.0f;
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
        g_cb->tinynn_log_cb(
                "output name: %s sum =%f, max=%f, ptr=%p, dim %zu, shape %zu "
                "%zu %zu "
                "%zu\n",
                &output_name_ptr[i], sum, max, output_ptr, layout.ndim,
                layout.shapes[0], layout.shapes[1], layout.shapes[2], layout.shapes[3]);
        EXAMPLE_CHECK(LITE_destroy_tensor(output), "destory output tensor");
    }
    return max;
}
int free_megcc_test() {
    EXAMPLE_CHECK(LITE_destroy_network(model), "free model failed \n");
    g_cb->tinynn_log_cb("free_megcc_test success\n");

    return 0;
}
