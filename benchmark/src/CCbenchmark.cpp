#include "CCbenchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <map>
#include "lite-c/common_enum_c.h"
#include "lite-c/global_c.h"
#include "lite-c/tensor_c.h"
const int number = 50;
const int warmup = 10;

#define LITE_CAPI_CHECK(error_, msg_)  \
    if (error_) {                      \
        printf(msg_);                  \
        LITE_destroy_network(m_model); \
        __builtin_trap();              \
    }

#define EXAMPLE_ASSERT(exp_, ...) \
    if (!(exp_)) {                \
        printf("" __VA_ARGS__);   \
        __builtin_trap();         \
    }

using namespace megcc;
using namespace Benchmark;

#if TINYNN_CALLBACK_ENABLE
#include <malloc.h>
#include "tinynn_callback.h"
static void test_timeimp(int32_t* sec, int32_t* usec) {
    struct timeval t;
    gettimeofday(&t, NULL);
    *sec = t.tv_sec;
    *usec = t.tv_usec;
}
static TinyNnCallBack g_cb = {
        .tinynn_log_cb = printf,
        .tinynn_gettime_cb = test_timeimp,
        .tinynn_malloc_cb = malloc,
        .tinynn_free_cb = free,
        .tinynn_fopen_cb = fopen,
        .tinynn_ftell_cb = ftell,
        .tinynn_fseek_cb = fseek,
        .tinynn_fclose_cb = fclose,
        .tinynn_fwrite_cb = fwrite,
        .tinynn_fread_cb = fread,
};
#endif

/////////////////// CCBenchmarker ////////////////
void CCBenchmarker::load_model() {
#if TINYNN_CALLBACK_ENABLE
    register_tinynn_cb(TINYNN_CB_VERSION, g_cb);
#endif
    LITE_CAPI_CHECK(
            LITE_make_network(&m_model, *default_config(), *default_network_io()),
            "create model error. \n");

    LITE_CAPI_CHECK(
            LITE_load_model_from_path(m_model, m_model_path.c_str()),
            "load model error. \n");
}

void CCBenchmarker::profile() {
    size_t nr_input = 0;
    LITE_CAPI_CHECK(
            LITE_get_all_input_name(m_model, &nr_input, NULL),
            "get input number failed\n");
    char* input_name_ptr[nr_input];
    LITE_CAPI_CHECK(
            LITE_get_all_input_name(m_model, NULL, (const char**)(&input_name_ptr)),
            "get input name failed\n");
    void* input_data[nr_input];
    for (size_t i = 0; i < nr_input; ++i) {
        LiteTensor input;
        LITE_CAPI_CHECK(
                LITE_get_io_tensor(m_model, input_name_ptr[i], LITE_INPUT, &input),
                "get input tensor failed\n");

        size_t length;
        LITE_CAPI_CHECK(
                LITE_get_tensor_total_size_in_byte(input, &length),
                "get input tensor size failed\n");
        input_data[i] = malloc(length);
        LITE_CAPI_CHECK(
                LITE_reset_tensor_memory(input, input_data[i], length),
                "set input ptr failed\n");
        LITE_CAPI_CHECK(LITE_destroy_tensor(input), "destory input tensor");
    }

    for (int i = 0; i < warmup; i++) {
        LITE_CAPI_CHECK(LITE_forward(m_model), "run model failed\n");
        LITE_CAPI_CHECK(LITE_wait(m_model), "wait model failed\n");
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < number; i++) {
        LITE_CAPI_CHECK(LITE_forward(m_model), "run model failed\n");
        LITE_CAPI_CHECK(LITE_wait(m_model), "wait model failed\n");
    }
    gettimeofday(&end, NULL);

    for (size_t i = 0; i < nr_input; ++i) {
        free(input_data[i]);
    }

    unsigned long diff =
            1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    float average_time = ((float)diff) / number / 1000;
    if (m_log_level == 3) {
        printf("the inference average time=%.3f ms\n", average_time);
    }
}

CCBenchmarker::~CCBenchmarker() {
    LITE_CAPI_CHECK(LITE_destroy_network(m_model), "delete model failed\n");
}