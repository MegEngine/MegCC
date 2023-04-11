#include <stdarg.h>
#include <stdlib.h>
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

#ifdef __arm__
volatile unsigned int* const UART0DR = (unsigned int*)0x10009000;
#else
volatile unsigned int* const UART0DR = (unsigned int*)0x09000000;
#endif

typedef int (*LogFunc)(const char* fmt, ...);
#define EXAMPLE_CHECK(error_, msg_...) \
    if (error_ != 0) {                 \
        log_func(msg_);                \
        LITE_destroy_network(model);   \
        __builtin_trap();              \
    }

static void print_uart0(const char* s) {
    while (*s != '\0') {               /* Loop until end of string */
        *UART0DR = (unsigned int)(*s); /* Transmit char */
        s++;                           /* Next char */
    }
}

static int print_uart0_ftm(const char* fmt, ...) {
#define MAX_LOG_SIZE 200
    va_list ap;
    char buff[MAX_LOG_SIZE];
    va_start(ap, fmt);
    vsnprintf(buff, sizeof(buff), fmt, ap);
    va_end(ap);
    //! MAX_LOG_SIZE may overflow
    buff[MAX_LOG_SIZE - 1] = '\n';
    print_uart0(buff);
    return 0;
}

static void test_timeimp(int32_t* sec, int32_t* usec) {
    (void)&sec;
    (void)&usec;
    print_uart0("not imp test_timeimp");
    __builtin_trap();
}

static TinyNnCallBack g_cb = {
        .tinynn_log_cb = print_uart0_ftm,
        .tinynn_gettime_cb = test_timeimp,
        .tinynn_malloc_cb = malloc,
        .tinynn_free_cb = free,
};

void run_tinynn_test(
        unsigned char* model_ptr, int model_len, unsigned char* inptr, int input_len,
        LogFunc log_func) {
    LiteNetwork model;
    size_t i, nr_input;
    char** input_name = NULL;
    EXAMPLE_CHECK(
            LITE_make_network(&model, *default_config(), *default_network_io()),
            "create model error. \n");

    EXAMPLE_CHECK(
            LITE_load_model_from_mem(model, model_ptr, model_len),
            "load model error. \n");

    LITE_get_all_input_name(model, &nr_input, NULL);
    input_name = (char**)g_cb.tinynn_malloc_cb(sizeof(char*) * nr_input);
    LITE_get_all_input_name(model, NULL, (const char**)input_name);
    //! TODO: The byte lengths of all inputs in the model must be same with `input_len`
    //! for now.
    //！ All inputs of the model share `inptr`.
    for (i = 0; i < nr_input; ++i) {
        LiteTensor input;
        EXAMPLE_CHECK(
                LITE_get_io_tensor(model, input_name[i], LITE_INPUT, &input),
                "get input tensor failed\n");
        EXAMPLE_CHECK(
                LITE_reset_tensor_memory(input, inptr, input_len),
                "set input ptr failed\n");
    }
    g_cb.tinynn_free_cb(input_name);

    EXAMPLE_CHECK(LITE_forward(model), "run model failed\n");
    EXAMPLE_CHECK(LITE_wait(model), "wait model failed\n");

    size_t nr_output = 0;
    EXAMPLE_CHECK(
            LITE_get_all_output_name(model, &nr_output, NULL),
            "get output number failed\n");
    char** output_name_ptr = (char**)g_cb.tinynn_malloc_cb(sizeof(char*) * nr_output);
    EXAMPLE_CHECK(
            LITE_get_all_output_name(model, NULL, (const char**)output_name_ptr),
            "get output name failed\n");

    for (size_t i = 0; i < nr_output; ++i) {
        float sum = 0.0f, max = 0.0f;
        LiteTensor output;
        EXAMPLE_CHECK(
                LITE_get_io_tensor(model, output_name_ptr[i], LITE_OUTPUT, &output),
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
                output_name_ptr[i], sum, max, output_ptr, layout.ndim, layout.shapes[0],
                layout.shapes[1], layout.shapes[2], layout.shapes[3]);
        EXAMPLE_CHECK(LITE_destroy_tensor(output), "destory output tensor");
    }
    g_cb.tinynn_free_cb(output_name_ptr);
    EXAMPLE_CHECK(LITE_destroy_network(model), "free model failed \n");
}

int main() {
    register_tinynn_cb(TINYNN_CB_VERSION, g_cb);
    LITE_set_log_level(LITE_DEBUG);
    print_uart0("Hello world!\n");
    run_tinynn_test(
            model_tiny, model_tiny_len, input_bin, input_bin_len, print_uart0_ftm);
    print_uart0("inference done\n");
    print_uart0("Bye world!\n");
    //! shutdown system by host call, delete it when real board
    //! reference resources: https://wiki.segger.com/Semihosting
#ifdef __aarch64__
    __asm("mov w0, 0x18");
    __asm("mov x1, #0x20000");
    __asm("add x1, x1, #0x26");
    __asm("HLT #0xF000");
#else
    register int reg0 asm("r0");
    register int reg1 asm("r1");
    reg0 = 0x18;
    reg1 = 0x20026;
    asm("svc 0x123456");
#endif
}
