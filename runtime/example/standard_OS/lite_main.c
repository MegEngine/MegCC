/**
 * \file runtime/test/lite_main.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string.h>
#include <getopt.h>
#include <sys/time.h>
#include "extern_c_opr.h"
#include "lite-c/common_enum_c.h"
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "lite-c/tensor_c.h"
#include "stdio.h"
#include "stdlib.h"

#define LITE_CAPI_CHECK(error_, msg_) \
    if (error_) {                     \
        printf(msg_);                 \
        LITE_destroy_network(model);  \
        __builtin_trap();             \
    }

#define EXAMPLE_ASSERT(exp_, ...) \
    if (!(exp_)) {                \
        printf(""__VA_ARGS__);    \
        __builtin_trap();         \
    }

static void* read_file(const char* file_name) {
    FILE* fin = fopen(file_name, "rb");
    if (!fin) {
        fprintf(stderr, "Open file error!!\n");
        return NULL;
    }
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    void* ptr = malloc(size);
    size_t read_bytes = fread(ptr, 1, size, fin);
    fclose(fin);
    EXAMPLE_ASSERT(read_bytes == size);
    return ptr;
}

static void write_file(const char* file_name, void* ptr, size_t length) {
    FILE* fout = fopen(file_name, "wb");
    if (!fout) {
        fprintf(stderr, "Open file error!!\n");
        return;
    }
    size_t write_bytes = fwrite(ptr, 1, length, fout);
    EXAMPLE_ASSERT(write_bytes == length);
    fclose(fout);
}

static inline void run_model(LiteNetwork model, const char* output_dir,
                             int instance_cnt, const int print_out) {
#if TINYNN_DUMP_TENSOR || DEBUG_MODE
    const int number = 1;
    const int warmup = 0;
#else
    const int number = 100;
    const int warmup = 20;
#endif
    for (int i = 0; i < warmup; i++) {
        LITE_CAPI_CHECK(LITE_forward(model), "run model failed\n");
        LITE_CAPI_CHECK(LITE_wait(model), "wait model failed\n");
        printf("warmup iter %d finished.\n\n", i);
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < number; i++) {
        LITE_CAPI_CHECK(LITE_forward(model), "run model failed\n");
        LITE_CAPI_CHECK(LITE_wait(model), "wait model failed\n");
        printf("execute iter %d finished.\n", i);
    }
    gettimeofday(&end, NULL);
    unsigned long diff =
            1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("every iter use time: %fms\n", ((float)diff) / number / 1000);
    size_t nr_output = 0;
    LITE_CAPI_CHECK(LITE_get_all_output_name(model, &nr_output, NULL),
                    "get output number failed\n");
    char* output_name_ptr[nr_output];
    LITE_CAPI_CHECK(LITE_get_all_output_name(model, NULL,
                                             (const char**)&output_name_ptr),
                    "get output name failed\n");

    for (size_t i = 0; i < nr_output; ++i) {
        LiteTensor output;
        LITE_CAPI_CHECK(LITE_get_io_tensor(model, output_name_ptr[i],
                                           LITE_OUTPUT, &output),
                        "get output tensor failed\n");
        float* output_ptr = NULL;
        LITE_CAPI_CHECK(LITE_get_tensor_memory(output, (void**)&output_ptr),
                        "get output tensor memory failed\n");
        size_t length = 0;
        LITE_CAPI_CHECK(LITE_get_tensor_total_size_in_byte(output, &length),
                        "get output tensor size failed\n");
        char path_buffer[200];
        if (output_dir) {
            snprintf(path_buffer, 200, "%s/%s_%d", output_dir,
                     output_name_ptr[i], instance_cnt);
            write_file(path_buffer, output_ptr, length);
        }
        if (print_out) {
            printf("output data: ");
        }
        float sum = 0.0f, max = 0.0f;
        for (size_t i = 0; i < length / sizeof(float); i++) {
            if (print_out) {
                if (i && i % 10 == 0) {
                    printf("\n");
                }
                printf("%f ", output_ptr[i]);
            }
            sum += output_ptr[i];
            if (max < output_ptr[i]) {
                max = output_ptr[i];
            }
        }
        LiteLayout layout;
        LITE_CAPI_CHECK(LITE_get_tensor_layout(output, &layout),
                        "get layout failed\n");
        printf("\nsum =%f, max=%f, ptr=%p, dim %zu, shape %zu %zu %zu "
               "%zu\n",
               sum, max, output_ptr, layout.ndim, layout.shapes[0],
               layout.shapes[1], layout.shapes[2], layout.shapes[3]);
        LITE_CAPI_CHECK(LITE_destroy_tensor(output), "destory output tensor");
    }
}

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

void usage(){
    fprintf(stderr, 
        "Usage:\n"
        "\t--input-model/-m: input model path\n"
        "\t--output-dir/-o: output file path\n"
        "\t--log-level/-l: 0:ERROR, 1:WARN, 2:INFO, 3:DEBUG\n"
        "\t--input-data/-d: var=path/to/data_file\n"
        "\t--data-shape/-s: data shape\n"
        "\t--c-opr-lib/-c: path to extern opr lib file(.so)\n"
        "\t--c-opr-init-interface/-i: the init API of your loader\n"
        );
}

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define RTLD_LAZY 0

static void* dlopen(const char* file, int) {
    return (void*)(LoadLibrary(file));
}

static void* dlsym(void* handle, const char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return (void*)symbol;
}

#else
#include <dlfcn.h>
#endif

int main(int argc, char** argv) {
    LITE_set_log_level(WARN);
#if TINYNN_CALLBACK_ENABLE
    register_tinynn_cb(TINYNN_CB_VERSION, g_cb);
#endif
    char* model_path = NULL;
    char* output_dir = NULL;
    int print_out = 0;
    char* data_str = NULL;
    char* data_shape_str = NULL;
    char* extern_so = NULL;
    const char* c_opr_lib_interface = "mgb_c_opr_init";

    const struct option long_options[] = {
        {"input-model", required_argument, 0, 'm'},
        {"output-dir", required_argument, 0, 'o'},
        {"log-level", required_argument, 0, 'l'},
        {"input-data", required_argument, 0, 'd'},
        {"data-shape", required_argument, 0, 's'},
        {"c-opr-lib", required_argument, 0, 'c'},
        {"c-opr-init-interface", required_argument, 0, 'i'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    const char* shortopt = "m:o:l:d:s:c:i:h";
    int c, option_idx = 0;
    while(1) {
        c = getopt_long(argc, argv, shortopt, long_options, &option_idx);
        if(c == -1){
            break;
        }
        switch(c){
            case 'm':
                model_path = optarg;
                break;
            case 'o':
                output_dir = optarg;
                break;
            case 'l':
                print_out = atoi(optarg);
                break;
            case 'd':
                data_str = optarg;
                break;
            case 's':
                data_shape_str = optarg;
                break;
            case 'c':
                extern_so = optarg;
                break;
            case 'i':
                c_opr_lib_interface = optarg;
                break;
            case 'h':
                usage();
                exit(0);
                break;
            default:
                abort();
        }
    }
    
    if (print_out == 2) {
        LITE_set_log_level(INFO);
    } else if (print_out == 3) {
        LITE_set_log_level(DEBUG);
    }

    if(extern_so){
        void* handle = dlopen(extern_so, RTLD_LAZY);
        EXAMPLE_ASSERT(handle, "load loader failed.\n");
        void (*func)(const MGBExternCOprApi* (*)(int)) = NULL;
        *(void**)&func = dlsym(handle, c_opr_lib_interface);
        EXAMPLE_ASSERT(func, "load init interface of loader failed.\n");
        func(mgb_get_extern_c_opr_api_versioned);
    }
    
    LiteNetwork model;
    LITE_CAPI_CHECK(
            LITE_make_network(&model, *default_config(), *default_network_io()),
            "create model error. \n");

    LITE_CAPI_CHECK(LITE_load_model_from_path(model, model_path),
                    "load model error. \n");

    size_t nr_input = 0;
    LITE_CAPI_CHECK(LITE_get_all_input_name(model, &nr_input, NULL),
                    "get input number failed\n");
    char* input_name_ptr[nr_input];
    LITE_CAPI_CHECK(LITE_get_all_input_name(model, NULL,
                                            (const char**)(&input_name_ptr)),
                    "get input name failed\n");
    printf("input %zu\n", nr_input);
    int instance_cnt = 0;
    char* data_instance_remain_str = NULL;
    char* data_instance =
            data_str ? strtok_r(data_str, ":", &data_instance_remain_str)
                     : NULL;
    char* instance_shape_remain_str = NULL;
    char* shape_instance = NULL;
    if (data_shape_str) {
        shape_instance =
                strtok_r(data_shape_str, ":", &instance_shape_remain_str);
    }
    while (data_instance) {
        char* data_remain_str = NULL;
        char* data_name = strtok_r(data_instance, "=;", &data_remain_str);
        size_t input_cnt = 0;
        float* data[nr_input];
        char* shape_remain_str = NULL;
        char* shape_name = NULL;
        if (shape_instance) {
            shape_name = strtok_r(shape_instance, "=;", &shape_remain_str);
        }
        for (size_t i = 0; i < nr_input; ++i) {
            if (data_name) {
                char* data_path = strtok_r(NULL, "=;", &data_remain_str);
                if (data_path) {
                    data[i] = read_file(data_path);
                    EXAMPLE_ASSERT(data[i] != NULL, "can not read input file");
                    LiteTensor input;
                    LITE_CAPI_CHECK(LITE_get_io_tensor(model, data_name,
                                                       LITE_INPUT, &input),
                                    "get input tensor failed\n");
                    if (shape_name) {
                        LiteLayout layout;
                        LITE_CAPI_CHECK(LITE_get_tensor_layout(input, &layout),
                                        "get input tensor layout failed\n");
                        char* shape_list_str =
                                strtok_r(NULL, "=;", &shape_remain_str);
                        char* shape_val_remain = NULL;
                        char* shape_val = strtok_r(shape_list_str, "(),",
                                                   &shape_val_remain);
                        int dim_cnt = 0;
                        while (shape_val) {
                            layout.shapes[dim_cnt++] = atoi(shape_val);
                            shape_val = strtok_r(NULL, ",", &shape_val_remain);
                        }
                        layout.ndim = dim_cnt;
                        LITE_CAPI_CHECK(LITE_set_tensor_layout(input, layout),
                                        "get input tensor failed\n");
                        shape_name = strtok_r(NULL, "=;", &shape_remain_str);
                    }
                    size_t length;
                    LITE_CAPI_CHECK(
                            LITE_get_tensor_total_size_in_byte(input, &length),
                            "get input tensor size failed\n");
                    LITE_CAPI_CHECK(
                            LITE_reset_tensor_memory(input, data[i], length),
                            "set input ptr failed\n");
                    LITE_CAPI_CHECK(LITE_destroy_tensor(input),
                                    "destory input tensor");
                    ++input_cnt;
                } else {
                    printf("invalid data %s\n", data_name);
                    return -1;
                }
                data_name = strtok_r(NULL, "=;", &data_remain_str);
            } else {
                printf("error: input from args not equal model input, need "
                       "%zu, get %zu\n",
                       nr_input, input_cnt);
            }
        }
        run_model(model, output_dir, instance_cnt, print_out);
        for (size_t i = 0; i < nr_input; ++i) {
            free(data[i]);
        }
        data_instance = strtok_r(NULL, ":", &data_instance_remain_str);
        if (shape_instance) {
            shape_instance = strtok_r(NULL, ":", &instance_shape_remain_str);
        }
        ++instance_cnt;
    }
    //! if no input data set, just run the model with random input data
    if (instance_cnt == 0) {
        run_model(model, output_dir, instance_cnt, print_out);
    }

    LITE_CAPI_CHECK(LITE_destroy_network(model), "delete model failed\n");

    return 0;
}

// vim: syntax=cpp.doxygen
