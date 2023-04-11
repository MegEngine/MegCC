#include "lite-c/common_enum_c.h"
#include "lite-c/global_c.h"
#include "lite-c/network_c.h"
#include "tinynn_callback.h"

#include <stdlib.h>

#define EXAMPLE_CHECK(error_, msg_...) \
    if (error_ != 0) {                 \
        g_cb->tinynn_log_cb(msg_);     \
        LITE_destroy_network(model);   \
        __builtin_trap();              \
    }

int init_megcc_test(const char* input_name, TinyNnCallBack* cb);
float run_megcc_test(int iter);
int free_megcc_test();
