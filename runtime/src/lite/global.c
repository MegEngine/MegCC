#include "lite-c/global_c.h"
#include "utils.h"
extern LiteLogLevel g_log_level;
//! TODO: imp more api define in lite-c/global_c.h

LITE_API int LITE_set_log_level(LiteLogLevel level) {
    g_log_level = level;
    return 0;
}

LITE_API int LITE_get_log_level(LiteLogLevel* level) {
    *level = g_log_level;
    return 0;
}
