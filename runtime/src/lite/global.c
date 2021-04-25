/**
 * \file runtime/src/lite/global.c
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
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
