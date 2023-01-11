/**
 * \file compiler/tools/kernel_exporter/utils.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "utils.h"
#include <stdarg.h>
#include <iostream>

inline constexpr const char* convert_fmt_str(const char* fmt) {
    return fmt;
}

std::string svsprintf(const char* fmt, va_list ap_orig) {
    fmt = convert_fmt_str(fmt);
    int size = 100; /* Guess we need no more than 100 bytes */
    char* p;

    if ((p = (char*)malloc(size)) == nullptr)
        goto err;

    for (;;) {
        va_list ap;
        va_copy(ap, ap_orig);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

        if (n < 0)
            goto err;

        if (n < size) {
            std::string rst(p);
            free(p);
            return rst;
        }

        size = n + 1;

        char* np = (char*)realloc(p, size);
        if (!np) {
            free(p);
            goto err;
        } else
            p = np;
    }

err:
    fprintf(stderr, "could not allocate memory for svsprintf; fmt=%s\n", fmt);
    __builtin_trap();
}

std::string ssprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    auto rst = svsprintf(fmt, ap);
    va_end(ap);
    return rst;
}
