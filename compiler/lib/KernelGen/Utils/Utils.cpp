/**
 * \file
 * compiler/lib/KernelGen/Utils/Utils.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <cstdarg>
#include <string>
#include "Utils.h"

using namespace megcc;
using namespace KernelGen;

namespace {
std::string svsprintf(const char* fmt, va_list ap_orig) {
    int size = 100; /* Guess we need no more than 100 bytes */
    char* p;

    if ((p = (char*)malloc(size)) == nullptr)
        return "svsprintf: malloc failed";

    for (;;) {
        va_list ap;
        va_copy(ap, ap_orig);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

        if (n < 0)
            return "svsprintf: vsnprintf failed";

        if (n < size) {
            std::string rst(p);
            free(p);
            return rst;
        }

        size = n + 1;

        char* np = (char*)realloc(p, size);
        if (!np) {
            free(p);
            return "svsprintf: realloc failed";
        } else
            p = np;
    }
}
}  // anonymous namespace

std::string Utils::ssprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    auto rst = svsprintf(fmt, ap);
    va_end(ap);
    return rst;
}

std::string Utils::DtypeHelper::max() {
    switch (m_dtype) {
        case DtypeEnum::float32:
            return "__FLT_MAX__";
        case DtypeEnum::int32:
            return "__INT_MAX__";
        default:
            CC_ASSERT(0) << "not support dtype " << (int)m_dtype;
    }
    return "";
}

std::string Utils::DtypeHelper::min() {
    switch (m_dtype) {
        case DtypeEnum::float32:
            return "-__FLT_MAX__";
        case DtypeEnum::int32:
            return "-__INT_MAX__";
        default:
            CC_ASSERT(0) << "not support dtype " << (int)m_dtype;
    }
    return "";
}

std::string Utils::DtypeHelper::one() {
    switch (m_dtype) {
        case DtypeEnum::float32:
            return "1.f";
        case DtypeEnum::int32:
            return "1";
        default:
            CC_ASSERT(0) << "not support dtype " << (int)m_dtype;
    }
    return "";
}

std::string Utils::DtypeHelper::zero() {
    switch (m_dtype) {
        case DtypeEnum::float32:
            return "0.f";
        case DtypeEnum::int32:
            return "0";
        default:
            CC_ASSERT(0) << "not support dtype " << (int)m_dtype;
    }
    return "";
}

std::string Utils::DtypeHelper::inline_max_func() {
    switch (m_dtype) {
        case DtypeEnum::float32:
            return R"(
static inline float max(float a, float b){
    return a > b? a:b;
})";
        case DtypeEnum::int32:
            return R"(
static inline int max(int a, int b){
    return a > b? a:b;
})";
        default:
            CC_ASSERT(0) << "not support dtype " << (int)m_dtype;
    }
    return "";
}

std::string Utils::DtypeHelper::inline_min_func() {
    switch (m_dtype) {
        case DtypeEnum::float32:
            return R"(
static inline float min(float a, float b){
    return a < b? a:b;
})";
        case DtypeEnum::int32:
            return R"(
static inline int min(int a, int b){
    return a < b? a:b;
})";
        default:
            CC_ASSERT(0) << "not support dtype " << (int)m_dtype;
    }
    return "";
}