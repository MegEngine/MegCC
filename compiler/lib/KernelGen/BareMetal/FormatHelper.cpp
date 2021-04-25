/**
 * \file
 * compiler/lib/KernelGen/BareMetal/FormatHelper.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <sstream>
#include "FormatHelper.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

std::string GenFormatIter::gen_inline_format_iter_symbol(
        std::string format_str) {
    return "get_linear_addr_" + format_str;
}

std::string GenFormatIter::gen_inline_format_iter_body(std::string format_str) {
    std::stringstream ss;
    ss << R"(static inline size_t )"
       << gen_inline_format_iter_symbol(format_str);
    ss << R"((const int n, const int c, const int h,
                                     const int w, const int* stride,
                                     const bool is_output) {)";
    if (format_str == "NCHW") {
        ss << R"(
            size_t offset = (size_t)n * stride[0] + c * stride[1] + h * stride[2] + w * stride[3];
            return offset;
        )";
    } else {
        CC_ASSERT(format_str == "NCHW44") << "format not support\n";
        ss << R"(
            return (size_t)n * stride[0] + c / 4 * stride[1] + h * stride[2] + w * stride[3] + (c % 4) * stride[4];
        )";
    }
    ss << "}\n";
    return ss.str();
}

// vim: syntax=cpp.doxygen
