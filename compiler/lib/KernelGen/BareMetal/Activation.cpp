/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Activation.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>
#include <string>
#include "Activation.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

std::string GenActivation::gen_func_dep(std::string mode) {
    if (mode == "RELU") {
        return R"(
            static inline float relu_act(float res){
                return res > 0? res:0;
            }
        )";
    } else if (mode.size() == 0 || mode == "IDENTITY") {
        return "";
    } else if (mode == "H_SWISH") {
        return R"(
static inline float min(float a, float b){
    return a > b? b:a;
}
static inline float max(float a, float b){
    return a > b? a:b;
}
static inline float hswish_act(float res){
    return res * min(max(res + 3, 0.f), 6.f) * (1.f / 6.f);
}

)";
    } else if (mode == "SIGMOID") {
        return R"(
            #include <math.h>
            static inline float sigmoid_act(float res){
                return 1.f / (expf(-res) + 1.f);
            }
        )";
    } else {
        CC_ABORT << "not support " << mode.c_str() << "\n";
    }
    return "";
}

std::string GenActivation::gen_func_call_with_typecvt_dep(
        std::string mode, std::string src_specifier,
        std::string dst_specifier) {
    auto act_dep = gen_func_dep(mode);
    if (src_specifier == "int" && dst_specifier == "int8_t") {
        return act_dep + R"(
            static inline int8_t fp32_to_int8(float src){
                int res = roundf(src);
                res = res > 127? 127:res;
                res = res < -128? -128:res;
                return (int8_t)(res);
            }
        )";
    } else if (src_specifier == dst_specifier) {
        return act_dep;
    } else {
        CC_ABORT << "not support type" << src_specifier << " to "
                 << dst_specifier << "\n";
    }
    return "";
}

std::string GenActivation::gen_func_call(std::string mode, std::string val) {
    if (mode == "RELU") {
        return "relu_act(" + val + ")";
    } else if (mode.size() == 0 || mode == "IDENTITY") {
        return val;
    } else if (mode == "SIGMOID") {
        return "sigmoid_act(" + val + ")";
    } else if (mode == "H_SWISH") {
        return "hswish_act(" + val + ")";
    } else {
        CC_ABORT << "not support " << mode.c_str() << val.c_str() << "\n";
    }
    return "";
}

std::string GenActivation::gen_func_call_with_typecvt(
        std::string mode, std::string args, std::string src_specifier,
        std::string dst_specifier, std::string scale_name,
        std::string flt_scale_name, std::string div_scale_name) {
    if (src_specifier == "int" && dst_specifier == "int8_t") {
        auto act_str = gen_func_call(mode, args + "*" + scale_name + "*" +
                                                   flt_scale_name + "/" +
                                                   div_scale_name);
        return "fp32_to_int8(" + act_str + ")";
    } else if (src_specifier == dst_specifier) {
        return gen_func_call(mode, args);
    } else {
        CC_ABORT << "not support type" << src_specifier << " to "
                 << dst_specifier << "\n";
    }
    return "";
}

// vim: syntax=cpp.doxygen
