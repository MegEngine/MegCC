/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Typecvt.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "../Utils/StringTemplate.h"
#include "../Utils/SymbolHelper.h"
#include "../Utils/Utils.h"
#include "FormatHelper.h"
#include "Typecvt.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool TypecvtKernel::IsAvailable(TContext* context) const {
    auto src_dtype =
            SymbolHelper::gen_valid_dtype(context->getAttrOprand("operand:0").dtype);
    auto dst_dtype =
            SymbolHelper::gen_valid_dtype(context->getAttrOprand("operand:1").dtype);
    bool ok_type =
            !(!Utils::is_quant_dtype(src_dtype) && !Utils::is_quant_dtype(dst_dtype) &&
              Utils::is_int_dtype(dst_dtype)) ||
            (Utils::is_float_dtype(src_dtype, 32) && Utils::is_int_dtype(dst_dtype, 8));
    if (Utils::is_quant_dtype(src_dtype)) {
        CC_ASSERT(context->getAttrOprand("operand:0").scale > 0);
    }
    if (Utils::is_quant_dtype(dst_dtype)) {
        CC_ASSERT(context->getAttrOprand("operand:1").scale > 0);
    }
    return ok_type;
}

//! kernel gen
std::string TypecvtKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_typecvt_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}
namespace {

std::string gen_staturate(const std::string& dst_dtype) {
    std::string naive_body_temp = R"(
        static inline int staturate(float val){
            return round(val);
        }
    )";
    if (dst_dtype == "qsi8" || dst_dtype == "si8") {
        naive_body_temp = R"(
        static inline int staturate(float val){
            int dst_val = roundf(val);
            dst_val = dst_val >= 127? 127:dst_val;
            dst_val = dst_val <= -128? -128:dst_val;
            return dst_val;
        }
    )";
    } else if (dst_dtype == "ui8") {
        naive_body_temp = R"(
        static inline int staturate(float val){
            int dst_val = roundf(val);
            dst_val = dst_val >= 255? 255:dst_val;
            dst_val = dst_val <= 0? 0:dst_val;
            return dst_val;
        })";
    }
    return naive_body_temp;
}

std::string gen_act(
        const std::string& src_dtype, const std::string& dst_dtype,
        const std::string& src_specifier, const std::string& dst_specifier) {
    std::string naive_body_temp;
    if (!Utils::is_quant_dtype(src_dtype) && !Utils::is_quant_dtype(dst_dtype)) {
        naive_body_temp = R"(
             static inline ${dst_specifier} act(${src_specifier} val, const float src_scale, const uint8_t src_zp, const float dst_scale, const uint8_t dst_zp){
                 return val;
             }
         )";
    } else if (!Utils::is_quant_dtype(src_dtype) && Utils::is_quant_dtype(dst_dtype)) {
        naive_body_temp = R"(
             static inline ${dst_specifier} act(${src_specifier} val, const float src_scale, const uint8_t src_zp, const float dst_scale, const uint8_t dst_zp){
                 return staturate(val / dst_scale);
             }
         )";
    } else if (Utils::is_quant_dtype(src_dtype) && !Utils::is_quant_dtype(dst_dtype)) {
        naive_body_temp = R"(
             static inline ${dst_specifier} act(${src_specifier} val, const float src_scale, const uint8_t src_zp, const float dst_scale, const uint8_t dst_zp){
                 return val * src_scale;
             }
         )";
    } else if (Utils::is_quant_dtype(src_dtype) && Utils::is_quant_dtype(dst_dtype)) {
        naive_body_temp = R"(
             static inline ${dst_specifier} act(${src_specifier} val, const float src_scale, const uint8_t src_zp, const float dst_scale, const uint8_t dst_zp){
                 return staturate(val * src_scale / dst_scale );
             }
         )";
    } else if (
            Utils::is_float_dtype(src_dtype, 32) && Utils::is_int_dtype(dst_dtype, 8)) {
        naive_body_temp = R"(
             static inline ${dst_specifier} act(${src_specifier} val, const float src_scale, const uint8_t src_zp, const float dst_scale, const uint8_t dst_zp){
                 return staturate(val * src_scale / dst_scale );
             }
         )";
    } else {
        CC_ABORT << "not support cvt " << src_dtype << "->" << dst_dtype << "\n";
    }
    return StringTemplate::StringTemplateArgs()
            .add("src_specifier", src_specifier)
            .add("dst_specifier", dst_specifier)
            .render(naive_body_temp);
}

}  // namespace

std::string TypecvtKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    auto src_dtype_str =
            SymbolHelper::gen_valid_dtype(context->getAttrOprand("operand:0").dtype);
    auto dst_dtype_str =
            SymbolHelper::gen_valid_dtype(context->getAttrOprand("operand:1").dtype);
    std::string src_specifier = Utils::cvt_dtype_specifier(src_dtype_str);
    std::string dst_specifier = Utils::cvt_dtype_specifier(dst_dtype_str);
    ss << R"(
        #include <math.h>
    )";
    ss << gen_staturate(dst_dtype_str);
    ss << gen_act(src_dtype_str, dst_dtype_str, src_specifier, dst_specifier);
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    const Tensor* src_tensor = inputs[0];
    const Tensor* dst_tensor = outputs[0];
    ${src_specifier}* src_data = (${src_specifier}*)(src_tensor->ptr);
    ${dst_specifier}* dst_data = (${dst_specifier}*)(dst_tensor->ptr);
    TINYNN_ASSERT(src_data);
    TINYNN_ASSERT(dst_data);
    
    const Layout src_layout = src_tensor->layout;
    const Layout dst_layout = dst_tensor->layout;
    const float src_scale = src_tensor->dtype.param.scale;
    const float dst_scale = dst_tensor->dtype.param.scale;
    const uint8_t src_zp = src_tensor->dtype.param.zero_point;
    const uint8_t dst_zp = dst_tensor->dtype.param.zero_point;

    size_t nr_elem = 1;
    for (int i = 0; i < src_layout.nr_dim; ++i) {
        nr_elem *= src_layout.dims[i];
    }
    for(size_t i = 0; i < nr_elem; ++i){
        ${src_specifier} val = src_data[i];
        dst_data[i] = act(val, src_scale, src_zp, dst_scale, dst_zp);
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("src_specifier", src_specifier)
                    .add("dst_specifier", dst_specifier)
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen
