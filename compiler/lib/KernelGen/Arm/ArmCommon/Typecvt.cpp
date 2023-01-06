/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Typecvt.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "NeonIntrinCompat.h"
#include "Typecvt.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool TypecvtKernel::IsAvailable(TContext* context) const {
    auto src_dtype = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:0").dtype);
    auto dst_dtype = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:1").dtype);
    bool ok_type =
            (Utils::is_quant_dtype(src_dtype, 8) &&
             Utils::is_quant_dtype(dst_dtype, 8)) ||
            (Utils::is_quant_dtype(src_dtype, 8) &&
             Utils::is_float_dtype(dst_dtype)) ||
            (Utils::is_float_dtype(src_dtype) &&
             Utils::is_quant_dtype(dst_dtype, 8)) ||
            (Utils::get_dtype_enum(src_dtype) == Utils::DtypeEnum::uint8 &&
             Utils::is_float_dtype(dst_dtype));
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
    ss << "ArmCommon_kernel_typecvt_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}
namespace {
std::string init_declare(const std::string& src_dtype,
                         const std::string& dst_dtype) {
    auto src_dtype_enum = Utils::get_dtype_enum(src_dtype);
    auto dst_dtype_enum = Utils::get_dtype_enum(dst_dtype);
    std::string body_temp = R"(
        static float scale;
        static float32x4_t vscale;
        static const size_t SIMD_WIDTH = 8;
    )";
    if (src_dtype_enum == Utils::DtypeEnum::uint8 &&
        dst_dtype_enum == Utils::DtypeEnum::float32) {
        body_temp = R"(
        static float scale;
        static float32x4_t vscale;
        static const size_t SIMD_WIDTH = 16;
    )";
    }
    return body_temp;
}

std::string gen_scale(const std::string& src_dtype,
                      const std::string& dst_dtype) {
    std::string body_temp;
    if (Utils::is_float_dtype(src_dtype)) {
        body_temp += R"(
            src_scale = 1;
         )";
    }
    if (Utils::is_float_dtype(dst_dtype)) {
        body_temp += R"(
           dst_scale = 1;
        )";
    }
    return body_temp;
}

std::string gen_cvt(const std::string& src_dtype,
                    const std::string& dst_dtype) {
    auto src_dtype_enum = Utils::get_dtype_enum(src_dtype);
    auto dst_dtype_enum = Utils::get_dtype_enum(dst_dtype);
    std::string body_temp;
    if (Utils::is_float_dtype(src_dtype) &&
        Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
            float32x4_t vitem0 = vmulq_f32(vld1q_f32(src), vscale);
            float32x4_t vitem1 = vmulq_f32(vld1q_f32(src + 4), vscale);

            int32x4_t vres0 = vcvtaq_s32_f32(vitem0);
            int32x4_t vres1 = vcvtaq_s32_f32(vitem1);

            vst1_s8(dst, vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1))));
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_float_dtype(dst_dtype)) {
        body_temp = R"(
            int16x8_t vsrc = vmovl_s8(vld1_s8(src));
            vst1q_f32(dst, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vsrc))), vscale));
            vst1q_f32(dst+4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vsrc))), vscale));
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
            int16x8_t vsrc = vmovl_s8(vld1_s8(src));
            float32x4_t vitem0 =
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vsrc))), vscale);
            float32x4_t vitem1 =
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vsrc))), vscale);            
            int32x4_t vres0 = vcvtaq_s32_f32(vitem0);
            int32x4_t vres1 = vcvtaq_s32_f32(vitem1);
            vst1_s8(dst, vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1))));
         )";
    } else if (src_dtype_enum == Utils::DtypeEnum::uint8 &&
               dst_dtype_enum == Utils::DtypeEnum::float32) {
        body_temp = R"(
            uint8x16_t u8_src = vld1q_u8(src);
            uint16x8_t vsrc0 = vmovl_u8(vget_low_u8(u8_src));
            uint16x8_t vsrc1 = vmovl_u8(vget_high_u8(u8_src));
            float32x4_t vitem0 = vcvtq_f32_u32(vmovl_u16(vget_low_s16(vsrc0)));
            float32x4_t vitem1 = vcvtq_f32_u32(vmovl_u16(vget_high_s16(vsrc0)));
            float32x4_t vitem2 = vcvtq_f32_u32(vmovl_u16(vget_low_s16(vsrc1)));
            float32x4_t vitem3 = vcvtq_f32_u32(vmovl_u16(vget_high_s16(vsrc1)));
            vst1q_f32(dst + 0 * 4, vitem0);
            vst1q_f32(dst + 1 * 4, vitem1);
            vst1q_f32(dst + 2 * 4, vitem2);
            vst1q_f32(dst + 3 * 4, vitem3);
         )";
    } else {
        CC_ABORT << "ArmCommon not support optimise cvt " << src_dtype << "->"
                 << dst_dtype << "\n";
    }
    return body_temp;
}

std::string gen_cvt_remain(const std::string& src_dtype,
                           const std::string& dst_dtype) {
    auto src_dtype_enum = Utils::get_dtype_enum(src_dtype);
    auto dst_dtype_enum = Utils::get_dtype_enum(dst_dtype);
    std::string body_temp;
    if (Utils::is_float_dtype(src_dtype) &&
        Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
                float val = (*src)*scale;
                int dst_val = roundf(val);
                dst_val = dst_val >= 127? 127:dst_val;
                dst_val = dst_val <= -128? -128:dst_val;
                *dst = dst_val;
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_float_dtype(dst_dtype)) {
        body_temp = R"(
            *dst = (*src)*scale;
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
            float val = (*src)*scale;
            int dst_val = roundf(val);
            dst_val = dst_val >= 127? 127:dst_val;
            dst_val = dst_val <= -128? -128:dst_val;
            *dst = dst_val;
         )";
    } else if (src_dtype_enum == Utils::DtypeEnum::uint8 &&
               dst_dtype_enum == Utils::DtypeEnum::float32) {
        body_temp = R"(
            *dst = (float)*src;
         )";
    } else {
        CC_ABORT << "ArmCommon not support optimise cvt " << src_dtype << "->"
                 << dst_dtype << "\n";
    }
    return body_temp;
}

}  // namespace

std::string TypecvtKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    auto src_dtype_str = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:0").dtype);
    auto dst_dtype_str = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:1").dtype);
    std::string src_specifier = Utils::cvt_dtype_specifier(src_dtype_str);
    std::string dst_specifier = Utils::cvt_dtype_specifier(dst_dtype_str);
    ss << R"(
    #include <arm_neon.h>
    #include <math.h>
    )";
    ss << gen_neon_intrin_compat();
    ss << init_declare(src_dtype_str, dst_dtype_str);
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    const Tensor* src_tensor = inputs[0];
    const Tensor* dst_tensor = outputs[0];
    ${src_specifier}* src = (${src_specifier}*)(src_tensor->ptr);
    ${dst_specifier}* dst = (${dst_specifier}*)(dst_tensor->ptr);
    TINYNN_ASSERT(src);
    TINYNN_ASSERT(dst);
    
    const Layout src_layout = src_tensor->layout;
    const Layout dst_layout = dst_tensor->layout;
    float src_scale = src_tensor->dtype.param.scale;
    float dst_scale = dst_tensor->dtype.param.scale;

    size_t nr_elem = 1;
    for (int i = 0; i < src_layout.nr_dim; ++i) {
        nr_elem *= src_layout.dims[i];
    }
    ${gen_scale}
    scale = src_scale/dst_scale;
    vscale = vdupq_n_f32(scale);
    size_t idx = 0;
    
    for(; idx + SIMD_WIDTH <= nr_elem; idx += SIMD_WIDTH){
        ${gen_cvt}
        src += SIMD_WIDTH;
        dst += SIMD_WIDTH;
    }

    for(;idx < nr_elem;++idx){
        ${gen_cvt_remain}
        ++src;
        ++dst;
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("src_specifier", src_specifier)
                    .add("dst_specifier", dst_specifier)
                    .add("gen_scale", gen_scale(src_dtype_str, dst_dtype_str))
                    .add("gen_cvt", gen_cvt(src_dtype_str, dst_dtype_str))
                    .add("gen_cvt_remain",
                         gen_cvt_remain(src_dtype_str, dst_dtype_str))
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen
