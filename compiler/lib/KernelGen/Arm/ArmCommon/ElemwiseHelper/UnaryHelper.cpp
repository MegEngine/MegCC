/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/ElemwiseHelper/UnaryHelper.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Arm/ArmCommon/ArmSimdHelper.h"
#include "ElemwiseHelper.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

std::string ElemwiseGenUnary::GenCodeBody(std::vector<std::string> strs) const {
    std::stringstream body_ss;
    if (m_inline_mode) {
        body_ss << R"(static inline void ${inline_func_name}(const ${src_specifier}* src, ${dst_specifier}* dst, size_t nr_elem)";
        if (m_i32_to_qs8) {
            body_ss << ", float src_scale, float dst_scale";
        }
        body_ss << "){";
    } else {
        body_ss << R"(
            Layout in_layout = inputs[0]->layout;
            size_t nr_elem = 1;
            for (int i = 0; i < in_layout.nr_dim; ++i) {
                nr_elem *= in_layout.dims[i];
            }
            const ${src_specifier} * src = ${source};
            ${dst_specifier}* dst = ${dst};
        )";
    }
    body_ss << R"(
        ${kernel_init()}

        size_t index = 0;
        for(; index + 7 < nr_elem; index += 8) {
            ${src_simd_specifier} vsrc0 = ${src_ld1q}(src);
            ${src_simd_specifier} vsrc1 = ${src_ld1q}(src + 4);
            ${kernel_simd_unroll(2, vsrc0, vdst0, vsrc1, vdst1)}
            ${dst_store(dst, vdst0)};
            ${dst_store(dst + 4, vdst1)};
            src += 8;
            dst += 8;
        }
        for(; index + 3 < nr_elem; index += 4) {
            ${src_simd_specifier} vsrc0 = ${src_ld1q}(src);
            ${kernel_simd_unroll(1, vsrc0, vdst0)}
            ${dst_store(dst, vdst0)};
            src += 4;
            dst += 4;
        }
        for(; index < nr_elem; index++) {
            ${kernel_naive_unroll(1, src, dst)}
            src += 1;
            dst += 1;
        })";
    if (m_inline_mode) {
        body_ss << "}";
    }
    auto kernel_init = [this](std::vector<std::string> strs) {
        return GenKernelSimdInit(strs);
    };
    auto kernel_simd_unroll = [this](std::vector<std::string> strs) {
        return GenKernelSimdUnroll(strs);
    };
    auto kernel_naive_unroll = [this](std::vector<std::string> strs) {
        return GenKernelNaiveUnroll(strs);
    };
    std::stringstream ss;
    auto body_render =
            StringTemplate::StringTemplateArgs()
                    .add("kernel_init", kernel_init)
                    .add("kernel_simd_unroll", kernel_simd_unroll)
                    .add("kernel_naive_unroll", kernel_naive_unroll)
                    .add("src_specifier",
                         Utils::cvt_dtype_specifier(m_src_dtype))
                    .add("dst_specifier",
                         Utils::cvt_dtype_specifier(m_dst_dtype))
                    .add("src_ld1q", m_src_simd->get_ld1q_symbol())
                    .add("dst_store",
                         [=](std::string ptr, std::string dst_reg) {
                             if (m_i32_to_qs8) {
                                 return "vst1_lane_s32((int32_t*)(" + ptr +
                                        ")," + dst_reg + ", 0)\n";
                             } else {
                                 return m_dst_simd->get_st1q_symbol() + "(" +
                                        ptr + "," + dst_reg + ")\n";
                             }
                         })
                    .add("dst_st1q", m_dst_simd->get_st1q_symbol())
                    .add("src_simd_specifier",
                         m_src_simd->get_specifier_q_symbol());

    if (m_inline_mode) {
        body_render.add("inline_func_name", GenInlineName());
    } else {
        auto input = strs[0];
        auto output = strs[1];
        body_render.add("source", input).add("dst", output);
    }
    ss << body_render.render(body_ss.str());

    return ss.str();
}

//! Relu
std::string ElemwiseGenUnaryRelu::GenInlineName() const {
    return "ElemwiseGenUnaryRelu";
}
std::string ElemwiseGenUnaryRelu::GenKernelSimdInit(
        std::vector<std::string>) const {
    std::stringstream writer;
    writer << "\nfloat32x4_t vzero = vdupq_n_f32(0.f);";
    return writer.str();
}

std::string ElemwiseGenUnaryRelu::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    std::stringstream writer;
    for (int i = 0; i < unroll; i++) {
        writer << "\n float32x4_t " << strs[2 * i + 2] << " = vmaxq_f32(("
               << strs[2 * i + 1] << "), vzero);";
    }
    return writer.str();
}

std::string ElemwiseGenUnaryRelu::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto input_ptr = strs[1];
    auto output_ptr = strs[2];
    std::stringstream writer;
    for (int i = 0; i < unroll; i++) {
        writer << "\n(" << output_ptr << ")[" << i << "] =  fmax((" << input_ptr
               << ")[" << i << "], 0.0f);";
    }
    return writer.str();
}

//! Exp
std::string ElemwiseGenUnaryExp::GenInlineName() const {
    return "ElemwiseGenUnaryExp";
}
std::string ElemwiseGenUnaryExp::GenKernelSimdInit(
        std::vector<std::string>) const {
    return " ";
}

std::string ElemwiseGenUnaryExp::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    std::stringstream writer;
    for (int i = 0; i < unroll; i++) {
        writer << "\n float32x4_t " << strs[2 * i + 2] << " = exp_ps_f32("
               << strs[2 * i + 1] << ");";
    }
    return writer.str();
}

std::string ElemwiseGenUnaryExp::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto input_ptr = strs[1];
    auto output_ptr = strs[2];
    std::stringstream writer;
    for (int i = 0; i < unroll; i++) {
        writer << "\n(" << output_ptr << ")[" << i << "] =  exp((" << input_ptr
               << ")[" << i << "]);";
    }
    return writer.str();
}

//! Sigmoid
std::string ElemwiseGenUnarySigmoid::GenInlineName() const {
    return "ElemwiseGenUnarySigmoid";
}
std::string ElemwiseGenUnarySigmoid::GenKernelSimdInit(
        std::vector<std::string>) const {
    std::stringstream writer;
    writer << "\nfloat32x4_t ones = vdupq_n_f32(1.f);";
    return writer.str();
}

std::string ElemwiseGenUnarySigmoid::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto input_ptr = strs[1];
    auto output_ptr = strs[2];
    std::stringstream writer;
    for (int i = 0; i < unroll; i++) {
        std::string input_reg_str = "";
        auto input_render =
                StringTemplate::StringTemplateArgs().add("idx", i).add(
                        "input_reg", strs[2 * i + 1]);
        if (m_i32_to_qs8) {
            input_reg_str = input_render.render(R"(
                float32x4_t input_reg_${idx} = vmulq_n_f32(vcvtq_f32_s32(${input_reg}), src_scale);
            )");
        } else {
            CC_ASSERT(Utils::is_float_dtype(m_dst_dtype, 32))
                    << "not support dst_type " << m_dst_dtype;
            input_reg_str = input_render.render(R"(
                float32x4_t input_reg_${idx} = ${input_reg};
            )");
        }
        writer << input_reg_str;
        std::string input_temp = R"(
                float32x4_t rcep_${idx} = vaddq_f32(ones, exp_ps_f32(vnegq_f32(input_reg_${idx})));
            )";
        writer << input_render.render(input_temp);

        writer << "\n float32x4_t temp_" << i << " = vrecpeq_f32(rcep_" << i
               << ");";
        writer << "\n float32x4_t dst_temp_" << i
               << " = vmulq_f32(vrecpsq_f32(rcep_" << i << ", temp_" << i
               << "), temp_" << i << ");";
        if (m_i32_to_qs8) {
            std::string quant_temp = R"(
                int16x4_t temp_s16_${idx} = vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(dst_temp_${idx}, dst_scale)));
                int8x8_t ${dst_reg_name} = vqmovn_s16(vcombine_s16(temp_s16_${idx}, temp_s16_${idx}));
            )";
            writer << StringTemplate::StringTemplateArgs()
                              .add("dst_reg_name", strs[2 * i + 2])
                              .add("idx", i)
                              .render(quant_temp);
        } else {
            CC_ASSERT(Utils::is_float_dtype(m_dst_dtype, 32));
            writer << "\n float32x4_t " << strs[2 * i + 2] << " = dst_temp_"
                   << i << ";";
        }
    }
    return writer.str();
}

std::string ElemwiseGenUnarySigmoid::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto input_ptr = strs[1];
    auto output_ptr = strs[2];
    std::stringstream writer;
    auto body_render = StringTemplate::StringTemplateArgs()
                               .add("dst_ptr", output_ptr)
                               .add("src_ptr", input_ptr);
    for (int i = 0; i < unroll; i++) {
        if (m_i32_to_qs8) {
            writer << body_render.render(R"(
                float res = 1.f / ( 1.f + expf(-(${src_ptr}[0] * src_scale)));
                int res_i32 = (int)roundf(res * dst_scale);
                res_i32 = res_i32 > 127 ? 127 : res_i32;
                res_i32 = res_i32 < -128 ? -128 : res_i32;
                (${dst_ptr})[0] = res_i32;
            )");
        } else {
            CC_ASSERT(Utils::is_float_dtype(m_dst_dtype, 32));
            writer << body_render.render(R"(
                (${dst_ptr})[0] =  1.f / ( 1.f + expf(-(${src_ptr})[0]));
            )");
        }
    }
    return writer.str();
}

//! HSWISH
std::string ElemwiseGenUnaryHswish::GenInlineName() const {
    return "ElemwiseGenUnaryHswish";
}
std::string ElemwiseGenUnaryHswish::GenKernelSimdInit(
        std::vector<std::string>) const {
    std::stringstream writer;
    writer << R"(
        float32x4_t v0 = vdupq_n_f32(0.f);
        float32x4_t v1 = vdupq_n_f32(1.f);
        float32x4_t v3 = vdupq_n_f32(3.f);
        float32x4_t v6 = vdupq_n_f32(6.f);
        float32x4_t v6_inv = vdupq_n_f32(1.f/6.f);
    )";
    return writer.str();
}

std::string ElemwiseGenUnaryHswish::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto input_ptr = strs[1];
    auto output_ptr = strs[2];
    std::stringstream writer;
    for (int i = 0; i < unroll; i++) {
        std::string input_reg_str = "";
        auto input_render =
                StringTemplate::StringTemplateArgs().add("idx", i).add(
                        "input_reg", strs[2 * i + 1]);
        if (m_i32_to_qs8) {
            input_reg_str = input_render.render(R"(
                float32x4_t input_reg_${idx} = vmulq_n_f32(vcvtq_f32_s32(${input_reg}), src_scale);
            )");
        } else {
            CC_ASSERT(Utils::is_float_dtype(m_dst_dtype, 32))
                    << "not support dst_type " << m_dst_dtype;
            input_reg_str = input_render.render(R"(
                float32x4_t input_reg_${idx} = ${input_reg};
            )");
        }
        writer << input_reg_str;
        std::string input_temp = R"(
                float32x4_t relu6_${idx} = vminq_f32(vmaxq_f32(vaddq_f32(input_reg_${idx}, v3), v0), v6);
                float32x4_t dst_temp_${idx} = vmulq_f32(vmulq_f32(relu6_${idx}, input_reg_${idx}), v6_inv);
            )";
        writer << input_render.render(input_temp);

        if (m_i32_to_qs8) {
            std::string quant_temp = R"(
                int16x4_t temp_s16_${idx} = vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(dst_temp_${idx}, dst_scale)));
                int8x8_t ${dst_reg_name} = vqmovn_s16(vcombine_s16(temp_s16_${idx}, temp_s16_${idx}));
            )";
            writer << StringTemplate::StringTemplateArgs()
                              .add("dst_reg_name", strs[2 * i + 2])
                              .add("idx", i)
                              .render(quant_temp);
        } else {
            CC_ASSERT(Utils::is_float_dtype(m_dst_dtype, 32));
            writer << "\n float32x4_t " << strs[2 * i + 2] << " = dst_temp_"
                   << i << ";";
        }
    }
    return writer.str();
}

std::string ElemwiseGenUnaryHswish::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto input_ptr = strs[1];
    auto output_ptr = strs[2];
    std::stringstream writer;
    auto body_render = StringTemplate::StringTemplateArgs()
                               .add("dst_ptr", output_ptr)
                               .add("src_ptr", input_ptr);
    for (int i = 0; i < unroll; i++) {
        if (m_i32_to_qs8) {
            writer << body_render.render(R"(
                float temp = ${src_ptr}[0] + 3;
                temp = temp > 6? 6 : temp;
                temp = temp < 0? 0 : temp;
                float res = ${src_ptr}[0] * temp / 6.f;
                int res_i32 = (int)roundf(res * dst_scale);
                res_i32 = res_i32 > 127 ? 127 : res_i32;
                res_i32 = res_i32 < -128 ? -128 : res_i32;
                (${dst_ptr})[0] = res_i32;
            )");
        } else {
            CC_ASSERT(Utils::is_float_dtype(m_dst_dtype, 32));
            writer << body_render.render(R"(
                float temp = ${src_ptr}[0] + 3;
                temp = temp > 6? 6 : temp;
                temp = temp < 0? 0 : temp;
                (${dst_ptr})[0] =  ${src_ptr}[0] * temp / 6.f;
            )");
        }
    }
    return writer.str();
}
// vim: syntax=cpp.doxygen
