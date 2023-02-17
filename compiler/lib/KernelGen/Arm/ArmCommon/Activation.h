/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Activation.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {

enum NonlineMode { IDENTITY, H_SWISH, RELU, SIGMOID };

struct ActivationGenIntrinsicBase {
    //! gen the const neon data, such as zero in relu
    virtual std::string GenIntrinsicInitFloat() const = 0;

    //! compute the input neon data and write to the output neon
    virtual std::string GenIntrinsicFloat(
            const std::string& input, const std::string& output) const = 0;
    //! compute the input neon data and write to the output ptr
    virtual std::string GenIntrinsicFloatStore(
            const std::string& input, const std::string& outptr) const = 0;

    //! compute the input neon data and write to the output ptr
    virtual std::string GenIntrinsicQuantStore(
            const std::string& input, const std::string& outptr,
            const std::string& scale_sym) const = 0;
};

template <NonlineMode mode>
struct ActivationGenIntrinsic : public ActivationGenIntrinsicBase {
public:
    //! gen the const neon data, such as zero in relu
    virtual std::string GenIntrinsicInitFloat() const override { return ""; }

    //! compute the input neon data and write to the output neon
    virtual std::string GenIntrinsicFloat(
            const std::string&, const std::string&) const override {
        return "";
    };
    std::string GenIntrinsicFloatStore(
            const std::string& input, const std::string& outptr) const override {
        std::stringstream writer;
        writer << "\n vst1q_f32(" << outptr << ", " << input << ");";
        return writer.str();
    }
    std::string GenIntrinsicQuantStore(
            const std::string& input, const std::string& outptr,
            const std::string& scale_sym) const override {
        std::string store_temp = R"(
            {
                float32x4_t f32_res = vcvtq_f32_s32(${input_reg});
                float32x4_t res = vmulq_n_f32(f32_res, ${scale_sym});
                int32x4_t s32_res = vcvtaq_s32_f32(res);
                int16x4_t s16_res = vqmovn_s32(s32_res);
                int8x8_t s8_res = vqmovn_s16(vcombine_s16(s16_res, s16_res));
                vst1_lane_s32((int32_t*)(${output_ptr}), vreinterpret_s32_s8(s8_res), 0);
            }        
        )";
        return StringTemplate::StringTemplateArgs()
                .add("input_reg", input)
                .add("scale_sym", scale_sym)
                .add("output_ptr", outptr)
                .render(store_temp);
    }
};

template <>
struct ActivationGenIntrinsic<NonlineMode::RELU> : public ActivationGenIntrinsicBase {
public:
    std::string GenIntrinsicInitFloat() const override {
        std::stringstream writer;
        writer << "\nfloat32x4_t vzero = vdupq_n_f32(0.f);";
        return writer.str();
    }
    std::string GenIntrinsicFloat(
            const std::string& input, const std::string& output) const override {
        std::stringstream writer;
        writer << "\n" << output << " = vmaxq_f32(" << input << ", vzero);";
        return writer.str();
    }
    std::string GenIntrinsicFloatStore(
            const std::string& input, const std::string& outptr) const override {
        std::stringstream writer;
        writer << "\n vst1q_f32(" << outptr << ", vmaxq_f32(" << input << ", vzero));";
        return writer.str();
    }
    std::string GenIntrinsicQuantStore(
            const std::string& input, const std::string& outptr,
            const std::string& scale_sym) const override {
        std::string store_temp = R"(
            {
                float32x4_t f32_res = vcvtq_f32_s32(${input_reg});
                float32x4_t res = vmaxq_f32(vmulq_n_f32(f32_res, ${scale_sym}), vzero);
                int32x4_t s32_res = vcvtaq_s32_f32(res);
                int16x4_t s16_res = vqmovn_s32(s32_res);
                int8x8_t s8_res = vqmovn_s16(vcombine_s16(s16_res, s16_res));
                vst1_lane_s32((int32_t*)(${output_ptr}), vreinterpret_s32_s8(s8_res), 0);
            }        
        )";
        return StringTemplate::StringTemplateArgs()
                .add("input_reg", input)
                .add("scale_sym", scale_sym)
                .add("output_ptr", outptr)
                .render(store_temp);
    }
};

template <>
struct ActivationGenIntrinsic<NonlineMode::H_SWISH>
        : public ActivationGenIntrinsicBase {
public:
    std::string GenIntrinsicInitFloat() const override {
        std::stringstream writer;
        writer << "\nfloat32x4_t vzero = vdupq_n_f32(0.f);";
        writer << "\nfloat32x4_t f6_v = vdupq_n_f32(6.f);";
        writer << "\nfloat32x4_t f3_v = vdupq_n_f32(3.f);";
        writer << "\nfloat32x4_t inv6_v = vdupq_n_f32(1/6.f);";
        return writer.str();
    }
    std::string GenIntrinsicFloat(
            const std::string& input, const std::string& output) const override {
        std::stringstream writer;
        auto input_temp = "hswish_temp";
        writer << "\n{";
        writer << "float32x4_t " << input_temp << " = vaddq_f32(" << input
               << ", f3_v);\n";
        writer << input_temp << " = vmaxq_f32(" << input_temp << ", vzero);\n";
        writer << input_temp << " = vminq_f32(" << input_temp << ", f6_v);\n";
        writer << input_temp << " = vmulq_f32(" << input << ", " << input_temp
               << ");\n";
        writer << "\n" << output << " = vmulq_f32(" << input_temp << ", inv6_v);";
        writer << "\n}";
        return writer.str();
    }
    std::string GenIntrinsicFloatStore(
            const std::string& input, const std::string& outptr) const override {
        std::stringstream writer;

        auto input_temp = "hswish_temp";
        writer << "\n{";
        writer << "float32x4_t " << input_temp << " = vaddq_f32(" << input
               << ", f3_v);\n";
        writer << input_temp << " = vmaxq_f32(" << input_temp << ", vzero);\n";
        writer << input_temp << " = vminq_f32(" << input_temp << ", f6_v);\n";
        writer << input_temp << " = vmulq_f32(" << input << ", " << input_temp
               << ");\n";
        writer << "\n vst1q_f32(" << outptr << ", vmulq_f32(" << input_temp
               << ", inv6_v));";
        writer << "\n}";
        return writer.str();
    }
    std::string GenIntrinsicQuantStore(
            const std::string& input, const std::string& outptr,
            const std::string& scale_sym) const override {
        CC_ASSERT(0) << "not impl quant hswish act\n";
        return "";
    }
};

std::shared_ptr<ActivationGenIntrinsicBase> create_activation_gener_instrinsic(
        std::string mode);

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
