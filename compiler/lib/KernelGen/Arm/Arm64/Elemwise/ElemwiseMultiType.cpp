#include <sstream>

#include "Common/ElemwiseCommon.h"
#include "ElemwiseMultiType.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;

namespace {

std::string gen_dep(std::string mode) {
    return R"(
        static inline int8_t fp32_to_int8(float src){
                int res = roundf(src);
                res = res > 127? 127:res;
                res = res < -128? -128:res;
                return (int8_t)(res);
        }
    )";
}
std::string gen_unary(std::string mode, bool unroll, const std::string& src_dtype) {
    if (mode == "QRELU") {
        if (unroll) {
            if (src_dtype == "int") {
                return R"(
                    int32x4_t v_in_0 = vld1q_s32(inptr);
                    float32x4_t vf_0 = vcvtq_f32_s32(v_in_0);
                    vf_0 = vmulq_f32(vf_0, v_scale_0);
                    vf_0 = vmaxq_f32(vf_0, v_zero);
                    vf_0 = vmulq_f32(vf_0, v_scale_div);
                    v_in_0 = vcvtaq_s32_f32(vf_0);
                    int16x4_t v_res_0 = vqmovn_s32(v_in_0);

                    int32x4_t v_in_1 = vld1q_s32(inptr + 4);
                    float32x4_t vf_1 = vcvtq_f32_s32(v_in_1);
                    vf_1 = vmulq_f32(vf_1, v_scale_0);
                    vf_1 = vmaxq_f32(vf_1, v_zero);
                    vf_1 = vmulq_f32(vf_1, v_scale_div);
                    v_in_1 = vcvtaq_s32_f32(vf_1);
                    int16x4_t v_res_1 = vqmovn_s32(v_in_1);

                    int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                    int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                    int32x4_t v_in_2 = vld1q_s32(inptr + 8);
                    float32x4_t vf_2 = vcvtq_f32_s32(v_in_2);
                    vf_2 = vmulq_f32(vf_2, v_scale_0);
                    vf_2 = vmaxq_f32(vf_2, v_zero);
                    vf_2 = vmulq_f32(vf_2, v_scale_div);
                    v_in_2 = vcvtaq_s32_f32(vf_2);
                    int16x4_t v_res_2 = vqmovn_s32(v_in_2);

                    int32x4_t v_in_3 = vld1q_s32(inptr + 12);
                    float32x4_t vf_3 = vcvtq_f32_s32(v_in_3);
                    vf_3 = vmulq_f32(vf_3, v_scale_0);
                    vf_3 = vmaxq_f32(vf_3, v_zero);
                    vf_3 = vmulq_f32(vf_3, v_scale_div);
                    v_in_3 = vcvtaq_s32_f32(vf_3);
                    int16x4_t v_res_3 = vqmovn_s32(v_in_3);

                    int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                    int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                    int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                    vst1q_s8(outptr, v_res);
                )";
            } else {
                CC_ASSERT(src_dtype == "int8_t");
                return R"(
                    int8x16_t v_in = vld1q_s8(inptr);
                    int16x8_t v16_in_01 = vmovl_s8(vget_low_s8(v_in));

                    int32x4_t v_in_0 = vmovl_s16(vget_low_s16(v16_in_01));
                    float32x4_t vf_0 = vcvtq_f32_s32(v_in_0);
                    vf_0 = vmulq_f32(vf_0, v_scale_0);
                    vf_0 = vmaxq_f32(vf_0, v_zero);
                    int32x4_t v_in_1 = vmovl_s16(vget_high_s16(v16_in_01));
                    vf_0 = vmulq_f32(vf_0, v_scale_div);
                    v_in_0 = vcvtaq_s32_f32(vf_0);
                    int16x4_t v_res_0 = vqmovn_s32(v_in_0);

                    int16x8_t v16_in_23 = vmovl_s8(vget_high_s8(v_in));
                    float32x4_t vf_1 = vcvtq_f32_s32(v_in_1);
                    vf_1 = vmulq_f32(vf_1, v_scale_0);
                    vf_1 = vmaxq_f32(vf_1, v_zero);
                    int32x4_t v_in_2 = vmovl_s16(vget_low_s16(v16_in_23));
                    vf_1 = vmulq_f32(vf_1, v_scale_div);
                    v_in_1 = vcvtaq_s32_f32(vf_1);
                    int16x4_t v_res_1 = vqmovn_s32(v_in_1);

                    int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                    int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                    int32x4_t v_in_3 = vmovl_s16(vget_high_s16(v16_in_23));
                    float32x4_t vf_2 = vcvtq_f32_s32(v_in_2);
                    vf_2 = vmulq_f32(vf_2, v_scale_0);
                    vf_2 = vmaxq_f32(vf_2, v_zero);
                    vf_2 = vmulq_f32(vf_2, v_scale_div);
                    v_in_2 = vcvtaq_s32_f32(vf_2);
                    int16x4_t v_res_2 = vqmovn_s32(v_in_2);

                    float32x4_t vf_3 = vcvtq_f32_s32(v_in_3);
                    vf_3 = vmulq_f32(vf_3, v_scale_0);
                    vf_3 = vmaxq_f32(vf_3, v_zero);
                    vf_3 = vmulq_f32(vf_3, v_scale_div);
                    v_in_3 = vcvtaq_s32_f32(vf_3);
                    int16x4_t v_res_3 = vqmovn_s32(v_in_3);

                    int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                    int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                    int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                    vst1q_s8(outptr, v_res);
                )";
            }
        } else {
            return "int8_t out_val = fp32_to_int8(((scale_0 * val_0) > 0?(scale_0 "
                   "* "
                   "val_0 ):0) * scale_div)";
        }
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary(
        std::string mode, bool unroll, const std::string& src_type, bool broadcast) {
    if (mode == "QADD") {
        if (unroll) {
            if (src_type == "int") {
                if (broadcast) {
                    return R"(
                        int32x4_t v_in_0_0 = vld1q_s32(inptr_0);
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int32x4_t v_in_0_1 = vld1q_s32(inptr_0 + 4);
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_2 = vld1q_s32(inptr_0 + 8);
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        int32x4_t v_in_0_3 = vld1q_s32(inptr_0 + 12);
                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                } else {
                    return R"(
                        int32x4_t v_in_0_0 = vld1q_s32(inptr_0);
                        int32x4_t v_in_1_0 = vld1q_s32(inptr_1);
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        float32x4_t vf_1_0 = vcvtq_f32_s32(v_in_1_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_1_0 = vmulq_f32(vf_1_0, v_scale_1);
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int32x4_t v_in_0_1 = vld1q_s32(inptr_0 + 4);
                        int32x4_t v_in_1_1 = vld1q_s32(inptr_1 + 4);
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        float32x4_t vf_1_1 = vcvtq_f32_s32(v_in_1_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_1_1 = vmulq_f32(vf_1_1, v_scale_1);
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_2 = vld1q_s32(inptr_0 + 8);
                        int32x4_t v_in_1_2 = vld1q_s32(inptr_1 + 8);
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        float32x4_t vf_1_2 = vcvtq_f32_s32(v_in_1_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_1_2 = vmulq_f32(vf_1_2, v_scale_1);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        int32x4_t v_in_0_3 = vld1q_s32(inptr_0 + 12);
                        int32x4_t v_in_1_3 = vld1q_s32(inptr_1 + 12);
                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        float32x4_t vf_1_3 = vcvtq_f32_s32(v_in_1_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_1_3 = vmulq_f32(vf_1_3, v_scale_1);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                }
            } else {
                CC_ASSERT(src_type == "int8_t");
                if (broadcast) {
                    return R"(
                        int8x16_t v8_in_0 = vld1q_s8(inptr_0);
                        int16x8_t v16_in_0_01 = vmovl_s8(vget_low_s8(v8_in_0));

                        int32x4_t v_in_0_0 = vmovl_s16(vget_low_s16(v16_in_0_01));
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1);
                        int32x4_t v_in_0_1 = vmovl_s16(vget_high_s16(v16_in_0_01));
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int16x8_t v16_in_0_23 = vmovl_s8(vget_high_s8(v8_in_0));
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        int32x4_t v_in_0_2 = vmovl_s16(vget_low_s16(v16_in_0_23));
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_3 = vmovl_s16(vget_high_s16(v16_in_0_23));
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                } else {
                    return R"(
                        int8x16_t v8_in_0 = vld1q_s8(inptr_0);
                        int8x16_t v8_in_1 = vld1q_s8(inptr_1);

                        int16x8_t v16_in_0_01 = vmovl_s8(vget_low_s8(v8_in_0));
                        int16x8_t v16_in_1_01 = vmovl_s8(vget_low_s8(v8_in_1));

                        int32x4_t v_in_0_0 = vmovl_s16(vget_low_s16(v16_in_0_01));
                        int32x4_t v_in_1_0 = vmovl_s16(vget_low_s16(v16_in_1_01));
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        float32x4_t vf_1_0 = vcvtq_f32_s32(v_in_1_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_1_0 = vmulq_f32(vf_1_0, v_scale_1);
                        int32x4_t v_in_0_1 = vmovl_s16(vget_high_s16(v16_in_0_01));
                        int32x4_t v_in_1_1 = vmovl_s16(vget_high_s16(v16_in_1_01));
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int16x8_t v16_in_0_23 = vmovl_s8(vget_high_s8(v8_in_0));
                        int16x8_t v16_in_1_23 = vmovl_s8(vget_high_s8(v8_in_1));

                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        float32x4_t vf_1_1 = vcvtq_f32_s32(v_in_1_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_1_1 = vmulq_f32(vf_1_1, v_scale_1);
                        int32x4_t v_in_0_2 = vmovl_s16(vget_low_s16(v16_in_0_23));
                        int32x4_t v_in_1_2 = vmovl_s16(vget_low_s16(v16_in_1_23));
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_3 = vmovl_s16(vget_high_s16(v16_in_0_23));
                        int32x4_t v_in_1_3 = vmovl_s16(vget_high_s16(v16_in_1_23));
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        float32x4_t vf_1_2 = vcvtq_f32_s32(v_in_1_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_1_2 = vmulq_f32(vf_1_2, v_scale_1);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        float32x4_t vf_1_3 = vcvtq_f32_s32(v_in_1_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_1_3 = vmulq_f32(vf_1_3, v_scale_1);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                }
            }
        } else {
            if (broadcast) {
                return "int8_t out_val = fp32_to_int8((scale_0 * val_0 + f_val_1) * "
                       "scale_div);";
            } else {
                return "int8_t out_val = fp32_to_int8((scale_0 * val_0 + scale_1 * "
                       "val_1) * scale_div);";
            }
        }
    } else if (mode == "QFUSE_ADD_RELU") {
        if (unroll) {
            if (src_type == "int") {
                if (broadcast) {
                    return R"(
                        int32x4_t v_in_0_0 = vld1q_s32(inptr_0);
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1);
                        vf_0_0 = vmaxq_f32(vf_0_0, v_zero);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int32x4_t v_in_0_1 = vld1q_s32(inptr_0 + 4);
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1);
                        vf_0_1 = vmaxq_f32(vf_0_1, v_zero);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_2 = vld1q_s32(inptr_0 + 8);
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1);
                        vf_0_2 = vmaxq_f32(vf_0_2, v_zero);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        int32x4_t v_in_0_3 = vld1q_s32(inptr_0 + 12);
                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1);
                        vf_0_3 = vmaxq_f32(vf_0_3, v_zero);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                } else {
                    return R"(
                        int32x4_t v_in_0_0 = vld1q_s32(inptr_0);
                        int32x4_t v_in_1_0 = vld1q_s32(inptr_1);
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        float32x4_t vf_1_0 = vcvtq_f32_s32(v_in_1_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_1_0 = vmulq_f32(vf_1_0, v_scale_1);
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1_0);
                        vf_0_0 = vmaxq_f32(vf_0_0, v_zero);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int32x4_t v_in_0_1 = vld1q_s32(inptr_0 + 4);
                        int32x4_t v_in_1_1 = vld1q_s32(inptr_1 + 4);
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        float32x4_t vf_1_1 = vcvtq_f32_s32(v_in_1_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_1_1 = vmulq_f32(vf_1_1, v_scale_1);
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1_1);
                        vf_0_1 = vmaxq_f32(vf_0_1, v_zero);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_2 = vld1q_s32(inptr_0 + 8);
                        int32x4_t v_in_1_2 = vld1q_s32(inptr_1 + 8);
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        float32x4_t vf_1_2 = vcvtq_f32_s32(v_in_1_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_1_2 = vmulq_f32(vf_1_2, v_scale_1);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1_2);
                        vf_0_2 = vmaxq_f32(vf_0_2, v_zero);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        int32x4_t v_in_0_3 = vld1q_s32(inptr_0 + 12);
                        int32x4_t v_in_1_3 = vld1q_s32(inptr_1 + 12);
                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        float32x4_t vf_1_3 = vcvtq_f32_s32(v_in_1_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_1_3 = vmulq_f32(vf_1_3, v_scale_1);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1_3);
                        vf_0_3 = vmaxq_f32(vf_0_3, v_zero);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                }
            } else {
                CC_ASSERT(src_type == "int8_t");
                if (broadcast) {
                    return R"(
                        int8x16_t v8_in_0 = vld1q_s8(inptr_0);
                        int16x8_t v16_in_0_01 = vmovl_s8(vget_low_s8(v8_in_0));

                        int32x4_t v_in_0_0 = vmovl_s16(vget_low_s16(v16_in_0_01));
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1);
                        int32x4_t v_in_0_1 = vmovl_s16(vget_high_s16(v16_in_0_01));
                        vf_0_0 = vmaxq_f32(vf_0_0, v_zero);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int16x8_t v16_in_0_23 = vmovl_s8(vget_high_s8(v8_in_0));
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1);
                        vf_0_1 = vmaxq_f32(vf_0_1, v_zero);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        int32x4_t v_in_0_2 = vmovl_s16(vget_low_s16(v16_in_0_23));
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_3 = vmovl_s16(vget_high_s16(v16_in_0_23));
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1);
                        vf_0_2 = vmaxq_f32(vf_0_2, v_zero);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1);
                        vf_0_3 = vmaxq_f32(vf_0_3, v_zero);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                } else {
                    return R"(
                        int8x16_t v8_in_0 = vld1q_s8(inptr_0);
                        int8x16_t v8_in_1 = vld1q_s8(inptr_1);

                        int16x8_t v16_in_0_01 = vmovl_s8(vget_low_s8(v8_in_0));
                        int16x8_t v16_in_1_01 = vmovl_s8(vget_low_s8(v8_in_1));

                        int32x4_t v_in_0_0 = vmovl_s16(vget_low_s16(v16_in_0_01));
                        int32x4_t v_in_1_0 = vmovl_s16(vget_low_s16(v16_in_1_01));
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        float32x4_t vf_1_0 = vcvtq_f32_s32(v_in_1_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_0);
                        vf_1_0 = vmulq_f32(vf_1_0, v_scale_1);
                        int32x4_t v_in_0_1 = vmovl_s16(vget_high_s16(v16_in_0_01));
                        int32x4_t v_in_1_1 = vmovl_s16(vget_high_s16(v16_in_1_01));
                        vf_0_0 = vaddq_f32(vf_0_0, vf_1_0);
                        vf_0_0 = vmaxq_f32(vf_0_0, v_zero);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale_div);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int16x8_t v16_in_0_23 = vmovl_s8(vget_high_s8(v8_in_0));
                        int16x8_t v16_in_1_23 = vmovl_s8(vget_high_s8(v8_in_1));

                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        float32x4_t vf_1_1 = vcvtq_f32_s32(v_in_1_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_0);
                        vf_1_1 = vmulq_f32(vf_1_1, v_scale_1);
                        int32x4_t v_in_0_2 = vmovl_s16(vget_low_s16(v16_in_0_23));
                        int32x4_t v_in_1_2 = vmovl_s16(vget_low_s16(v16_in_1_23));
                        vf_0_1 = vaddq_f32(vf_0_1, vf_1_1);
                        vf_0_1 = vmaxq_f32(vf_0_1, v_zero);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale_div);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_3 = vmovl_s16(vget_high_s16(v16_in_0_23));
                        int32x4_t v_in_1_3 = vmovl_s16(vget_high_s16(v16_in_1_23));
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        float32x4_t vf_1_2 = vcvtq_f32_s32(v_in_1_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_0);
                        vf_1_2 = vmulq_f32(vf_1_2, v_scale_1);
                        vf_0_2 = vaddq_f32(vf_0_2, vf_1_2);
                        vf_0_2 = vmaxq_f32(vf_0_2, v_zero);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale_div);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        float32x4_t vf_1_3 = vcvtq_f32_s32(v_in_1_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_0);
                        vf_1_3 = vmulq_f32(vf_1_3, v_scale_1);
                        vf_0_3 = vaddq_f32(vf_0_3, vf_1_3);
                        vf_0_3 = vmaxq_f32(vf_0_3, v_zero);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale_div);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                }
            }
        } else {
            if (broadcast) {
                return R"(
                float val0 = scale_0 * val_0;
                int8_t out_val  = fp32_to_int8( ((val0 + f_val_1) > 0? (val0 + f_val_1):0) * scale_div);
                )";
            } else {
                return R"(
                float val0 = scale_0 * val_0;
                float val1 = scale_1 * val_1;     
                int8_t out_val  = fp32_to_int8( ((val0 + val1) > 0? (val0 + val1):0) * scale_div);
                )";
            }
        }
    } else if (mode == "QMUL") {
        if (unroll) {
            if (src_type == "int") {
                if (broadcast) {
                    return R"(
                        int32x4_t v_in_0_0 = vld1q_s32(inptr_0);
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, vf_qmul_1);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int32x4_t v_in_0_1 = vld1q_s32(inptr_0 + 4);
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, vf_qmul_1);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_2 = vld1q_s32(inptr_0 + 8);
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, vf_qmul_1);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        int32x4_t v_in_0_3 = vld1q_s32(inptr_0 + 12);
                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, vf_qmul_1);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                } else {
                    return R"(
                        int32x4_t v_in_0_0 = vld1q_s32(inptr_0);
                        int32x4_t v_in_1_0 = vld1q_s32(inptr_1);
                        float32x4_t vf_0_0 = vcvtq_f32_s32(vmulq_s32(v_in_0_0, v_in_1_0));
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale);
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int32x4_t v_in_0_1 = vld1q_s32(inptr_0 + 4);
                        int32x4_t v_in_1_1 = vld1q_s32(inptr_1 + 4);
                        float32x4_t vf_0_1 = vcvtq_f32_s32(vmulq_s32(v_in_0_1, v_in_1_1));
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale);
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_2 = vld1q_s32(inptr_0 + 8);
                        int32x4_t v_in_1_2 = vld1q_s32(inptr_1 + 8);
                        float32x4_t vf_0_2 = vcvtq_f32_s32(vmulq_s32(v_in_0_2, v_in_1_2));
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        int32x4_t v_in_0_3 = vld1q_s32(inptr_0 + 12);
                        int32x4_t v_in_1_3 = vld1q_s32(inptr_1 + 12);
                        float32x4_t vf_0_3 = vcvtq_f32_s32(vmulq_s32(v_in_0_3, v_in_1_3));
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                }
            } else {
                CC_ASSERT(src_type == "int8_t");
                if (broadcast) {
                    return R"(
                        int8x16_t v8_in_0 = vld1q_s8(inptr_0);
                        int16x8_t v16_in_0_01 = vmovl_s8(vget_low_s8(v8_in_0));

                        int32x4_t v_in_0_0 = vmovl_s16(vget_low_s16(v16_in_0_01));
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, vf_qmul_1);
                        int32x4_t v_in_0_1 = vmovl_s16(vget_high_s16(v16_in_0_01));
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int16x8_t v16_in_0_23 = vmovl_s8(vget_high_s8(v8_in_0));
                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, vf_qmul_1);
                        int32x4_t v_in_0_2 = vmovl_s16(vget_low_s16(v16_in_0_23));
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_3 = vmovl_s16(vget_high_s16(v16_in_0_23));
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, vf_qmul_1);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, vf_qmul_1);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                } else {
                    return R"(
                        int8x16_t v8_in_0 = vld1q_s8(inptr_0);
                        int8x16_t v8_in_1 = vld1q_s8(inptr_1);

                        int16x8_t v16_in_mul_01 = vmull_s8(vget_low_s8(v8_in_0), vget_low_s8(v8_in_1));

                        int32x4_t v_in_0_0 = vmovl_s16(vget_low_s16(v16_in_mul_01));
                        float32x4_t vf_0_0 = vcvtq_f32_s32(v_in_0_0);
                        vf_0_0 = vmulq_f32(vf_0_0, v_scale);
                        int32x4_t v_in_0_1 = vmovl_s16(vget_high_s16(v16_in_mul_01));
                        v_in_0_0 = vcvtaq_s32_f32(vf_0_0);
                        int16x4_t v_res_0 = vqmovn_s32(v_in_0_0);

                        int16x8_t v16_in_mul_23 = vmull_s8(vget_high_s8(v8_in_0), vget_high_s8(v8_in_1));

                        float32x4_t vf_0_1 = vcvtq_f32_s32(v_in_0_1);
                        vf_0_1 = vmulq_f32(vf_0_1, v_scale);
                        int32x4_t v_in_0_2 = vmovl_s16(vget_low_s16(v16_in_mul_23));
                        v_in_0_1 = vcvtaq_s32_f32(vf_0_1);
                        int16x4_t v_res_1 = vqmovn_s32(v_in_0_1);

                        int16x8_t v_res_01 = vcombine_s16(v_res_0, v_res_1);
                        int8x8_t v8_res_01 = vqmovn_s16(v_res_01);

                        int32x4_t v_in_0_3 = vmovl_s16(vget_high_s16(v16_in_mul_23));
                        float32x4_t vf_0_2 = vcvtq_f32_s32(v_in_0_2);
                        vf_0_2 = vmulq_f32(vf_0_2, v_scale);
                        v_in_0_2 = vcvtaq_s32_f32(vf_0_2);
                        int16x4_t v_res_2 = vqmovn_s32(v_in_0_2);

                        float32x4_t vf_0_3 = vcvtq_f32_s32(v_in_0_3);
                        vf_0_3 = vmulq_f32(vf_0_3, v_scale);
                        v_in_0_3 = vcvtaq_s32_f32(vf_0_3);
                        int16x4_t v_res_3 = vqmovn_s32(v_in_0_3);

                        int16x8_t v_res_23 = vcombine_s16(v_res_2, v_res_3);
                        int8x8_t v8_res_23 = vqmovn_s16(v_res_23);

                        int8x16_t v_res = vcombine_s8(v8_res_01, v8_res_23);

                        vst1q_s8(outptr, v_res);
                    )";
                }
            }
        } else {
            if (broadcast) {
                return R"(
                    int8_t out_val = fp32_to_int8(val_0 * f_qmul_val_1);
                )";
            } else {
                return R"(
                    int8_t out_val = fp32_to_int8(val_0 * val_1 * scale_mul);
                )";
            }
        }
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

}  // namespace

bool ElemwiseMultiTypeKernel::IsAvailable(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    auto nr_operands = context->getAttrInt("nr_operands");
    bool nr_operands_ok = nr_operands == 2 || nr_operands == 3;
    bool mode_ok_unary = nr_operands == 2 && mode == "QRELU";
    bool dtype_ok_unary =
            nr_operands == 2 &&
            (Utils::is_quant_dtype(context->getAttrOprand("operand:0").dtype, 8) ||
             Utils::is_quant_dtype(context->getAttrOprand("operand:0").dtype, 32)) &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:1").dtype, 8);
    bool mode_ok_binary =
            nr_operands == 3 &&
            (mode == "QADD" || mode == "QFUSE_ADD_RELU" || mode == "QMUL");
    bool dtype_ok_binary =
            nr_operands == 3 &&
            ((Utils::is_quant_dtype(context->getAttrOprand("operand:0").dtype, 8) &&
              Utils::is_quant_dtype(context->getAttrOprand("operand:1").dtype, 8)) ||
             (Utils::is_quant_dtype(context->getAttrOprand("operand:0").dtype, 32) &&
              Utils::is_quant_dtype(context->getAttrOprand("operand:1").dtype, 32))) &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:2").dtype, 8);
    const auto& op0_shape = context->getAttrOprand("operand:0").shape;
    const auto& op1_shape = context->getAttrOprand("operand:1").shape;
    size_t op1_nr_elem = 1;
    for (auto dim : op1_shape) {
        op1_nr_elem *= dim;
    }
    //! broadcast mode 0: op0 shape: (a, b, c, d, ...), op1 shape: (1, b, 1, 1, ...)
    //! broadcast mode 1: op0 shape: (a, b, c, d, ...), op1_nr_elem = 1
    bool shape_ok_binary =
            nr_operands == 3 &&
            ((op0_shape == op1_shape) ||
             (op0_shape.size() == op1_shape.size() && op0_shape.size() > 2 &&
              op0_shape[1] == op1_shape[1] && op1_nr_elem == op1_shape[1]) ||
             (op1_nr_elem == 1));
    return nr_operands_ok && ((mode_ok_unary && dtype_ok_unary) ||
                              (mode_ok_binary && dtype_ok_binary && shape_ok_binary));
}

std::string ElemwiseMultiTypeKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Arm64_kernel_elementwise_multitype";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string ElemwiseMultiTypeKernel::GetKernelBody(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    std::stringstream writer;
    writer << "#include <arm_neon.h> \n";
    writer << "#include <math.h> \n";
    writer << gen_dep(mode);
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);
    if (context->getAttrInt("nr_operands") == 2) {
        auto op0 = context->getAttrOprand("operand:0");
        auto dst = context->getAttrOprand("operand:1");
        auto op0_specifier = Utils::cvt_dtype_specifier(op0.dtype);
        auto dst_specifier = Utils::cvt_dtype_specifier(dst.dtype);
        std::string unary_str = R"({
                ${op0_specifier}* input_0 = (${op0_specifier}*)inputs[0]->ptr;
                float scale_0 = inputs[0]->dtype.param.scale;
                float32x4_t v_scale_0 = vdupq_n_f32(scale_0);
                float32x4_t v_zero = vdupq_n_f32(0.f);
                TINYNN_ASSERT(input_0);
                ${dst_specifier}* output_data = (${dst_specifier}*)outputs[0]->ptr;
                float scale_dst = outputs[0]->dtype.param.scale;
                TINYNN_ASSERT(output_data);
                float scale_div = 1.f / scale_dst;
                float32x4_t v_scale_div = vdupq_n_f32(scale_div);

                Layout in_layout = inputs[0]->layout;
                size_t nr_elem = 1;
                for (int i = 0; i < in_layout.nr_dim; ++i) {
                    nr_elem *= in_layout.dims[i];
                }
                size_t i = 0;
                for(; i + 15 < nr_elem; i += 16){
                    ${op0_specifier} *inptr = input_0 + i;
                    ${dst_specifier} *outptr = output_data + i;
                    ${act_unroll};
                }
                for(; i < nr_elem; ++i){
                    ${op0_specifier} val_0 = input_0[i];
                    ${act};
                    output_data[i] = out_val;
                }
                return TinyNN_SUCCESS;
                }
            )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("op0_specifier", op0_specifier)
                          .add("dst_specifier", dst_specifier)
                          .add("act", gen_unary(mode, false, op0_specifier))
                          .add("act_unroll", gen_unary(mode, true, op0_specifier))
                          .render(unary_str);
    } else if (context->getAttrInt("nr_operands") == 3) {
        auto op0 = context->getAttrOprand("operand:0");
        auto op1 = context->getAttrOprand("operand:1");
        auto dst = context->getAttrOprand("operand:2");
        auto op0_specifier = Utils::cvt_dtype_specifier(op0.dtype);
        auto op1_specifier = Utils::cvt_dtype_specifier(op1.dtype);
        auto dst_specifier = Utils::cvt_dtype_specifier(dst.dtype);
        std::string binary_str = R"({
                ${op0_specifier}* input_0 = (${op0_specifier}*)inputs[0]->ptr;
                float scale_0 = inputs[0]->dtype.param.scale;
                float32x4_t v_scale_0 = vdupq_n_f32(scale_0);
                TINYNN_ASSERT(input_0);
                ${op1_specifier}* input_1 = (${op1_specifier}*)inputs[1]->ptr;
                float scale_1 = inputs[1]->dtype.param.scale;
                float32x4_t v_scale_1 = vdupq_n_f32(scale_1);
                TINYNN_ASSERT(input_1);
                ${dst_specifier}* output_data = (${dst_specifier}*)outputs[0]->ptr;
                float scale_dst = outputs[0]->dtype.param.scale;
                TINYNN_ASSERT(output_data);
                float scale_div = 1.f / scale_dst;
                float32x4_t v_scale_div = vdupq_n_f32(scale_div);
                float scale_mul = scale_0 * scale_1 * scale_div;
                float32x4_t v_scale = vdupq_n_f32(scale_mul);
                float32x4_t v_zero = vdupq_n_f32(0.f);

                Layout in_layout0 = inputs[0]->layout;
                size_t nr_elem0 = 1;
                for (int i = 0; i < in_layout0.nr_dim; ++i) {
                    nr_elem0 *= in_layout0.dims[i];
                }
                Layout in_layout1 = inputs[1]->layout;
                size_t nr_elem1 = 1;
                for (int i = 0; i < in_layout1.nr_dim; ++i) {
                    nr_elem1 *= in_layout1.dims[i];
                }
                if (nr_elem0 == nr_elem1) {
                    size_t i = 0;
                    for(; i + 15 < nr_elem0; i += 16){
                        ${op0_specifier} *inptr_0 = input_0 + i;
                        ${op1_specifier} *inptr_1 = input_1 + i;
                        ${dst_specifier} *outptr = output_data + i;
                        ${act_unroll};
                    }
                    for(; i < nr_elem0; ++i){
                        ${op0_specifier} val_0 = input_0[i];
                        ${op1_specifier} val_1 = input_1[i];
                        ${act};
                        output_data[i] = out_val;
                    }
                } else if (nr_elem1 == 1) {
                    size_t i = 0;
                    float f_val_1 = input_1[0] * scale_1;
                    float32x4_t vf_1 = vdupq_n_f32(f_val_1);
                    float f_qmul_val_1 = input_1[0] * scale_mul;
                    float32x4_t vf_qmul_1 = vdupq_n_f32(f_qmul_val_1);
                    for (; i + 15 < nr_elem0; i += 16) {
                        ${op0_specifier} *inptr_0 = input_0 + i;
                        ${dst_specifier} *outptr = output_data + i;
                        ${act_unroll_broadcast};
                    }
                    for(; i < nr_elem0; ++i){
                        ${op0_specifier} val_0 = input_0[i];
                        ${act_broadcast};
                        output_data[i] = out_val;
                    }
                } else {
                    TINYNN_ASSERT(nr_elem0 > nr_elem1);
                    for (int i = 0; i < in_layout0.dims[0]; ++i) {
                        for (int j = 0; j < in_layout0.dims[1]; ++j) {
                            float f_val_1 = input_1[j] * scale_1;
                            float32x4_t vf_1 = vdupq_n_f32(f_val_1);
                            float f_qmul_val_1 = input_1[j] * scale_mul;
                            float32x4_t vf_qmul_1 = vdupq_n_f32(f_qmul_val_1);
                            size_t k = 0;
                            int idx_base = i * in_layout0.stride[0] + j * in_layout0.stride[1];
                            for (; k + 15 < in_layout0.stride[1]; k += 16) {
                                int idx = idx_base + k;
                                ${op0_specifier} *inptr_0 = input_0 + idx;
                                ${dst_specifier} *outptr = output_data + idx;
                                ${act_unroll_broadcast};
                            }
                            for (int k = 0; k < in_layout0.stride[1]; ++k) {
                                int idx = idx_base + k;
                                ${op0_specifier} val_0 = input_0[idx];
                                ${act_broadcast};
                                output_data[idx] = out_val;
                            }
                        }
                    }
                }
                return TinyNN_SUCCESS;
                }
            )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("op0_specifier", op0_specifier)
                          .add("op1_specifier", op1_specifier)
                          .add("dst_specifier", dst_specifier)
                          .add("act_unroll",
                               gen_binary(mode, true, op0_specifier, false))
                          .add("act", gen_binary(mode, false, op0_specifier, false))
                          .add("act_unroll_broadcast",
                               gen_binary(mode, true, op0_specifier, true))
                          .add("act_broadcast",
                               gen_binary(mode, false, op0_specifier, true))
                          .render(binary_str);
    } else {
        CC_ABORT << "not support operands size " << context->getAttrInt("nr_operands")
                 << "\n";
    }
    return writer.str();
}

// vim: syntax=cpp.doxygen
