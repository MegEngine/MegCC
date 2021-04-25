/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/CvtColor.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>
#include <unordered_map>

#include "CvtColor.h"

#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;
namespace {

struct CvtColorGen {
    std::string aux_func;
    std::string cvt_code;
};

CvtColorGen gen_rgb_yuv(TContext* ctx) {
    CvtColorGen res;
    res.aux_func = R"(
        #define descale(x, n) (((x) + (1 << ((n)-1))) >> (n))
        static inline uint8_t saturate_cast_ui8(int x){
            return (uint8_t)((unsigned)x <= UCHAR_MAX ? x
                                                    : x > 0 ? UCHAR_MAX : 0);
        }
    )";
    res.cvt_code = R"(
        #define yuv_shift 14
        const int coeffs[] = {1868, 9617, 4899, 8061, 14369};
        const int delta = 128 << yuv_shift;

        const int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3],
                C4 = coeffs[4];

        int16x4_t v_c0, v_c1, v_c2;
        int32x4_t v_c3, v_c4, v_delta, v_delta2;
        v_c0 = vdup_n_s16(coeffs[0]);
        v_c1 = vdup_n_s16(coeffs[1]);
        v_c2 = vdup_n_s16(coeffs[2]);
        v_c3 = vdupq_n_s32(coeffs[3]);
        v_c4 = vdupq_n_s32(coeffs[4]);
        v_delta = vdupq_n_s32(128 << yuv_shift);
        v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));

        for (size_t r = 0; r < rows; ++r) {
            const uint8_t* psrc = sptr + r * cols * ch;
            uint8_t* pdst = dptr + r * cols * ch;
            const uint8_t* const pend = psrc + cols * 3;

            // pack 8 pixels (24 uchar)
            for (; psrc <= pend - 8 * 3; psrc += 8 * 3, pdst += 8 * 3) {
                uint8x8x3_t v_dst;
                int16x8x3_t v_src16;

                uint8x8x3_t v_src = vld3_u8(psrc);
                v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
                v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
                v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

                int16x4x3_t v_src0;
                v_src0.val[0] = vget_low_s16(v_src16.val[0]);
                v_src0.val[1] = vget_low_s16(v_src16.val[1]);
                v_src0.val[2] = vget_low_s16(v_src16.val[2]);

                int32x4_t v_Y0 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0),
                                                    v_src0.val[1], v_c1),
                                        v_src0.val[2], v_c2);
                v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta2), yuv_shift);
                int32x4_t v_Cr0 = vmlaq_s32(
                        v_delta, vsubq_s32(vmovl_s16(v_src0.val[0]), v_Y0), v_c3);
                v_Cr0 = vshrq_n_s32(vaddq_s32(v_Cr0, v_delta2), yuv_shift);
                int32x4_t v_Cb0 = vmlaq_s32(
                        v_delta, vsubq_s32(vmovl_s16(v_src0.val[2]), v_Y0), v_c4);
                v_Cb0 = vshrq_n_s32(vaddq_s32(v_Cb0, v_delta2), yuv_shift);

                v_src0.val[0] = vget_high_s16(v_src16.val[0]);
                v_src0.val[1] = vget_high_s16(v_src16.val[1]);
                v_src0.val[2] = vget_high_s16(v_src16.val[2]);

                int32x4_t v_Y1 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0),
                                                    v_src0.val[1], v_c1),
                                        v_src0.val[2], v_c2);
                v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta2), yuv_shift);
                int32x4_t v_Cr1 = vmlaq_s32(
                        v_delta, vsubq_s32(vmovl_s16(v_src0.val[0]), v_Y1), v_c3);
                v_Cr1 = vshrq_n_s32(vaddq_s32(v_Cr1, v_delta2), yuv_shift);
                int32x4_t v_Cb1 = vmlaq_s32(
                        v_delta, vsubq_s32(vmovl_s16(v_src0.val[2]), v_Y1), v_c4);
                v_Cb1 = vshrq_n_s32(vaddq_s32(v_Cb1, v_delta2), yuv_shift);

                v_dst.val[0] = vqmovun_s16(
                        vcombine_s16(vqmovn_s32(v_Y0), vqmovn_s32(v_Y1)));
                v_dst.val[1] = vqmovun_s16(
                        vcombine_s16(vqmovn_s32(v_Cr0), vqmovn_s32(v_Cr1)));
                v_dst.val[2] = vqmovun_s16(
                        vcombine_s16(vqmovn_s32(v_Cb0), vqmovn_s32(v_Cb1)));

                vst3_u8(pdst, v_dst);
            }
            for (; psrc < pend; psrc += 3, pdst += 3) {
                int Y = descale(psrc[0] * C0 + psrc[1] * C1 + psrc[2] * C2,
                                yuv_shift);
                int Cr = descale((psrc[0] - Y) * C3 + delta, yuv_shift);
                int Cb = descale((psrc[2] - Y) * C4 + delta, yuv_shift);
                pdst[0] = saturate_cast_ui8(Y);
                pdst[1] = saturate_cast_ui8(Cr);
                pdst[2] = saturate_cast_ui8(Cb);
            }
        }
    )";
    return res;
}

CvtColorGen gen_rgb_bgr(TContext* ctx) {
    CvtColorGen res;
    res.aux_func = R"(
    )";
    res.cvt_code = R"(
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* psrc = sptr + (r) * cols * ch;
        uint8_t* pdst = dptr + (r) * cols * ch;
        const uint8_t* const pend = psrc + cols * 3;

        for (; psrc <= pend - 48; pdst += 48, psrc += 48) {
            uint8x16x3_t v_src = vld3q_u8(psrc), v_dst;
            v_dst.val[0] = v_src.val[2];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[0];
            vst3q_u8(pdst, v_dst);
        }
        for (; psrc <= pend - 24; pdst += 24, psrc += 24) {
            uint8x8x3_t v_src = vld3_u8(psrc), v_dst;
            v_dst.val[0] = v_src.val[2];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[0];
            vst3_u8(pdst, v_dst);
        }
        for (; psrc < pend; pdst += 3, psrc += 3) {
            uint8_t t0 = psrc[0], t1 = psrc[1], t2 = psrc[2];
            pdst[0] = t2;
            pdst[1] = t1;
            pdst[2] = t0;
        }
    }

    )";
    return res;
}

CvtColorGen gen_yuv_bgr_nv21(TContext* ctx) {
    CvtColorGen res;
    res.aux_func = R"(
        #define SET_COLOR(out, index)   \
        {   out[index++] = B;     \
            out[index++] = G;     \
            out[index++] = R;     \
        }
    )";
    res.cvt_code = R"(
    uint8x16_t v_y;
    int32x4_t v_y_s32_0, v_y_s32_1, v_y_s32_2, v_y_s32_3;
    uint8x8x2_t v_vu;
    int32x4_t v_RV0, v_RV1, v_RV2, v_RV3;
    int32x4_t v_GVU0, v_GVU1, v_GVU2, v_GVU3;
    int32x4_t v_BU0, v_BU1, v_BU2, v_BU3;

    int32x4x4_t v_R;
    int32x4x4_t v_G;
    int32x4x4_t v_B;
    uint8x16x3_t v_RGB, v_BGR;

    int16x8_t v_128;
    v_128 = vdupq_n_s16(128);

    int16x4_t v_359, v_88, v_183, v_454;
    v_359 = vdup_n_s16(359);
    v_88 = vdup_n_s16(88);
    v_183 = vdup_n_s16(183);
    v_454 = vdup_n_s16(454);
    int width = dst->cols;
    int height = dst->rows;
    int src_step = cols * ch;
    int dst_step = width * dst->channels;
    const unsigned char* pY = sptr;
    const unsigned char* pU;
    const unsigned char* pV;

    pV = sptr + height * src_step;
    //! only used if is_planar is false
    pU = sptr + (height + height / 4) * src_step;

    for (size_t r = 0; r < height; r += 2, pY += (src_step << 1)) {
        unsigned char* dst0 = dptr + r * dst_step;
        unsigned char* dst1 = dptr + (r + 1) * dst_step;
        size_t index0 = 0;
        size_t index1 = 0;
        int c = 0;
        for (; c <= (int)(width - 16); c += 16, index0 += 48, index1 += 48) {
            int16x8x2_t v_vu_s16;

            v_vu = vld2_u8(pV + c);
            v_vu_s16.val[0] =
                    vreinterpretq_s16_u16(vmovl_u8(v_vu.val[0]));
            v_vu_s16.val[1] =
                    vreinterpretq_s16_u16(vmovl_u8(v_vu.val[1]));
                
            v_vu_s16.val[0] = vsubq_s16(v_vu_s16.val[0], v_128);
            v_vu_s16.val[1] = vsubq_s16(v_vu_s16.val[1], v_128);

            int16x4_t v_v0, v_u0;
            int16x4_t v_v1, v_u1;
            v_v0 = vget_low_s16(v_vu_s16.val[0]);
            v_v1 = vget_high_s16(v_vu_s16.val[0]);
            v_u0 = vget_low_s16(v_vu_s16.val[1]);
            v_u1 = vget_high_s16(v_vu_s16.val[1]);

            v_RV1 = vshrq_n_s32(vmull_s16(v_v0, v_359), 8);
            v_RV3 = vshrq_n_s32(vmull_s16(v_v1, v_359), 8);
            v_GVU1 = vshrq_n_s32(
                    vaddq_s32(vmull_s16(v_u0, v_88), vmull_s16(v_v0, v_183)),
                    8);
            v_GVU3 = vshrq_n_s32(
                    vaddq_s32(vmull_s16(v_u1, v_88), vmull_s16(v_v1, v_183)),
                    8);
            v_BU1 = vshrq_n_s32(vmull_s16(v_u0, v_454), 8);
            v_BU3 = vshrq_n_s32(vmull_s16(v_u1, v_454), 8);

            int32x4x2_t temp;
            temp = vzipq_s32(v_RV1, v_RV1);
            v_RV0 = temp.val[0];
            v_RV1 = temp.val[1];
            temp = vzipq_s32(v_RV3, v_RV3);
            v_RV2 = temp.val[0];
            v_RV3 = temp.val[1];

            temp = vzipq_s32(v_GVU1, v_GVU1);
            v_GVU0 = temp.val[0];
            v_GVU1 = temp.val[1];
            temp = vzipq_s32(v_GVU3, v_GVU3);
            v_GVU2 = temp.val[0];
            v_GVU3 = temp.val[1];

            temp = vzipq_s32(v_BU1, v_BU1);
            v_BU0 = temp.val[0];
            v_BU1 = temp.val[1];
            temp = vzipq_s32(v_BU3, v_BU3);
            v_BU2 = temp.val[0];
            v_BU3 = temp.val[1];

            v_y = vld1q_u8(pY + c);
            uint8x8_t v_y_half;
            v_y_half = vget_low_u8(v_y);
            int16x8_t v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_0 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_1 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_y_half = vget_high_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_2 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_3 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_R.val[0] = vaddq_s32(v_y_s32_0, v_RV0);
            v_R.val[1] = vaddq_s32(v_y_s32_1, v_RV1);
            v_R.val[2] = vaddq_s32(v_y_s32_2, v_RV2);
            v_R.val[3] = vaddq_s32(v_y_s32_3, v_RV3);

            v_G.val[0] = vsubq_s32(v_y_s32_0, v_GVU0);
            v_G.val[1] = vsubq_s32(v_y_s32_1, v_GVU1);
            v_G.val[2] = vsubq_s32(v_y_s32_2, v_GVU2);
            v_G.val[3] = vsubq_s32(v_y_s32_3, v_GVU3);

            v_B.val[0] = vaddq_s32(v_y_s32_0, v_BU0);
            v_B.val[1] = vaddq_s32(v_y_s32_1, v_BU1);
            v_B.val[2] = vaddq_s32(v_y_s32_2, v_BU2);
            v_B.val[3] = vaddq_s32(v_y_s32_3, v_BU3);

   
            v_BGR.val[0] = vcombine_u8(
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                vmovn_s32(v_B.val[1]))),
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                vmovn_s32(v_B.val[3]))));
            v_BGR.val[1] = vcombine_u8(
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                vmovn_s32(v_G.val[1]))),
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                vmovn_s32(v_G.val[3]))));
            v_BGR.val[2] = vcombine_u8(
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                vmovn_s32(v_R.val[1]))),
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                vmovn_s32(v_R.val[3]))));
            vst3q_u8((dst0 + c * 3), v_BGR);
            

            v_y = vld1q_u8(pY + src_step + c);
            v_y_half = vget_low_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_0 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_1 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_y_half = vget_high_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_2 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_3 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_R.val[0] = vaddq_s32(v_y_s32_0, v_RV0);
            v_R.val[1] = vaddq_s32(v_y_s32_1, v_RV1);
            v_R.val[2] = vaddq_s32(v_y_s32_2, v_RV2);
            v_R.val[3] = vaddq_s32(v_y_s32_3, v_RV3);

            v_G.val[0] = vsubq_s32(v_y_s32_0, v_GVU0);
            v_G.val[1] = vsubq_s32(v_y_s32_1, v_GVU1);
            v_G.val[2] = vsubq_s32(v_y_s32_2, v_GVU2);
            v_G.val[3] = vsubq_s32(v_y_s32_3, v_GVU3);

            v_B.val[0] = vaddq_s32(v_y_s32_0, v_BU0);
            v_B.val[1] = vaddq_s32(v_y_s32_1, v_BU1);
            v_B.val[2] = vaddq_s32(v_y_s32_2, v_BU2);
            v_B.val[3] = vaddq_s32(v_y_s32_3, v_BU3);

            v_BGR.val[0] = vcombine_u8(
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                vmovn_s32(v_B.val[1]))),
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                vmovn_s32(v_B.val[3]))));
            v_BGR.val[1] = vcombine_u8(
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                vmovn_s32(v_G.val[1]))),
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                vmovn_s32(v_G.val[3]))));
            v_BGR.val[2] = vcombine_u8(
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                vmovn_s32(v_R.val[1]))),
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                vmovn_s32(v_R.val[3]))));
            vst3q_u8((dst1 + c * 3), v_BGR);
        }

        for (; c < (int)width; c += 2) {
            int Y00, Y01, Y10, Y11, U, V;
            int R, G, B;
            Y00 = *((pY) + c);
            Y01 = *((pY) + c + 1);
            Y10 = *((pY) + src_step + c);
            Y11 = *((pY) + src_step + c + 1);

               
            V = *(pV + c);
            U = *(pV + c + 1);
                
            

            int ruv, guv, buv;
            ruv = ((359 * (V - 128)) >> 8);
            guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
            buv = ((454 * (U - 128)) >> 8);

            R = Y00 + ruv;
            G = Y00 + guv;
            B = Y00 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst0, index0);

            R = Y01 + ruv;
            G = Y01 + guv;
            B = Y01 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst0, index0);

            ruv = ((359 * (V - 128)) >> 8);
            guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
            buv = ((454 * (U - 128)) >> 8);
            R = Y10 + ruv;
            G = Y10 + guv;
            B = Y10 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst1, index1);

            R = Y11 + ruv;
            G = Y11 + guv;
            B = Y11 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst1, index1);
        }
        pV += src_step;
    }
    )";
    return res;
}

struct CvtColorReg {
    using GenFunc = std::function<CvtColorGen(TContext*)>;
    struct CvtColorRegEntry {
        GenFunc func;
        std::string mode_symbol;
    };

    CvtColorReg() {
        reg_map = {{"RGB2YUV", {gen_rgb_yuv, "rgb2yuv"}},
                   {"YUV2BGR_NV21", {gen_yuv_bgr_nv21, "yuv2bgr_nv21"}},
                   {"RGB2BGR", {gen_rgb_bgr, "rgb2bgr"}}};
    }

    bool usable(const std::string& mode) {
        return reg_map.find(mode) != reg_map.end();
    }
    GenFunc get_func(const std::string& mode) { return reg_map[mode].func; }
    std::string get_mode_sym(const std::string& mode) {
        return reg_map[mode].mode_symbol;
    }
    std::unordered_map<std::string, CvtColorRegEntry> reg_map;
};

static CvtColorReg g_cvt_reg;
}  // namespace

bool CvtColorKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    bool mode_ok = g_cvt_reg.usable(context->getAttrStr("mode"));
    return dtype_ok && mode_ok;
}

//! kernel gen
std::string CvtColorKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto mode = context->getAttrStr("mode");
    auto mode_str = g_cvt_reg.get_mode_sym(mode);
    ss << "tinycv_cvt_" << mode_str << "_" << src_dtype;
    return ss.str();
}

std::string CvtColorKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst)";
}

std::string CvtColorKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    auto mode = context->getAttrStr("mode");
    auto gen_func = g_cvt_reg.get_func(mode);
    auto code = gen_func(context);
    std::string body_temp = R"(
        #include <arm_neon.h>
        #include <limits.h>
        #include <string.h>
        #include "tinycv_c.h"
        ${aux_func}
        void ${kernel_sig}{
            uint8_t * sptr = src->data;
            uint8_t * dptr = dst->data;
            size_t rows = src->rows;
            size_t cols = src->cols;
            size_t hw = rows * cols;
            size_t ch = src->channels;
            
            ${cvt_code}
            
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("aux_func", code.aux_func)
            .add("cvt_code", code.cvt_code)
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
