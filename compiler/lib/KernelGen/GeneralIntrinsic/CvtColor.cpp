/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/CvtColor.cpp
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
using namespace GeneralIntrinsic;
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

        GI_INT16_t v_c0, v_c1, v_c2;
        GI_INT32_t v_c3, v_c4, v_delta, v_delta2, v_zero;
        v_zero = GiBroadcastInt32(0); 
        v_c0 = GiBroadcastInt16(coeffs[0]);
        v_c1 = GiBroadcastInt16(coeffs[1]);
        v_c2 = GiBroadcastInt16(coeffs[2]);
        v_c3 = GiBroadcastInt32(coeffs[3]);
        v_c4 = GiBroadcastInt32(coeffs[4]);
        v_delta = GiBroadcastInt32(128 << yuv_shift);
        v_delta2 = GiBroadcastInt32(1 << (yuv_shift - 1));

        for (size_t r = 0; r < rows; ++r) {
            const uint8_t* psrc = sptr + r * cols * ch;
            uint8_t* pdst = dptr + r * cols * ch;
            const uint8_t* const pend = psrc + cols * 3;

            // pack 16 pixels (48 uchar)
            for (; psrc <= pend - 16 * 3; psrc += 16 * 3, pdst += 16 * 3) {
                GI_UINT8_t v_src0 = GiLoadUzip0V3Uint8(psrc);
                GI_UINT8_t v_src1 = GiLoadUzip1V3Uint8(psrc);
                GI_UINT8_t v_src2 = GiLoadUzip2V3Uint8(psrc);

                GI_INT16_t v_src00_16 = GiCvtUint8toInt16Low(v_src0);
                GI_INT16_t v_src01_16 = GiCvtUint8toInt16Low(v_src1);
                GI_INT16_t v_src02_16 = GiCvtUint8toInt16Low(v_src2);

                GI_INT16_t v_src10_16 = GiCvtUint8toInt16High(v_src0);
                GI_INT16_t v_src11_16 = GiCvtUint8toInt16High(v_src1);
                GI_INT16_t v_src12_16 = GiCvtUint8toInt16High(v_src2);
                //! v_src0x_16 low part 
                GI_INT32_t v_Y0 = GiMultiplyAddInt16LongLow(
                    GiMultiplyAddInt16LongLow(
                    GiMultiplyAddInt16LongLow(v_zero, v_src00_16, v_c0),
                    v_src01_16, v_c1),
                    v_src02_16, v_c2);

                v_Y0 = GiShiftRightInt32(GiAddInt32(v_Y0, v_delta2), yuv_shift);

                GI_INT32_t v_Cr0 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveLowLongInt16(v_src00_16), v_Y0), v_c3);
                v_Cr0 = GiShiftRightInt32(GiAddInt32(v_Cr0, v_delta2), yuv_shift);

                GI_INT32_t v_Cb0 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveLowLongInt16(v_src02_16), v_Y0), v_c4);
                v_Cb0 = GiShiftRightInt32(GiAddInt32(v_Cb0, v_delta2), yuv_shift);
                //! v_src0x_16 high part
                GI_INT32_t v_Y1 = GiMultiplyAddInt16LongHigh(
                    GiMultiplyAddInt16LongHigh(
                    GiMultiplyAddInt16LongHigh(v_zero, v_src00_16, v_c0),
                    v_src01_16, v_c1),
                    v_src02_16, v_c2);

                v_Y1 = GiShiftRightInt32(GiAddInt32(v_Y1, v_delta2), yuv_shift);

                GI_INT32_t v_Cr1 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveHighLongInt16(v_src00_16), v_Y1), v_c3);
                v_Cr1 = GiShiftRightInt32(GiAddInt32(v_Cr1, v_delta2), yuv_shift);

                GI_INT32_t v_Cb1 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveHighLongInt16(v_src02_16), v_Y1), v_c4);
                v_Cb1 = GiShiftRightInt32(GiAddInt32(v_Cb1, v_delta2), yuv_shift);

                //! v_src1x_16 low part 
                GI_INT32_t v_Y2 = GiMultiplyAddInt16LongLow(
                    GiMultiplyAddInt16LongLow(
                    GiMultiplyAddInt16LongLow(v_zero, v_src10_16, v_c0),
                    v_src11_16, v_c1),
                    v_src12_16, v_c2);

                v_Y2 = GiShiftRightInt32(GiAddInt32(v_Y2, v_delta2), yuv_shift);

                GI_INT32_t v_Cr2 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveLowLongInt16(v_src10_16), v_Y2), v_c3);
                v_Cr2 = GiShiftRightInt32(GiAddInt32(v_Cr2, v_delta2), yuv_shift);

                GI_INT32_t v_Cb2 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveLowLongInt16(v_src12_16), v_Y2), v_c4);
                v_Cb2 = GiShiftRightInt32(GiAddInt32(v_Cb2, v_delta2), yuv_shift);
                //! v_src1x_16 high part
                GI_INT32_t v_Y3 = GiMultiplyAddInt16LongHigh(
                    GiMultiplyAddInt16LongHigh(
                    GiMultiplyAddInt16LongHigh(v_zero, v_src10_16, v_c0),
                    v_src11_16, v_c1),
                    v_src12_16, v_c2);

                v_Y3 = GiShiftRightInt32(GiAddInt32(v_Y3, v_delta2), yuv_shift);

                GI_INT32_t v_Cr3 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveHighLongInt16(v_src10_16), v_Y3), v_c3);
                v_Cr3 = GiShiftRightInt32(GiAddInt32(v_Cr3, v_delta2), yuv_shift);

                GI_INT32_t v_Cb3 = GiMultiplyAddInt32(
                        v_delta, GiSubtractInt32(GiMoveHighLongInt16(v_src12_16), v_Y3), v_c4);
                v_Cb3 = GiShiftRightInt32(GiAddInt32(v_Cb3, v_delta2), yuv_shift);

                GI_UINT8_t v_dst0 = GiCvtFromInt32V4ToUint8(v_Y0, v_Y1, v_Y2, v_Y3);
                GI_UINT8_t v_dst1 = GiCvtFromInt32V4ToUint8(v_Cr0, v_Cr1, v_Cr2, v_Cr3);
                GI_UINT8_t v_dst2 = GiCvtFromInt32V4ToUint8(v_Cb0, v_Cb1, v_Cb2, v_Cb3);

                GiStoreZipUint8V3(pdst, v_dst0, v_dst1, v_dst2);
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
            GI_UINT8_t v_src0, v_src1, v_src2, v_dst0, v_dst1, v_dst2; 
            v_src0 = GiLoadUzip0V3Uint8(psrc);
            v_src1 = GiLoadUzip1V3Uint8(psrc);
            v_src2 = GiLoadUzip2V3Uint8(psrc); 
            v_dst0 = v_src2;
            v_dst1 = v_src1;
            v_dst2 = v_src0;
            GiStoreZipUint8V3(pdst, v_dst0, v_dst1, v_dst2);
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
    GI_UINT8_t v_y;
    GI_INT32_t v_y_s32_0, v_y_s32_1, v_y_s32_2, v_y_s32_3;
    GI_UINT8_t v_vu;
    GI_INT32_t v_RV0, v_RV1, v_RV2, v_RV3;
    GI_INT32_t v_GVU0, v_GVU1, v_GVU2, v_GVU3;
    GI_INT32_t v_BU0, v_BU1, v_BU2, v_BU3;

    GI_INT32_t v_R0, v_R1, v_R2, v_R3;
    GI_INT32_t v_G0, v_G1, v_G2, v_G3;
    GI_INT32_t v_B0, v_B1, v_B2, v_B3;
    GI_UINT8_t v_RGB_r, v_RGB_g, v_RGB_b, v_BGR_b, v_BGR_g, v_BGR_r;

    GI_INT16_t v_128 = GiBroadcastInt16(128);
    GI_INT32_t v_zero = GiBroadcastInt32(0); 

    GI_INT16_t v_359, v_88, v_183, v_454;
    v_359 = GiBroadcastInt16(359);
    v_88 = GiBroadcastInt16(88);
    v_183 = GiBroadcastInt16(183);
    v_454 = GiBroadcastInt16(454);
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
            GI_INT16_t v_vu_s160, v_vu_s161;

            v_vu = GiLoadUint8(pV + c);
            v_vu = GiInterleave2Uint8(v_vu);
            v_vu_s160 = GiCvtUint8toInt16Low(v_vu);
            v_vu_s161 = GiCvtUint8toInt16High(v_vu);
                
            v_vu_s160 = GiSubtractInt16(v_vu_s160, v_128);
            v_vu_s161 = GiSubtractInt16(v_vu_s161, v_128);

            v_RV1 = GiShiftRightInt32(GiMultiplyAddInt16LongLow(v_zero, v_vu_s160, v_359), 8);
            v_RV3 = GiShiftRightInt32(GiMultiplyAddInt16LongHigh(v_zero, v_vu_s160, v_359), 8);
            v_GVU1 = GiShiftRightInt32(
                    GiAddInt32(GiMultiplyAddInt16LongLow(v_zero, v_vu_s161, v_88), GiMultiplyAddInt16LongLow(v_zero, v_vu_s160, v_183)),
                    8);
            v_GVU3 = GiShiftRightInt32(
                    GiAddInt32(GiMultiplyAddInt16LongHigh(v_zero, v_vu_s161, v_88), GiMultiplyAddInt16LongHigh(v_zero, v_vu_s160, v_183)),
                    8);
            v_BU1 = GiShiftRightInt32(GiMultiplyAddInt16LongLow(v_zero, v_vu_s161, v_454), 8);
            v_BU3 = GiShiftRightInt32(GiMultiplyAddInt16LongHigh(v_zero, v_vu_s161, v_454), 8);

            v_RV0 = GiZipV0Int32(v_RV1, v_RV1);
            v_RV1 = GiZipV1Int32(v_RV1, v_RV1);
            v_RV2 = GiZipV0Int32(v_RV3, v_RV3);
            v_RV3 = GiZipV1Int32(v_RV3, v_RV3);

            v_GVU0 = GiZipV0Int32(v_GVU1, v_GVU1);
            v_GVU1 = GiZipV1Int32(v_GVU1, v_GVU1);
            v_GVU2 = GiZipV0Int32(v_GVU3, v_GVU3);
            v_GVU3 = GiZipV1Int32(v_GVU3, v_GVU3);
            v_BU0 = GiZipV0Int32(v_BU1, v_BU1);
            v_BU1 = GiZipV1Int32(v_BU1, v_BU1);
            v_BU2 = GiZipV0Int32(v_BU3, v_BU3);
            v_BU3 = GiZipV1Int32(v_BU3, v_BU3);

            v_y = GiLoadUint8(pY + c);
            GI_INT16_t v_y_2quarter = GiCvtUint8toInt16Low(v_y);
            v_y_s32_0 = GiMoveLowLongInt16(v_y_2quarter);
            v_y_s32_1 = GiMoveHighLongInt16(v_y_2quarter);

            v_y_2quarter = GiCvtUint8toInt16High(v_y);
            v_y_s32_2 = GiMoveLowLongInt16(v_y_2quarter);
            v_y_s32_3 = GiMoveHighLongInt16(v_y_2quarter);

            v_R0 = GiAddInt32(v_y_s32_0, v_RV0);
            v_R1 = GiAddInt32(v_y_s32_1, v_RV1);
            v_R2 = GiAddInt32(v_y_s32_2, v_RV2);
            v_R3 = GiAddInt32(v_y_s32_3, v_RV3);

            v_G0 = GiSubtractInt32(v_y_s32_0, v_GVU0);
            v_G1 = GiSubtractInt32(v_y_s32_1, v_GVU1);
            v_G2 = GiSubtractInt32(v_y_s32_2, v_GVU2);
            v_G3 = GiSubtractInt32(v_y_s32_3, v_GVU3);

            v_B0 = GiAddInt32(v_y_s32_0, v_BU0);
            v_B1 = GiAddInt32(v_y_s32_1, v_BU1);
            v_B2 = GiAddInt32(v_y_s32_2, v_BU2);
            v_B3 = GiAddInt32(v_y_s32_3, v_BU3);

   
            v_BGR_b = GiCvtFromInt32V4ToUint8(v_B0, v_B1, v_B2, v_B3);
            v_BGR_g = GiCvtFromInt32V4ToUint8(v_G0, v_G1, v_G2, v_G3);
            v_BGR_r = GiCvtFromInt32V4ToUint8(v_R0, v_R1, v_R2, v_R3);
            GiStoreZipUint8V3((dst0 + c * 3), v_BGR_b, v_BGR_g, v_BGR_r);
            

            v_y = GiLoadUint8(pY + src_step + c);
            v_y_2quarter = GiCvtUint8toInt16Low(v_y);
            v_y_s32_0 = GiMoveLowLongInt16(v_y_2quarter);
            v_y_s32_1 = GiMoveHighLongInt16(v_y_2quarter);

            v_y_2quarter = GiCvtUint8toInt16High(v_y);
            v_y_s32_2 = GiMoveLowLongInt16(v_y_2quarter);
            v_y_s32_3 = GiMoveHighLongInt16(v_y_2quarter);

            v_R0 = GiAddInt32(v_y_s32_0, v_RV0);
            v_R1 = GiAddInt32(v_y_s32_1, v_RV1);
            v_R2 = GiAddInt32(v_y_s32_2, v_RV2);
            v_R3 = GiAddInt32(v_y_s32_3, v_RV3);

            v_G0 = GiSubtractInt32(v_y_s32_0, v_GVU0);
            v_G1 = GiSubtractInt32(v_y_s32_1, v_GVU1);
            v_G2 = GiSubtractInt32(v_y_s32_2, v_GVU2);
            v_G3 = GiSubtractInt32(v_y_s32_3, v_GVU3);

            v_B0 = GiAddInt32(v_y_s32_0, v_BU0);
            v_B1 = GiAddInt32(v_y_s32_1, v_BU1);
            v_B2 = GiAddInt32(v_y_s32_2, v_BU2);
            v_B3 = GiAddInt32(v_y_s32_3, v_BU3);
            //! WARNING: GiCvtFromInt32V4ToUint8 use v_qmovn_s32 which is slower than v_movn_s32 but get the right value anyway
            //! if the value of int32 is certainly in the range of uint8, v_movn_s32 is a better choice 
            v_BGR_b = GiCvtFromInt32V4ToUint8(v_B0, v_B1, v_B2, v_B3);
            v_BGR_g = GiCvtFromInt32V4ToUint8(v_G0, v_G1, v_G2, v_G3);
            v_BGR_r = GiCvtFromInt32V4ToUint8(v_R0, v_R1, v_R2, v_R3);
            GiStoreZipUint8V3((dst1 + c * 3), v_BGR_b, v_BGR_g, v_BGR_r);
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
    std::stringstream writer;
    writer << R"(
        #include "gi_int.h"
        #include <limits.h>
        #include <string.h>
        #include "tinycv_c.h"
    )";
    
    std::string body_temp = R"(
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

    writer << StringTemplate::StringTemplateArgs()
            .add("aux_func", code.aux_func)
            .add("cvt_code", code.cvt_code)
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
    return writer.str();
}

// vim: syntax=cpp.doxygen
