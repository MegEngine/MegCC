
/**
 * \file
 * compiler/lib/KernelGen/BareMetal/CvtColor.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>
#include <unordered_map>

#include "CvtColor.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;
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
        const int yuv_shift = 14;
        const int coef[] = {1868, 9617, 4899, 8061, 14369};
        const int delta = 128 << yuv_shift;
        for(size_t hw_idx = 0; hw_idx < hw; ++hw_idx){
            const uint8_t* v = sptr + hw_idx * ch;
            int Y = descale(v[0] * coef[0] + v[1] * coef[1] + v[2] * coef[2],
                            yuv_shift);
            int Cr = descale((v[0] - Y) * coef[3] + delta, yuv_shift);
            int Cb = descale((v[2] - Y) * coef[4] + delta, yuv_shift);
            uint8_t* target = dptr + hw_idx * ch;
            target[0] = saturate_cast_ui8(Y);
            target[1] = saturate_cast_ui8(Cr);
            target[2] = saturate_cast_ui8(Cb);
        }
    )";
    return res;
}

CvtColorGen gen_rgb_gray(TContext* ctx) {
    CvtColorGen res;

    res.cvt_code = R"(
        const int yuv_shift = 14, R2Y = 4899, G2Y = 9617, B2Y = 1868;
        for(size_t hw_idx = 0; hw_idx < hw; ++hw_idx){
            uint8_t x0 = sptr[0];
            uint8_t x1 = sptr[1];
            uint8_t x2 = sptr[2];
            dptr[0] =
                    (x0 * R2Y + x1 * G2Y + x2 * B2Y + (1 << (yuv_shift - 1))) >>
                    yuv_shift;
            sptr += 3;
            dptr += 1;
        }
    )";
    return res;
}

CvtColorGen gen_rgb_bgr(TContext* ctx) {
    CvtColorGen res;
    res.aux_func = R"(
    )";
    res.cvt_code = R"(
        for(size_t hw_idx = 0; hw_idx < hw; ++hw_idx){
            const uint8_t* rgb = sptr + hw_idx * ch;
            
            uint8_t* bgr = dptr + hw_idx * ch;
            bgr[0] = rgb[2];
            bgr[1] = rgb[1];
            bgr[2] = rgb[0];
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
        const uint8_t* pY;
        const uint8_t* pU;
        const uint8_t* pV;
        int Y00, Y01, U, V;
        int Y10, Y11;
        int i, j;
        int ruv, guv, buv;
        int R, G, B;

        bool is_uv = false;
        bool is_planar = false;
        int width = dst->cols;
        int height = dst->rows;
        int src_step = cols * ch;
        int dst_step = width * dst->channels;
        pY = sptr;
        if (is_uv) {
            pU = pY + height * src_step;
            pV = pY + (height + height / 4) * src_step;
        } else {
            pV = pY + height * src_step;
            pU = pY + (height + height / 4) * src_step;
        }
        for (i = 0; i < height; i += 2, pY += src_step * 2) {
            size_t index = 0;
            size_t index1 = 0;
            uint8_t* out = dptr + i * dst_step;
            uint8_t* out1 = dptr + (i + 1) * dst_step;
            int jV = 0;
            for (j = 0; j < width; j += 2) {
                Y00 = *((pY) + j);
                Y01 = *((pY) + j + 1);
                Y10 = *((pY) + src_step + j);
                Y11 = *((pY) + src_step + j + 1);
                if (is_planar) {
                    V = *(pV + jV);
                    U = *(pU + jV);
                    jV++;
                } else {
                    if (is_uv) {
                        U = *(pU + j);
                        V = *(pU + j + 1);
                    } else {
                        V = *(pV + j);
                        U = *(pV + j + 1);
                    }
                }

                ruv = ((359 * (V - 128)) >> 8);
                guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
                buv = ((454 * (U - 128)) >> 8);

                R = Y00 + ruv;
                G = Y00 + guv;
                B = Y00 + buv;
                R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
                G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
                B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

                SET_COLOR(out, index)

                R = Y01 + ruv;
                G = Y01 + guv;
                B = Y01 + buv;
                R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
                G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
                B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

                SET_COLOR(out, index)

                ruv = ((359 * (V - 128)) >> 8);
                guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
                buv = ((454 * (U - 128)) >> 8);
                R = Y10 + ruv;
                G = Y10 + guv;
                B = Y10 + buv;
                R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
                G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
                B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

                SET_COLOR(out1, index1)

                R = Y11 + ruv;
                G = Y11 + guv;
                B = Y11 + buv;
                R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
                G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
                B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

                SET_COLOR(out1, index1)
            }
            if (is_planar) {
                pV += src_step / 2;
                pU += src_step / 2;
            } else {
                if (is_uv) {
                    pU += src_step;
                } else {
                    pV += src_step;
                }
            }
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
        reg_map = {
                {"RGB2YUV", {gen_rgb_yuv, "rgb2yuv"}},
                {"YUV2BGR_NV21", {gen_yuv_bgr_nv21, "yuv2bgr_nv21"}},
                {"RGB2BGR", {gen_rgb_bgr, "rgb2bgr"}},
                {"RGB2GRAY", {gen_rgb_gray, "rgb2gray"}}

        };
    }

    bool usable(const std::string& mode) { return reg_map.find(mode) != reg_map.end(); }
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
    return GetCVKernelSymbol(context) + "(const TinyMat* src, const TinyMat* dst)";
}

std::string CvtColorKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    auto mode = context->getAttrStr("mode");
    auto gen_func = g_cvt_reg.get_func(mode);
    auto code = gen_func(context);
    std::string body_temp = R"(
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
