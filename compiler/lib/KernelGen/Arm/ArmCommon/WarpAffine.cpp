/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/WarpAffine.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "WarpAffine.h"
#include <cmath>
#include "../../Utils/StringTemplate.h"
#include "../../Utils/Utils.h"
#include "Common/CVRemapTable.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool WarpAffineKernel::IsAvailable(TContext* ctx) const {
    auto nr_operands = ctx->getAttrInt("nr_operands");
    auto src_layout = ctx->getAttrOprand("operand:0");
    auto mat_layout = ctx->getAttrOprand("operand:1");
    auto dst_layout = ctx->getAttrOprand("operand:2");
    auto format = ctx->getAttrStr("format");
    auto bmode = ctx->getAttrStr("border_mode");
    bool is_nhwc = format == "NHWC";
    bool is_u8 = src_layout.dtype == "ui8";
    if (is_u8 && is_nhwc && bmode == "CONSTANT") {
        float border_val = ctx->getAttrFloat("border_val");
        CC_ASSERT(std::abs(std::round(border_val) - border_val) < 1e-5)
                << "u8 const only support int as border_val not " << border_val << "\n";
    }
    CC_ASSERT(nr_operands == 3);

    bool dtype_valid = ((src_layout.dtype == "f32" && dst_layout.dtype == "f32") ||
                        (src_layout.dtype == "ui8" && dst_layout.dtype == "ui8")) &&
                       mat_layout.dtype == "f32";
    return dtype_valid;
}
//! kernel gen
std::string WarpAffineKernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    auto border_val_str = std::to_string(ctx->getAttrFloat("border_val"));
    border_val_str[border_val_str.find('.')] = '_';
    ss << "ArmCommon_kernel_warpaffine";
    ss << "_" << ctx->getAttrStr("format");
    ss << "_" << ctx->getAttrStr("imode");
    ss << "_" << ctx->getAttrStr("border_mode");
    ss << "_" << ctx->getAttrOprand("operand:0").dtype;
    ss << "_" << border_val_str;
    return ss.str();
}

namespace {

std::string gen_get_real_coord(const std::string& bmode) {
    std::string body_temp = R"(
            static inline int get_real_coord(int p, int len){
                if ((unsigned)p >= (unsigned)len){
                    ${core_temp}
                }
                return p;
            }
        )";
    std::string core_temp;
    if (bmode == "REFLECT") {
        core_temp = R"(
            if (len == 1)
                return 0;
            do {
                if (p < 0)
                    p = -p - 1;
                else
                    p = len - 1 - (p - len);
            } while ((unsigned)p >= (unsigned)len);
        )";
    } else if (bmode == "REFLECT_101") {
        core_temp = R"(
            if (len == 1)
                return 0;
            do {
                if (p < 0)
                    p = -p - 1 + 1;
                else
                    p = len - 1 - (p - len) - 1;
            } while ((unsigned)p >= (unsigned)len);
        )";
    } else if (bmode == "REPLICATE") {
        core_temp = R"(
            p = p < 0 ? 0 : len - 1;
        )";
    } else if (bmode == "CONSTANT") {
        core_temp = R"(
            p = -1;
        )";
    } else if (bmode == "WRAP") {
        core_temp = R"(
            if (p < 0)
                p -= ((p - len + 1) / len) * len;
            
            while (p >= len) {
                p -= len;
            }
        )";
    } else {
        CC_ABORT << "no support bmode " << bmode << "\n";
    }
    return StringTemplate::StringTemplateArgs()
            .add("core_temp", core_temp)
            .render(body_temp);
}

std::string gen_visit(
        const std::string& bmode, float border_val, const std::string& dtype_c_str,
        const std::string& dtype) {
    std::stringstream temp_body;
    if (bmode != "CONSTANT") {
        temp_body << R"(
            static inline float visit_src(const ${dtype_c_str}* sptr,int c, int h, int w, size_t sstrd[3]){
                return sptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w];
            }
        )";
    } else {
        temp_body << R"(
            static inline float visit_src(const ${dtype_c_str}* sptr,int c, int h, int w, size_t sstrd[3]){
                if (h != -1 && w != -1){
                    return sptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w];
                }else{
                    return ${border_val};
                }
            }
        )";
    }
    if (dtype == "ui8") {
        temp_body << R"(
            static inline uint8_t saturate_cast_ui8(int x){
                return (uint8_t)((unsigned)x <= UCHAR_MAX ? x
                                                        : x > 0 ? UCHAR_MAX : 0);
            }
            static inline uint8_t output_cvt(float val){
                int ival = (int)roundf(val);
                return saturate_cast_ui8(val);
            }
        )";
    } else {
        CC_ASSERT(dtype == "f32");
        temp_body << R"(
            static inline ${dtype_c_str} output_cvt(float val){
                return val;
            }
        )";
    }
    temp_body << R"(
        static inline void visit_dst(${dtype_c_str}* dptr,int c, int h, int w, size_t sstrd[3], float val){
            dptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w] = output_cvt(val);
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("border_val", std::to_string(border_val))
            .add("dtype_c_str", dtype_c_str)
            .render(temp_body.str());
}

std::string gen_warp_nhwc_u8(std::string bmode) {
    std::string const_boarder_str = R"(
        int sx0 = get_real_coord(sx, swidth);
        int sx1 = get_real_coord(sx + 1, swidth);
        int sy0 = get_real_coord(sy, sheight);
        int sy1 = get_real_coord(sy + 1, sheight);
        const uint8_t* v0 = S0 + sy0 * sstep + sx0 * CH;
        const uint8_t* v1 = S0 + sy0 * sstep + sx1 * CH;
        const uint8_t* v2 = S0 + sy1 * sstep + sx0 * CH;
        const uint8_t* v3 = S0 + sy1 * sstep + sx1 * CH;
    )";
    std::string assign_str = R"(
        D[ch] = int2u8_with_shift((int)(v0[ch] * w[0] + v1[ch] * w[1] +
                                        v2[ch] * w[2] + v3[ch] * w[3]));
    )";
    if (bmode == "CONSTANT") {
        const_boarder_str = R"(
            const uint8_t* v0; 
            const uint8_t* v1; 
            const uint8_t* v2; 
            const uint8_t* v3;
            int sx0 = get_real_coord(sx, swidth);
            int sx1 = get_real_coord(sx + 1, swidth);
            int sy0 = get_real_coord(sy, sheight);
            int sy1 = get_real_coord(sy + 1, sheight);
            v0 = (sy0 != -1 && sx0 != -1) ? S0 + sy0 * sstep + sx0 * CH : bval;
            v1 = (sy0 != -1 && sx1 != -1) ? S0 + sy0 * sstep + sx1 * CH : bval;
            v2 = (sy1 != -1 && sx0 != -1) ? S0 + sy1 * sstep + sx0 * CH : bval;
            v3 = (sy1 != -1 && sx1 != -1) ? S0 + sy1 * sstep + sx1 * CH : bval;
        )";
    }

    std::string body_temp = R"(

static inline int min(int x, int y) {
    return x < y ? x : y;
}
static inline int max(int x, int y) {
    return x > y ? x : y;
}
static inline int div_ceil(int x, int r) {
    return (x + r - 1) / r;
}

#define SATURATE_CAST_SHORT(X) (short)min(max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X) \
    (int)min(max((int)((X) + ((X) >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

#define INTER_BITS 5
#define AB_BITS 10
static const int AB_SCALE = 1 << AB_BITS;
#define INTER_TAB_SIZE  (1 << INTER_BITS)
#define INTER_TAB_SIZE2  (INTER_TAB_SIZE * INTER_TAB_SIZE)
static const int BLOCK_SZ = 64;
static const int INTER_REMAP_COEF_BITS = 15;
static const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;
static short global_table[INTER_TAB_SIZE2 * 2 * 2];
static bool init_table = false;

static inline uint8_t int2u8_with_shift(int val) {
    //! opt it
    const int shift = INTER_REMAP_COEF_BITS;
    const int delta = 1 << (INTER_REMAP_COEF_BITS - 1);
    return (uint8_t)((val + delta) >> shift);
}

static inline int saturate_bound(int x, int lower, int upper) {
    return (x < lower ? lower : (x >= upper ? upper - 1 : x));
}

static inline void remapBilinear_c1(const uint8_t* src, uint8_t* dst, int swidth,
                                 int sheight, int sstep, int dwidth,
                                 int dheight, int dstride, const short* xy,
                                 int xy_stride, const unsigned short* fxy,
                                 int fxy_stride, const short* wtab, const uint8_t* bval) {
    const uint8_t* S0 = src;
    unsigned int width1 = max(swidth - 1, 0);
    unsigned int height1 = max(sheight - 1, 0);
    int CH = 1;
    for (int dy = 0; dy < dheight; dy++) {
        uint8_t* D = dst + dy * dstride;
        const short* XY = xy + dy * xy_stride;
        const unsigned short* FXY = fxy + dy * fxy_stride;
        int X0 = 0;
        bool prevInlier = false;

        for (int dx = 0; dx <= dwidth; dx++) {
            bool curInlier =
                    dx < dwidth ? (unsigned)XY[dx * 2] < width1 &&
                                          (unsigned)XY[dx * 2 + 1] < height1
                                : !prevInlier;
            if (curInlier == prevInlier)
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if (!curInlier) {
                int len = 0;
                D += len * CH;
                dx += len;
                for (; dx < X1; dx++, D++) {
                    int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                    const short* w = wtab + FXY[dx] * 4;
                    const uint8_t* S = S0 + sy * sstep + sx;
                    *D = int2u8_with_shift((int)(S[0] * w[0] + S[1] * w[1] +
                                                 S[sstep] * w[2] +
                                                 S[sstep + 1] * w[3]));
                }
            } else {
                for (; dx < X1; dx++, D++) {
                    int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                    const short* w = wtab + FXY[dx] * 4;
                    const int ch = 0;
                    ${const_boarder_str}
                    ${assign_str}
                }
            }
        }
    }
}

static inline void remapBilinear_c3(const uint8_t* src, uint8_t* dst, int swidth,
                                 int sheight, int sstep, int dwidth,
                                 int dheight, int dstride, const short* xy,
                                 int xy_stride, const unsigned short* fxy,
                                 int fxy_stride, const short* wtab, const uint8_t* bval) {
    const uint8_t* S0 = src;
    unsigned int width1 = max(swidth - 1, 0);
    unsigned int height1 = max(sheight - 1, 0);
    const int CH = 3;
    for (int dy = 0; dy < dheight; dy++) {
        uint8_t* D = dst + dy * dstride;
        const short* XY = xy + dy * xy_stride;
        const unsigned short* FXY = fxy + dy * fxy_stride;
        int X0 = 0;
        bool prevInlier = false;

        for (int dx = 0; dx <= dwidth; dx++) {
            bool curInlier =
                    dx < dwidth ? (unsigned)XY[dx * 2] < width1 &&
                                          (unsigned)XY[dx * 2 + 1] < height1
                                : !prevInlier;
            if (curInlier == prevInlier)
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if (!curInlier) {
                int len = 0;
                D += len * CH;
                dx += len;
                for (; dx < X1; dx++, D += 3) {
                    int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                    const short* w = wtab + FXY[dx] * 4;
                    const uint8_t* S = S0 + sy * sstep + sx * 3;
                    int t0 = S[0] * w[0] + S[3] * w[1] + S[sstep] * w[2] +
                            S[sstep + 3] * w[3];
                    int t1 = S[1] * w[0] + S[4] * w[1] +
                            S[sstep + 1] * w[2] + S[sstep + 4] * w[3];
                    int t2 = S[2] * w[0] + S[5] * w[1] +
                            S[sstep + 2] * w[2] + S[sstep + 5] * w[3];
                    D[0] = int2u8_with_shift(t0);
                    D[1] = int2u8_with_shift(t1);
                    D[2] = int2u8_with_shift(t2);
                }
            } else {
                for (; dx < X1; dx++, D+=CH) {
                    int sx = XY[dx * 2];
                    int sy = XY[dx * 2 + 1];
                    const short* w = wtab + FXY[dx] * 4;
                    ${const_boarder_str}
                    for(int ch = 0; ch < CH; ++ch){
                        ${assign_str}
                    }
                }
            }
        }
    }
}


static inline void remapBilinear_cx(const uint8_t* src, uint8_t* dst, int swidth,
                                 int sheight, int sstep, int dwidth,
                                 int dheight, int dstride, const short* xy,
                                 int xy_stride, const unsigned short* fxy,
                                 int fxy_stride, const short* wtab, int CH, const uint8_t* bval) {
    const uint8_t* S0 = src;
    unsigned int width1 = max(swidth - 1, 0);
    unsigned int height1 = max(sheight - 1, 0);
    for (int dy = 0; dy < dheight; dy++) {
        uint8_t* D = dst + dy * dstride;
        const short* XY = xy + dy * xy_stride;
        const unsigned short* FXY = fxy + dy * fxy_stride;
        int X0 = 0;
        bool prevInlier = false;

        for (int dx = 0; dx <= dwidth; dx++) {
            bool curInlier =
                    dx < dwidth ? (unsigned)XY[dx * 2] < width1 &&
                                          (unsigned)XY[dx * 2 + 1] < height1
                                : !prevInlier;
            if (curInlier == prevInlier)
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if (!curInlier) {
                int len = 0;
                D += len * CH;
                dx += len;
                for (; dx < X1; dx++) {
                    int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                    const short* w = wtab + FXY[dx] * 4;
                    const uint8_t* S = S0 + sy * sstep + sx * CH;
                    for(int ch = 0; ch < CH; ++ch) {
                        *D = int2u8_with_shift((int)(S[0 + ch] * w[0] + S[1 * CH + ch] * w[1] +
                                                    S[sstep + ch] * w[2] +
                                                    S[sstep + 1 * CH + ch] * w[3]));
                        ++D;
                    }
                }
            } else {
                for (; dx < X1; dx++) {
                    int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                    const short* w = wtab + FXY[dx] * 4;
                    ${const_boarder_str}
                    for(int ch = 0; ch < CH; ++ch){
                        ${assign_str}
                    }
                    D += CH;
                }
            }
        }
    }
}


static inline void remap(const uint8_t* src_ptr, uint8_t* dst_ptr, int ih,
                         int iw, int oh, int ow, int bh, int bw, int ch,
                         const short* xy, unsigned short* mat_a, const short* wtab, const uint8_t* bval) {
    //! xy = [bh, bw, 2]
    //! mat_a = [bh, bw, 1]
    int x = 0;
    int y = 0;
    const int buf_size = 1 << 14;
    int brows0 = bh;
    int bcols0 = bw;
    unsigned short bufa[brows0 * bcols0];
    for (int y1 = 0; y1 < brows0; ++y1) {
        unsigned short* A = &bufa[y1 * bcols0];
        unsigned short* sA = mat_a + (y + y1) * bw + x;
        uint16x8_t v_scale = vdupq_n_u16(INTER_TAB_SIZE2 - 1);
        int x1 = 0;
        for (; x1 <= bcols0 - 8; x1 += 8) {
            vst1q_u16(A + x1, vandq_u16(vld1q_u16(sA + x1), v_scale));
        }
        for (; x1 < bcols0; ++x1)
            A[x1] = (unsigned short)(sA[x1] & (INTER_TAB_SIZE2 - 1));
    }
    //! ifunc
    if(ch == 1){
        remapBilinear_c1(src_ptr, dst_ptr, iw, ih, iw * ch, bw, bh, ow * ch,
                    xy + y * bw * 2 + x * 2, bw * 2, bufa, bw, wtab, bval);
    }else if(ch==3){
        remapBilinear_c3(src_ptr, dst_ptr, iw, ih, iw * ch, bw, bh, ow * ch,
                    xy + y * bw * 2 + x * 2, bw * 2, bufa, bw, wtab, bval);
    }else{
        remapBilinear_cx(src_ptr, dst_ptr, iw, ih, iw * ch, bw, bh, ow * ch,
                    xy + y * bw * 2 + x * 2, bw * 2, bufa, bw, wtab, ch, bval);
    }
}

static inline void warpaffine_img(const uint8_t* src_ptr, int ic, int ih, int iw, uint8_t* dst_ptr, int oh ,int ow, const double* M, const short* ctab, const uint8_t* board_val){
    int _adelta[ow * 2];
    int* adelta = &_adelta[0];
    int* bdelta = &_adelta[ow];
    int BLOCK_SZ_H = min(BLOCK_SZ / 2, oh);
    int BLOCK_SZ_W = min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_H, ow);
    BLOCK_SZ_H = min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_W, oh);
    int all_task = div_ceil(oh, BLOCK_SZ_H) * div_ceil(ow, BLOCK_SZ_W);
    int width_block_size = div_ceil(ow, BLOCK_SZ_W);
    short XY[BLOCK_SZ * BLOCK_SZ * 2];
    short A[BLOCK_SZ * BLOCK_SZ];
    int round_delta = AB_SCALE / INTER_TAB_SIZE / 2;

    for (int x = 0; x < ow; ++x) {
        adelta[x] = SATURATE_CAST_INT(M[0] * x * AB_SCALE);
        bdelta[x] = SATURATE_CAST_INT(M[3] * x * AB_SCALE);
    }
    
    for (int task_id = 0; task_id < all_task; ++task_id) {
        int y = (task_id / width_block_size) * BLOCK_SZ_H;
        int x = (task_id % width_block_size) * BLOCK_SZ_W;
        int bw = min(BLOCK_SZ_W, ow - x);
        int bh = min(BLOCK_SZ_H, oh - y);
        for (int y1 = 0; y1 < bh; ++y1) {
            short* xy = XY + y1 * bw * 2;
            int X0 = SATURATE_CAST_INT((M[1] * (y + y1) + M[2]) * AB_SCALE) +
                    round_delta;
            int Y0 = SATURATE_CAST_INT((M[4] * (y + y1) + M[5]) * AB_SCALE) +
                    round_delta;

            // if imode is not INTER_NEAREST
            short* alpha = A + y1 * bw;
            int32x4_t v__X0 = vdupq_n_s32(X0), v__Y0 = vdupq_n_s32(Y0),
                    v_mask = vdupq_n_s32(INTER_TAB_SIZE - 1);
            int x1 = 0;
            for (; x1 + 8 <= bw; x1 += 8) {
                int32x4_t v_X0 = vshrq_n_s32(
                        vaddq_s32(v__X0, vld1q_s32(adelta + x + x1)),
                        AB_BITS - INTER_BITS);
                int32x4_t v_Y0 = vshrq_n_s32(
                        vaddq_s32(v__Y0, vld1q_s32(bdelta + x + x1)),
                        AB_BITS - INTER_BITS);
                int32x4_t v_X1 = vshrq_n_s32(
                        vaddq_s32(v__X0, vld1q_s32(adelta + x + x1 + 4)),
                        AB_BITS - INTER_BITS);
                int32x4_t v_Y1 = vshrq_n_s32(
                        vaddq_s32(v__Y0, vld1q_s32(bdelta + x + x1 + 4)),
                        AB_BITS - INTER_BITS);

                int16x8x2_t v_xy;
                v_xy.val[0] =
                        vcombine_s16(vqmovn_s32(vshrq_n_s32(v_X0, INTER_BITS)),
                                    vqmovn_s32(vshrq_n_s32(v_X1, INTER_BITS)));
                v_xy.val[1] =
                        vcombine_s16(vqmovn_s32(vshrq_n_s32(v_Y0, INTER_BITS)),
                                    vqmovn_s32(vshrq_n_s32(v_Y1, INTER_BITS)));

                vst2q_s16(xy + (x1 << 1), v_xy);

                int16x4_t v_alpha0 = vmovn_s32(vaddq_s32(
                        vshlq_n_s32(vandq_s32(v_Y0, v_mask), INTER_BITS),
                        vandq_s32(v_X0, v_mask)));
                int16x4_t v_alpha1 = vmovn_s32(vaddq_s32(
                        vshlq_n_s32(vandq_s32(v_Y1, v_mask), INTER_BITS),
                        vandq_s32(v_X1, v_mask)));
                vst1q_s16(alpha + x1, vcombine_s16(v_alpha0, v_alpha1));
            }

            for (; x1 < bw; x1++) {
                int X = (X0 + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                int Y = (Y0 + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                xy[x1 * 2] = SATURATE_CAST_SHORT(X >> INTER_BITS);
                xy[x1 * 2 + 1] = SATURATE_CAST_SHORT(Y >> INTER_BITS);
                alpha[x1] =
                        (short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                (X & (INTER_TAB_SIZE - 1)));
            }
        }
        //! remap                
        remap(src_ptr, dst_ptr + y * ow * ic + x * ic, ih, iw, oh, ow, bh, bw,
            ic, &XY[0], (unsigned short*)&A[0], ctab, board_val);
        
    }
}
    )";
    return StringTemplate::StringTemplateArgs()
            .add("const_boarder_str", const_boarder_str)
            .add("assign_str", assign_str)
            .render(body_temp);
}

std::string to_lower_case(std::string data) {
    for (auto& c : data) {
        c = tolower(c);
    }
    return data;
}
}  // namespace
std::vector<KernelObj> WarpAffineKernel::GetDependInternalSymbol(TContext* ctx) const {
    Common::CVRemapTableKernel remap_table;
    auto format = ctx->getAttrStr("format");
    auto dtype_str = ctx->getAttrOprand("operand:0").dtype;
    bool is_nhwc = format == "NHWC";
    bool is_u8 = dtype_str == "ui8";
    bool is_nhwc_u8 = is_nhwc && is_u8;
    if (is_nhwc_u8) {
        return {
                {remap_table.GetKernelSymbol(ctx), remap_table.GetKernelBody(ctx),
                 remap_table.GetBodyGuardBegin(ctx), remap_table.GetBodyGuardEnd(ctx),
                 remap_table.GetDependInternalSymbol(ctx)}};
    } else {
        return {};
    }
}
std::string WarpAffineKernel::GetKernelBody(TContext* ctx) const {
    auto format = ctx->getAttrStr("format");
    auto imode = ctx->getAttrStr("imode");
    auto bmode = ctx->getAttrStr("border_mode");
    auto input = ctx->getAttrOprand("operand:0");
    float border_val = ctx->getAttrFloat("border_val");
    auto dtype_str = ctx->getAttrOprand("operand:0").dtype;
    bool is_nhwc = format == "NHWC";
    bool is_u8 = input.dtype == "ui8";
    bool is_nhwc_u8 = is_nhwc && is_u8;
    bool is_const_bmode = bmode == "CONSTANT";
    std::string dtype_c_str = Utils::cvt_dtype_specifier(dtype_str);
    uint32_t spatial_start = 2;
    uint32_t batch_pos = 0;
    uint32_t channel_pos = 1;
    std::stringstream ss;
    ss << R"(
        #include <arm_neon.h>
        #include <limits.h>
        #include <math.h>
        #include <stdbool.h>
        #include <string.h>
        #define rep(i, n) for (int i = 0; i < (n); ++i)
    )";
    ss << gen_get_real_coord(bmode);
    std::string board_u8_str = "0";
    if (is_nhwc_u8) {
        if (is_const_bmode) {
            board_u8_str = std::to_string((int)std::round(border_val));
        }
        ss << "short* internal_get_cv_table();\n";
        ss << gen_warp_nhwc_u8(bmode);
    } else {
        ss << gen_visit(bmode, border_val, dtype_c_str, dtype_str);
    }

    std::string stride_str = R"(
        size_t sstrd[3] = {ih * iw, iw, 1};
        size_t dstrd[3] = {oh * ow, ow, 1};
    )";
    if (is_nhwc) {
        spatial_start = 1;
        channel_pos = 3;
        stride_str = R"(
        size_t sstrd[3] = {1, iw * ic, ic};
        size_t dstrd[3] = {1, ow * ic, ic};
    )";
    } else {
        CC_ASSERT(format == "NCHW");
    }
    ss << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    ss << "const uint32_t spatial_start = " << spatial_start << ";\n";
    ss << "const uint32_t batch_pos = " << batch_pos << ";\n";
    ss << "const uint32_t channel_pos = " << channel_pos << ";\n";
    std::string gen_boarder_init = "";
    if (is_nhwc_u8 && is_const_bmode) {
        gen_boarder_init = R"(
            uint8_t boarder_array[ic];
            for(int i = 0; i < ic; ++i){
                boarder_array[i] = board_u8;
            }
            board_val = &boarder_array[0];
        )";
    }
    std::stringstream body_ss;
    body_ss << R"(
        const Tensor* src_tensor = inputs[0];
        TINYNN_ASSERT(src_tensor);
        const Tensor* weight_tensor = inputs[1];
        TINYNN_ASSERT(weight_tensor);
        const Tensor* dst_tensor = outputs[0];
        TINYNN_ASSERT(dst_tensor);

        const ${dtype_c_str}* src_ptr = src_tensor->ptr;
        TINYNN_ASSERT(src_ptr);
        const float* weight_ptr = weight_tensor->ptr;
        TINYNN_ASSERT(weight_ptr);
        ${dtype_c_str}* dst_ptr = dst_tensor->ptr;
        TINYNN_ASSERT(dst_ptr);
        const int* mid_ptr = NULL;

        const Layout src_layout = src_tensor->layout;
        const Layout dst_layout = dst_tensor->layout;

        const int batch = dst_layout.dims[batch_pos];
        const int ic = src_layout.dims[channel_pos];
        const int ih = src_layout.dims[spatial_start];
        const int iw = src_layout.dims[spatial_start + 1];
        
        const int oc = dst_layout.dims[channel_pos];
        const int oh = dst_layout.dims[spatial_start];
        const int ow = dst_layout.dims[spatial_start + 1];
        TINYNN_ASSERT(ic == oc);

        ${stride_str}

        const size_t in_batch_stride = (size_t)ic * ih * iw;
        const size_t out_batch_stride = (size_t)ic * oh * ow;
        )";
    std::string nhwc_u8_temp = R"(
        const int src_batch_stride = ic * ih * iw;
        const int dst_batch_stride = ic * oh * ow;

        const uint8_t board_u8 = ${board_u8_str};
        const uint8_t *board_val = 0;
        ${gen_boarder_init}

        short* ctab = internal_get_cv_table();
        for(int batch_idx = 0; batch_idx < batch; ++batch_idx){
            double M[6];
            rep(i, 6) M[i] = (double)weight_ptr[i];
            weight_ptr += 6;
            warpaffine_img(src_ptr + batch_idx * src_batch_stride, ic, ih, iw, dst_ptr + batch_idx * dst_batch_stride, oh, ow, &M[0], ctab, board_val);
        }
    )";
    std::string normal_temp_body = R"(
        rep(batch_idx, batch){
            const float* mptr = weight_ptr + batch_idx * 2 * 3;
            const ${dtype_c_str}* batch_src_ptr = src_ptr + batch_idx * in_batch_stride;

            rep(oh_idx, oh)
            rep(ow_idx, ow){
                float alphaw = mptr[0] * ow_idx + mptr[1] * oh_idx + mptr[2];
                float alphah = mptr[3] * ow_idx + mptr[4] * oh_idx + mptr[5];

                int iw0 = get_real_coord(floorf(alphaw) + 0, iw);
                int iw1 = get_real_coord(floorf(alphaw) + 1, iw);
                int ih0 = get_real_coord(floorf(alphah) + 0, ih);
                int ih1 = get_real_coord(floorf(alphah) + 1, ih);

                alphaw -= floorf(alphaw);
                alphah -= floorf(alphah);
                float alphaw_p = 1.0f - alphaw;
                float alphah_p = 1.0f - alphah;
                rep(ic_idx, ic){
                    float val = visit_src(batch_src_ptr, ic_idx, ih0, iw0, sstrd) * alphaw_p * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih0, iw1, sstrd) * alphaw * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw0, sstrd) * alphaw_p * alphah +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw1, sstrd) * alphaw * alphah;
                    visit_dst(dst_ptr, ic_idx, oh_idx, ow_idx, dstrd, val);
                }
            }
            dst_ptr += out_batch_stride;
        }
    )";
    if (is_nhwc_u8) {
        body_ss << nhwc_u8_temp;
    } else {
        body_ss << normal_temp_body;
    }
    body_ss << R"(
        return TinyNN_SUCCESS;
    }
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("gen_boarder_init", gen_boarder_init)
                    .add("stride_str", stride_str)
                    .add("dtype_c_str", dtype_c_str)
                    .add("board_u8_str", board_u8_str)
                    .render(body_ss.str());
    return ss.str();
}

bool WarpAffineKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    bool mode_ok = context->getAttrStr("format") == "NHWC" &&
                   context->getAttrStr("imode") == "LINEAR";
    auto format = context->getAttrStr("format");
    auto bmode = context->getAttrStr("border_mode");
    bool is_nhwc = format == "NHWC";
    bool is_u8 = src_dtype == "ui8";
    if (is_u8 && is_nhwc && bmode == "CONSTANT") {
        float border_val = context->getAttrFloat("border_val");
        CC_ASSERT(std::abs(std::round(border_val) - border_val) < 1e-5)
                << "u8 const only support int as border_val not " << border_val << "\n";
    }
    return dtype_ok && mode_ok;
}

std::string WarpAffineKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto bmode = context->getAttrStr("border_mode");
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_warp_affine_" << to_lower_case(bmode) << "_linear_" << src_dtype;
    return ss.str();
}

std::string WarpAffineKernel::GetCVKernelSignature(TContext* context) const {
    auto bmode = context->getAttrStr("border_mode");
    bool is_const_bmode = bmode == "CONSTANT";
    std::string const_bmode_arg = is_const_bmode ? ", uint8_t const_board_val" : "";
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, const double* trans" +
           const_bmode_arg + ")";
}

std::string WarpAffineKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    auto bmode = context->getAttrStr("border_mode");
    auto input = context->getAttrOprand("operand:0");
    auto dtype_str = context->getAttrOprand("operand:0").dtype;
    auto format = context->getAttrStr("format");
    bool is_nhwc = format == "NHWC";
    bool is_u8 = input.dtype == "ui8";
    bool is_nhwc_u8 = is_nhwc && is_u8;
    bool is_const_bmode = bmode == "CONSTANT";
    std::string dtype_c_str = Utils::cvt_dtype_specifier(dtype_str);
    std::stringstream ss;
    ss << R"(
        #include <arm_neon.h>
        #include <limits.h>
        #include <math.h>
        #include "tinycv_c.h"
        #define rep(i, n) for (int i = 0; i < (n); ++i)
        #define rep_step(i, n, s) for (int i = 0; i < (n); ++i)
    )";
    ss << "short* internal_get_cv_table();\n";
    ss << gen_get_real_coord(bmode);
    ss << gen_warp_nhwc_u8(bmode);
    std::string gen_boarder_init = "";
    std::string board_u8 = "0";
    if (is_nhwc_u8 && is_const_bmode) {
        board_u8 = "const_board_val";
        gen_boarder_init = R"(
            uint8_t boarder_array[ic];
            for(int i = 0; i < ic; ++i){
                boarder_array[i] = board_u8;
            }
            board_val = &boarder_array[0];
        )";
    }

    std::string body_temp = R"(
        void ${kernel_sig}{
            uint8_t * src_ptr = src->data;
            uint8_t * dst_ptr = dst->data;
            const int ih = src->rows;
            const int iw = src->cols;
            
            const int ic = src->channels;
            const int oh = dst->rows;
            const int ow = dst->cols;
            const double* M = trans;

            const uint8_t board_u8 = ${board_u8};
            const uint8_t *board_val = 0;
            ${gen_boarder_init}
            short* ctab = internal_get_cv_table();
            warpaffine_img(src_ptr, ic, ih, iw, dst_ptr, oh, ow, &M[0], ctab, board_val);
        }
    )";

    ss << StringTemplate::StringTemplateArgs()
                    .add("gen_boarder_init", gen_boarder_init)
                    .add("kernel_sig", kernel_sig)
                    .add("board_u8", board_u8)
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen
