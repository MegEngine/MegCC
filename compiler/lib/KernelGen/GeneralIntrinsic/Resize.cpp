#include <sstream>

#include "Common/Resize.h"
#include "Resize.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool ResizeKernel::IsAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok_f32 = src_dtype == "f32";
    bool dtype_ok_f16 = src_dtype == "f16";
    bool mode_ok = context->getAttrStr("imode") == "LINEAR";
    bool format_ok_f32 = context->getAttrStr("format") == "NCHW" ||
                         context->getAttrStr("format") == "NCHW44";
    bool format_ok_f16 = context->getAttrStr("format") == "NCHW88";
    return mode_ok &&
           ((dtype_ok_f32 && format_ok_f32) || (dtype_ok_f16 && format_ok_f16));
}
//! kernel gen
std::string ResizeKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto fmt = context->getAttrStr("format");
    auto imode = context->getAttrStr("imode");
    ss << "GI_kernel_resize_linear_" << fmt << "_" << imode << "_" << src_dtype;
    return ss.str();
}

std::string ResizeKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto fmt = context->getAttrStr("format");
    auto specifier = Utils::cvt_dtype_specifier(src_dtype);
    auto imode = context->getAttrStr("imode");
    ss << R"(
        #include <math.h>
        #include <stdalign.h>
    )";
    if (specifier == "gi_float16_t") {
        ss << R"(
            #include "gi_float16.h"
        )";
    }
    auto coord_str = ResizeHelper::GenCoordHelper(imode, specifier);
    auto gen_layout_dims = ResizeHelper::GenLayoutDims(fmt);
    auto get_offset = ResizeHelper::GenGetOffset(fmt);
    ss << StringTemplate::StringTemplateArgs()
                    .add("coord_helper_str", coord_str)
                    .add("get_offset", get_offset)
                    .render(R"(
        static inline float output_converter(float x){
            return x;
        }
        ${coord_helper_str}
        ${get_offset}
        #define rep(i, n) for (int i = 0; i < (n); ++i)
    )");
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
        const Tensor* src_tensor = inputs[0];
        const Tensor* dst_tensor = outputs[0];
        ${specifier}* sptr = (${specifier}*)(src_tensor->ptr);
        ${specifier}* dptr = (${specifier}*)(dst_tensor->ptr);
        TINYNN_ASSERT(sptr);
        TINYNN_ASSERT(dptr);
        
        const Layout src_layout = src_tensor->layout;
        const Layout dst_layout = dst_tensor->layout;
        ${gen_layout_dims}
        float scale_h = (float)(OH) / IH;
        float scale_w = (float)(OW) / IW;

        ${normal_impl}
        return TinyNN_SUCCESS;
    })";
    auto normal_impl = ResizeHelper::GenNormImpl(fmt, specifier);
    ss << StringTemplate::StringTemplateArgs()
                    .add("specifier", specifier)
                    .add("normal_impl", normal_impl)
                    .add("gen_layout_dims", gen_layout_dims)
                    .render(body_temp);
    return ss.str();
}

bool ResizeKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "ui8" || Utils::is_float_dtype(src_dtype);
    bool mode_ok = context->getAttrStr("imode") == "LINEAR" &&
                   context->getAttrStr("format") == "NHWC";
    return dtype_ok && mode_ok;
}

//! kernel gen
std::string ResizeKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_resize_linear_" << src_dtype;
    return ss.str();
}

std::string ResizeKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) + "(const TinyMat* src, const TinyMat* dst)";
}

inline std::string gen_gi_interpolate_c1_func(const bool is_ui8) {
    std::string gi_interpolate_c1 = is_ui8 ? R"(
                    GI_INT32_t v_rx = GiBroadcastInt32(rx);
                    GI_INT32_t v_irx = GiBroadcastInt32(irx);
#define RSCALE (SCALE + SCALE - 16)
                    for (; dy + 16 <= dstcols; dy += 16) {
                        GI_INT32_t v_cache0_0;
                        GI_INT32_t v_cache1_0;
                        GI_INT32_t v_cache0_4;
                        GI_INT32_t v_cache1_4;
                        GI_INT32_t v_cache0_8;
                        GI_INT32_t v_cache1_8;
                        GI_INT32_t v_cache0_c;
                        GI_INT32_t v_cache1_c;

                        v_cache0_0 = GiLoadInt32(cache0_ptr + dy + 0x0);
                        v_cache1_0 = GiLoadInt32(cache1_ptr + dy + 0x0);
                        v_cache0_4 = GiLoadInt32(cache0_ptr + dy + 0x4);
                        v_cache1_4 = GiLoadInt32(cache1_ptr + dy + 0x4);
                        v_cache0_8 = GiLoadInt32(cache0_ptr + dy + 0x8);
                        v_cache1_8 = GiLoadInt32(cache1_ptr + dy + 0x8);
                        v_cache0_c = GiLoadInt32(cache0_ptr + dy + 0xc);
                        v_cache1_c = GiLoadInt32(cache1_ptr + dy + 0xc);

                        GI_INT16_t v_ans0, v_ans4, v_ans8, v_ansc;
                        v_ans0 = GiCvtInt32ToInt16(GiShiftRightInt32(GiMultiplyAddInt32(GiMultiplyInt32(v_rx, v_cache1_0),
                                                        v_irx, v_cache0_0),
                                            16));
                        v_ans4 = GiCvtInt32ToInt16(GiShiftRightInt32(GiMultiplyAddInt32(GiMultiplyInt32(v_rx, v_cache1_4),
                                                        v_irx, v_cache0_4),
                                            16));
                        v_ans8 = GiCvtInt32ToInt16(GiShiftRightInt32(GiMultiplyAddInt32(GiMultiplyInt32(v_rx, v_cache1_8),
                                                        v_irx, v_cache0_8),
                                            16));
                        v_ansc = GiCvtInt32ToInt16(GiShiftRightInt32(GiMultiplyAddInt32(GiMultiplyInt32(v_rx, v_cache1_c),
                                                        v_irx, v_cache0_c),
                                            16));

                        GI_INT16_t v_half16_0, v_half16_1;
                        v_half16_0 = GiCombineInt16Low(v_ans0, v_ans4);  // x0
                        v_half16_1 = GiCombineInt16Low(v_ans8, v_ansc);  // y0

                        GI_UINT8_t v_half8_0, v_half8_1;
                        v_half8_0 = GiShiftRightInt16ToUint8(v_half16_0, RSCALE);
                        v_half8_1 = GiShiftRightInt16ToUint8(v_half16_1, RSCALE);
                        GI_UINT8_t v_all8_0 = GiCombineUint8Low(v_half8_0, v_half8_1);

                        GiStoreUint8(pdst + dy, v_all8_0);
                    }
#undef RSCALE
    )"
                                           : R"(
#define EXPAND(x)                                  \
    v_cache0 = GiLoadFloat32(cache0_ptr + dy + x); \
    v_cache1 = GiLoadFloat32(cache1_ptr + dy + x); \
    GiStoreFloat32(                                \
            pdst + dy + x,                         \
            GiMlaqFloat32(GiMultiplyFloat32(v_rx, v_cache1), v_irx, v_cache0));

                    GI_FLOAT32_t v_rx = GiBroadcastFloat32(rx);
                    GI_FLOAT32_t v_irx = GiBroadcastFloat32(irx);
                    for (; dy + 8 <= dstcols; dy += 8) {
                        GI_FLOAT32_t v_cache0;
                        GI_FLOAT32_t v_cache1;
                        EXPAND(0);
                        EXPAND(4);
                    }
                    if (dy + 4 <= dstcols) {
                        GI_FLOAT32_t v_cache0;
                        GI_FLOAT32_t v_cache1;
                        EXPAND(0);
                        dy += 4;
                    }
#undef EXPAND
    )";
    return gi_interpolate_c1;
}

inline std::string gen_gi_interpolate_c3_func(const bool is_ui8) {
    std::string gi_interpolate_c3 = is_ui8 ? R"(
                    for (; dy < dstcols; dy += 3) {
                        uint8_t* pcdst = pdst + dy;
                        pcdst[0] = (rx * cache1[dy + 0] + irx * cache0[dy + 0]) >>
                                (SCALE + SCALE);
                        pcdst[1] = (rx * cache1[dy + 1] + irx * cache0[dy + 1]) >>
                                (SCALE + SCALE);
                        pcdst[2] = (rx * cache1[dy + 2] + irx * cache0[dy + 2]) >>
                                (SCALE + SCALE);
                    }
    )"
                                           : R"(
#define EXPAND(x)                                                                 \
    v_cache0 = GiLoadUzipFloat32V3(cache0_ptr + dy + (x)*3);                      \
    v_cache1 = GiLoadUzipFloat32V3(cache1_ptr + dy + (x)*3);                      \
    a0 = GiMlaqFloat32(                                                           \
            GiMultiplyFloat32(v_rx, GiGetSubVectorFloat32V3(v_cache1, 0)), v_irx, \
            GiGetSubVectorFloat32V3(v_cache0, 0));                                \
    GiSetSubVectorFloat32V3(v_dst, 0, a0);                                        \
    a1 = GiMlaqFloat32(                                                           \
            GiMultiplyFloat32(v_rx, GiGetSubVectorFloat32V3(v_cache1, 1)), v_irx, \
            GiGetSubVectorFloat32V3(v_cache0, 1));                                \
    GiSetSubVectorFloat32V3(v_dst, 1, a1);                                        \
    a2 = GiMlaqFloat32(                                                           \
            GiMultiplyFloat32(v_rx, GiGetSubVectorFloat32V3(v_cache1, 2)), v_irx, \
            GiGetSubVectorFloat32V3(v_cache0, 2));                                \
    GiSetSubVectorFloat32V3(v_dst, 2, a2);                                        \
    GiStoreZipFloat32V3(pdst + dy + (x)*3, v_dst);
                    GI_FLOAT32_t v_rx = GiBroadcastFloat32(rx);
                    GI_FLOAT32_t v_irx = GiBroadcastFloat32(irx);
                    GI_FLOAT32_t a0, a1, a2;
                    for (; dy + 8 * 3 <= dstcols; dy += 8 * 3) {
                        GI_FLOAT32_V3_t v_cache0;
                        GI_FLOAT32_V3_t v_cache1;
                        GI_FLOAT32_V3_t v_dst;

                        EXPAND(0);
                        EXPAND(4);
                    }

                    if (dy + 4 * 3 <= dstcols) {
                        GI_FLOAT32_V3_t v_cache0;
                        GI_FLOAT32_V3_t v_cache1;
                        GI_FLOAT32_V3_t v_dst;

                        EXPAND(0);

                        dy += 4 * 3;
                    }
#undef EXPAND
                    for (; dy < dstcols; dy += 3) {
                        float* pcdst = pdst + dy;
                        pcdst[0] = rx * cache1[dy + 0] + irx * cache0[dy + 0];
                        pcdst[1] = rx * cache1[dy + 1] + irx * cache0[dy + 1];
                        pcdst[2] = rx * cache1[dy + 2] + irx * cache0[dy + 2];
                    }
    )";
    return gi_interpolate_c3;
}

std::string ResizeKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    auto src_dtype =
            Utils::cvt_dtype_specifier(context->getAttrOprand("operand:0").dtype);
    std::stringstream writer;
    writer << R"(
        #include <string.h>
        #include "tinycv_c.h"
        )";
    std::string body_temp = R"(
        #include <math.h>
        #include <stdalign.h>
        #include "${gi_header}"
        #define rep(i, n) for (int i = 0; i < (n); ++i)
        ${define_scale}
        ${output_converter_func}
        
        static inline void get_nearest_linear_coord(float scale, int size, int idx, float* ah0, int* ih0, float* ah1, int* ih1){
            if (size == 1) {
                *ah0 = 1.f;
                *ih0 = 0;
                *ah1 = 0.f;
                *ih1 = 0;
            }

            float alpha = (idx + 0.5f) / scale - 0.5f;
            int origin_idx = (int)(floorf(alpha));
            alpha -= origin_idx;

            if (origin_idx < 0) {
                origin_idx = 0;
                alpha = 0;
            } else if (origin_idx + 1 >= size) {
                origin_idx = size - 2;
                alpha = 1;
            }

            *ah0 = 1 - alpha;
            *ih0 = origin_idx;
            *ah1 = alpha;
            *ih1 = origin_idx + 1;
        }

        static inline void calc_cache_${func_tag}c3_1(const ${src_dtype}* src, const int src_h_stride, const ${src_dtype}* dst, const int OW,
                       const int* tabsx,
                       const int* tabsy,
                       const ${calc_dtype}* tabrx,
                       const ${calc_dtype}* tabry, int dx,
                       ${calc_dtype}* cache0, ${calc_dtype}* cache1) {
            (void)tabrx;
            const ${src_dtype}* psrc1 = src + (tabsx[dx] + 1) * src_h_stride;

            int dy = 0, dy3 = 0;

            for (; dy < OW; ++dy, dy3 += 3) {
                const ${src_dtype}* pcsrc10 = psrc1 + (tabsy[dy] + 0) * 3;
                const ${src_dtype}* pcsrc11 = psrc1 + (tabsy[dy] + 1) * 3;
                ${calc_dtype} ry = tabry[dy];
                ${calc_dtype} iry = ${calc_iry_func};
                cache1[dy3 + 0] = pcsrc11[0] * ry + pcsrc10[0] * iry;
                cache1[dy3 + 1] = pcsrc11[1] * ry + pcsrc10[1] * iry;
                cache1[dy3 + 2] = pcsrc11[2] * ry + pcsrc10[2] * iry;
            }
        }
        static inline void calc_cache_${func_tag}c3_2(const ${src_dtype}* src, const int src_h_stride, const ${src_dtype}* dst, const int OW,
                       const int*  tabsx,
                       const int*  tabsy,
                       const ${calc_dtype}*  tabrx,
                       const ${calc_dtype}*  tabry, int dx,
                       ${calc_dtype}*  cache0, 
                       ${calc_dtype}*  cache1) {
            (void)tabrx;
            const ${src_dtype}* psrc0 = src + (tabsx[dx] + 0) * src_h_stride;
            const ${src_dtype}* psrc1 = src + (tabsx[dx] + 1) * src_h_stride;
            int dstcols = OW;
            int dy = 0, dy3 = 0;

            // 4 pixels each time
            for (; dy < dstcols; ++dy, dy3 += 3) {
                const ${src_dtype}* pcsrc00 = psrc0 + (tabsy[dy] + 0) * 3;
                const ${src_dtype}* pcsrc01 = psrc0 + (tabsy[dy] + 1) * 3;
                const ${src_dtype}* pcsrc10 = psrc1 + (tabsy[dy] + 0) * 3;
                const ${src_dtype}* pcsrc11 = psrc1 + (tabsy[dy] + 1) * 3;
                ${calc_dtype} ry = tabry[dy];
                ${calc_dtype} iry = ${calc_iry_func};
                cache0[dy3 + 0] = pcsrc01[0] * ry + pcsrc00[0] * iry;
                cache1[dy3 + 0] = pcsrc11[0] * ry + pcsrc10[0] * iry;
                cache0[dy3 + 1] = pcsrc01[1] * ry + pcsrc00[1] * iry;
                cache1[dy3 + 1] = pcsrc11[1] * ry + pcsrc10[1] * iry;
                cache0[dy3 + 2] = pcsrc01[2] * ry + pcsrc00[2] * iry;
                cache1[dy3 + 2] = pcsrc11[2] * ry + pcsrc10[2] * iry;
            }
        }

        static inline void calc_cache_${func_tag}c2_1(const ${src_dtype}* src, const int src_h_stride, const ${src_dtype}* dst, const int OW,
                       const int* tabsx,
                       const int* tabsy,
                       const ${calc_dtype}* tabrx,
                       const ${calc_dtype}* tabry, int dx,
                       ${calc_dtype}* cache0, ${calc_dtype}* cache1) {
            (void)tabrx;
            const ${src_dtype}* psrc1 = src + (tabsx[dx] + 1) * src_h_stride;

            int dy = 0, dy2 = 0;

            for (; dy < OW; ++dy, dy2 += 2) {
                const ${src_dtype}* pcsrc10 = psrc1 + (tabsy[dy] + 0) * 2;
                const ${src_dtype}* pcsrc11 = psrc1 + (tabsy[dy] + 1) * 2;
                ${calc_dtype} ry = tabry[dy];
                ${calc_dtype} iry = ${calc_iry_func};
                cache1[dy2 + 0] = pcsrc11[0] * ry + pcsrc10[0] * iry;
                cache1[dy2 + 1] = pcsrc11[1] * ry + pcsrc10[1] * iry;
            }
        }
        static inline void calc_cache_${func_tag}c2_2(const ${src_dtype}* src, const int src_h_stride, const ${src_dtype}* dst, const int OW,
                       const int*  tabsx,
                       const int*  tabsy,
                       const ${calc_dtype}*  tabrx,
                       const ${calc_dtype}*  tabry, int dx,
                       ${calc_dtype}*  cache0, 
                       ${calc_dtype}*  cache1) {
            (void)tabrx;
            const ${src_dtype}* psrc0 = src + (tabsx[dx] + 0) * src_h_stride;
            const ${src_dtype}* psrc1 = src + (tabsx[dx] + 1) * src_h_stride;
            int dstcols = OW;
            int dy = 0, dy2 = 0;

            // 4 pixels each time
            for (; dy < dstcols; ++dy, dy2 += 2) {
                const ${src_dtype}* pcsrc00 = psrc0 + (tabsy[dy] + 0) * 2;
                const ${src_dtype}* pcsrc01 = psrc0 + (tabsy[dy] + 1) * 2;
                const ${src_dtype}* pcsrc10 = psrc1 + (tabsy[dy] + 0) * 2;
                const ${src_dtype}* pcsrc11 = psrc1 + (tabsy[dy] + 1) * 2;
                ${calc_dtype} ry = tabry[dy];
                ${calc_dtype} iry = ${calc_iry_func};
                cache0[dy2 + 0] = pcsrc01[0] * ry + pcsrc00[0] * iry;
                cache1[dy2 + 0] = pcsrc11[0] * ry + pcsrc10[0] * iry;
                cache0[dy2 + 1] = pcsrc01[1] * ry + pcsrc00[1] * iry;
                cache1[dy2 + 1] = pcsrc11[1] * ry + pcsrc10[1] * iry;
            }
        }

        static inline void build_tabs_linear_${func_tag}(int IH, int IW, int OH, int OW,
                          int* tabsx, 
                          int* tabsy,
                          ${calc_dtype}* tabrx,
                          ${calc_dtype}* tabry) {

            const float fx = (float)(OH) / IH;
            const float fy = (float)(OW) / IW;
            const float ifx = 1.0f / fx;
            const float ify = 1.0f / fy;
            for (int dx = 0; dx < OH; ++dx) {
                float rx = (dx + 0.5f) * ifx - 0.5f;
                int sx = (int)(floor(rx));
                rx -= sx;
                if (sx < 0) {
                    sx = 0;
                    rx = 0;
                } else if (sx + 1 >= IH) {
                    sx = IH - 2;
                    rx = 1;
                }
                tabsx[dx] = sx;
                tabrx[dx] = ${calc_rx_func};
            }
            for (int dy = 0; dy < OW; ++dy) {
                float ry = (dy + 0.5f) * ify - 0.5f;
                int sy = (int)(floor(ry));
                ry -= sy;
                if (sy < 0) {
                    sy = 0;
                    ry = 0;
                } else if (sy + 1 >= IW) {
                    sy = IW - 2;
                    ry = 1;
                }
                tabsy[dy] = sy;
                tabry[dy] = ${calc_ry_func};
            }
        }

        static inline void calc_cache_${func_tag}c1_1(const ${src_dtype}* src, const int src_h_stride, const ${src_dtype}* dst, const int OW,
                       const int* tabsx,
                       const int* tabsy,
                       const ${calc_dtype}* tabrx,
                       const ${calc_dtype}* tabry, int dx,
                       ${calc_dtype}* cache0, 
                       ${calc_dtype}* cache1) {
            (void)tabrx;
            const ${src_dtype}* psrc1 = src + (tabsx[dx] + 1) * src_h_stride;
            int dstcols = OW;
            int dy = 0;

            
            for (; dy < dstcols; ++dy) {
                const ${src_dtype}* pcsrc10 = psrc1 + (tabsy[dy] + 0);
                const ${src_dtype}* pcsrc11 = psrc1 + (tabsy[dy] + 1);
                ${calc_dtype} ry = tabry[dy];
                ${calc_dtype} iry = ${calc_iry_func};
                cache1[dy] = pcsrc11[0] * ry + pcsrc10[0] * iry;
            }
        }

        static inline void calc_cache_${func_tag}c1_2(const ${src_dtype}* src, const int src_h_stride, const ${src_dtype}* dst, const int OW,
                            const int* tabsx,
                            const int* tabsy,
                            const ${calc_dtype}* tabrx,
                            const ${calc_dtype}* tabry, int dx,
                            ${calc_dtype}* cache0, 
                            ${calc_dtype}* cache1) {
            (void)tabrx;
            const ${src_dtype}* psrc0 = src + (tabsx[dx] + 0) * src_h_stride;
            const ${src_dtype}* psrc1 = src + (tabsx[dx] + 1) * src_h_stride;
            int dstcols = OW;
            int dy = 0;

            // 4 pixels each time
            for (; dy < dstcols; ++dy) {
                const ${src_dtype}* pcsrc00 = psrc0 + (tabsy[dy] + 0);
                const ${src_dtype}* pcsrc01 = psrc0 + (tabsy[dy] + 1);
                const ${src_dtype}* pcsrc10 = psrc1 + (tabsy[dy] + 0);
                const ${src_dtype}* pcsrc11 = psrc1 + (tabsy[dy] + 1);
                ${calc_dtype} ry = tabry[dy];
                ${calc_dtype} iry = ${calc_iry_func};
                cache0[dy] = pcsrc01[0] * ry + pcsrc00[0] * iry;
                cache1[dy] = pcsrc11[0] * ry + pcsrc10[0] * iry;
            }
        }

        void ${kernel_sig}{
            ${src_dtype}* sptr = src->data;
            ${src_dtype}* dptr = dst->data;
            int IH = src->rows;
            int IW = src->cols;
            int C = src->channels;
            int src_h_stride = IW * C;
            int OH = dst->rows;
            int OW = dst->cols;

            
            bool use_cache = OH >= 2 && OW >= 2 && IH >= 2 && IW >= 2;

            if (C == 1 && use_cache) {
                alignas(16) int tabsx[OH];
                alignas(16) int tabsy[OW];
                alignas(16) ${calc_dtype} tabrx[OH];
                alignas(16) ${calc_dtype} tabry[OW];
                build_tabs_linear_${func_tag}(IH, IW, OH, OW, tabsx, tabsy, tabrx, tabry);

                int dstrows = OH;
                int dstcols = OW;
                alignas(16) ${calc_dtype} cache0_body[dstcols];
                alignas(16) ${calc_dtype} cache1_body[dstcols];
                ${calc_dtype}* cache0 = &cache0_body[0];
                ${calc_dtype}* cache1 = &cache1_body[0];
                for (int dx = 0; dx < dstrows; ++dx) {
                    if (dx == 0 || tabsx[dx] != tabsx[dx - 1]) {
                        if (dx > 0 && tabsx[dx] == tabsx[dx - 1] + 1) {
                            ${calc_dtype}* temp = cache0;
                            cache0 = cache1;
                            cache1 = temp;
                            calc_cache_${func_tag}c1_1(sptr, src_h_stride, dptr, OW, tabsx, tabsy, tabrx, tabry, dx,
                                            cache0, cache1);
                        } else {
                            calc_cache_${func_tag}c1_2(sptr, src_h_stride, dptr, OW, tabsx, tabsy, tabrx, tabry, dx,
                                            cache0, cache1);
                        }
                    }
                    ${calc_dtype} rx = tabrx[dx];
                    ${calc_dtype} irx = ${calc_irx_func};
                    ${src_dtype}* pdst = dptr + (dx) * OW;
                    int dy = 0;

                    const ${calc_dtype}* cache0_ptr = cache0;
                    const ${calc_dtype}* cache1_ptr = cache1;
                    ${gi_interpolate_c1}
                    for (; dy < dstcols; ++dy) {
                        ${src_dtype}* pcdst = pdst + dy;
                        pcdst[0] =
                                (rx * cache1[dy] + irx * cache0[dy]) ${right_shift_func};
                    }
                }
            }else if (C == 2 && use_cache) {
                alignas(16) int tabsx[OH];
                alignas(16) int tabsy[OW];
                alignas(16) ${calc_dtype} tabrx[OH];
                alignas(16) ${calc_dtype} tabry[OW];
                build_tabs_linear_${func_tag}(IH, IW, OH, OW, tabsx, tabsy, tabrx, tabry);
                int dstrows = OH;
                int dstcols = OW * 2;
                alignas(16) ${calc_dtype} cache0_body[dstcols];
                alignas(16) ${calc_dtype} cache1_body[dstcols];
                ${calc_dtype}* cache0 = &cache0_body[0];
                ${calc_dtype}* cache1 = &cache1_body[0];
                for (int dx = 0; dx < dstrows; ++dx) {
                    if (dx == 0 || tabsx[dx] != tabsx[dx - 1]) {
                        if (dx > 0 && tabsx[dx] == tabsx[dx - 1] + 1) {
                            ${calc_dtype}* temp = cache0;
                            cache0 = cache1;
                            cache1 = temp;
                            calc_cache_${func_tag}c2_1(sptr, src_h_stride, dptr, OW, tabsx, tabsy, tabrx, tabry, dx,
                                            cache0, cache1);
                        } else {
                            calc_cache_${func_tag}c2_2(sptr, src_h_stride, dptr, OW, tabsx, tabsy, tabrx, tabry, dx,
                                            cache0, cache1);
                        }
                    }
                    ${calc_dtype} rx = tabrx[dx];
                    ${calc_dtype} irx = ${calc_irx_func};
                    ${src_dtype}* pdst = dptr + dx * OW * C;
                    int dy = 0;

                    for (; dy < dstcols; dy += 2) {
                        ${src_dtype}* pcdst = pdst + dy;
                        pcdst[0] = (rx * cache1[dy + 0] + irx * cache0[dy + 0]) ${right_shift_func};
                        pcdst[1] = (rx * cache1[dy + 1] + irx * cache0[dy + 1]) ${right_shift_func};
                    }
                }
            } else if (C == 3 && use_cache) {
                alignas(16) int tabsx[OH];
                alignas(16) int tabsy[OW];
                alignas(16) ${calc_dtype} tabrx[OH];
                alignas(16) ${calc_dtype} tabry[OW];
                build_tabs_linear_${func_tag}(IH, IW, OH, OW, tabsx, tabsy, tabrx, tabry);
                int dstrows = OH;
                int dstcols = OW * 3;
                alignas(16) ${calc_dtype} cache0_body[dstcols];
                alignas(16) ${calc_dtype} cache1_body[dstcols];
                ${calc_dtype}* cache0 = &cache0_body[0];
                ${calc_dtype}* cache1 = &cache1_body[0];
                for (int dx = 0; dx < dstrows; ++dx) {
                    if (dx == 0 || tabsx[dx] != tabsx[dx - 1]) {
                        if (dx > 0 && tabsx[dx] == tabsx[dx - 1] + 1) {
                            ${calc_dtype}* temp = cache0;
                            cache0 = cache1;
                            cache1 = temp;
                            calc_cache_${func_tag}c3_1(sptr, src_h_stride, dptr, OW, tabsx, tabsy, tabrx, tabry, dx,
                                            cache0, cache1);
                        } else {
                            calc_cache_${func_tag}c3_2(sptr, src_h_stride, dptr, OW, tabsx, tabsy, tabrx, tabry, dx,
                                            cache0, cache1);
                        }
                    }
                    ${calc_dtype} rx = tabrx[dx];
                    ${calc_dtype} irx = ${calc_irx_func};
                    ${src_dtype}* pdst = dptr + dx * OW * C;
                    int dy = 0;
                    const ${calc_dtype}* cache0_ptr = cache0;
                    const ${calc_dtype}* cache1_ptr = cache1;
                    ${gi_interpolate_c3}
                }
            } else {
                float scale_h = (float)(OH) / IH;
                float scale_w = (float)(OW) / IW;
                alignas(16) float ah0_cache[OH];
                alignas(16) int ih0_cache[OH];
                alignas(16) float ah1_cache[OH];
                alignas(16) int ih1_cache[OH];
                alignas(16) float aw0_cache[OW];
                alignas(16) int iw0_cache[OW];
                alignas(16) float aw1_cache[OW];
                alignas(16) int iw1_cache[OW];

                rep(oh, OH) {
                    get_nearest_linear_coord(scale_h, IH, oh, &ah0_cache[oh], &ih0_cache[oh], &ah1_cache[oh], &ih1_cache[oh]);
                }
                rep(ow, OW) {
                    get_nearest_linear_coord(scale_w, IW, ow, &aw0_cache[ow], &iw0_cache[ow], &aw1_cache[ow], &iw1_cache[ow]);
                }
                rep(oh, OH) {
                    int ih0 = ih0_cache[oh];
                    int ih1 = ih1_cache[oh];
                    float ah0 = ah0_cache[oh];
                    float ah1 = ah1_cache[oh];
                    rep(ow, OW) {
                        int iw0 = iw0_cache[ow];
                        int iw1 = iw1_cache[ow];
                        float aw0 = aw0_cache[ow];
                        float aw1 = aw1_cache[ow];
                        rep(c, C) {
                            dptr[(oh * OW + ow) * C + c] = output_converter(
                                    sptr[(ih0 * IW + iw0) * C + c] * ah0 * aw0 +
                                    sptr[(ih0 * IW + iw1) * C + c] * ah0 * aw1 +
                                    sptr[(ih1 * IW + iw0) * C + c] * ah1 * aw0 +
                                    sptr[(ih1 * IW + iw1) * C + c] * ah1 * aw1);
                        }
                    }
                }
            }
        }
    )";

    bool is_ui8 = src_dtype == "uint8_t";
    std::string gi_header = is_ui8 ? "gi_int.h" : "gi_float.h";
    std::string calc_dtype = is_ui8 ? "int" : "float";
    std::string func_tag = is_ui8 ? "8u" : "32f";
    std::string define_scale = is_ui8 ? R"(
        #define SCALE 11
    )"
                                      : "";
    std::string output_converter_func = is_ui8 ? R"(
        static inline uint8_t output_converter(float x){
            x = fmin(255.0f, fmax(0.0f, x));
            return (uint8_t) roundf(x);
        }
    )"
                                               : R"(
        static inline float output_converter(float x){
            return x;
        }
    )";
    std::string calc_rx_func = is_ui8 ? "(int)(rx * (1 << SCALE))" : "rx";
    std::string calc_ry_func = is_ui8 ? "(int)(ry * (1 << SCALE))" : "ry";
    std::string calc_irx_func = is_ui8 ? "(1 << SCALE) - rx" : "1.0f - rx";
    std::string calc_iry_func = is_ui8 ? "(1 << SCALE) - ry" : "1.0f - ry";
    std::string right_shift_func = is_ui8 ? ">> (SCALE + SCALE)" : "";
    std::string gi_interpolate_c1 = gen_gi_interpolate_c1_func(is_ui8);
    std::string gi_interpolate_c3 = gen_gi_interpolate_c3_func(is_ui8);
    writer << StringTemplate::StringTemplateArgs()
                      .add("kernel_sig", kernel_sig)
                      .add("gi_header", gi_header)
                      .add("src_dtype", src_dtype)
                      .add("calc_dtype", calc_dtype)
                      .add("func_tag", func_tag)
                      .add("define_scale", define_scale)
                      .add("calc_rx_func", calc_rx_func)
                      .add("calc_ry_func", calc_ry_func)
                      .add("calc_irx_func", calc_irx_func)
                      .add("calc_iry_func", calc_iry_func)
                      .add("right_shift_func", right_shift_func)
                      .add("right_shift_func", right_shift_func)
                      .add("gi_interpolate_c1", gi_interpolate_c1)
                      .add("gi_interpolate_c3", gi_interpolate_c3)
                      .add("output_converter_func", output_converter_func)
                      .render(body_temp);
    return writer.str();
}

// vim: syntax=cpp.doxygen
