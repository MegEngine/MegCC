#pragma once
#include <string>
#include "Arm/ArmCommon/ArmSimdHelper.h"
#include "Utils/StringTemplate.h"
namespace megcc {
namespace KernelGen {
namespace ArmCommon {
namespace {

std::string gen_nchw44_im2col_kern(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    ArmSimdHelper simd_helper(dtype);
    static std::string nchw44_im2col_temp = R"(
    #define rep(i, n) for (int i = 0; i < (n); ++i)
    static inline void img2col(const ${specifier}* __restrict src, ${specifier}* __restrict dst,
                        const int OW, const int IC,
                        const int IH, const int IW, const int FH, const int FW,
                        const int SH, const int SW, const int cur_index,
                        const int block_size) {
        int start_h = cur_index / OW;
        int cur_remain_w = cur_index % OW;
        int end_h = (cur_index + block_size) / OW;
        int end_remain_w = (cur_index + block_size) % OW;
        bool same_line = false;
        if (start_h == end_h) {
            same_line = true;
        }

        size_t newIC = IC / 4;
        size_t i = 0;
        
        if (same_line) {
            rep(ic, newIC) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            size_t index = 4 * (ic * IH * IW +
                                                (start_h * SH + fh) * IW +
                                                (w * SW + fw));
                            dst[i++] = src[index];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }
                    }
                }
            }
        } else {
            rep(ic, newIC) {
                rep(fh, FH) {
                    rep(fw, FW) {                      
                        for (int w = cur_remain_w; w < OW; w++) {
                            size_t index =4 * (ic * IH * IW +
                                        (start_h * SH + fh) * IW +
                                        (w * SW + fw));
                            dst[i++] = src[index + 0];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                size_t index = 4 * (ic * IH * IW +
                                                    (h * SH + fh) * IW +
                                                    (ow * SW + fw));
                                dst[i++] = src[index + 0];
                                dst[i++] = src[index + 1];
                                dst[i++] = src[index + 2];
                                dst[i++] = src[index + 3];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            size_t index = 4 * (ic * IH * IW +
                                                (end_h * SH + fh) * IW +
                                                (w * SW + fw));
                            dst[i++] = src[index + 0];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }
                    }
                }
            }
        }
        
    }
)";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", Utils::cvt_dtype_specifier(dtype))
            .render(nchw44_im2col_temp);
}
std::string gen_nchw44_pad_src_kern(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    ArmSimdHelper simd_helper(dtype);
    static std::string nchw44_pad_src_temp = R"(
    static inline void pad_src(${specifier}* inptr, ${specifier}* outptr, int ic, int ih, int iw, int pad_h, int pad_w){
        const int pack_c_size = 4;
        const int paded_iw = iw + 2 * pad_w;
        const int nr_pad_top_bottom = pad_h * paded_iw * pack_c_size;
        const ${simd_specifier} vzero = ${simd_dup1q}(0);
        for(int ic_idx = 0; ic_idx < ic; ic_idx += pack_c_size){
            memset(outptr, 0, sizeof(${specifier}) * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            
            for (int ih_idx = 0; ih_idx < ih; ++ih_idx){
                for(int i = 0; i < pad_w; ++i){
                    ${simd_st1q}(outptr, vzero);
                    outptr += 4;
                }
                memcpy(outptr, inptr + ih_idx * iw * pack_c_size, sizeof(${specifier}) * iw * pack_c_size);
                outptr += iw * pack_c_size;
                for(int i = 0; i < pad_w; ++i){
                    ${simd_st1q}(outptr, vzero);
                    outptr += 4;
                }
            }
            memset(outptr, 0, sizeof(${specifier}) * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            inptr += ih * iw * pack_c_size;
        }
    }
)";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", Utils::cvt_dtype_specifier(dtype))
            .add("simd_specifier", simd_helper.get_specifier_q_symbol())
            .add("simd_dup1q", simd_helper.get_dupq_n_symbol())
            .add("simd_st1q", simd_helper.get_st1q_symbol())
            .render(nchw44_pad_src_temp);
}

std::string gen_nchw_im2col_kern(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    std::string nchw_im2col_kern = R"(
    #define rep(i, n) for (int i = 0; i < (n); ++i)
    
    static inline void img2col(const ${specifier}* __restrict src, ${specifier}* __restrict dst,
                    const int OW, const int IC,
                    const int IH, const int IW, const int FH, const int FW,
                    const int SH, const int SW, const int cur_index,
                    const int block_size) {                                
        int start_h = cur_index / OW;
        int cur_remain_w = cur_index % OW;
        int end_h = (cur_index + block_size) / OW;
        int end_remain_w = (cur_index + block_size) % OW;
        bool same_line = false;
        if (start_h == end_h) {
            same_line = true;
        }

        size_t i = 0;
        if (same_line) {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            dst[i++] =
                                    src[ic * IH * IW + (start_h * SH + fh) * IW +
                                        (w * SW + fw)];
                        }
                    }
                }
            }
        } else {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {

                        for (int w = cur_remain_w; w < OW; w++) {
                            dst[i++] =
                                    src[ic * IH * IW + (start_h * SH + fh) * IW +
                                        (w * SW + fw)];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                dst[i++] = src[ic * IH * IW + (h * SH + fh) * IW +
                                            (ow * SW + fw)];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            dst[i++] = src[ic * IH * IW + (end_h * SH + fh) * IW +
                                        (w * SW + fw)];
                        }
                    }
                }
            }
        }
    }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", Utils::cvt_dtype_specifier(dtype))
            .render(nchw_im2col_kern);
}

std::string gen_nchw_im2col_s1_kern(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    std::string nchw_im2col_s1_kern = R"(
    #define rep(i, n) for (int i = 0; i < (n); ++i)
    static inline void img2col(const ${specifier}* __restrict src, ${specifier}* __restrict dst, 
             const int OW, const int IC, const int IH,
             const int IW, const int FH, const int FW, const int SH, const int SW, const int cur_index,
             const int block_size) {
        int start_h = cur_index / OW;
        int cur_remain_w = cur_index % OW;
        int end_h = (cur_index + block_size) / OW;
        int end_remain_w = (cur_index + block_size) % OW;
        bool is_xcorr = true;
        bool same_line = false;
        if (start_h == end_h) {
            same_line = true;
        }
        int i = 0;
        if (same_line) {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {

                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            dst[i++] = src[ic * IH * IW + (start_h + fh) * IW +
                                        (w + fw)];
                        }
                    }
                }
            }
        } else {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        
                        for (int w = cur_remain_w; w < OW; w++) {
                            dst[i++] = src[ic * IH * IW + (start_h + fh) * IW +
                                        (w + fw)];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                dst[i++] = src[ic * IH * IW + (h + fh) * IW +
                                            (ow + fw)];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            dst[i++] = src[ic * IH * IW + (end_h + fh) * IW +
                                        (w + fw)];
                        }
                    }
                }
            }
        }
    }
)";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", Utils::cvt_dtype_specifier(dtype))
            .render(nchw_im2col_s1_kern);
}

std::string gen_nchw_pad_src_kern(TContext* ctx) {
    std::string dtype = ctx->getAttrStr("dtype");
    std::string nchw_pad_src_kern = R"(
    static inline void pad_src(${specifier}* inptr, ${specifier}* outptr, int ic, int ih, int iw, int pad_h, int pad_w){
        size_t out_idx = 0;
        int paded_iw = iw + 2 * pad_w;
        int nr_pad_top_bottom = pad_h * paded_iw;

        for(int ic_idx = 0; ic_idx < ic; ic_idx++){
            memset(outptr, 0, sizeof(${specifier}) * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            
            for (int ih_idx = 0; ih_idx < ih; ++ih_idx){
                for(int i = 0; i < pad_w; ++i){
                    *outptr++ = 0;
                }
                memcpy(outptr, inptr + ih_idx * iw, sizeof(${specifier}) * iw);
                outptr += iw;
                for(int i = 0; i < pad_w; ++i){
                    *outptr++ = 0;
                }
            }
            memset(outptr, 0, sizeof(${specifier}) * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            inptr += ih * iw;
        }
    }
)";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", Utils::cvt_dtype_specifier(dtype))
            .render(nchw_pad_src_kern);
}

}  // namespace
}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc