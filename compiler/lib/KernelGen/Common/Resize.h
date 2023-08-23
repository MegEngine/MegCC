#pragma once
#include <string>
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {

class ResizeHelper {
public:
    static std::string GenCoordHelper(
            const std::string& imode, const std::string& specifier) {
        CC_ASSERT(imode == "LINEAR");
        CC_ASSERT(specifier == "float");
        std::string body = R"(
        #include <math.h>
        static inline void get_coord(float scale, int size, int idx, float* ah0, int* ih0, float* ah1, int* ih1){
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
    )";
        return body;
    }
    static std::string GenNormImpl(const std::string& format) {
        std::string ret;
        ret = R"(
            {
                alignas(16) float ah0_cache[OH];
                alignas(16) int ih0_cache[OH];
                alignas(16) float ah1_cache[OH];
                alignas(16) int ih1_cache[OH];
                alignas(16) float aw0_cache[OW];
                alignas(16) int iw0_cache[OW];
                alignas(16) float aw1_cache[OW];
                alignas(16) int iw1_cache[OW];

                rep(oh, OH) {
                    get_coord(scale_h, IH, oh, &ah0_cache[oh], &ih0_cache[oh], &ah1_cache[oh], &ih1_cache[oh]);
                }
                rep(ow, OW) {
                    get_coord(scale_w, IW, ow, &aw0_cache[ow], &iw0_cache[ow], &aw1_cache[ow], &iw1_cache[ow]);
                }
                int oc_stride = OH * OW;
                int out_batch_stride = C * oc_stride;
                int ic_stride = IH * IW;
                int in_batch_stride = C * ic_stride;
                
                rep(n, N){
                    rep(c, C) {
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
                                dptr[get_offset(oh, ow, c, OH, OW, C)] = output_converter(
                                        sptr[get_offset(ih0, iw0, c, IH, IW, C)] * ah0 * aw0 +
                                        sptr[get_offset(ih0, iw1, c, IH, IW, C)] * ah0 * aw1 +
                                        sptr[get_offset(ih1, iw0, c, IH, IW, C)] * ah1 * aw0 +
                                        sptr[get_offset(ih1, iw1, c, IH, IW, C)] * ah1 * aw1);
                                
                            }
                        }
                    }
                    sptr += in_batch_stride;
                    dptr += out_batch_stride;
                }
            }
            )";

        return ret;
    }

    static std::string GenNearestImpl() {
        std::string ret;
        ret = R"(
            rep(n, N) {
                rep(oh, OH) rep(ow, OW) {
                    int ih = MIN((int)(oh / scale_h), IH - 1);
                    int iw = MIN((int)(ow / scale_w), IW - 1);

                    rep(c, C) {
                        dptr[c * OH * OW + oh * OW + ow] =
                                sptr[c * S_IC + ih * S_IH + iw * S_IW];
                    }
                }
                sptr += S_IN;
                dptr += C * OH * OW;
            }
        )";
        return ret;
    }

    static std::string GenLayoutDims(const std::string& format) {
        std::string ret;
        if (format == "NCHW") {
            ret = R"(
                int N = src_layout.dims[0];
                int C = src_layout.dims[1];
                int IH = src_layout.dims[2];
                int IW = src_layout.dims[3];
                int OH = dst_layout.dims[2];
                int OW = dst_layout.dims[3];
                int S_IN = src_layout.stride[0];
                int S_IC = src_layout.stride[1];
                int S_IH = src_layout.stride[2];
                int S_IW = src_layout.stride[3];
            )";
        } else {
            CC_ASSERT(format == "NCHW44");
            ret = R"(
                int N = src_layout.dims[0];
                int C = src_layout.dims[1]*4;
                int IH = src_layout.dims[2];
                int IW = src_layout.dims[3];
                int OH = dst_layout.dims[2];
                int OW = dst_layout.dims[3];
            )";
        }
        return ret;
    }
    static std::string GenGetOffset(const std::string& format) {
        std::string ret;
        if (format == "NCHW") {
            ret = R"(
                static inline size_t get_offset(size_t h, size_t w, size_t c, size_t H, size_t W,
                               size_t C){
                    return c * H * W + h * W + w;
                }
            )";
        } else {
            CC_ASSERT(format == "NCHW44");
            ret = R"(
                static inline size_t get_offset(size_t h, size_t w, size_t c, size_t H, size_t W,
                               size_t C){
                    return (((c >> 2) * H * W + h * W + w) << 2) + (c & 3);
                }
            )";
        }
        return ret;
    }
};

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
