#include <float.h>
#include <sstream>

#include "Flip.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool FlipKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    return dtype_ok;
}

//! kernel gen
std::string FlipKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_flip_" << src_dtype;
    return ss.str();
}

std::string FlipKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, bool vertical, bool "
           "horizontal)";
}

std::string FlipKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include <arm_neon.h>
        #include <string.h>
        #include "tinycv_c.h"
        #define rep(i, n) for (size_t i = 0; i < (n); ++i)

        static void flip_horizontal_on_channel_c1(uint8_t* dst, const uint8_t* src, size_t W) {
            size_t j = 0;
            for (; j + 15 < W; j += 16) {
                    uint8x16_t src_vec = vld1q_u8(src + j);
                    uint8x16_t dst_vec = vrev64q_u8(src_vec);
                    dst_vec = vcombine_u8(vget_high_u8(dst_vec), vget_low_u8(dst_vec));
                    vst1q_u8((dst + W - j - 16), dst_vec);
            }
            for (; j < W; ++j) {
                    *(dst + W - j - 1) = *(src + j);
            }
        }
         static void flip_horizontal_on_channel_c3(uint8_t* dst, const uint8_t* src, size_t W) {
            size_t j = 0;
            for (; j + 15 < W; j += 16) {
                uint8x16x3_t src_vec = vld3q_u8(src + j * 3);
                uint8x16x3_t dst_vec;
                uint8x16_t rev_src0 = vrev64q_u8(src_vec.val[0]);
                dst_vec.val[0] =
                            vcombine_u8(vget_high_u8(rev_src0), vget_low_u8(rev_src0));

                uint8x16_t rev_src1 = vrev64q_u8(src_vec.val[1]);
                dst_vec.val[1] =
                        vcombine_u8(vget_high_u8(rev_src1), vget_low_u8(rev_src1));

                uint8x16_t rev_src2 = vrev64q_u8(src_vec.val[2]);
                dst_vec.val[2] =
                        vcombine_u8(vget_high_u8(rev_src2), vget_low_u8(rev_src2));
                vst3q_u8((dst + W * 3 - 3 * j - 48), dst_vec);
            }
            for (; j < W; ++j) {
                *(dst + W * 3 - j * 3 - 3) = *(src + j * 3);
                *(dst + W * 3 - j * 3 - 2) = *(src + j * 3 + 1);
                *(dst + W * 3 - j * 3 - 1) = *(src + j * 3 + 2);
                
            }
        }

        static void flip_horizontal_naive(uint8_t* dst, const uint8_t* src, size_t W,
                                size_t C) {
            size_t sc = 0;
            size_t dc = W * C;
            for (; sc + 8 * C <= W * C; sc += 8 * C, dc -= 8 * C) {
                rep(ic, C) dst[dc - 1 * C + ic] = src[sc + 0 * C + ic];
                rep(ic, C) dst[dc - 2 * C + ic] = src[sc + 1 * C + ic];
                rep(ic, C) dst[dc - 3 * C + ic] = src[sc + 2 * C + ic];
                rep(ic, C) dst[dc - 4 * C + ic] = src[sc + 3 * C + ic];
                rep(ic, C) dst[dc - 5 * C + ic] = src[sc + 4 * C + ic];
                rep(ic, C) dst[dc - 6 * C + ic] = src[sc + 5 * C + ic];
                rep(ic, C) dst[dc - 7 * C + ic] = src[sc + 6 * C + ic];
                rep(ic, C) dst[dc - 8 * C + ic] = src[sc + 7 * C + ic];
            }
            for (; sc < W * C; sc += C, dc -= C) {
                rep(ic, C) dst[dc - C + ic] = src[sc + ic];
            }
        }

        void ${kernel_sig} {
            uint8_t* src_base_ptr = src->data;
            uint8_t* dst_base_ptr = dst->data;
            size_t H = src->rows;
            size_t W = src->cols;
            size_t C = src->channels;
            size_t src_step = W * C;
            size_t dst_step = W * C;
            for (size_t sr = 0; sr < H; ++sr) {

                const uint8_t* sptr = src_base_ptr + sr * src_step;
                size_t dr = (vertical ? H - sr - 1 : sr);
                uint8_t* dptr = dst_base_ptr + dr * dst_step;

                if (!horizontal) {
                    memcpy(dptr, sptr, sizeof(uint8_t) * W * C);
                } else {
                    size_t sc = 0;
                    size_t dc = W * C;
                    if (C == 1) {
                        flip_horizontal_on_channel_c1(dptr, sptr, W);
                    } else if (C == 3) {
                        flip_horizontal_on_channel_c3(dptr, sptr, W);
                    } else {
                        flip_horizontal_naive(dptr, sptr, W, C);
                    }
                }
            }
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
