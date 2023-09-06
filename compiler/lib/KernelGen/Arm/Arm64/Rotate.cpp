#include "Rotate.h"
#include <float.h>
#include <sstream>
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;

bool RotateKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "f16";
    return dtype_ok;
}

//! kernel gen
std::string RotateKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_rotate_" << src_dtype;
    return ss.str();
}

std::string RotateKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, bool clockwise)";
}

std::string RotateKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include <arm_neon.h>
        #include <string.h>
        #include "tinycv_c.h"
#if defined(__aarch64__)
static inline float64x2x4_t zip_f64_f32(float16x8x2_t rotate0, float16x8x2_t rotate1) {
    float32x4_t rotate0_32 = vreinterpretq_f32_f16(rotate0.val[0]);
    float32x4_t rotate1_32 = vreinterpretq_f32_f16(rotate0.val[1]);
    float32x4_t rotate2_32 = vreinterpretq_f32_f16(rotate1.val[0]);
    float32x4_t rotate3_32 = vreinterpretq_f32_f16(rotate1.val[1]);
    float32x4x2_t rotate00 = vzipq_f32(rotate0_32, rotate2_32);
    float32x4x2_t rotate10 = vzipq_f32(rotate1_32, rotate3_32);
    float64x2x4_t ans;
    ans.val[0] = vreinterpretq_f64_f32(rotate00.val[0]);
    ans.val[1] = vreinterpretq_f64_f32(rotate00.val[1]);
    ans.val[2] = vreinterpretq_f64_f32(rotate10.val[0]);
    ans.val[3] = vreinterpretq_f64_f32(rotate10.val[1]);
    return ans;
}


static void rotate_clockwise_fp16_8x8(float16_t* sptr, float16_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    float16_t* src = sptr + ih * W + iw;
    float16x8_t src0 = vld1q_f16(src + 0 * W);
    float16x8_t src1 = vld1q_f16(src + 1 * W);
    float16x8_t src2 = vld1q_f16(src + 2 * W);
    float16x8_t src3 = vld1q_f16(src + 3 * W);
    float16x8_t src4 = vld1q_f16(src + 4 * W);
    float16x8_t src5 = vld1q_f16(src + 5 * W);
    float16x8_t src6 = vld1q_f16(src + 6 * W);
    float16x8_t src7 = vld1q_f16(src + 7 * W);

    float16x8x2_t rotate3 = vzipq_f16(src1, src0);
    float16x8x2_t rotate2 = vzipq_f16(src3, src2);
    float16x8x2_t rotate1 = vzipq_f16(src5, src4);
    float16x8x2_t rotate0 = vzipq_f16(src7, src6);

    float64x2x4_t dstA = zip_f64_f32(rotate0, rotate1);
    float64x2x4_t dstB = zip_f64_f32(rotate2, rotate3);

    float64x2_t dst00 = vzip1q_f64(dstA.val[0], dstB.val[0]);
    float64x2_t dst01 = vzip2q_f64(dstA.val[0], dstB.val[0]);

    float64x2_t dst10 = vzip1q_f64(dstA.val[1], dstB.val[1]);
    float64x2_t dst11 = vzip2q_f64(dstA.val[1], dstB.val[1]);

    float64x2_t dst20 = vzip1q_f64(dstA.val[2], dstB.val[2]);
    float64x2_t dst21 = vzip2q_f64(dstA.val[2], dstB.val[2]);

    float64x2_t dst30 = vzip1q_f64(dstA.val[3], dstB.val[3]);
    float64x2_t dst31 = vzip2q_f64(dstA.val[3], dstB.val[3]);

    float16_t* dst = dptr + iw * H + H - ih - 8;

    vst1q_f64((float64_t *) (dst + 0 * H), dst00);
    vst1q_f64((float64_t *) (dst + 1 * H), dst01);
    vst1q_f64((float64_t *) (dst + 2 * H), dst10);
    vst1q_f64((float64_t *) (dst + 3 * H), dst11);
    vst1q_f64((float64_t *) (dst + 4 * H), dst20);
    vst1q_f64((float64_t *) (dst + 5 * H), dst21);
    vst1q_f64((float64_t *) (dst + 6 * H), dst30);
    vst1q_f64((float64_t *) (dst + 7 * H), dst31);

}


static void rotate_countclockwise_fp16_8x8(float16_t* sptr, float16_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    float16_t* src = sptr + ih * W + iw;
    float16x8_t src0 = vld1q_f16(src + 0 * W);
    float16x8_t src1 = vld1q_f16(src + 1 * W);
    float16x8_t src2 = vld1q_f16(src + 2 * W);
    float16x8_t src3 = vld1q_f16(src + 3 * W);
    float16x8_t src4 = vld1q_f16(src + 4 * W);
    float16x8_t src5 = vld1q_f16(src + 5 * W);
    float16x8_t src6 = vld1q_f16(src + 6 * W);
    float16x8_t src7 = vld1q_f16(src + 7 * W);

    float16x8x2_t rotate0 = vzipq_f16(src0, src1);
    float16x8x2_t rotate1 = vzipq_f16(src2, src3);
    float16x8x2_t rotate2 = vzipq_f16(src4, src5);
    float16x8x2_t rotate3 = vzipq_f16(src6, src7);

    float64x2x4_t dstA = zip_f64_f32(rotate0, rotate1);
    float64x2x4_t dstB = zip_f64_f32(rotate2, rotate3);

    float64x2_t dst00 = vzip1q_f64(dstA.val[0], dstB.val[0]);
    float64x2_t dst01 = vzip2q_f64(dstA.val[0], dstB.val[0]);

    float64x2_t dst10 = vzip1q_f64(dstA.val[1], dstB.val[1]);
    float64x2_t dst11 = vzip2q_f64(dstA.val[1], dstB.val[1]);

    float64x2_t dst20 = vzip1q_f64(dstA.val[2], dstB.val[2]);
    float64x2_t dst21 = vzip2q_f64(dstA.val[2], dstB.val[2]);

    float64x2_t dst30 = vzip1q_f64(dstA.val[3], dstB.val[3]);
    float64x2_t dst31 = vzip2q_f64(dstA.val[3], dstB.val[3]);

    float16_t* dst = dptr + (W - iw - 8) * H + ih;

    vst1q_f64((float64_t *) (dst + 0 * H), dst31);
    vst1q_f64((float64_t *) (dst + 1 * H), dst30);
    vst1q_f64((float64_t *) (dst + 2 * H), dst21);
    vst1q_f64((float64_t *) (dst + 3 * H), dst20);
    vst1q_f64((float64_t *) (dst + 4 * H), dst11);
    vst1q_f64((float64_t *) (dst + 5 * H), dst10);
    vst1q_f64((float64_t *) (dst + 6 * H), dst01);
    vst1q_f64((float64_t *) (dst + 7 * H), dst00);

}

static void rotate_clockwise_fp16x3_8x8(float16_t* sptr, float16_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    float16_t* src = sptr + ih * W*3 + iw*3;
    
    float16x8x3_t src0 = vld3q_f16(src + 0  * W);
    float16x8x3_t src1 = vld3q_f16(src + 3  * W);
    float16x8x3_t src2 = vld3q_f16(src + 6  * W);
    float16x8x3_t src3 = vld3q_f16(src + 9  * W);
    float16x8x3_t src4 = vld3q_f16(src + 12 * W);
    float16x8x3_t src5 = vld3q_f16(src + 15 * W);
    float16x8x3_t src6 = vld3q_f16(src + 18 * W);
    float16x8x3_t src7 = vld3q_f16(src + 21 * W);

    float64x2x3_t ans0, ans1, ans2, ans3, ans4, ans5, ans6, ans7;
    float16x8x3_t a0, a1, a2, a3, a4, a5, a6, a7;
    for(size_t idx = 0; idx <3; ++idx){
        float16x8x2_t rotate3 = vzipq_f16(src1.val[idx], src0.val[idx]);
        float16x8x2_t rotate2 = vzipq_f16(src3.val[idx], src2.val[idx]);
        float16x8x2_t rotate1 = vzipq_f16(src5.val[idx], src4.val[idx]);
        float16x8x2_t rotate0 = vzipq_f16(src7.val[idx], src6.val[idx]);

        float64x2x4_t dstA = zip_f64_f32(rotate0, rotate1);
        float64x2x4_t dstB = zip_f64_f32(rotate2, rotate3);

        float64x2_t dst00 = vzip1q_f64(dstA.val[0], dstB.val[0]);
        float64x2_t dst01 = vzip2q_f64(dstA.val[0], dstB.val[0]);

        float64x2_t dst10 = vzip1q_f64(dstA.val[1], dstB.val[1]);
        float64x2_t dst11 = vzip2q_f64(dstA.val[1], dstB.val[1]);

        float64x2_t dst20 = vzip1q_f64(dstA.val[2], dstB.val[2]);
        float64x2_t dst21 = vzip2q_f64(dstA.val[2], dstB.val[2]);

        float64x2_t dst30 = vzip1q_f64(dstA.val[3], dstB.val[3]);
        float64x2_t dst31 = vzip2q_f64(dstA.val[3], dstB.val[3]); 
        a0.val[idx] = vreinterpretq_f16_f64(dst00);
        a1.val[idx] = vreinterpretq_f16_f64(dst01);
        a2.val[idx] = vreinterpretq_f16_f64(dst10);
        a3.val[idx] = vreinterpretq_f16_f64(dst11);
        a4.val[idx] = vreinterpretq_f16_f64(dst20);
        a5.val[idx] = vreinterpretq_f16_f64(dst21);
        a6.val[idx] = vreinterpretq_f16_f64(dst30);
        a7.val[idx] = vreinterpretq_f16_f64(dst31);
    }
    float16_t* dst = dptr + iw * H*3 + (H - ih - 8)*3;

    vst3q_f16((float16_t *)(dst + 0  * H), a0);
    vst3q_f16((float16_t *)(dst + 3  * H), a1);
    vst3q_f16((float16_t *)(dst + 6  * H), a2);
    vst3q_f16((float16_t *)(dst + 9  * H), a3);
    vst3q_f16((float16_t *)(dst + 12 * H), a4);
    vst3q_f16((float16_t *)(dst + 15 * H), a5);
    vst3q_f16((float16_t *)(dst + 18 * H), a6);
    vst3q_f16((float16_t *)(dst + 21 * H), a7);
}

static void rotate_countclockwise_fp16x3_8x8(float16_t* sptr, float16_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    float16_t* src = sptr + ih * W*3 + iw*3;

    float16x8x3_t src0 = vld3q_f16(src + 0  * W);
    float16x8x3_t src1 = vld3q_f16(src + 3  * W);
    float16x8x3_t src2 = vld3q_f16(src + 6  * W);
    float16x8x3_t src3 = vld3q_f16(src + 9  * W);
    float16x8x3_t src4 = vld3q_f16(src + 12 * W);
    float16x8x3_t src5 = vld3q_f16(src + 15 * W);
    float16x8x3_t src6 = vld3q_f16(src + 18 * W);
    float16x8x3_t src7 = vld3q_f16(src + 21 * W); 

    float16x8x3_t a0, a1, a2, a3, a4, a5, a6, a7;
    for(size_t idx = 0; idx <3; ++idx){ 
        float16x8x2_t rotate0 = vzipq_f16(src0.val[idx], src1.val[idx]);
        float16x8x2_t rotate1 = vzipq_f16(src2.val[idx], src3.val[idx]);
        float16x8x2_t rotate2 = vzipq_f16(src4.val[idx], src5.val[idx]);
        float16x8x2_t rotate3 = vzipq_f16(src6.val[idx], src7.val[idx]);

        float64x2x4_t dstA = zip_f64_f32(rotate0, rotate1);
        float64x2x4_t dstB = zip_f64_f32(rotate2, rotate3);

        float64x2_t dst00 = vzip1q_f64(dstA.val[0], dstB.val[0]);
        float64x2_t dst01 = vzip2q_f64(dstA.val[0], dstB.val[0]);

        float64x2_t dst10 = vzip1q_f64(dstA.val[1], dstB.val[1]);
        float64x2_t dst11 = vzip2q_f64(dstA.val[1], dstB.val[1]);

        float64x2_t dst20 = vzip1q_f64(dstA.val[2], dstB.val[2]);
        float64x2_t dst21 = vzip2q_f64(dstA.val[2], dstB.val[2]);

        float64x2_t dst30 = vzip1q_f64(dstA.val[3], dstB.val[3]);
        float64x2_t dst31 = vzip2q_f64(dstA.val[3], dstB.val[3]); 

        a0.val[idx] = vreinterpretq_f16_f64(dst00);
        a1.val[idx] = vreinterpretq_f16_f64(dst01);
        a2.val[idx] = vreinterpretq_f16_f64(dst10);
        a3.val[idx] = vreinterpretq_f16_f64(dst11);
        a4.val[idx] = vreinterpretq_f16_f64(dst20);
        a5.val[idx] = vreinterpretq_f16_f64(dst21);
        a6.val[idx] = vreinterpretq_f16_f64(dst30);
        a7.val[idx] = vreinterpretq_f16_f64(dst31);
    }
    float16_t* dst = dptr + (W - iw - 8) * H*3 + ih*3;
    vst3q_f16((float16_t *)(dst + 0  * H), a7);
    vst3q_f16((float16_t *)(dst + 3  * H), a6);
    vst3q_f16((float16_t *)(dst + 6  * H), a5);
    vst3q_f16((float16_t *)(dst + 9  * H), a4);
    vst3q_f16((float16_t *)(dst + 12 * H), a3);
    vst3q_f16((float16_t *)(dst + 15 * H), a2);
    vst3q_f16((float16_t *)(dst + 18 * H), a1);
    vst3q_f16((float16_t *)(dst + 21 * H), a0); 
}

#endif
        static void rotate_pixel(float16_t* sptr, float16_t* dptr, size_t ih, size_t iw, size_t IH, size_t IW, size_t C, bool clockwise){
            size_t ow, oh;
            if(clockwise){
                ow = IH - ih - 1;
                oh = iw;  
            }else{
                ow = ih;
                oh = IW - iw - 1;
            }

            if(C == 1){
                dptr[oh * IH + ow] = sptr[ih * IW + iw];
            }else if(C == 3){
                size_t dst_offset = oh * IH * 3 + ow * 3;
                size_t src_offset = ih * IW * 3 + iw * 3;
                dptr[dst_offset + 0] = sptr[src_offset + 0];
                dptr[dst_offset + 1] = sptr[src_offset + 1];
                dptr[dst_offset + 2] = sptr[src_offset + 2];
            }else{
                size_t dst_offset = oh * IH * C + ow * C;
                size_t src_offset = ih * IW * C + iw * C;
                for (size_t ic = 0; ic < C; ++ic) {
                    dptr[dst_offset + ic] = sptr[src_offset + ic];
                }
            }
        }

        static void rotate_clockwise(float16_t* sptr, float16_t* dptr, size_t IH, size_t IW, size_t C) {
            size_t ih = 0, OH = IW, OW = IH;
            if(C == 1){
                for (; ih + 7 < IH; ih += 8) {
                    size_t iw = 0;
                    for (; iw + 7 < IW; iw += 8) {
                        rotate_clockwise_fp16_8x8(sptr, dptr,ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0; i < 8; ++i){
                           rotate_pixel(sptr, dptr, ih+i, iw, IH, IW, 1, true);
                        }
                    }
                }
                for (; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                       rotate_pixel(sptr, dptr, ih, iw, IH, IW, 1, true);
                    }
                }
            }
#if defined(__aarch64__)
            else if( C == 3){  
                for (; ih + 7 < IH; ih += 8) {
                    size_t iw = 0;
                    for (; iw + 7 < IW; iw += 8) {
                        rotate_clockwise_fp16x3_8x8(sptr, dptr,ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0;i<8;++i){
                           rotate_pixel(sptr, dptr, ih+i, iw, IH, IW, 3, true);
                        }
                    }
                }
                for (; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                       rotate_pixel(sptr, dptr, ih, iw, IH, IW, 3, true);
                    }
                }
            }
#endif            
            else{
                for (size_t ih = 0; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                        rotate_pixel(sptr, dptr, ih, iw, IH, IW, C, true);
                    }
                }
            }
        }

        static void rotate_countclockwise(float16_t* sptr, float16_t* dptr, size_t IH, size_t IW,
                                size_t C) {
            size_t ih = 0, OH = IW, OW = IH;
            if(C == 1){
                for (; ih + 7 < IH; ih += 8) {
                    size_t iw = 0;
                    for (; iw + 7 < IW; iw += 8) {
                            rotate_countclockwise_fp16_8x8(sptr, dptr, ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0;i<8;++i){
                            rotate_pixel(sptr, dptr, ih + i, iw, IH, IW, 1, false);
                        }
                    }
                }

                for (; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                       rotate_pixel(sptr, dptr, ih, iw, IH, IW, 1, false);
                    }
                }
            }
#if defined(__aarch64__)
            else if( C == 3){
                for (; ih + 7 < IH; ih += 8) {
                    size_t iw = 0;
                    for (; iw + 7 < IW; iw += 8) {
                        rotate_countclockwise_fp16x3_8x8(sptr, dptr,ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0;i<8;++i){
                            rotate_pixel(sptr, dptr, ih + i, iw, IH, IW, 3, false);
                        }
                    }
                }
                for (; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                      rotate_pixel(sptr, dptr, ih, iw, IH, IW, 3, false);
                    }
                }

            }
#endif            
            else{
               for (size_t ih = 0; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                       rotate_pixel(sptr, dptr, ih, iw, IH, IW, C, false);
                    }
                }
            }
        }

        void ${kernel_sig}{
            float16_t * sptr = src->data;
            float16_t * dptr = dst->data;
            size_t IH = src->rows;
            size_t IW = src->cols;
            size_t C = src->channels;
            if(clockwise){
                rotate_clockwise(sptr, dptr, IH, IW, C);
            }
            else{
                rotate_countclockwise(sptr, dptr, IH, IW, C);
            }
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}
// vim: syntax=cpp.doxygen
