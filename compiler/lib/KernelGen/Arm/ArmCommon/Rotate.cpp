/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Rotate.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Rotate.h"
#include <float.h>
#include <sstream>
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool RotateKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "ui8";
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
static inline uint32x4x4_t zip_u32_u16(uint8x16x2_t rotate0, uint8x16x2_t rotate1) {
    uint16x8_t rotate0_16 = vreinterpretq_u16_u8(rotate0.val[0]);
    uint16x8_t rotate1_16 = vreinterpretq_u16_u8(rotate0.val[1]);
    uint16x8_t rotate2_16 = vreinterpretq_u16_u8(rotate1.val[0]);
    uint16x8_t rotate3_16 = vreinterpretq_u16_u8(rotate1.val[1]);
    uint16x8x2_t rotate00 = vzipq_u16(rotate0_16, rotate2_16);
    uint16x8x2_t rotate10 = vzipq_u16(rotate1_16, rotate3_16);
    uint32x4x4_t ans;
    ans.val[0] = vreinterpretq_u32_u16(rotate00.val[0]);
    ans.val[1] = vreinterpretq_u32_u16(rotate00.val[1]);
    ans.val[2] = vreinterpretq_u32_u16(rotate10.val[0]);
    ans.val[3] = vreinterpretq_u32_u16(rotate10.val[1]);
    return ans;
}

static inline uint64x2x4_t zip_u64_u32(uint32x4_t rotate0, uint32x4_t rotate1, uint32x4_t rotate2, uint32x4_t rotate3) {
    uint32x4x2_t rotate01, rotate23;
    rotate01.val[0] = vzip1q_u32(rotate0, rotate1);
    rotate01.val[1] = vzip2q_u32(rotate0, rotate1);
    rotate23.val[0] = vzip1q_u32(rotate2, rotate3);
    rotate23.val[1] = vzip2q_u32(rotate2, rotate3);
    
    uint64x2_t dst_0 = vreinterpretq_u64_u32(rotate01.val[0]);
    uint64x2_t dst_1 = vreinterpretq_u64_u32(rotate01.val[1]);
    uint64x2_t dst_2 = vreinterpretq_u64_u32(rotate23.val[0]);
    uint64x2_t dst_3 = vreinterpretq_u64_u32(rotate23.val[1]);
    uint64x2x4_t ans;
    ans.val[0] = vzip1q_u64(dst_0, dst_2);
    ans.val[1] = vzip2q_u64(dst_0, dst_2);
    ans.val[2] = vzip1q_u64(dst_1, dst_3);
    ans.val[3] = vzip2q_u64(dst_1, dst_3);
    return ans;
}
        
static void rotate_clockwise_u8_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    uint8_t* src = sptr + ih * W + iw;
    uint8x16_t src0 = vld1q_u8(src + 0  * W);  
    uint8x16_t src1 = vld1q_u8(src + 1  * W);  
    uint8x16_t src2 = vld1q_u8(src + 2  * W);  
    uint8x16_t src3 = vld1q_u8(src + 3  * W);  
    uint8x16_t src4 = vld1q_u8(src + 4  * W);  
    uint8x16_t src5 = vld1q_u8(src + 5  * W);  
    uint8x16_t src6 = vld1q_u8(src + 6  * W);  
    uint8x16_t src7 = vld1q_u8(src + 7  * W);   
    uint8x16_t src8 = vld1q_u8(src + 8  * W);  
    uint8x16_t src9 = vld1q_u8(src + 9  * W);  
    uint8x16_t srcA = vld1q_u8(src + 10 * W); 
    uint8x16_t srcB = vld1q_u8(src + 11 * W); 
    uint8x16_t srcC = vld1q_u8(src + 12 * W); 
    uint8x16_t srcD = vld1q_u8(src + 13 * W); 
    uint8x16_t srcE = vld1q_u8(src + 14 * W); 
    uint8x16_t srcF = vld1q_u8(src + 15 * W); 
    
    uint8x16x2_t rotate7 = vzipq_u8(src1, src0);
    uint8x16x2_t rotate6 = vzipq_u8(src3, src2);
    uint8x16x2_t rotate5 = vzipq_u8(src5, src4);
    uint8x16x2_t rotate4 = vzipq_u8(src7, src6);  

    uint8x16x2_t rotate3 = vzipq_u8(src9, src8);
    uint8x16x2_t rotate2 = vzipq_u8(srcB, srcA);
    uint8x16x2_t rotate1 = vzipq_u8(srcD, srcC); 
    uint8x16x2_t rotate0 = vzipq_u8(srcF, srcE);  
    
    uint32x4x4_t dstA = zip_u32_u16(rotate0, rotate1);
    uint32x4x4_t dstB = zip_u32_u16(rotate2, rotate3);
    uint32x4x4_t dstC = zip_u32_u16(rotate4, rotate5);
    uint32x4x4_t dstD = zip_u32_u16(rotate6, rotate7);

    uint64x2x4_t dst0 = zip_u64_u32(dstA.val[0], dstB.val[0], dstC.val[0],dstD.val[0]);
    uint64x2x4_t dst1 = zip_u64_u32(dstA.val[1], dstB.val[1], dstC.val[1],dstD.val[1]);
    uint64x2x4_t dst2 = zip_u64_u32(dstA.val[2], dstB.val[2], dstC.val[2],dstD.val[2]);
    uint64x2x4_t dst3 = zip_u64_u32(dstA.val[3], dstB.val[3], dstC.val[3],dstD.val[3]);

    uint8_t* dst = dptr + iw * H + H - ih -16;
    vst1q_u64((uint64_t*)(dst + 0  * H), dst0.val[0]);
    vst1q_u64((uint64_t*)(dst + 1  * H), dst0.val[1]);
    vst1q_u64((uint64_t*)(dst + 2  * H), dst0.val[2]);
    vst1q_u64((uint64_t*)(dst + 3  * H), dst0.val[3]);
    vst1q_u64((uint64_t*)(dst + 4  * H), dst1.val[0]);
    vst1q_u64((uint64_t*)(dst + 5  * H), dst1.val[1]);
    vst1q_u64((uint64_t*)(dst + 6  * H), dst1.val[2]);
    vst1q_u64((uint64_t*)(dst + 7  * H), dst1.val[3]);
    vst1q_u64((uint64_t*)(dst + 8  * H), dst2.val[0]);
    vst1q_u64((uint64_t*)(dst + 9  * H), dst2.val[1]);
    vst1q_u64((uint64_t*)(dst + 10 * H), dst2.val[2]);
    vst1q_u64((uint64_t*)(dst + 11 * H), dst2.val[3]);
    vst1q_u64((uint64_t*)(dst + 12 * H), dst3.val[0]);
    vst1q_u64((uint64_t*)(dst + 13 * H), dst3.val[1]);
    vst1q_u64((uint64_t*)(dst + 14 * H), dst3.val[2]);
    vst1q_u64((uint64_t*)(dst + 15 * H), dst3.val[3]);
}

static void rotate_countclockwise_u8_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw,
                                size_t H, size_t W) {
    uint8_t* src = sptr + ih * W + iw;
    uint8x16_t src0 = vld1q_u8(src + 0  * W); 
    uint8x16_t src1 = vld1q_u8(src + 1  * W);  
    uint8x16_t src2 = vld1q_u8(src + 2  * W);
    uint8x16_t src3 = vld1q_u8(src + 3  * W);
    uint8x16_t src4 = vld1q_u8(src + 4  * W);
    uint8x16_t src5 = vld1q_u8(src + 5  * W);
    uint8x16_t src6 = vld1q_u8(src + 6  * W);
    uint8x16_t src7 = vld1q_u8(src + 7  * W); 
    uint8x16_t src8 = vld1q_u8(src + 8  * W);
    uint8x16_t src9 = vld1q_u8(src + 9  * W);
    uint8x16_t srcA = vld1q_u8(src + 10 * W);
    uint8x16_t srcB = vld1q_u8(src + 11 * W);
    uint8x16_t srcC = vld1q_u8(src + 12 * W);
    uint8x16_t srcD = vld1q_u8(src + 13 * W);
    uint8x16_t srcE = vld1q_u8(src + 14 * W);
    uint8x16_t srcF = vld1q_u8(src + 15 * W);

    uint8x16x2_t rotate0 = vzipq_u8(src0, src1); 
    uint8x16x2_t rotate1 = vzipq_u8(src2, src3); 
    uint8x16x2_t rotate2 = vzipq_u8(src4, src5); 
    uint8x16x2_t rotate3 = vzipq_u8(src6, src7); 
    uint8x16x2_t rotate4 = vzipq_u8(src8, src9); 
    uint8x16x2_t rotate5 = vzipq_u8(srcA, srcB); 
    uint8x16x2_t rotate6 = vzipq_u8(srcC, srcD); 
    uint8x16x2_t rotate7 = vzipq_u8(srcE, srcF); 

    uint32x4x4_t dstA = zip_u32_u16(rotate0, rotate1);
    uint32x4x4_t dstB = zip_u32_u16(rotate2, rotate3);
    uint32x4x4_t dstC = zip_u32_u16(rotate4, rotate5);
    uint32x4x4_t dstD = zip_u32_u16(rotate6, rotate7);

    uint64x2x4_t dst0 = zip_u64_u32(dstA.val[0], dstB.val[0], dstC.val[0],dstD.val[0]);
    uint64x2x4_t dst1 = zip_u64_u32(dstA.val[1], dstB.val[1], dstC.val[1],dstD.val[1]);
    uint64x2x4_t dst2 = zip_u64_u32(dstA.val[2], dstB.val[2], dstC.val[2],dstD.val[2]);
    uint64x2x4_t dst3 = zip_u64_u32(dstA.val[3], dstB.val[3], dstC.val[3],dstD.val[3]);

    uint8_t* dst = dptr + (W - iw - 16) * H + ih;
    vst1q_u64((uint64_t*)(dst + 0  * H), dst3.val[3]);
    vst1q_u64((uint64_t*)(dst + 1  * H), dst3.val[2]);
    vst1q_u64((uint64_t*)(dst + 2  * H), dst3.val[1]);
    vst1q_u64((uint64_t*)(dst + 3  * H), dst3.val[0]);
    vst1q_u64((uint64_t*)(dst + 4  * H), dst2.val[3]);
    vst1q_u64((uint64_t*)(dst + 5  * H), dst2.val[2]);
    vst1q_u64((uint64_t*)(dst + 6  * H), dst2.val[1]);
    vst1q_u64((uint64_t*)(dst + 7  * H), dst2.val[0]);
    vst1q_u64((uint64_t*)(dst + 8  * H), dst1.val[3]);
    vst1q_u64((uint64_t*)(dst + 9  * H), dst1.val[2]);
    vst1q_u64((uint64_t*)(dst + 10 * H), dst1.val[1]);
    vst1q_u64((uint64_t*)(dst + 11 * H), dst1.val[0]);
    vst1q_u64((uint64_t*)(dst + 12 * H), dst0.val[3]);
    vst1q_u64((uint64_t*)(dst + 13 * H), dst0.val[2]);
    vst1q_u64((uint64_t*)(dst + 14 * H), dst0.val[1]);
    vst1q_u64((uint64_t*)(dst + 15 * H), dst0.val[0]); 
}

static void rotate_clockwise_u8x3_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw,
                            size_t H, size_t W) {
    uint8_t* src = sptr + ih * W*3 + iw*3;
    uint8x16x3_t src0 = vld3q_u8(src + 0  * W);
    uint8x16x3_t src1 = vld3q_u8(src + 3  * W);
    uint8x16x3_t src2 = vld3q_u8(src + 6  * W);
    uint8x16x3_t src3 = vld3q_u8(src + 9  * W);
    uint8x16x3_t src4 = vld3q_u8(src + 12 * W);
    uint8x16x3_t src5 = vld3q_u8(src + 15 * W);
    uint8x16x3_t src6 = vld3q_u8(src + 18 * W);
    uint8x16x3_t src7 = vld3q_u8(src + 21 * W);
    uint8x16x3_t src8 = vld3q_u8(src + 24 * W);
    uint8x16x3_t src9 = vld3q_u8(src + 27 * W);
    uint8x16x3_t srcA = vld3q_u8(src + 30 * W);
    uint8x16x3_t srcB = vld3q_u8(src + 33 * W);
    uint8x16x3_t srcC = vld3q_u8(src + 36 * W);
    uint8x16x3_t srcD = vld3q_u8(src + 39 * W);
    uint8x16x3_t srcE = vld3q_u8(src + 42 * W);
    uint8x16x3_t srcF = vld3q_u8(src + 45 * W);

    uint8x16x3_t ans0, ans1, ans2, ans3, ans4, ans5, ans6, ans7;
    uint8x16x3_t ans8, ans9, ansA, ansB, ansC, ansD, ansE, ansF;
    for(size_t idx = 0; idx <3; ++idx){

        uint8x16x2_t rotate7 = vzipq_u8(src1.val[idx], src0.val[idx]);
        uint8x16x2_t rotate6 = vzipq_u8(src3.val[idx], src2.val[idx]);
        uint8x16x2_t rotate5 = vzipq_u8(src5.val[idx], src4.val[idx]);
        uint8x16x2_t rotate4 = vzipq_u8(src7.val[idx], src6.val[idx]);  
        uint8x16x2_t rotate3 = vzipq_u8(src9.val[idx], src8.val[idx]);
        uint8x16x2_t rotate2 = vzipq_u8(srcB.val[idx], srcA.val[idx]);
        uint8x16x2_t rotate1 = vzipq_u8(srcD.val[idx], srcC.val[idx]); 
        uint8x16x2_t rotate0 = vzipq_u8(srcF.val[idx], srcE.val[idx]);  
        
        uint32x4x4_t dstA = zip_u32_u16(rotate0, rotate1);
        uint32x4x4_t dstB = zip_u32_u16(rotate2, rotate3);
        uint32x4x4_t dstC = zip_u32_u16(rotate4, rotate5);
        uint32x4x4_t dstD = zip_u32_u16(rotate6, rotate7);

        uint64x2x4_t dst0 = zip_u64_u32(dstA.val[0], dstB.val[0], dstC.val[0],dstD.val[0]);
        uint64x2x4_t dst1 = zip_u64_u32(dstA.val[1], dstB.val[1], dstC.val[1],dstD.val[1]);
        uint64x2x4_t dst2 = zip_u64_u32(dstA.val[2], dstB.val[2], dstC.val[2],dstD.val[2]);
        uint64x2x4_t dst3 = zip_u64_u32(dstA.val[3], dstB.val[3], dstC.val[3],dstD.val[3]);

        ans0.val[idx] = vreinterpretq_u8_u64(dst0.val[0]);
        ans1.val[idx] = vreinterpretq_u8_u64(dst0.val[1]);
        ans2.val[idx] = vreinterpretq_u8_u64(dst0.val[2]);
        ans3.val[idx] = vreinterpretq_u8_u64(dst0.val[3]);
        ans4.val[idx] = vreinterpretq_u8_u64(dst1.val[0]);
        ans5.val[idx] = vreinterpretq_u8_u64(dst1.val[1]);
        ans6.val[idx] = vreinterpretq_u8_u64(dst1.val[2]);
        ans7.val[idx] = vreinterpretq_u8_u64(dst1.val[3]);
        ans8.val[idx] = vreinterpretq_u8_u64(dst2.val[0]);
        ans9.val[idx] = vreinterpretq_u8_u64(dst2.val[1]);
        ansA.val[idx] = vreinterpretq_u8_u64(dst2.val[2]);
        ansB.val[idx] = vreinterpretq_u8_u64(dst2.val[3]);
        ansC.val[idx] = vreinterpretq_u8_u64(dst3.val[0]);
        ansD.val[idx] = vreinterpretq_u8_u64(dst3.val[1]);
        ansE.val[idx] = vreinterpretq_u8_u64(dst3.val[2]);
        ansF.val[idx] = vreinterpretq_u8_u64(dst3.val[3]);
    }

    uint8_t* dst = dptr + iw * H*3 + (H - ih - 16)*3;
    vst3q_u8(dst + 0  * H, ans0);
    vst3q_u8(dst + 3  * H, ans1);
    vst3q_u8(dst + 6  * H, ans2);
    vst3q_u8(dst + 9  * H, ans3);
    vst3q_u8(dst + 12 * H, ans4);
    vst3q_u8(dst + 15 * H, ans5);
    vst3q_u8(dst + 18 * H, ans6);
    vst3q_u8(dst + 21 * H, ans7);
    vst3q_u8(dst + 24 * H, ans8);
    vst3q_u8(dst + 27 * H, ans9);
    vst3q_u8(dst + 30 * H, ansA);
    vst3q_u8(dst + 33 * H, ansB);
    vst3q_u8(dst + 36 * H, ansC);
    vst3q_u8(dst + 39 * H, ansD);
    vst3q_u8(dst + 42 * H, ansE);
    vst3q_u8(dst + 45 * H, ansF);
}

static void rotate_countclockwise_u8x3_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw,
                                    size_t H, size_t W) {
    uint8_t* src = sptr + ih * W*3 + iw*3;
    uint8x16x3_t src0 = vld3q_u8(src + 0  * W);
    uint8x16x3_t src1 = vld3q_u8(src + 3  * W);
    uint8x16x3_t src2 = vld3q_u8(src + 6  * W);
    uint8x16x3_t src3 = vld3q_u8(src + 9  * W);
    uint8x16x3_t src4 = vld3q_u8(src + 12 * W);
    uint8x16x3_t src5 = vld3q_u8(src + 15 * W);
    uint8x16x3_t src6 = vld3q_u8(src + 18 * W);
    uint8x16x3_t src7 = vld3q_u8(src + 21 * W);
    uint8x16x3_t src8 = vld3q_u8(src + 24 * W);
    uint8x16x3_t src9 = vld3q_u8(src + 27 * W);
    uint8x16x3_t srcA = vld3q_u8(src + 30 * W);
    uint8x16x3_t srcB = vld3q_u8(src + 33 * W);
    uint8x16x3_t srcC = vld3q_u8(src + 36 * W);
    uint8x16x3_t srcD = vld3q_u8(src + 39 * W);
    uint8x16x3_t srcE = vld3q_u8(src + 42 * W);
    uint8x16x3_t srcF = vld3q_u8(src + 45 * W);

    uint8x16x3_t ans0, ans1, ans2, ans3, ans4, ans5, ans6, ans7;
    uint8x16x3_t ans8, ans9, ansA, ansB, ansC, ansD, ansE, ansF;
    for(size_t idx = 0; idx <3; ++idx){
        
        uint8x16x2_t rotate0 = vzipq_u8(src0.val[idx], src1.val[idx]);
        uint8x16x2_t rotate1 = vzipq_u8(src2.val[idx], src3.val[idx]);
        uint8x16x2_t rotate2 = vzipq_u8(src4.val[idx], src5.val[idx]);
        uint8x16x2_t rotate3 = vzipq_u8(src6.val[idx], src7.val[idx]);  
        uint8x16x2_t rotate4 = vzipq_u8(src8.val[idx], src9.val[idx]);
        uint8x16x2_t rotate5 = vzipq_u8(srcA.val[idx], srcB.val[idx]);
        uint8x16x2_t rotate6 = vzipq_u8(srcC.val[idx], srcD.val[idx]); 
        uint8x16x2_t rotate7 = vzipq_u8(srcE.val[idx], srcF.val[idx]);  
        
        uint32x4x4_t dstA = zip_u32_u16(rotate0, rotate1);
        uint32x4x4_t dstB = zip_u32_u16(rotate2, rotate3);
        uint32x4x4_t dstC = zip_u32_u16(rotate4, rotate5);
        uint32x4x4_t dstD = zip_u32_u16(rotate6, rotate7);

        uint64x2x4_t dst0 = zip_u64_u32(dstA.val[0], dstB.val[0], dstC.val[0],dstD.val[0]);
        uint64x2x4_t dst1 = zip_u64_u32(dstA.val[1], dstB.val[1], dstC.val[1],dstD.val[1]);
        uint64x2x4_t dst2 = zip_u64_u32(dstA.val[2], dstB.val[2], dstC.val[2],dstD.val[2]);
        uint64x2x4_t dst3 = zip_u64_u32(dstA.val[3], dstB.val[3], dstC.val[3],dstD.val[3]);

        ans0.val[idx] = vreinterpretq_u8_u64(dst0.val[0]);
        ans1.val[idx] = vreinterpretq_u8_u64(dst0.val[1]);
        ans2.val[idx] = vreinterpretq_u8_u64(dst0.val[2]);
        ans3.val[idx] = vreinterpretq_u8_u64(dst0.val[3]);
        ans4.val[idx] = vreinterpretq_u8_u64(dst1.val[0]);
        ans5.val[idx] = vreinterpretq_u8_u64(dst1.val[1]);
        ans6.val[idx] = vreinterpretq_u8_u64(dst1.val[2]);
        ans7.val[idx] = vreinterpretq_u8_u64(dst1.val[3]);
        ans8.val[idx] = vreinterpretq_u8_u64(dst2.val[0]);
        ans9.val[idx] = vreinterpretq_u8_u64(dst2.val[1]);
        ansA.val[idx] = vreinterpretq_u8_u64(dst2.val[2]);
        ansB.val[idx] = vreinterpretq_u8_u64(dst2.val[3]);
        ansC.val[idx] = vreinterpretq_u8_u64(dst3.val[0]);
        ansD.val[idx] = vreinterpretq_u8_u64(dst3.val[1]);
        ansE.val[idx] = vreinterpretq_u8_u64(dst3.val[2]);
        ansF.val[idx] = vreinterpretq_u8_u64(dst3.val[3]);
    }

    uint8_t* dst = dptr + (W - iw - 16) * H*3 + ih*3;
    vst3q_u8(dst + 0  * H, ansF);
    vst3q_u8(dst + 3  * H, ansE);
    vst3q_u8(dst + 6  * H, ansD);
    vst3q_u8(dst + 9  * H, ansC);
    vst3q_u8(dst + 12 * H, ansB);
    vst3q_u8(dst + 15 * H, ansA);
    vst3q_u8(dst + 18 * H, ans9);
    vst3q_u8(dst + 21 * H, ans8);
    vst3q_u8(dst + 24 * H, ans7);
    vst3q_u8(dst + 27 * H, ans6);
    vst3q_u8(dst + 30 * H, ans5);
    vst3q_u8(dst + 33 * H, ans4);
    vst3q_u8(dst + 36 * H, ans3);
    vst3q_u8(dst + 39 * H, ans2);
    vst3q_u8(dst + 42 * H, ans1);
    vst3q_u8(dst + 45 * H, ans0);
}
#else

static inline void rotate_clockwise_u8_16x16(
        const uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    const uint8_t* src = sptr + ih * W + iw;
    const uint8_t* dst = dptr + iw * H + (H - ih - 16);
    int src_step = W;
    int dst_step = H;
    asm volatile(
            "\n"
            "vld1.8 {d0, d1}, [%[src]], %[src_step] \n"
            "vld1.8 {d2, d3}, [%[src]], %[src_step] \n"
            "vld1.8 {d4, d5}, [%[src]], %[src_step] \n"
            "vld1.8 {d6, d7}, [%[src]], %[src_step] \n"
            "vld1.8 {d8, d9}, [%[src]], %[src_step] \n"
            "vld1.8 {d10, d11}, [%[src]], %[src_step] \n"
            "vld1.8 {d12, d13}, [%[src]], %[src_step] \n"
            "vld1.8 {d14, d15}, [%[src]], %[src_step] \n"
            "vld1.8 {d16, d17}, [%[src]], %[src_step] \n"
            "vld1.8 {d18, d19}, [%[src]], %[src_step] \n"
            "vld1.8 {d20, d21}, [%[src]], %[src_step] \n"
            "vld1.8 {d22, d23}, [%[src]], %[src_step] \n"
            "vld1.8 {d24, d25}, [%[src]], %[src_step] \n"
            "vld1.8 {d26, d27}, [%[src]], %[src_step] \n"
            "vld1.8 {d28, d29}, [%[src]], %[src_step] \n"
            "vld1.8 {d30, d31}, [%[src]], %[src_step] \n"
            "vtrn.8 q0, q1 \n"
            "vtrn.8 q2, q3 \n"
            "vtrn.8 q4, q5 \n"
            "vtrn.8 q6, q7 \n"
            "vtrn.8 q8, q9 \n"
            "vtrn.8 q10, q11 \n"
            "vtrn.8 q12, q13 \n"
            "vtrn.8 q14, q15 \n"
            "vtrn.16 q0, q2 \n"
            "vtrn.16 q1, q3 \n"
            "vtrn.16 q4, q6 \n"
            "vtrn.16 q5, q7 \n"
            "vtrn.16 q8, q10 \n"
            "vtrn.16 q9, q11 \n"
            "vtrn.16 q12, q14 \n"
            "vtrn.16 q13, q15 \n"
            "vtrn.32 q0, q4 \n"
            "vtrn.32 q1, q5 \n"
            "vtrn.32 q2, q6 \n"
            "vtrn.32 q3, q7 \n"
            "vtrn.32 q8, q12 \n"
            "vtrn.32 q9, q13 \n"
            "vtrn.32 q10, q14 \n"
            "vtrn.32 q11, q15 \n"
            "vswp d1, d16 \n"
            "vswp d3, d18 \n"
            "vswp d5, d20 \n"
            "vswp d7, d22 \n"
            "vswp d9, d24 \n"
            "vswp d11, d26 \n"
            "vswp d13, d28 \n"
            "vswp d15, d30 \n"
            "vswp d0, d1 \n"
            "vswp d2, d3 \n"
            "vswp d4, d5 \n"
            "vswp d6, d7 \n"
            "vswp d8, d9 \n"
            "vswp d10, d11 \n"
            "vswp d12, d13 \n"
            "vswp d14, d15 \n"
            "vswp d16, d17 \n"
            "vswp d18, d19 \n"
            "vswp d20, d21 \n"
            "vswp d22, d23 \n"
            "vswp d24, d25 \n"
            "vswp d26, d27 \n"
            "vswp d28, d29 \n"
            "vswp d30, d31 \n"
            "vrev64.8 q0, q0\n"
            "vrev64.8 q1, q1\n"
            "vrev64.8 q2, q2\n"
            "vrev64.8 q3, q3\n"
            "vrev64.8 q4, q4\n"
            "vrev64.8 q5, q5\n"
            "vrev64.8 q6, q6\n"
            "vrev64.8 q7, q7\n"
            "vrev64.8 q8, q8\n"
            "vrev64.8 q9, q9\n"
            "vrev64.8 q10, q10\n"
            "vrev64.8 q11, q11\n"
            "vrev64.8 q12, q12\n"
            "vrev64.8 q13, q13\n"
            "vrev64.8 q14, q14\n"
            "vrev64.8 q15, q15\n"
            "vst1.8 {d0, d1}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d2, d3}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d4, d5}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d6, d7}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d8, d9}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d10, d11}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d12, d13}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d14, d15}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d16, d17}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d18, d19}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d20, d21}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d22, d23}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d24, d25}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d26, d27}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d28, d29}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d30, d31}, [%[dst]], %[dst_step] \n"
            : [src] "+r"(src), [dst] "+r"(dst)
            : [src_step] "r"(src_step), [dst_step] "r"(dst_step)
            : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
              "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18",
              "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31");
}

static inline void rotate_countclockwise_u8_16x16(
        const uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    const uint8_t* src = sptr + ih * W + iw;
    const uint8_t* dst = dptr + (W - iw - 16) * H + ih;
    int src_step = W;
    int dst_step = H;
    asm volatile(
            "\n"
            "vld1.8 {d0, d1}, [%[src]], %[src_step] \n"
            "vld1.8 {d2, d3}, [%[src]], %[src_step] \n"
            "vld1.8 {d4, d5}, [%[src]], %[src_step] \n"
            "vld1.8 {d6, d7}, [%[src]], %[src_step] \n"
            "vld1.8 {d8, d9}, [%[src]], %[src_step] \n"
            "vld1.8 {d10, d11}, [%[src]], %[src_step] \n"
            "vld1.8 {d12, d13}, [%[src]], %[src_step] \n"
            "vld1.8 {d14, d15}, [%[src]], %[src_step] \n"
            "vld1.8 {d16, d17}, [%[src]], %[src_step] \n"
            "vld1.8 {d18, d19}, [%[src]], %[src_step] \n"
            "vld1.8 {d20, d21}, [%[src]], %[src_step] \n"
            "vld1.8 {d22, d23}, [%[src]], %[src_step] \n"
            "vld1.8 {d24, d25}, [%[src]], %[src_step] \n"
            "vld1.8 {d26, d27}, [%[src]], %[src_step] \n"
            "vld1.8 {d28, d29}, [%[src]], %[src_step] \n"
            "vld1.8 {d30, d31}, [%[src]], %[src_step] \n"
            "vtrn.8 q0, q1 \n"
            "vtrn.8 q2, q3 \n"
            "vtrn.8 q4, q5 \n"
            "vtrn.8 q6, q7 \n"
            "vtrn.8 q8, q9 \n"
            "vtrn.8 q10, q11 \n"
            "vtrn.8 q12, q13 \n"
            "vtrn.8 q14, q15 \n"
            "vtrn.16 q0, q2 \n"
            "vtrn.16 q1, q3 \n"
            "vtrn.16 q4, q6 \n"
            "vtrn.16 q5, q7 \n"
            "vtrn.16 q8, q10 \n"
            "vtrn.16 q9, q11 \n"
            "vtrn.16 q12, q14 \n"
            "vtrn.16 q13, q15 \n"
            "vtrn.32 q0, q4 \n"
            "vtrn.32 q1, q5 \n"
            "vtrn.32 q2, q6 \n"
            "vtrn.32 q3, q7 \n"
            "vtrn.32 q8, q12 \n"
            "vtrn.32 q9, q13 \n"
            "vtrn.32 q10, q14 \n"
            "vtrn.32 q11, q15 \n"
            "vswp d1, d16 \n"
            "vswp d3, d18 \n"
            "vswp d5, d20 \n"
            "vswp d7, d22 \n"
            "vswp d9, d24 \n"
            "vswp d11, d26 \n"
            "vswp d13, d28 \n"
            "vswp d15, d30 \n"
            "vst1.8 {d30, d31}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d28, d29}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d26, d27}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d24, d25}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d22, d23}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d20, d21}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d18, d19}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d16, d17}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d14, d15}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d12, d13}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d10, d11}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d8, d9}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d6, d7}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d4, d5}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d2, d3}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d0, d1}, [%[dst]], %[dst_step] \n"
            : [src] "+r"(src), [dst] "+r"(dst)
            : [src_step] "r"(src_step), [dst_step] "r"(dst_step)
            : "r0", "r1", "r2", "r3", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
              "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18",
              "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31");
}

#endif
        static void rotate_pixel(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw, size_t IH, size_t IW, size_t C, bool clockwise){
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

        static void rotate_clockwise(uint8_t* sptr, uint8_t* dptr, size_t IH, size_t IW, size_t C) {
            size_t ih = 0, OH = IW, OW = IH;
            if(C == 1){
                for (; ih + 15 < IH; ih += 16) {
                    size_t iw = 0;
                    for (; iw + 15 < IW; iw += 16) {
                        rotate_clockwise_u8_16x16(sptr, dptr,ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0; i < 16; ++i){
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
                for (; ih + 15 < IH; ih += 16) {
                    size_t iw = 0;
                    for (; iw + 15 < IW; iw += 16) {
                        rotate_clockwise_u8x3_16x16(sptr, dptr,ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0;i<16;++i){
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

        static void rotate_countclockwise(uint8_t* sptr, uint8_t* dptr, size_t IH, size_t IW,
                                size_t C) {
            size_t ih = 0, OH = IW, OW = IH;
            if(C == 1){
                for (; ih + 15 < IH; ih += 16) {
                    size_t iw = 0;
                    for (; iw + 15 < IW; iw += 16) {
                            rotate_countclockwise_u8_16x16(sptr, dptr, ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0;i<16;++i){
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
                for (; ih + 15 < IH; ih += 16) {
                    size_t iw = 0;
                    for (; iw + 15 < IW; iw += 16) {
                        rotate_countclockwise_u8x3_16x16(sptr, dptr,ih, iw, IH, IW);
                    }
                    for (; iw < IW; ++iw) {
                        for(size_t i = 0;i<16;++i){
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
            uint8_t * sptr = src->data;
            uint8_t * dptr = dst->data;
            size_t IH = src->rows;
            size_t IW = src->cols;
            size_t C = src->channels;
            if(clockwise){
                rotate_clockwise(sptr, dptr, IH, IW, C);
            }else{
                rotate_countclockwise(sptr, dptr, IH, IW, C);
            }
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}
// vim: syntax=cpp.doxygen
