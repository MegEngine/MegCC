/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsicRotate.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Rotate.h"
#include <sstream>
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

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
    std::stringstream writer;
    writer << R"(
        #include <string.h>
        #include "gi_int.h"
        #include "tinycv_c.h")";
    writer << R"(
static inline GI_INT32_t zip_i32_i16_low(GI_INT8_t rotate0, GI_INT8_t rotate1) {
    GI_INT16_t rotate0_16 = GiReinterpretInt8AsInt16(rotate0);
    GI_INT16_t rotate1_16 = GiReinterpretInt8AsInt16(rotate1);
    GI_INT16_t rotate00 = GiZipV0Int16(rotate0_16, rotate1_16);
    return GiReinterpretInt16AsInt32(rotate00);
}
static inline GI_INT32_t zip_i32_i16_high(GI_INT8_t rotate0, GI_INT8_t rotate1) {
    GI_INT16_t rotate0_16 = GiReinterpretInt8AsInt16(rotate0);
    GI_INT16_t rotate1_16 = GiReinterpretInt8AsInt16(rotate1);
    GI_INT16_t rotate00 = GiZipV1Int16(rotate0_16, rotate1_16);
    return GiReinterpretInt16AsInt32(rotate00);
}
        
static void rotate_clockwise_i8_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw, size_t H, size_t W) {
    int8_t* src = (int8_t*)sptr + ih * W + iw;
    GI_INT8_t src0 = GiLoadInt8(src + 0  * W);  
    GI_INT8_t src1 = GiLoadInt8(src + 1  * W);  
    GI_INT8_t src2 = GiLoadInt8(src + 2  * W);  
    GI_INT8_t src3 = GiLoadInt8(src + 3  * W);  
    GI_INT8_t src4 = GiLoadInt8(src + 4  * W);  
    GI_INT8_t src5 = GiLoadInt8(src + 5  * W);  
    GI_INT8_t src6 = GiLoadInt8(src + 6  * W);  
    GI_INT8_t src7 = GiLoadInt8(src + 7  * W);   
    GI_INT8_t src8 = GiLoadInt8(src + 8  * W);  
    GI_INT8_t src9 = GiLoadInt8(src + 9  * W);  
    GI_INT8_t srcA = GiLoadInt8(src + 10 * W); 
    GI_INT8_t srcB = GiLoadInt8(src + 11 * W); 
    GI_INT8_t srcC = GiLoadInt8(src + 12 * W); 
    GI_INT8_t srcD = GiLoadInt8(src + 13 * W); 
    GI_INT8_t srcE = GiLoadInt8(src + 14 * W); 
    GI_INT8_t srcF = GiLoadInt8(src + 15 * W); 
    
    GI_INT8_t rotate70 = GiZipV0Int8(src1, src0);
    GI_INT8_t rotate60 = GiZipV0Int8(src3, src2);
    GI_INT8_t rotate50 = GiZipV0Int8(src5, src4);
    GI_INT8_t rotate40 = GiZipV0Int8(src7, src6);
    GI_INT8_t rotate71 = GiZipV1Int8(src1, src0);
    GI_INT8_t rotate61 = GiZipV1Int8(src3, src2);
    GI_INT8_t rotate51 = GiZipV1Int8(src5, src4);
    GI_INT8_t rotate41 = GiZipV1Int8(src7, src6);   

    GI_INT8_t rotate30 = GiZipV0Int8(src9, src8);
    GI_INT8_t rotate20 = GiZipV0Int8(srcB, srcA);
    GI_INT8_t rotate10 = GiZipV0Int8(srcD, srcC); 
    GI_INT8_t rotate00 = GiZipV0Int8(srcF, srcE);
    GI_INT8_t rotate31 = GiZipV1Int8(src9, src8);
    GI_INT8_t rotate21 = GiZipV1Int8(srcB, srcA);
    GI_INT8_t rotate11 = GiZipV1Int8(srcD, srcC); 
    GI_INT8_t rotate01 = GiZipV1Int8(srcF, srcE);    
    
    GI_INT32_t dstA0 = zip_i32_i16_low(rotate00, rotate10);
    GI_INT32_t dstB0 = zip_i32_i16_low(rotate20, rotate30);
    GI_INT32_t dstC0 = zip_i32_i16_low(rotate40, rotate50);
    GI_INT32_t dstD0 = zip_i32_i16_low(rotate60, rotate70);

    GI_INT32_t dstA1 = zip_i32_i16_high(rotate00, rotate10);
    GI_INT32_t dstB1 = zip_i32_i16_high(rotate20, rotate30);
    GI_INT32_t dstC1 = zip_i32_i16_high(rotate40, rotate50);
    GI_INT32_t dstD1 = zip_i32_i16_high(rotate60, rotate70);

    GI_INT32_t dstA2 = zip_i32_i16_low(rotate01, rotate11);
    GI_INT32_t dstB2 = zip_i32_i16_low(rotate21, rotate31);
    GI_INT32_t dstC2 = zip_i32_i16_low(rotate41, rotate51);
    GI_INT32_t dstD2 = zip_i32_i16_low(rotate61, rotate71);
    
    GI_INT32_t dstA3 = zip_i32_i16_high(rotate01, rotate11);
    GI_INT32_t dstB3 = zip_i32_i16_high(rotate21, rotate31);
    GI_INT32_t dstC3 = zip_i32_i16_high(rotate41, rotate51);
    GI_INT32_t dstD3 = zip_i32_i16_high(rotate61, rotate71);

    GI_INT32_t dst00 = GiCombineInt32Low(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0));
    GI_INT32_t dst10 = GiCombineInt32Low(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1));
    GI_INT32_t dst20 = GiCombineInt32Low(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2));
    GI_INT32_t dst30 = GiCombineInt32Low(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3));
  
    GI_INT32_t dst01 = GiCombineInt32High(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0));
    GI_INT32_t dst11 = GiCombineInt32High(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1));
    GI_INT32_t dst21 = GiCombineInt32High(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2));
    GI_INT32_t dst31 = GiCombineInt32High(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3));
  
    GI_INT32_t dst02 = GiCombineInt32Low(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0));
    GI_INT32_t dst12 = GiCombineInt32Low(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1));
    GI_INT32_t dst22 = GiCombineInt32Low(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2));
    GI_INT32_t dst32 = GiCombineInt32Low(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3));
  
    GI_INT32_t dst03 = GiCombineInt32High(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0));
    GI_INT32_t dst13 = GiCombineInt32High(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1));
    GI_INT32_t dst23 = GiCombineInt32High(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2));
    GI_INT32_t dst33 = GiCombineInt32High(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3));

  
    int8_t* dst = (int8_t*)dptr + iw * H + H - ih -16;
    GiStoreInt8(dst + 0  * H, GiReinterInt32ToInt8(dst00));
    GiStoreInt8(dst + 1  * H, GiReinterInt32ToInt8(dst01));
    GiStoreInt8(dst + 2  * H, GiReinterInt32ToInt8(dst02));
    GiStoreInt8(dst + 3  * H, GiReinterInt32ToInt8(dst03));
    GiStoreInt8(dst + 4  * H, GiReinterInt32ToInt8(dst10));
    GiStoreInt8(dst + 5  * H, GiReinterInt32ToInt8(dst11));
    GiStoreInt8(dst + 6  * H, GiReinterInt32ToInt8(dst12));
    GiStoreInt8(dst + 7  * H, GiReinterInt32ToInt8(dst13));
    GiStoreInt8(dst + 8  * H, GiReinterInt32ToInt8(dst20));
    GiStoreInt8(dst + 9  * H, GiReinterInt32ToInt8(dst21));
    GiStoreInt8(dst + 10 * H, GiReinterInt32ToInt8(dst22));
    GiStoreInt8(dst + 11 * H, GiReinterInt32ToInt8(dst23));
    GiStoreInt8(dst + 12 * H, GiReinterInt32ToInt8(dst30));
    GiStoreInt8(dst + 13 * H, GiReinterInt32ToInt8(dst31));
    GiStoreInt8(dst + 14 * H, GiReinterInt32ToInt8(dst32));
    GiStoreInt8(dst + 15 * H, GiReinterInt32ToInt8(dst33));
}

static void rotate_countclockwise_u8_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw,
                                size_t H, size_t W) {
    int8_t* src = (int8_t*)sptr + ih * W + iw;
    GI_INT8_t src0 = GiLoadInt8(src + 0  * W); 
    GI_INT8_t src1 = GiLoadInt8(src + 1  * W);  
    GI_INT8_t src2 = GiLoadInt8(src + 2  * W);
    GI_INT8_t src3 = GiLoadInt8(src + 3  * W);
    GI_INT8_t src4 = GiLoadInt8(src + 4  * W);
    GI_INT8_t src5 = GiLoadInt8(src + 5  * W);
    GI_INT8_t src6 = GiLoadInt8(src + 6  * W);
    GI_INT8_t src7 = GiLoadInt8(src + 7  * W); 
    GI_INT8_t src8 = GiLoadInt8(src + 8  * W);
    GI_INT8_t src9 = GiLoadInt8(src + 9  * W);
    GI_INT8_t srcA = GiLoadInt8(src + 10 * W);
    GI_INT8_t srcB = GiLoadInt8(src + 11 * W);
    GI_INT8_t srcC = GiLoadInt8(src + 12 * W);
    GI_INT8_t srcD = GiLoadInt8(src + 13 * W);
    GI_INT8_t srcE = GiLoadInt8(src + 14 * W);
    GI_INT8_t srcF = GiLoadInt8(src + 15 * W);

    GI_INT8_t rotate00 = GiZipV0Int8(src0, src1); 
    GI_INT8_t rotate10 = GiZipV0Int8(src2, src3); 
    GI_INT8_t rotate20 = GiZipV0Int8(src4, src5); 
    GI_INT8_t rotate30 = GiZipV0Int8(src6, src7); 
    GI_INT8_t rotate40 = GiZipV0Int8(src8, src9); 
    GI_INT8_t rotate50 = GiZipV0Int8(srcA, srcB); 
    GI_INT8_t rotate60 = GiZipV0Int8(srcC, srcD); 
    GI_INT8_t rotate70 = GiZipV0Int8(srcE, srcF); 

    GI_INT8_t rotate01 = GiZipV1Int8(src0, src1);
    GI_INT8_t rotate11 = GiZipV1Int8(src2, src3);
    GI_INT8_t rotate21 = GiZipV1Int8(src4, src5);
    GI_INT8_t rotate31 = GiZipV1Int8(src6, src7);
    GI_INT8_t rotate41 = GiZipV1Int8(src8, src9);
    GI_INT8_t rotate51 = GiZipV1Int8(srcA, srcB);
    GI_INT8_t rotate61 = GiZipV1Int8(srcC, srcD);
    GI_INT8_t rotate71 = GiZipV1Int8(srcE, srcF);

    GI_INT32_t dstA0 = zip_i32_i16_low(rotate00, rotate10);
    GI_INT32_t dstB0 = zip_i32_i16_low(rotate20, rotate30);
    GI_INT32_t dstC0 = zip_i32_i16_low(rotate40, rotate50);
    GI_INT32_t dstD0 = zip_i32_i16_low(rotate60, rotate70);

    GI_INT32_t dstA1 = zip_i32_i16_high(rotate00, rotate10);
    GI_INT32_t dstB1 = zip_i32_i16_high(rotate20, rotate30);
    GI_INT32_t dstC1 = zip_i32_i16_high(rotate40, rotate50);
    GI_INT32_t dstD1 = zip_i32_i16_high(rotate60, rotate70);

    GI_INT32_t dstA2 = zip_i32_i16_low(rotate01, rotate11);
    GI_INT32_t dstB2 = zip_i32_i16_low(rotate21, rotate31);
    GI_INT32_t dstC2 = zip_i32_i16_low(rotate41, rotate51);
    GI_INT32_t dstD2 = zip_i32_i16_low(rotate61, rotate71);
    
    GI_INT32_t dstA3 = zip_i32_i16_high(rotate01, rotate11);
    GI_INT32_t dstB3 = zip_i32_i16_high(rotate21, rotate31);
    GI_INT32_t dstC3 = zip_i32_i16_high(rotate41, rotate51);
    GI_INT32_t dstD3 = zip_i32_i16_high(rotate61, rotate71);

    GI_INT32_t dst00 = GiCombineInt32Low(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0));
    GI_INT32_t dst10 = GiCombineInt32Low(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1));
    GI_INT32_t dst20 = GiCombineInt32Low(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2));
    GI_INT32_t dst30 = GiCombineInt32Low(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3));
  
    GI_INT32_t dst01 = GiCombineInt32High(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0));
    GI_INT32_t dst11 = GiCombineInt32High(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1));
    GI_INT32_t dst21 = GiCombineInt32High(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2));
    GI_INT32_t dst31 = GiCombineInt32High(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3));
  
    GI_INT32_t dst02 = GiCombineInt32Low(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0));
    GI_INT32_t dst12 = GiCombineInt32Low(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1));
    GI_INT32_t dst22 = GiCombineInt32Low(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2));
    GI_INT32_t dst32 = GiCombineInt32Low(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3));
  
    GI_INT32_t dst03 = GiCombineInt32High(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0));
    GI_INT32_t dst13 = GiCombineInt32High(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1));
    GI_INT32_t dst23 = GiCombineInt32High(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2));
    GI_INT32_t dst33 = GiCombineInt32High(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3));

    int8_t* dst = (int8_t*)(dptr + (W - iw - 16) * H + ih);
    GiStoreInt8(dst + 0  * H, GiReinterInt32ToInt8(dst33));
    GiStoreInt8(dst + 1  * H, GiReinterInt32ToInt8(dst32));
    GiStoreInt8(dst + 2  * H, GiReinterInt32ToInt8(dst31));
    GiStoreInt8(dst + 3  * H, GiReinterInt32ToInt8(dst30));
    GiStoreInt8(dst + 4  * H, GiReinterInt32ToInt8(dst23));
    GiStoreInt8(dst + 5  * H, GiReinterInt32ToInt8(dst22));
    GiStoreInt8(dst + 6  * H, GiReinterInt32ToInt8(dst21));
    GiStoreInt8(dst + 7  * H, GiReinterInt32ToInt8(dst20));
    GiStoreInt8(dst + 8  * H, GiReinterInt32ToInt8(dst13));
    GiStoreInt8(dst + 9  * H, GiReinterInt32ToInt8(dst12));
    GiStoreInt8(dst + 10 * H, GiReinterInt32ToInt8(dst11));
    GiStoreInt8(dst + 11 * H, GiReinterInt32ToInt8(dst10));
    GiStoreInt8(dst + 12 * H, GiReinterInt32ToInt8(dst03));
    GiStoreInt8(dst + 13 * H, GiReinterInt32ToInt8(dst02));
    GiStoreInt8(dst + 14 * H, GiReinterInt32ToInt8(dst01));
    GiStoreInt8(dst + 15 * H, GiReinterInt32ToInt8(dst00)); 
}

static void rotate_clockwise_u8x3_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw,
                            size_t H, size_t W) {
    int8_t* src = (int8_t*)sptr + ih * W*3 + iw*3;
    GI_INT8_V3_t src0 = GiLoadUzipInt8V3(src + 0  * W);
    GI_INT8_V3_t src1 = GiLoadUzipInt8V3(src + 3  * W);
    GI_INT8_V3_t src2 = GiLoadUzipInt8V3(src + 6  * W);
    GI_INT8_V3_t src3 = GiLoadUzipInt8V3(src + 9  * W);
    GI_INT8_V3_t src4 = GiLoadUzipInt8V3(src + 12 * W);
    GI_INT8_V3_t src5 = GiLoadUzipInt8V3(src + 15 * W);
    GI_INT8_V3_t src6 = GiLoadUzipInt8V3(src + 18 * W);
    GI_INT8_V3_t src7 = GiLoadUzipInt8V3(src + 21 * W);
    GI_INT8_V3_t src8 = GiLoadUzipInt8V3(src + 24 * W);
    GI_INT8_V3_t src9 = GiLoadUzipInt8V3(src + 27 * W);
    GI_INT8_V3_t srcA = GiLoadUzipInt8V3(src + 30 * W);
    GI_INT8_V3_t srcB = GiLoadUzipInt8V3(src + 33 * W);
    GI_INT8_V3_t srcC = GiLoadUzipInt8V3(src + 36 * W);
    GI_INT8_V3_t srcD = GiLoadUzipInt8V3(src + 39 * W);
    GI_INT8_V3_t srcE = GiLoadUzipInt8V3(src + 42 * W);
    GI_INT8_V3_t srcF = GiLoadUzipInt8V3(src + 45 * W);

    GI_INT8_t src00 = GiGetSubVectorInt8V3(src0, 0);
    GI_INT8_t src10 = GiGetSubVectorInt8V3(src1, 0);
    GI_INT8_t src20 = GiGetSubVectorInt8V3(src2, 0);
    GI_INT8_t src30 = GiGetSubVectorInt8V3(src3, 0);
    GI_INT8_t src40 = GiGetSubVectorInt8V3(src4, 0);
    GI_INT8_t src50 = GiGetSubVectorInt8V3(src5, 0);
    GI_INT8_t src60 = GiGetSubVectorInt8V3(src6, 0);
    GI_INT8_t src70 = GiGetSubVectorInt8V3(src7, 0);
    GI_INT8_t src80 = GiGetSubVectorInt8V3(src8, 0);
    GI_INT8_t src90 = GiGetSubVectorInt8V3(src9, 0);
    GI_INT8_t srcA0 = GiGetSubVectorInt8V3(srcA, 0);
    GI_INT8_t srcB0 = GiGetSubVectorInt8V3(srcB, 0);
    GI_INT8_t srcC0 = GiGetSubVectorInt8V3(srcC, 0);
    GI_INT8_t srcD0 = GiGetSubVectorInt8V3(srcD, 0);
    GI_INT8_t srcE0 = GiGetSubVectorInt8V3(srcE, 0);
    GI_INT8_t srcF0 = GiGetSubVectorInt8V3(srcF, 0);

    GI_INT8_t src01 = GiGetSubVectorInt8V3(src0, 1);
    GI_INT8_t src11 = GiGetSubVectorInt8V3(src1, 1);
    GI_INT8_t src21 = GiGetSubVectorInt8V3(src2, 1);
    GI_INT8_t src31 = GiGetSubVectorInt8V3(src3, 1);
    GI_INT8_t src41 = GiGetSubVectorInt8V3(src4, 1);
    GI_INT8_t src51 = GiGetSubVectorInt8V3(src5, 1);
    GI_INT8_t src61 = GiGetSubVectorInt8V3(src6, 1);
    GI_INT8_t src71 = GiGetSubVectorInt8V3(src7, 1);
    GI_INT8_t src81 = GiGetSubVectorInt8V3(src8, 1);
    GI_INT8_t src91 = GiGetSubVectorInt8V3(src9, 1);
    GI_INT8_t srcA1 = GiGetSubVectorInt8V3(srcA, 1);
    GI_INT8_t srcB1 = GiGetSubVectorInt8V3(srcB, 1);
    GI_INT8_t srcC1 = GiGetSubVectorInt8V3(srcC, 1);
    GI_INT8_t srcD1 = GiGetSubVectorInt8V3(srcD, 1);
    GI_INT8_t srcE1 = GiGetSubVectorInt8V3(srcE, 1);
    GI_INT8_t srcF1 = GiGetSubVectorInt8V3(srcF, 1);

    GI_INT8_t src02 = GiGetSubVectorInt8V3(src0, 2);
    GI_INT8_t src12 = GiGetSubVectorInt8V3(src1, 2);
    GI_INT8_t src22 = GiGetSubVectorInt8V3(src2, 2);
    GI_INT8_t src32 = GiGetSubVectorInt8V3(src3, 2);
    GI_INT8_t src42 = GiGetSubVectorInt8V3(src4, 2);
    GI_INT8_t src52 = GiGetSubVectorInt8V3(src5, 2);
    GI_INT8_t src62 = GiGetSubVectorInt8V3(src6, 2);
    GI_INT8_t src72 = GiGetSubVectorInt8V3(src7, 2);
    GI_INT8_t src82 = GiGetSubVectorInt8V3(src8, 2);
    GI_INT8_t src92 = GiGetSubVectorInt8V3(src9, 2);
    GI_INT8_t srcA2 = GiGetSubVectorInt8V3(srcA, 2);
    GI_INT8_t srcB2 = GiGetSubVectorInt8V3(srcB, 2);
    GI_INT8_t srcC2 = GiGetSubVectorInt8V3(srcC, 2);
    GI_INT8_t srcD2 = GiGetSubVectorInt8V3(srcD, 2);
    GI_INT8_t srcE2 = GiGetSubVectorInt8V3(srcE, 2);
    GI_INT8_t srcF2 = GiGetSubVectorInt8V3(srcF, 2);


    GI_INT8_t ans00, ans10, ans20, ans30, ans40, ans50, ans60, ans70;
    GI_INT8_t ans01, ans11, ans21, ans31, ans41, ans51, ans61, ans71;
    GI_INT8_t ans02, ans12, ans22, ans32, ans42, ans52, ans62, ans72;
    GI_INT8_t ans80, ans90, ansA0, ansB0, ansC0, ansD0, ansE0, ansF0;
    GI_INT8_t ans81, ans91, ansA1, ansB1, ansC1, ansD1, ansE1, ansF1;
    GI_INT8_t ans82, ans92, ansA2, ansB2, ansC2, ansD2, ansE2, ansF2;
#define rotate_interleave(idx) \
        { \
        GI_INT8_t rotate70 = GiZipV0Int8(src1##idx, src0##idx); \
        GI_INT8_t rotate60 = GiZipV0Int8(src3##idx, src2##idx); \
        GI_INT8_t rotate50 = GiZipV0Int8(src5##idx, src4##idx); \
        GI_INT8_t rotate40 = GiZipV0Int8(src7##idx, src6##idx); \
        GI_INT8_t rotate30 = GiZipV0Int8(src9##idx, src8##idx); \
        GI_INT8_t rotate20 = GiZipV0Int8(srcB##idx, srcA##idx); \
        GI_INT8_t rotate10 = GiZipV0Int8(srcD##idx, srcC##idx); \
        GI_INT8_t rotate00 = GiZipV0Int8(srcF##idx, srcE##idx); \
        \
        GI_INT8_t rotate71 = GiZipV1Int8(src1##idx, src0##idx); \
        GI_INT8_t rotate61 = GiZipV1Int8(src3##idx, src2##idx); \
        GI_INT8_t rotate51 = GiZipV1Int8(src5##idx, src4##idx); \
        GI_INT8_t rotate41 = GiZipV1Int8(src7##idx, src6##idx); \
        GI_INT8_t rotate31 = GiZipV1Int8(src9##idx, src8##idx); \
        GI_INT8_t rotate21 = GiZipV1Int8(srcB##idx, srcA##idx); \
        GI_INT8_t rotate11 = GiZipV1Int8(srcD##idx, srcC##idx); \
        GI_INT8_t rotate01 = GiZipV1Int8(srcF##idx, srcE##idx); \
        \
        GI_INT32_t dstA0 = zip_i32_i16_low(rotate00, rotate10); \
        GI_INT32_t dstB0 = zip_i32_i16_low(rotate20, rotate30); \
        GI_INT32_t dstC0 = zip_i32_i16_low(rotate40, rotate50); \
        GI_INT32_t dstD0 = zip_i32_i16_low(rotate60, rotate70); \
        \
        GI_INT32_t dstA1 = zip_i32_i16_high(rotate00, rotate10); \
        GI_INT32_t dstB1 = zip_i32_i16_high(rotate20, rotate30); \
        GI_INT32_t dstC1 = zip_i32_i16_high(rotate40, rotate50); \
        GI_INT32_t dstD1 = zip_i32_i16_high(rotate60, rotate70); \
        \
        GI_INT32_t dstA2 = zip_i32_i16_low(rotate01, rotate11); \
        GI_INT32_t dstB2 = zip_i32_i16_low(rotate21, rotate31); \
        GI_INT32_t dstC2 = zip_i32_i16_low(rotate41, rotate51); \
        GI_INT32_t dstD2 = zip_i32_i16_low(rotate61, rotate71); \
        \
        GI_INT32_t dstA3 = zip_i32_i16_high(rotate01, rotate11); \
        GI_INT32_t dstB3 = zip_i32_i16_high(rotate21, rotate31); \
        GI_INT32_t dstC3 = zip_i32_i16_high(rotate41, rotate51); \
        GI_INT32_t dstD3 = zip_i32_i16_high(rotate61, rotate71); \
        \
        GI_INT32_t dst00 = GiCombineInt32Low(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0)); \
        GI_INT32_t dst10 = GiCombineInt32Low(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1)); \
        GI_INT32_t dst20 = GiCombineInt32Low(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2)); \
        GI_INT32_t dst30 = GiCombineInt32Low(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3)); \
        \
        GI_INT32_t dst01 = GiCombineInt32High(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0)); \
        GI_INT32_t dst11 = GiCombineInt32High(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1)); \
        GI_INT32_t dst21 = GiCombineInt32High(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2)); \
        GI_INT32_t dst31 = GiCombineInt32High(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3)); \
        \
        GI_INT32_t dst02 = GiCombineInt32Low(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0)); \
        GI_INT32_t dst12 = GiCombineInt32Low(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1)); \
        GI_INT32_t dst22 = GiCombineInt32Low(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2)); \
        GI_INT32_t dst32 = GiCombineInt32Low(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3)); \
        \
        GI_INT32_t dst03 = GiCombineInt32High(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0)); \
        GI_INT32_t dst13 = GiCombineInt32High(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1)); \
        GI_INT32_t dst23 = GiCombineInt32High(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2)); \
        GI_INT32_t dst33 = GiCombineInt32High(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3)); \
        \
        ans0##idx = GiReinterInt32ToInt8(dst00); \
        ans1##idx = GiReinterInt32ToInt8(dst01); \
        ans2##idx = GiReinterInt32ToInt8(dst02); \
        ans3##idx = GiReinterInt32ToInt8(dst03); \
        ans4##idx = GiReinterInt32ToInt8(dst10); \
        ans5##idx = GiReinterInt32ToInt8(dst11); \
        ans6##idx = GiReinterInt32ToInt8(dst12); \
        ans7##idx = GiReinterInt32ToInt8(dst13); \
        ans8##idx = GiReinterInt32ToInt8(dst20); \
        ans9##idx = GiReinterInt32ToInt8(dst21); \
        ansA##idx = GiReinterInt32ToInt8(dst22); \
        ansB##idx = GiReinterInt32ToInt8(dst23); \
        ansC##idx = GiReinterInt32ToInt8(dst30); \
        ansD##idx = GiReinterInt32ToInt8(dst31); \
        ansE##idx = GiReinterInt32ToInt8(dst32); \
        ansF##idx = GiReinterInt32ToInt8(dst33); \
}

rotate_interleave(0)
rotate_interleave(1)
rotate_interleave(2)

#undef rotate_interleave 
    int8_t* dst =(int8_t*) dptr + iw * H*3 + (H - ih - 16)*3;
    GiStoreZipInt8V3(dst + 0  * H, ans00, ans01, ans02);
    GiStoreZipInt8V3(dst + 3  * H, ans10, ans11, ans12);
    GiStoreZipInt8V3(dst + 6  * H, ans20, ans21, ans22);
    GiStoreZipInt8V3(dst + 9  * H, ans30, ans31, ans32);
    GiStoreZipInt8V3(dst + 12 * H, ans40, ans41, ans42);
    GiStoreZipInt8V3(dst + 15 * H, ans50, ans51, ans52);
    GiStoreZipInt8V3(dst + 18 * H, ans60, ans61, ans62);
    GiStoreZipInt8V3(dst + 21 * H, ans70, ans71, ans72);
    GiStoreZipInt8V3(dst + 24 * H, ans80, ans81, ans82);
    GiStoreZipInt8V3(dst + 27 * H, ans90, ans91, ans92);
    GiStoreZipInt8V3(dst + 30 * H, ansA0, ansA1, ansA2);
    GiStoreZipInt8V3(dst + 33 * H, ansB0, ansB1, ansB2);
    GiStoreZipInt8V3(dst + 36 * H, ansC0, ansC1, ansC2);
    GiStoreZipInt8V3(dst + 39 * H, ansD0, ansD1, ansD2);
    GiStoreZipInt8V3(dst + 42 * H, ansE0, ansE1, ansE2);
    GiStoreZipInt8V3(dst + 45 * H, ansF0, ansF1, ansF2);
}

static void rotate_countclockwise_u8x3_16x16(uint8_t* sptr, uint8_t* dptr, size_t ih, size_t iw,
                                    size_t H, size_t W) {
    int8_t* src = (int8_t*)sptr + ih * W*3 + iw*3;
    int32_t test[4];
    GI_INT8_V3_t src0 = GiLoadUzipInt8V3(src + 0  * W);
    GI_INT8_V3_t src1 = GiLoadUzipInt8V3(src + 3  * W);
    GI_INT8_V3_t src2 = GiLoadUzipInt8V3(src + 6  * W);
    GI_INT8_V3_t src3 = GiLoadUzipInt8V3(src + 9  * W);
    GI_INT8_V3_t src4 = GiLoadUzipInt8V3(src + 12 * W);
    GI_INT8_V3_t src5 = GiLoadUzipInt8V3(src + 15 * W);
    GI_INT8_V3_t src6 = GiLoadUzipInt8V3(src + 18 * W);
    GI_INT8_V3_t src7 = GiLoadUzipInt8V3(src + 21 * W);
    GI_INT8_V3_t src8 = GiLoadUzipInt8V3(src + 24 * W);
    GI_INT8_V3_t src9 = GiLoadUzipInt8V3(src + 27 * W);
    GI_INT8_V3_t srcA = GiLoadUzipInt8V3(src + 30 * W);
    GI_INT8_V3_t srcB = GiLoadUzipInt8V3(src + 33 * W);
    GI_INT8_V3_t srcC = GiLoadUzipInt8V3(src + 36 * W);
    GI_INT8_V3_t srcD = GiLoadUzipInt8V3(src + 39 * W);
    GI_INT8_V3_t srcE = GiLoadUzipInt8V3(src + 42 * W);
    GI_INT8_V3_t srcF = GiLoadUzipInt8V3(src + 45 * W);

    GI_INT8_t src00 = GiGetSubVectorInt8V3(src0, 0);
    GI_INT8_t src10 = GiGetSubVectorInt8V3(src1, 0);
    GI_INT8_t src20 = GiGetSubVectorInt8V3(src2, 0);
    GI_INT8_t src30 = GiGetSubVectorInt8V3(src3, 0);
    GI_INT8_t src40 = GiGetSubVectorInt8V3(src4, 0);
    GI_INT8_t src50 = GiGetSubVectorInt8V3(src5, 0);
    GI_INT8_t src60 = GiGetSubVectorInt8V3(src6, 0);
    GI_INT8_t src70 = GiGetSubVectorInt8V3(src7, 0);
    GI_INT8_t src80 = GiGetSubVectorInt8V3(src8, 0);
    GI_INT8_t src90 = GiGetSubVectorInt8V3(src9, 0);
    GI_INT8_t srcA0 = GiGetSubVectorInt8V3(srcA, 0);
    GI_INT8_t srcB0 = GiGetSubVectorInt8V3(srcB, 0);
    GI_INT8_t srcC0 = GiGetSubVectorInt8V3(srcC, 0);
    GI_INT8_t srcD0 = GiGetSubVectorInt8V3(srcD, 0);
    GI_INT8_t srcE0 = GiGetSubVectorInt8V3(srcE, 0);
    GI_INT8_t srcF0 = GiGetSubVectorInt8V3(srcF, 0);

    GI_INT8_t src01 = GiGetSubVectorInt8V3(src0, 1);
    GI_INT8_t src11 = GiGetSubVectorInt8V3(src1, 1);
    GI_INT8_t src21 = GiGetSubVectorInt8V3(src2, 1);
    GI_INT8_t src31 = GiGetSubVectorInt8V3(src3, 1);
    GI_INT8_t src41 = GiGetSubVectorInt8V3(src4, 1);
    GI_INT8_t src51 = GiGetSubVectorInt8V3(src5, 1);
    GI_INT8_t src61 = GiGetSubVectorInt8V3(src6, 1);
    GI_INT8_t src71 = GiGetSubVectorInt8V3(src7, 1);
    GI_INT8_t src81 = GiGetSubVectorInt8V3(src8, 1);
    GI_INT8_t src91 = GiGetSubVectorInt8V3(src9, 1);
    GI_INT8_t srcA1 = GiGetSubVectorInt8V3(srcA, 1);
    GI_INT8_t srcB1 = GiGetSubVectorInt8V3(srcB, 1);
    GI_INT8_t srcC1 = GiGetSubVectorInt8V3(srcC, 1);
    GI_INT8_t srcD1 = GiGetSubVectorInt8V3(srcD, 1);
    GI_INT8_t srcE1 = GiGetSubVectorInt8V3(srcE, 1);
    GI_INT8_t srcF1 = GiGetSubVectorInt8V3(srcF, 1);

    GI_INT8_t src02 = GiGetSubVectorInt8V3(src0, 2);
    GI_INT8_t src12 = GiGetSubVectorInt8V3(src1, 2);
    GI_INT8_t src22 = GiGetSubVectorInt8V3(src2, 2);
    GI_INT8_t src32 = GiGetSubVectorInt8V3(src3, 2);
    GI_INT8_t src42 = GiGetSubVectorInt8V3(src4, 2);
    GI_INT8_t src52 = GiGetSubVectorInt8V3(src5, 2);
    GI_INT8_t src62 = GiGetSubVectorInt8V3(src6, 2);
    GI_INT8_t src72 = GiGetSubVectorInt8V3(src7, 2);
    GI_INT8_t src82 = GiGetSubVectorInt8V3(src8, 2);
    GI_INT8_t src92 = GiGetSubVectorInt8V3(src9, 2);
    GI_INT8_t srcA2 = GiGetSubVectorInt8V3(srcA, 2);
    GI_INT8_t srcB2 = GiGetSubVectorInt8V3(srcB, 2);
    GI_INT8_t srcC2 = GiGetSubVectorInt8V3(srcC, 2);
    GI_INT8_t srcD2 = GiGetSubVectorInt8V3(srcD, 2);
    GI_INT8_t srcE2 = GiGetSubVectorInt8V3(srcE, 2);
    GI_INT8_t srcF2 = GiGetSubVectorInt8V3(srcF, 2);


    GI_INT8_t ans00, ans10, ans20, ans30, ans40, ans50, ans60, ans70;
    GI_INT8_t ans01, ans11, ans21, ans31, ans41, ans51, ans61, ans71;
    GI_INT8_t ans02, ans12, ans22, ans32, ans42, ans52, ans62, ans72;
    GI_INT8_t ans80, ans90, ansA0, ansB0, ansC0, ansD0, ansE0, ansF0;
    GI_INT8_t ans81, ans91, ansA1, ansB1, ansC1, ansD1, ansE1, ansF1;
    GI_INT8_t ans82, ans92, ansA2, ansB2, ansC2, ansD2, ansE2, ansF2;
#define rotate_interleave(idx) \
    { \
        GI_INT8_t rotate00 = GiZipV0Int8(src0##idx, src1##idx); \
        GI_INT8_t rotate10 = GiZipV0Int8(src2##idx, src3##idx); \
        GI_INT8_t rotate20 = GiZipV0Int8(src4##idx, src5##idx); \
        GI_INT8_t rotate30 = GiZipV0Int8(src6##idx, src7##idx); \
        GI_INT8_t rotate40 = GiZipV0Int8(src8##idx, src9##idx); \
        GI_INT8_t rotate50 = GiZipV0Int8(srcA##idx, srcB##idx); \
        GI_INT8_t rotate60 = GiZipV0Int8(srcC##idx, srcD##idx); \
        GI_INT8_t rotate70 = GiZipV0Int8(srcE##idx, srcF##idx); \
        \
        GI_INT8_t rotate01 = GiZipV1Int8(src0##idx, src1##idx); \
        GI_INT8_t rotate11 = GiZipV1Int8(src2##idx, src3##idx); \
        GI_INT8_t rotate21 = GiZipV1Int8(src4##idx, src5##idx); \
        GI_INT8_t rotate31 = GiZipV1Int8(src6##idx, src7##idx); \
        GI_INT8_t rotate41 = GiZipV1Int8(src8##idx, src9##idx); \
        GI_INT8_t rotate51 = GiZipV1Int8(srcA##idx, srcB##idx); \
        GI_INT8_t rotate61 = GiZipV1Int8(srcC##idx, srcD##idx); \
        GI_INT8_t rotate71 = GiZipV1Int8(srcE##idx, srcF##idx); \
        \
        GI_INT32_t dstA0 = zip_i32_i16_low(rotate00, rotate10); \
        GI_INT32_t dstB0 = zip_i32_i16_low(rotate20, rotate30); \
        GI_INT32_t dstC0 = zip_i32_i16_low(rotate40, rotate50); \
        GI_INT32_t dstD0 = zip_i32_i16_low(rotate60, rotate70); \
        \
        GI_INT32_t dstA1 = zip_i32_i16_high(rotate00, rotate10); \
        GI_INT32_t dstB1 = zip_i32_i16_high(rotate20, rotate30); \
        GI_INT32_t dstC1 = zip_i32_i16_high(rotate40, rotate50); \
        GI_INT32_t dstD1 = zip_i32_i16_high(rotate60, rotate70); \
        \
        GI_INT32_t dstA2 = zip_i32_i16_low(rotate01, rotate11); \
        GI_INT32_t dstB2 = zip_i32_i16_low(rotate21, rotate31); \
        GI_INT32_t dstC2 = zip_i32_i16_low(rotate41, rotate51); \
        GI_INT32_t dstD2 = zip_i32_i16_low(rotate61, rotate71); \
        \
        GI_INT32_t dstA3 = zip_i32_i16_high(rotate01, rotate11); \
        GI_INT32_t dstB3 = zip_i32_i16_high(rotate21, rotate31); \
        GI_INT32_t dstC3 = zip_i32_i16_high(rotate41, rotate51); \
        GI_INT32_t dstD3 = zip_i32_i16_high(rotate61, rotate71); \
        \
        GI_INT32_t dst00 = GiCombineInt32Low(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0)); \
        GI_INT32_t dst10 = GiCombineInt32Low(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1)); \
        GI_INT32_t dst20 = GiCombineInt32Low(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2)); \
        GI_INT32_t dst30 = GiCombineInt32Low(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3)); \
        \
        GI_INT32_t dst01 = GiCombineInt32High(GiZipV0Int32(dstA0, dstB0), GiZipV0Int32(dstC0, dstD0)); \
        GI_INT32_t dst11 = GiCombineInt32High(GiZipV0Int32(dstA1, dstB1), GiZipV0Int32(dstC1, dstD1)); \
        GI_INT32_t dst21 = GiCombineInt32High(GiZipV0Int32(dstA2, dstB2), GiZipV0Int32(dstC2, dstD2)); \
        GI_INT32_t dst31 = GiCombineInt32High(GiZipV0Int32(dstA3, dstB3), GiZipV0Int32(dstC3, dstD3)); \
        \
        GI_INT32_t dst02 = GiCombineInt32Low(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0)); \
        GI_INT32_t dst12 = GiCombineInt32Low(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1)); \
        GI_INT32_t dst22 = GiCombineInt32Low(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2)); \
        GI_INT32_t dst32 = GiCombineInt32Low(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3)); \
        \
        GI_INT32_t dst03 = GiCombineInt32High(GiZipV1Int32(dstA0, dstB0), GiZipV1Int32(dstC0, dstD0)); \
        GI_INT32_t dst13 = GiCombineInt32High(GiZipV1Int32(dstA1, dstB1), GiZipV1Int32(dstC1, dstD1)); \
        GI_INT32_t dst23 = GiCombineInt32High(GiZipV1Int32(dstA2, dstB2), GiZipV1Int32(dstC2, dstD2)); \
        GI_INT32_t dst33 = GiCombineInt32High(GiZipV1Int32(dstA3, dstB3), GiZipV1Int32(dstC3, dstD3)); \
        \
        ans0##idx = GiReinterInt32ToInt8(dst00); \
        ans1##idx = GiReinterInt32ToInt8(dst01); \
        ans2##idx = GiReinterInt32ToInt8(dst02); \
        ans3##idx = GiReinterInt32ToInt8(dst03); \
        ans4##idx = GiReinterInt32ToInt8(dst10); \
        ans5##idx = GiReinterInt32ToInt8(dst11); \
        ans6##idx = GiReinterInt32ToInt8(dst12); \
        ans7##idx = GiReinterInt32ToInt8(dst13); \
        ans8##idx = GiReinterInt32ToInt8(dst20); \
        ans9##idx = GiReinterInt32ToInt8(dst21); \
        ansA##idx = GiReinterInt32ToInt8(dst22); \
        ansB##idx = GiReinterInt32ToInt8(dst23); \
        ansC##idx = GiReinterInt32ToInt8(dst30); \
        ansD##idx = GiReinterInt32ToInt8(dst31); \
        ansE##idx = GiReinterInt32ToInt8(dst32); \
        ansF##idx = GiReinterInt32ToInt8(dst33); \
    }

rotate_interleave(0)
rotate_interleave(1)
rotate_interleave(2)

#undef rotate_interleave 

    int8_t* dst = (int8_t*)dptr + (W - iw - 16) * H*3 + ih*3;
    GiStoreZipInt8V3(dst + 0  * H, ansF0, ansF1, ansF2);
    GiStoreZipInt8V3(dst + 3  * H, ansE0, ansE1, ansE2);
    GiStoreZipInt8V3(dst + 6  * H, ansD0, ansD1, ansD2);
    GiStoreZipInt8V3(dst + 9  * H, ansC0, ansC1, ansC2);
    GiStoreZipInt8V3(dst + 12 * H, ansB0, ansB1, ansB2);
    GiStoreZipInt8V3(dst + 15 * H, ansA0, ansA1, ansA2);
    GiStoreZipInt8V3(dst + 18 * H, ans90, ans91, ans92);
    GiStoreZipInt8V3(dst + 21 * H, ans80, ans81, ans82);
    GiStoreZipInt8V3(dst + 24 * H, ans70, ans71, ans72);
    GiStoreZipInt8V3(dst + 27 * H, ans60, ans61, ans62);
    GiStoreZipInt8V3(dst + 30 * H, ans50, ans51, ans52);
    GiStoreZipInt8V3(dst + 33 * H, ans40, ans41, ans42);
    GiStoreZipInt8V3(dst + 36 * H, ans30, ans31, ans32);
    GiStoreZipInt8V3(dst + 39 * H, ans20, ans21, ans22);
    GiStoreZipInt8V3(dst + 42 * H, ans10, ans11, ans12);
    GiStoreZipInt8V3(dst + 45 * H, ans00, ans01, ans02);
}

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
                        rotate_clockwise_i8_16x16(sptr, dptr,ih, iw, IH, IW);
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
            else{
               for (size_t ih = 0; ih < IH; ++ih) {
                    for (size_t iw = 0; iw < IW; ++iw) {
                       rotate_pixel(sptr, dptr, ih, iw, IH, IW, C, false);
                    }
                }
            }
        })";
    std::string body_temp = R"(
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

    writer << StringTemplate::StringTemplateArgs()
                      .add("kernel_sig", kernel_sig)
                      .render(body_temp);
    return writer.str();
}
// vim: syntax=cpp.doxygen
