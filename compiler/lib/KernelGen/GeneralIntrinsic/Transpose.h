/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/Transpose.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {
namespace {
std::string transpose_naive(std::string spec) {
    std::string temp = R"(
static inline void transpose_naive(${spec}* src, ${spec}* dst, int m, int n, int ch, int src_stride, int dst_stride){
    for(int row_id = 0; row_id < m; ++row_id)
    for(int col_id = 0; col_id < n; ++col_id){
        ${spec} * dst_ptr = dst + row_id * dst_stride + col_id * ch;
        ${spec} * src_ptr = src + col_id * src_stride + row_id * ch;
        for(int channel_id = 0 ; channel_id < ch; ++channel_id){
            dst_ptr[channel_id] = src_ptr[channel_id];
        }
    }
}
    )";

    return StringTemplate::StringTemplateArgs().add("spec", spec).render(temp);
}

std::string trans_3x32_i8_contig_src() {
    return R"(

static inline void trans_3x32_i8_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    int8_t* src_ptr = (int8_t*)src;
    int8_t* dst_ptr = (int8_t*)dst;
    GI_INT8_V3_t src0 = GiLoadUzipInt8V3(src_ptr + 0 * 16); 
    GI_INT8_t src00 = GiGetSubVectorInt8V3(src0, 0);
    GI_INT8_t src01 = GiGetSubVectorInt8V3(src0, 1);
    GI_INT8_t src02 = GiGetSubVectorInt8V3(src0, 2);
    GI_INT8_V3_t src1 = GiLoadUzipInt8V3(src_ptr + 3 * 16);
    GI_INT8_t src10 = GiGetSubVectorInt8V3(src1, 0);
    GI_INT8_t src11 = GiGetSubVectorInt8V3(src1, 1);
    GI_INT8_t src12 = GiGetSubVectorInt8V3(src1, 2);

    GiStoreInt8(dst_ptr + 0 * dst_step + 0,  src00);
    GiStoreInt8(dst_ptr + 0 * dst_step + 16, src10);
    GiStoreInt8(dst_ptr + 1 * dst_step + 0,  src01);
    GiStoreInt8(dst_ptr + 1 * dst_step + 16, src11);
    GiStoreInt8(dst_ptr + 2 * dst_step + 0,  src02);
    GiStoreInt8(dst_ptr + 2 * dst_step + 16, src12);
}
    )";
}

std::string trans_3x64_i8_contig_src() {
    return R"(

static inline void trans_3x64_i8_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    int8_t* src_ptr = (int8_t*)src;
    int8_t* dst_ptr = (int8_t*)dst;
    GI_INT8_V3_t src0 = GiLoadUzipInt8V3(src_ptr + 0 * 16); 
    GI_INT8_t src00 = GiGetSubVectorInt8V3(src0, 0);
    GI_INT8_t src01 = GiGetSubVectorInt8V3(src0, 1);
    GI_INT8_t src02 = GiGetSubVectorInt8V3(src0, 2);
    GI_INT8_V3_t src1 = GiLoadUzipInt8V3(src_ptr + 3 * 16);
    GI_INT8_t src10 = GiGetSubVectorInt8V3(src1, 0);
    GI_INT8_t src11 = GiGetSubVectorInt8V3(src1, 1);
    GI_INT8_t src12 = GiGetSubVectorInt8V3(src1, 2);

    GI_INT8_V3_t src2 = GiLoadUzipInt8V3(src_ptr + 6 * 16); 
    GI_INT8_t src20 = GiGetSubVectorInt8V3(src2, 0);
    GI_INT8_t src21 = GiGetSubVectorInt8V3(src2, 1);
    GI_INT8_t src22 = GiGetSubVectorInt8V3(src2, 2);
    GI_INT8_V3_t src3 = GiLoadUzipInt8V3(src_ptr + 9 * 16);
    GI_INT8_t src30 = GiGetSubVectorInt8V3(src3, 0);
    GI_INT8_t src31 = GiGetSubVectorInt8V3(src3, 1);
    GI_INT8_t src32 = GiGetSubVectorInt8V3(src3, 2);

    GiStoreInt8(dst_ptr + 0 * dst_step + 0 * 16, src00);
    GiStoreInt8(dst_ptr + 0 * dst_step + 1 * 16, src10);
    GiStoreInt8(dst_ptr + 0 * dst_step + 2 * 16, src20);
    GiStoreInt8(dst_ptr + 0 * dst_step + 3 * 16, src30);
    GiStoreInt8(dst_ptr + 1 * dst_step + 0 * 16, src01);
    GiStoreInt8(dst_ptr + 1 * dst_step + 1 * 16, src11);
    GiStoreInt8(dst_ptr + 1 * dst_step + 2 * 16, src21);
    GiStoreInt8(dst_ptr + 1 * dst_step + 3 * 16, src31);
    GiStoreInt8(dst_ptr + 2 * dst_step + 0 * 16, src02);
    GiStoreInt8(dst_ptr + 2 * dst_step + 1 * 16, src12);
    GiStoreInt8(dst_ptr + 2 * dst_step + 2 * 16, src22);
    GiStoreInt8(dst_ptr + 2 * dst_step + 3 * 16, src32);
}
    )";
}

std::string trans_16x16_i8() {
    return R"(
static inline void trans_16x16_i8(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    // 16x16

    GI_INT8_t q0 =  GiLoadInt8(src + 0 * src_step);
    GI_INT8_t q1 =  GiLoadInt8(src + 1 * src_step);
    GI_INT8_t q2 =  GiLoadInt8(src + 2 * src_step);
    GI_INT8_t q3 =  GiLoadInt8(src + 3 * src_step);
    GI_INT8_t q4 =  GiLoadInt8(src + 4 * src_step);
    GI_INT8_t q5 =  GiLoadInt8(src + 5 * src_step);
    GI_INT8_t q6 =  GiLoadInt8(src + 6 * src_step);
    GI_INT8_t q7 =  GiLoadInt8(src + 7 * src_step);
    GI_INT8_t q8 =  GiLoadInt8(src + 8 * src_step);
    GI_INT8_t q9 =  GiLoadInt8(src + 9 * src_step);
    GI_INT8_t qa =  GiLoadInt8(src + 10 * src_step);
    GI_INT8_t qb =  GiLoadInt8(src + 11 * src_step);
    GI_INT8_t qc =  GiLoadInt8(src + 12 * src_step);
    GI_INT8_t qd =  GiLoadInt8(src + 13 * src_step);
    GI_INT8_t qe =  GiLoadInt8(src + 14 * src_step);
    GI_INT8_t qf =  GiLoadInt8(src + 15 * src_step);

    GI_INT8_t tmp = q0; 
    q0 = GiZipV0Int8(q0, q1); 
    q1 = GiZipV1Int8(tmp, q1);
    tmp = q2; 
    q2 = GiZipV0Int8(q2, q3);
    q3 = GiZipV1Int8(tmp, q3);
    tmp = q4; 
    q4 = GiZipV0Int8(q4, q5);
    q5 = GiZipV1Int8(tmp, q5);
    tmp = q6; 
    q6 = GiZipV0Int8(q6, q7);
    q7 = GiZipV1Int8(tmp, q7);
    tmp = q8; 
    q8 = GiZipV0Int8(q8, q9);
    q9 = GiZipV1Int8(tmp, q9);
    tmp = qa; 
    qa = GiZipV0Int8(qa, qb);
    qb = GiZipV1Int8(tmp, qb);
    tmp = qc; 
    qc = GiZipV0Int8(qc, qd);
    qd = GiZipV1Int8(tmp, qd);
    tmp = qe; 
    qe = GiZipV0Int8(qe, qf);  
    qf = GiZipV1Int8(tmp, qf);

    GI_INT16_t q00 = GiReinterpretInt8AsInt16(q0); 
    GI_INT16_t q10 = GiReinterpretInt8AsInt16(q2);
    GI_INT16_t q20 = GiReinterpretInt8AsInt16(q1);
    GI_INT16_t q30 = GiReinterpretInt8AsInt16(q3);
    GI_INT16_t q40 = GiReinterpretInt8AsInt16(q4);
    GI_INT16_t q50 = GiReinterpretInt8AsInt16(q6);
    GI_INT16_t q60 = GiReinterpretInt8AsInt16(q5);
    GI_INT16_t q70 = GiReinterpretInt8AsInt16(q7);
    GI_INT16_t q80 = GiReinterpretInt8AsInt16(q8);
    GI_INT16_t q90 = GiReinterpretInt8AsInt16(qa);
    GI_INT16_t qa0 = GiReinterpretInt8AsInt16(q9);
    GI_INT16_t qb0 = GiReinterpretInt8AsInt16(qb);
    GI_INT16_t qc0 = GiReinterpretInt8AsInt16(qc);
    GI_INT16_t qd0 = GiReinterpretInt8AsInt16(qe);
    GI_INT16_t qe0 = GiReinterpretInt8AsInt16(qd);  
    GI_INT16_t qf0 = GiReinterpretInt8AsInt16(qf);

    GI_INT16_t tmp0 = q00;    
    q00 = GiZipV0Int16(q00, q10); 
    q10 = GiZipV1Int16(tmp0, q10);

    tmp0 = q20; 
    q20 = GiZipV0Int16(q20, q30);
    q30 = GiZipV1Int16(tmp0, q30);
    tmp0 = q40;
    q40 = GiZipV0Int16(q40, q50);
    q50 = GiZipV1Int16(tmp0, q50);
    tmp0 = q60;
    q60 = GiZipV0Int16(q60, q70);
    q70 = GiZipV1Int16(tmp0, q70);
    tmp0 = q80;
    q80 = GiZipV0Int16(q80, q90);
    q90 = GiZipV1Int16(tmp0, q90);
    tmp0 = qa0;
    qa0 = GiZipV0Int16(qa0, qb0);
    qb0 = GiZipV1Int16(tmp0, qb0);
    tmp0 = qc0;
    qc0 = GiZipV0Int16(qc0, qd0);
    qd0 = GiZipV1Int16(tmp0, qd0);
    tmp0 = qe0;
    qe0 = GiZipV0Int16(qe0, qf0);  
    qf0 = GiZipV1Int16(tmp0, qf0);

    GI_INT32_t q01 = GiReinterpretInt16AsInt32(q00); 
    GI_INT32_t q11 = GiReinterpretInt16AsInt32(q40);
    GI_INT32_t q21 = GiReinterpretInt16AsInt32(q10);
    GI_INT32_t q31 = GiReinterpretInt16AsInt32(q50);
    GI_INT32_t q41 = GiReinterpretInt16AsInt32(q20);
    GI_INT32_t q51 = GiReinterpretInt16AsInt32(q60);
    GI_INT32_t q61 = GiReinterpretInt16AsInt32(q30);
    GI_INT32_t q71 = GiReinterpretInt16AsInt32(q70);
    GI_INT32_t q81 = GiReinterpretInt16AsInt32(q80);
    GI_INT32_t q91 = GiReinterpretInt16AsInt32(qc0);
    GI_INT32_t qa1 = GiReinterpretInt16AsInt32(q90);
    GI_INT32_t qb1 = GiReinterpretInt16AsInt32(qd0);
    GI_INT32_t qc1 = GiReinterpretInt16AsInt32(qa0);
    GI_INT32_t qd1 = GiReinterpretInt16AsInt32(qe0);
    GI_INT32_t qe1 = GiReinterpretInt16AsInt32(qb0);  
    GI_INT32_t qf1 = GiReinterpretInt16AsInt32(qf0);

    GI_INT32_t tmp1 = q01;  
    q01 = GiZipV0Int32(q01, q11); 
    q11 = GiZipV1Int32(tmp1, q11);
    tmp1 = q21;
    q21 = GiZipV0Int32(q21, q31);
    q31 = GiZipV1Int32(tmp1, q31);
    
    tmp1 = q41;
    q41 = GiZipV0Int32(q41, q51);
    q51 = GiZipV1Int32(tmp1, q51);
    tmp1 = q61;
    q61 = GiZipV0Int32(q61, q71);
    q71 = GiZipV1Int32(tmp1, q71);
    tmp1 = q81;
    q81 = GiZipV0Int32(q81, q91);
    q91 = GiZipV1Int32(tmp1, q91);
    tmp1 = qa1;
    qa1 = GiZipV0Int32(qa1, qb1);
    qb1 = GiZipV1Int32(tmp1, qb1);
    tmp1 = qc1;
    qc1 = GiZipV0Int32(qc1, qd1);
    qd1 = GiZipV1Int32(tmp1, qd1);
    tmp1 = qe1;
    qe1 = GiZipV0Int32(qe1, qf1);  
    qf1 = GiZipV1Int32(tmp1, qf1);
    
    tmp1 = q01;
    q01 = GiCombineInt32Low(q01, q81); 
    q81 = GiCombineInt32High(tmp1, q81);
    tmp1 = q11;
    q11 = GiCombineInt32Low(q11, q91);
    q91 = GiCombineInt32High(tmp1, q91);
    tmp1 = q21;
    q21 = GiCombineInt32Low(q21, qa1);
    qa1 = GiCombineInt32High(tmp1, qa1);
    tmp1 = q31;
    q31 = GiCombineInt32Low(q31, qb1);
    qb1 = GiCombineInt32High(tmp1, qb1);
    tmp1 = q41;
    q41 = GiCombineInt32Low(q41, qc1);
    qc1 = GiCombineInt32High(tmp1, qc1);
    tmp1 = q51;
    q51 = GiCombineInt32Low(q51, qd1);
    qd1 = GiCombineInt32High(tmp1, qd1);
    tmp1 = q61;
    q61 = GiCombineInt32Low(q61, qe1);
    qe1 = GiCombineInt32High(tmp1, qe1);
    tmp1 = q71;
    q71 = GiCombineInt32Low(q71, qf1);  
    qf1 = GiCombineInt32High(tmp1, qf1);
    GiStoreInt8(dst + 0 * dst_step,  GiReinterInt32ToInt8(q01));
    GiStoreInt8(dst + 1 * dst_step,  GiReinterInt32ToInt8(q81));
    GiStoreInt8(dst + 2 * dst_step,  GiReinterInt32ToInt8(q11));
    GiStoreInt8(dst + 3 * dst_step,  GiReinterInt32ToInt8(q91));
    GiStoreInt8(dst + 4 * dst_step,  GiReinterInt32ToInt8(q21));
    GiStoreInt8(dst + 5 * dst_step,  GiReinterInt32ToInt8(qa1));
    GiStoreInt8(dst + 6 * dst_step,  GiReinterInt32ToInt8(q31));
    GiStoreInt8(dst + 7 * dst_step,  GiReinterInt32ToInt8(qb1));
    GiStoreInt8(dst + 8 * dst_step,  GiReinterInt32ToInt8(q41));
    GiStoreInt8(dst + 9 * dst_step,  GiReinterInt32ToInt8(qc1));
    GiStoreInt8(dst + 10 * dst_step, GiReinterInt32ToInt8(q51));
    GiStoreInt8(dst + 11 * dst_step, GiReinterInt32ToInt8(qd1));
    GiStoreInt8(dst + 12 * dst_step, GiReinterInt32ToInt8(q61));
    GiStoreInt8(dst + 13 * dst_step, GiReinterInt32ToInt8(qe1));
    GiStoreInt8(dst + 14 * dst_step, GiReinterInt32ToInt8(q71));
    GiStoreInt8(dst + 15 * dst_step, GiReinterInt32ToInt8(qf1));


}
    )";
}

std::string trans_4x16_i8_contig_src() {
    return R"(
static inline void trans_4x16_i8_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    int8_t* src_ptr = (int8_t*)src;
    int8_t* dst_ptr = (int8_t*)dst;
    GI_INT8_t src0 = GiLoadInt8(src_ptr + 0 * 16);  // A0A1A2A3B0B1B2B3C0C1C2C3D0D1D2D3
    GI_INT8_t src1 = GiLoadInt8(src_ptr + 1 * 16);  // E0E1E2E3F0F1F2F3G0G1G2G3H0H1H2H3
    GI_INT8_t src2 = GiLoadInt8(src_ptr + 2 * 16);  // I0I1I2I3J0J1J2J3K0K1K2K3L0L1L2L3
    GI_INT8_t src3 = GiLoadInt8(src_ptr + 3 * 16);  // M0M1M2M3N0N1N2N3O0O1O2O3P0P1P2P3

    GI_INT32_t n048c = GiReinterpretInt8AsInt32(GiInterleave4Int8(src0));     //A0B0C0D0A1B1C1D1A2B2C2D2A3B3C3D3
    GI_INT32_t n159d = GiReinterpretInt8AsInt32(GiInterleave4Int8(src1));     //E0F0G0H0E1F1G1H1E2F2G2H2E3F3G3H3
    GI_INT32_t n26ae = GiReinterpretInt8AsInt32(GiInterleave4Int8(src2));     //I0J0K0L0I1J1K1L1I2J2K2L2I3J3K3L3
    GI_INT32_t n37bf = GiReinterpretInt8AsInt32(GiInterleave4Int8(src3));     //M0N0O0P0M1N1O1P1M2N2O2P2M3N3O3P3

    GI_INT32_t n0145 = GiZipV0Int32(n048c, n159d);
    GI_INT32_t n89cd = GiZipV1Int32(n048c, n159d);
    GI_INT32_t n2367 = GiZipV0Int32(n26ae, n37bf);
    GI_INT32_t nabef = GiZipV1Int32(n26ae, n37bf);

    GI_INT8_t n0123 = GiReinterInt32ToInt8(GiCombineInt32Low(n0145, n2367));
    GI_INT8_t n4567 = GiReinterInt32ToInt8(GiCombineInt32High(n0145, n2367));
    GI_INT8_t n89ab = GiReinterInt32ToInt8(GiCombineInt32Low(n89cd, nabef));
    GI_INT8_t ncdef = GiReinterInt32ToInt8(GiCombineInt32High(n89cd, nabef));

    GiStoreInt8(dst_ptr + 0 * dst_step,  n0123);
    GiStoreInt8(dst_ptr + 1 * dst_step,  n4567);
    GiStoreInt8(dst_ptr + 2 * dst_step,  n89ab);
    GiStoreInt8(dst_ptr + 3 * dst_step,  ncdef);
}
    )";
}

std::string trans_8x8_i16() {
    return R"(
static inline void trans_8x8_i16(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    int16_t* src_ptr = (int16_t*)src;
    int8_t* dst_ptr = (int8_t*)dst;
    size_t i16_dst_step = dst_step * 2;
    GI_INT16_t src0 = GiLoadInt16(src_ptr + 0 * src_step);  // A0A1A2A3A4A5A6A7
    GI_INT16_t src1 = GiLoadInt16(src_ptr + 1 * src_step);  // B0B1B2B3B4B5B6B7
    GI_INT16_t src2 = GiLoadInt16(src_ptr + 2 * src_step);  // C0C1C2C3C4C5C6C7
    GI_INT16_t src3 = GiLoadInt16(src_ptr + 3 * src_step);  // D0D1D2D3D4D5D6D7
    GI_INT16_t src4 = GiLoadInt16(src_ptr + 4 * src_step);  // E0E1E2E3E4E5E6E7
    GI_INT16_t src5 = GiLoadInt16(src_ptr + 5 * src_step);  // F0F1F2F3F4F5F6F7
    GI_INT16_t src6 = GiLoadInt16(src_ptr + 6 * src_step);  // G0G1G2G3G4G5G6G7
    GI_INT16_t src7 = GiLoadInt16(src_ptr + 7 * src_step);  // H0H1H2H3H4H5H6H7

    GI_INT16_t ab_low = GiZipV0Int16(src0, src1);   // A0B0A1B1A2B2A3B3
    GI_INT16_t ab_high = GiZipV1Int16(src0, src1);  // A4B4A5B5A6B6A7B7
    GI_INT16_t cd_low = GiZipV0Int16(src2, src3);   // C0D0C1D1C2D2C3D3
    GI_INT16_t cd_high = GiZipV1Int16(src2, src3);  // C4D4C5D5C6D6C7D7
    GI_INT16_t ef_low = GiZipV0Int16(src4, src5);   // E0F0E1F1E2F2E3F3
    GI_INT16_t ef_high = GiZipV1Int16(src4, src5);  // E4F4E5F5E6F6E7F7
    GI_INT16_t gh_low = GiZipV0Int16(src6, src7);   // G0H0G1H1G2H2G3H3
    GI_INT16_t gh_high = GiZipV1Int16(src6, src7);  // G4H4G5H5G6H6G7H7

    GI_INT32_t abcd_0 = GiZipV0Int32(GiReinterpretInt16AsInt32(ab_low),
            GiReinterpretInt16AsInt32(cd_low));  // A0B0C0D0A1B1C1D1
    GI_INT32_t abcd_2 = GiZipV1Int32(GiReinterpretInt16AsInt32(ab_low),
            GiReinterpretInt16AsInt32(cd_low));  // A2B2C2D2A3B3C3D3
    GI_INT32_t abcd_4 =GiZipV0Int32(GiReinterpretInt16AsInt32(ab_high),
            GiReinterpretInt16AsInt32(cd_high));  // A4B4C4D4A5B5C5D5
    GI_INT32_t abcd_6 = GiZipV1Int32(GiReinterpretInt16AsInt32(ab_high),
            GiReinterpretInt16AsInt32(cd_high));  // A6B6C6D6A7B7C7D7
    GI_INT32_t efgh_0 = GiZipV0Int32(GiReinterpretInt16AsInt32(ef_low),
            GiReinterpretInt16AsInt32(gh_low));  // E0F0G0H0E1F1G1H1
    GI_INT32_t efgh_2 = GiZipV1Int32(GiReinterpretInt16AsInt32(ef_low),
            GiReinterpretInt16AsInt32(gh_low));  // E2F2G2H2E3F3G3H3
    GI_INT32_t efgh_4 = GiZipV0Int32(GiReinterpretInt16AsInt32(ef_high),
            GiReinterpretInt16AsInt32(gh_high));  // E4F4G4H4E5F5G5H5
    GI_INT32_t efgh_6 = GiZipV1Int32(GiReinterpretInt16AsInt32(ef_high),
            GiReinterpretInt16AsInt32(gh_high));  // E6F6G6H6E7F7G7H7

    GI_INT32_t row_0 = GiCombineInt32Low(abcd_0, efgh_0);
    GI_INT32_t row_1 = GiCombineInt32High(abcd_0, efgh_0);
    GI_INT32_t row_2 = GiCombineInt32Low(abcd_2, efgh_2);
    GI_INT32_t row_3 = GiCombineInt32High(abcd_2, efgh_2);
    GI_INT32_t row_4 = GiCombineInt32Low(abcd_4, efgh_4);
    GI_INT32_t row_5 = GiCombineInt32High(abcd_4, efgh_4);
    GI_INT32_t row_6 = GiCombineInt32Low(abcd_6, efgh_6);
    GI_INT32_t row_7 = GiCombineInt32High(abcd_6, efgh_6);

    // int32 store may cause bus error for unaligned dst address
    GiStoreInt8(dst_ptr + 0 * i16_dst_step, GiReinterInt32ToInt8(row_0));
    GiStoreInt8(dst_ptr + 1 * i16_dst_step, GiReinterInt32ToInt8(row_1));
    GiStoreInt8(dst_ptr + 2 * i16_dst_step, GiReinterInt32ToInt8(row_2));
    GiStoreInt8(dst_ptr + 3 * i16_dst_step, GiReinterInt32ToInt8(row_3));
    GiStoreInt8(dst_ptr + 4 * i16_dst_step, GiReinterInt32ToInt8(row_4));
    GiStoreInt8(dst_ptr + 5 * i16_dst_step, GiReinterInt32ToInt8(row_5));
    GiStoreInt8(dst_ptr + 6 * i16_dst_step, GiReinterInt32ToInt8(row_6));
    GiStoreInt8(dst_ptr + 7 * i16_dst_step, GiReinterInt32ToInt8(row_7));
}

    )";
}

std::string trans_16x16_i8x3() {
    return R"(
static inline void trans_16x16_i8x3(const void* src, void* dst, const size_t src_step,
                      const size_t dst_step) {
    int8_t* src_ptr = (int8_t*)src;
    int8_t* dst_ptr = (int8_t*)dst;
    GI_INT8_V3_t src0 = GiLoadUzipInt8V3(src_ptr + 0 * src_step);
    GI_INT8_V3_t src1 = GiLoadUzipInt8V3(src_ptr + 1 * src_step);
    GI_INT8_V3_t src2 = GiLoadUzipInt8V3(src_ptr + 2 * src_step);
    GI_INT8_V3_t src3 = GiLoadUzipInt8V3(src_ptr + 3 * src_step);
    GI_INT8_V3_t src4 = GiLoadUzipInt8V3(src_ptr + 4 * src_step);
    GI_INT8_V3_t src5 = GiLoadUzipInt8V3(src_ptr + 5 * src_step);
    GI_INT8_V3_t src6 = GiLoadUzipInt8V3(src_ptr + 6 * src_step);
    GI_INT8_V3_t src7 = GiLoadUzipInt8V3(src_ptr + 7 * src_step);
    GI_INT8_V3_t src8 = GiLoadUzipInt8V3(src_ptr + 8 * src_step);
    GI_INT8_V3_t src9 = GiLoadUzipInt8V3(src_ptr + 9 * src_step);
    GI_INT8_V3_t srca = GiLoadUzipInt8V3(src_ptr + 10 * src_step);
    GI_INT8_V3_t srcb = GiLoadUzipInt8V3(src_ptr + 11 * src_step);
    GI_INT8_V3_t srcc = GiLoadUzipInt8V3(src_ptr + 12 * src_step);
    GI_INT8_V3_t srcd = GiLoadUzipInt8V3(src_ptr + 13 * src_step);
    GI_INT8_V3_t srce = GiLoadUzipInt8V3(src_ptr + 14 * src_step);
    GI_INT8_V3_t srcf = GiLoadUzipInt8V3(src_ptr + 15 * src_step);

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
    GI_INT8_t srca0 = GiGetSubVectorInt8V3(srca, 0);
    GI_INT8_t srcb0 = GiGetSubVectorInt8V3(srcb, 0);
    GI_INT8_t srcc0 = GiGetSubVectorInt8V3(srcc, 0);
    GI_INT8_t srcd0 = GiGetSubVectorInt8V3(srcd, 0);
    GI_INT8_t srce0 = GiGetSubVectorInt8V3(srce, 0);
    GI_INT8_t srcf0 = GiGetSubVectorInt8V3(srcf, 0);

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
    GI_INT8_t srca1 = GiGetSubVectorInt8V3(srca, 1);
    GI_INT8_t srcb1 = GiGetSubVectorInt8V3(srcb, 1);
    GI_INT8_t srcc1 = GiGetSubVectorInt8V3(srcc, 1);
    GI_INT8_t srcd1 = GiGetSubVectorInt8V3(srcd, 1);
    GI_INT8_t srce1 = GiGetSubVectorInt8V3(srce, 1);
    GI_INT8_t srcf1 = GiGetSubVectorInt8V3(srcf, 1);

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
    GI_INT8_t srca2 = GiGetSubVectorInt8V3(srca, 2);
    GI_INT8_t srcb2 = GiGetSubVectorInt8V3(srcb, 2);
    GI_INT8_t srcc2 = GiGetSubVectorInt8V3(srcc, 2);
    GI_INT8_t srcd2 = GiGetSubVectorInt8V3(srcd, 2);
    GI_INT8_t srce2 = GiGetSubVectorInt8V3(srce, 2);
    GI_INT8_t srcf2 = GiGetSubVectorInt8V3(srcf, 2);



    GI_INT8_t c0_ab = GiZipV0Int8(src00, src10);
    GI_INT8_t c0_cd = GiZipV0Int8(src20, src30);
    GI_INT8_t c0_ef = GiZipV0Int8(src40, src50);
    GI_INT8_t c0_gh = GiZipV0Int8(src60, src70);
    GI_INT8_t c0_ij = GiZipV0Int8(src80, src90);
    GI_INT8_t c0_kl = GiZipV0Int8(srca0, srcb0);
    GI_INT8_t c0_mn = GiZipV0Int8(srcc0, srcd0);
    GI_INT8_t c0_op = GiZipV0Int8(srce0, srcf0);

    GI_INT8_t c1_ab = GiZipV0Int8(src01, src11);
    GI_INT8_t c1_cd = GiZipV0Int8(src21, src31);
    GI_INT8_t c1_ef = GiZipV0Int8(src41, src51);
    GI_INT8_t c1_gh = GiZipV0Int8(src61, src71);
    GI_INT8_t c1_ij = GiZipV0Int8(src81, src91);
    GI_INT8_t c1_kl = GiZipV0Int8(srca1, srcb1);
    GI_INT8_t c1_mn = GiZipV0Int8(srcc1, srcd1);
    GI_INT8_t c1_op = GiZipV0Int8(srce1, srcf1);


    GI_INT8_t c2_ab = GiZipV0Int8(src02, src12);
    GI_INT8_t c2_cd = GiZipV0Int8(src22, src32);
    GI_INT8_t c2_ef = GiZipV0Int8(src42, src52);
    GI_INT8_t c2_gh = GiZipV0Int8(src62, src72);
    GI_INT8_t c2_ij = GiZipV0Int8(src82, src92);
    GI_INT8_t c2_kl = GiZipV0Int8(srca2, srcb2);
    GI_INT8_t c2_mn = GiZipV0Int8(srcc2, srcd2);
    GI_INT8_t c2_op = GiZipV0Int8(srce2, srcf2);


    GI_INT8_t c0_ab1 = GiZipV1Int8(src00, src10);
    GI_INT8_t c0_cd1 = GiZipV1Int8(src20, src30);
    GI_INT8_t c0_ef1 = GiZipV1Int8(src40, src50);
    GI_INT8_t c0_gh1 = GiZipV1Int8(src60, src70);
    GI_INT8_t c0_ij1 = GiZipV1Int8(src80, src90);
    GI_INT8_t c0_kl1 = GiZipV1Int8(srca0, srcb0);
    GI_INT8_t c0_mn1 = GiZipV1Int8(srcc0, srcd0);
    GI_INT8_t c0_op1 = GiZipV1Int8(srce0, srcf0);


    GI_INT8_t c1_ab1 = GiZipV1Int8(src01, src11);
    GI_INT8_t c1_cd1 = GiZipV1Int8(src21, src31);
    GI_INT8_t c1_ef1 = GiZipV1Int8(src41, src51);
    GI_INT8_t c1_gh1 = GiZipV1Int8(src61, src71);
    GI_INT8_t c1_ij1 = GiZipV1Int8(src81, src91);
    GI_INT8_t c1_kl1 = GiZipV1Int8(srca1, srcb1);
    GI_INT8_t c1_mn1 = GiZipV1Int8(srcc1, srcd1);
    GI_INT8_t c1_op1 = GiZipV1Int8(srce1, srcf1);


    GI_INT8_t c2_ab1 = GiZipV1Int8(src02, src12);
    GI_INT8_t c2_cd1 = GiZipV1Int8(src22, src32);
    GI_INT8_t c2_ef1 = GiZipV1Int8(src42, src52);
    GI_INT8_t c2_gh1 = GiZipV1Int8(src62, src72);
    GI_INT8_t c2_ij1 = GiZipV1Int8(src82, src92);
    GI_INT8_t c2_kl1 = GiZipV1Int8(srca2, srcb2);
    GI_INT8_t c2_mn1 = GiZipV1Int8(srcc2, srcd2);
    GI_INT8_t c2_op1 = GiZipV1Int8(srce2, srcf2);

    GI_INT16_t c0_abcd_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_ab), GiReinterpretInt8AsInt16(c0_cd));
    GI_INT16_t c0_abcd_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_ab), GiReinterpretInt8AsInt16(c0_cd));
    GI_INT16_t c0_efgh_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_ef), GiReinterpretInt8AsInt16(c0_gh));
    GI_INT16_t c0_efgh_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_ef), GiReinterpretInt8AsInt16(c0_gh));

    GI_INT16_t c1_abcd_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_ab), GiReinterpretInt8AsInt16(c1_cd));
    GI_INT16_t c1_abcd_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_ab), GiReinterpretInt8AsInt16(c1_cd));
    GI_INT16_t c1_efgh_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_ef), GiReinterpretInt8AsInt16(c1_gh));
    GI_INT16_t c1_efgh_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_ef), GiReinterpretInt8AsInt16(c1_gh));

    GI_INT16_t c2_abcd_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_ab), GiReinterpretInt8AsInt16(c2_cd));
    GI_INT16_t c2_abcd_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_ab), GiReinterpretInt8AsInt16(c2_cd));
    GI_INT16_t c2_efgh_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_ef), GiReinterpretInt8AsInt16(c2_gh));
    GI_INT16_t c2_efgh_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_ef), GiReinterpretInt8AsInt16(c2_gh));

    GI_INT32_t c0_0  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_abcd_low ), GiReinterpretInt16AsInt32(c0_efgh_low ));
    GI_INT32_t c0_1  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_abcd_low ), GiReinterpretInt16AsInt32(c0_efgh_low ));
    GI_INT32_t c0_2  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_abcd_high), GiReinterpretInt16AsInt32(c0_efgh_high));
    GI_INT32_t c0_3  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_abcd_high), GiReinterpretInt16AsInt32(c0_efgh_high));

    GI_INT32_t c1_0  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_abcd_low ), GiReinterpretInt16AsInt32(c1_efgh_low ));
    GI_INT32_t c1_1  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_abcd_low ), GiReinterpretInt16AsInt32(c1_efgh_low ));
    GI_INT32_t c1_2  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_abcd_high), GiReinterpretInt16AsInt32(c1_efgh_high));
    GI_INT32_t c1_3  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_abcd_high), GiReinterpretInt16AsInt32(c1_efgh_high));

    GI_INT32_t c2_0  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_abcd_low ), GiReinterpretInt16AsInt32(c2_efgh_low ));   
    GI_INT32_t c2_1  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_abcd_low ), GiReinterpretInt16AsInt32(c2_efgh_low ));   
    GI_INT32_t c2_2  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_abcd_high), GiReinterpretInt16AsInt32(c2_efgh_high));   
    GI_INT32_t c2_3  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_abcd_high), GiReinterpretInt16AsInt32(c2_efgh_high));

    GI_INT16_t c0_abcd1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_ab1), GiReinterpretInt8AsInt16(c0_cd1));
    GI_INT16_t c0_abcd1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_ab1), GiReinterpretInt8AsInt16(c0_cd1));
    GI_INT16_t c0_efgh1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_ef1), GiReinterpretInt8AsInt16(c0_gh1));
    GI_INT16_t c0_efgh1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_ef1), GiReinterpretInt8AsInt16(c0_gh1));

    GI_INT16_t c1_abcd1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_ab1), GiReinterpretInt8AsInt16(c1_cd1));
    GI_INT16_t c1_abcd1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_ab1), GiReinterpretInt8AsInt16(c1_cd1));
    GI_INT16_t c1_efgh1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_ef1), GiReinterpretInt8AsInt16(c1_gh1));
    GI_INT16_t c1_efgh1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_ef1), GiReinterpretInt8AsInt16(c1_gh1));

    GI_INT16_t c2_abcd1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_ab1), GiReinterpretInt8AsInt16(c2_cd1));
    GI_INT16_t c2_abcd1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_ab1), GiReinterpretInt8AsInt16(c2_cd1));
    GI_INT16_t c2_efgh1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_ef1), GiReinterpretInt8AsInt16(c2_gh1));
    GI_INT16_t c2_efgh1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_ef1), GiReinterpretInt8AsInt16(c2_gh1));

    GI_INT32_t c0_10  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_abcd1_low ), GiReinterpretInt16AsInt32(c0_efgh1_low ));
    GI_INT32_t c0_11  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_abcd1_low ), GiReinterpretInt16AsInt32(c0_efgh1_low ));
    GI_INT32_t c0_12  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_abcd1_high), GiReinterpretInt16AsInt32(c0_efgh1_high));
    GI_INT32_t c0_13  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_abcd1_high), GiReinterpretInt16AsInt32(c0_efgh1_high));

    GI_INT32_t c1_10  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_abcd1_low ), GiReinterpretInt16AsInt32(c1_efgh1_low ));
    GI_INT32_t c1_11  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_abcd1_low ), GiReinterpretInt16AsInt32(c1_efgh1_low ));
    GI_INT32_t c1_12  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_abcd1_high), GiReinterpretInt16AsInt32(c1_efgh1_high));
    GI_INT32_t c1_13  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_abcd1_high), GiReinterpretInt16AsInt32(c1_efgh1_high));

    GI_INT32_t c2_10  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_abcd1_low ), GiReinterpretInt16AsInt32(c2_efgh1_low ));   
    GI_INT32_t c2_11  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_abcd1_low ), GiReinterpretInt16AsInt32(c2_efgh1_low ));   
    GI_INT32_t c2_12  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_abcd1_high), GiReinterpretInt16AsInt32(c2_efgh1_high));   
    GI_INT32_t c2_13  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_abcd1_high), GiReinterpretInt16AsInt32(c2_efgh1_high));

    GI_INT16_t c0_ijkl_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_ij), GiReinterpretInt8AsInt16(c0_kl));
    GI_INT16_t c0_ijkl_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_ij), GiReinterpretInt8AsInt16(c0_kl));
    GI_INT16_t c0_mnop_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_mn), GiReinterpretInt8AsInt16(c0_op));
    GI_INT16_t c0_mnop_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_mn), GiReinterpretInt8AsInt16(c0_op));

    GI_INT16_t c1_ijkl_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_ij), GiReinterpretInt8AsInt16(c1_kl));
    GI_INT16_t c1_ijkl_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_ij), GiReinterpretInt8AsInt16(c1_kl));
    GI_INT16_t c1_mnop_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_mn), GiReinterpretInt8AsInt16(c1_op));
    GI_INT16_t c1_mnop_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_mn), GiReinterpretInt8AsInt16(c1_op));

    GI_INT16_t c2_ijkl_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_ij), GiReinterpretInt8AsInt16(c2_kl));
    GI_INT16_t c2_ijkl_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_ij), GiReinterpretInt8AsInt16(c2_kl));
    GI_INT16_t c2_mnop_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_mn), GiReinterpretInt8AsInt16(c2_op));
    GI_INT16_t c2_mnop_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_mn), GiReinterpretInt8AsInt16(c2_op));

    GI_INT32_t c0_20  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_ijkl_low ), GiReinterpretInt16AsInt32(c0_mnop_low ));
    GI_INT32_t c0_21  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_ijkl_low ), GiReinterpretInt16AsInt32(c0_mnop_low ));
    GI_INT32_t c0_22  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_ijkl_high), GiReinterpretInt16AsInt32(c0_mnop_high));
    GI_INT32_t c0_23  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_ijkl_high), GiReinterpretInt16AsInt32(c0_mnop_high));

    GI_INT32_t c1_20  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_ijkl_low ), GiReinterpretInt16AsInt32(c1_mnop_low ));
    GI_INT32_t c1_21  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_ijkl_low ), GiReinterpretInt16AsInt32(c1_mnop_low ));
    GI_INT32_t c1_22  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_ijkl_high), GiReinterpretInt16AsInt32(c1_mnop_high));
    GI_INT32_t c1_23  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_ijkl_high), GiReinterpretInt16AsInt32(c1_mnop_high));

    GI_INT32_t c2_20  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_ijkl_low ), GiReinterpretInt16AsInt32(c2_mnop_low ));   
    GI_INT32_t c2_21  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_ijkl_low ), GiReinterpretInt16AsInt32(c2_mnop_low ));   
    GI_INT32_t c2_22  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_ijkl_high), GiReinterpretInt16AsInt32(c2_mnop_high));   
    GI_INT32_t c2_23  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_ijkl_high), GiReinterpretInt16AsInt32(c2_mnop_high));

    GI_INT16_t c0_ijkl1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_ij1), GiReinterpretInt8AsInt16(c0_kl1));
    GI_INT16_t c0_ijkl1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_ij1), GiReinterpretInt8AsInt16(c0_kl1));
    GI_INT16_t c0_mnop1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c0_mn1), GiReinterpretInt8AsInt16(c0_op1));
    GI_INT16_t c0_mnop1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c0_mn1), GiReinterpretInt8AsInt16(c0_op1));

    GI_INT16_t c1_ijkl1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_ij1), GiReinterpretInt8AsInt16(c1_kl1));
    GI_INT16_t c1_ijkl1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_ij1), GiReinterpretInt8AsInt16(c1_kl1));
    GI_INT16_t c1_mnop1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c1_mn1), GiReinterpretInt8AsInt16(c1_op1));
    GI_INT16_t c1_mnop1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c1_mn1), GiReinterpretInt8AsInt16(c1_op1));

    GI_INT16_t c2_ijkl1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_ij1), GiReinterpretInt8AsInt16(c2_kl1));
    GI_INT16_t c2_ijkl1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_ij1), GiReinterpretInt8AsInt16(c2_kl1));
    GI_INT16_t c2_mnop1_low  = GiZipV0Int16(GiReinterpretInt8AsInt16(c2_mn1), GiReinterpretInt8AsInt16(c2_op1));
    GI_INT16_t c2_mnop1_high = GiZipV1Int16(GiReinterpretInt8AsInt16(c2_mn1), GiReinterpretInt8AsInt16(c2_op1));

    GI_INT32_t c0_30  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_ijkl1_low ), GiReinterpretInt16AsInt32(c0_mnop1_low ));
    GI_INT32_t c0_31  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_ijkl1_low ), GiReinterpretInt16AsInt32(c0_mnop1_low ));
    GI_INT32_t c0_32  = GiZipV0Int32(GiReinterpretInt16AsInt32(c0_ijkl1_high), GiReinterpretInt16AsInt32(c0_mnop1_high));
    GI_INT32_t c0_33  = GiZipV1Int32(GiReinterpretInt16AsInt32(c0_ijkl1_high), GiReinterpretInt16AsInt32(c0_mnop1_high));

    GI_INT32_t c1_30  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_ijkl1_low ), GiReinterpretInt16AsInt32(c1_mnop1_low ));
    GI_INT32_t c1_31  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_ijkl1_low ), GiReinterpretInt16AsInt32(c1_mnop1_low ));
    GI_INT32_t c1_32  = GiZipV0Int32(GiReinterpretInt16AsInt32(c1_ijkl1_high), GiReinterpretInt16AsInt32(c1_mnop1_high));
    GI_INT32_t c1_33  = GiZipV1Int32(GiReinterpretInt16AsInt32(c1_ijkl1_high), GiReinterpretInt16AsInt32(c1_mnop1_high));

    GI_INT32_t c2_30  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_ijkl1_low ), GiReinterpretInt16AsInt32(c2_mnop1_low ));   
    GI_INT32_t c2_31  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_ijkl1_low ), GiReinterpretInt16AsInt32(c2_mnop1_low ));   
    GI_INT32_t c2_32  = GiZipV0Int32(GiReinterpretInt16AsInt32(c2_ijkl1_high), GiReinterpretInt16AsInt32(c2_mnop1_high));   
    GI_INT32_t c2_33  = GiZipV1Int32(GiReinterpretInt16AsInt32(c2_ijkl1_high), GiReinterpretInt16AsInt32(c2_mnop1_high));

    GI_INT8_t c0_ans0 = GiReinterInt32ToInt8(GiCombineInt32Low(c0_0, c0_20));
    GI_INT8_t c0_ans2 = GiReinterInt32ToInt8(GiCombineInt32Low(c0_1, c0_21));
    GI_INT8_t c0_ans4 = GiReinterInt32ToInt8(GiCombineInt32Low(c0_2, c0_22));
    GI_INT8_t c0_ans6 = GiReinterInt32ToInt8(GiCombineInt32Low(c0_3, c0_23));
    GI_INT8_t c0_ans8 = GiReinterInt32ToInt8(GiCombineInt32Low(c0_10, c0_30));
    GI_INT8_t c0_ansa = GiReinterInt32ToInt8(GiCombineInt32Low(c0_11, c0_31));
    GI_INT8_t c0_ansc = GiReinterInt32ToInt8(GiCombineInt32Low(c0_12, c0_32));
    GI_INT8_t c0_anse = GiReinterInt32ToInt8(GiCombineInt32Low(c0_13, c0_33));
    GI_INT8_t c0_ans1 = GiReinterInt32ToInt8(GiCombineInt32High(c0_0, c0_20));
    GI_INT8_t c0_ans3 = GiReinterInt32ToInt8(GiCombineInt32High(c0_1, c0_21));
    GI_INT8_t c0_ans5 = GiReinterInt32ToInt8(GiCombineInt32High(c0_2, c0_22));
    GI_INT8_t c0_ans7 = GiReinterInt32ToInt8(GiCombineInt32High(c0_3, c0_23));
    GI_INT8_t c0_ans9 = GiReinterInt32ToInt8(GiCombineInt32High(c0_10, c0_30));
    GI_INT8_t c0_ansb = GiReinterInt32ToInt8(GiCombineInt32High(c0_11, c0_31));
    GI_INT8_t c0_ansd = GiReinterInt32ToInt8(GiCombineInt32High(c0_12, c0_32));
    GI_INT8_t c0_ansf = GiReinterInt32ToInt8(GiCombineInt32High(c0_13, c0_33));

    GI_INT8_t c1_ans0 = GiReinterInt32ToInt8(GiCombineInt32Low(c1_0, c1_20));
    GI_INT8_t c1_ans2 = GiReinterInt32ToInt8(GiCombineInt32Low(c1_1, c1_21));
    GI_INT8_t c1_ans4 = GiReinterInt32ToInt8(GiCombineInt32Low(c1_2, c1_22));
    GI_INT8_t c1_ans6 = GiReinterInt32ToInt8(GiCombineInt32Low(c1_3, c1_23));
    GI_INT8_t c1_ans8 = GiReinterInt32ToInt8(GiCombineInt32Low(c1_10, c1_30));
    GI_INT8_t c1_ansa = GiReinterInt32ToInt8(GiCombineInt32Low(c1_11, c1_31));
    GI_INT8_t c1_ansc = GiReinterInt32ToInt8(GiCombineInt32Low(c1_12, c1_32));
    GI_INT8_t c1_anse = GiReinterInt32ToInt8(GiCombineInt32Low(c1_13, c1_33));
    GI_INT8_t c1_ans1 = GiReinterInt32ToInt8(GiCombineInt32High(c1_0, c1_20));
    GI_INT8_t c1_ans3 = GiReinterInt32ToInt8(GiCombineInt32High(c1_1, c1_21));
    GI_INT8_t c1_ans5 = GiReinterInt32ToInt8(GiCombineInt32High(c1_2, c1_22));
    GI_INT8_t c1_ans7 = GiReinterInt32ToInt8(GiCombineInt32High(c1_3, c1_23));
    GI_INT8_t c1_ans9 = GiReinterInt32ToInt8(GiCombineInt32High(c1_10, c1_30));
    GI_INT8_t c1_ansb = GiReinterInt32ToInt8(GiCombineInt32High(c1_11, c1_31));
    GI_INT8_t c1_ansd = GiReinterInt32ToInt8(GiCombineInt32High(c1_12, c1_32));
    GI_INT8_t c1_ansf = GiReinterInt32ToInt8(GiCombineInt32High(c1_13, c1_33));

   
    GI_INT8_t c2_ans0 = GiReinterInt32ToInt8(GiCombineInt32Low(c2_0, c2_20));
    GI_INT8_t c2_ans2 = GiReinterInt32ToInt8(GiCombineInt32Low(c2_1, c2_21));
    GI_INT8_t c2_ans4 = GiReinterInt32ToInt8(GiCombineInt32Low(c2_2, c2_22));
    GI_INT8_t c2_ans6 = GiReinterInt32ToInt8(GiCombineInt32Low(c2_3, c2_23));
    GI_INT8_t c2_ans8 = GiReinterInt32ToInt8(GiCombineInt32Low(c2_10, c2_30));
    GI_INT8_t c2_ansa = GiReinterInt32ToInt8(GiCombineInt32Low(c2_11, c2_31));
    GI_INT8_t c2_ansc = GiReinterInt32ToInt8(GiCombineInt32Low(c2_12, c2_32));
    GI_INT8_t c2_anse = GiReinterInt32ToInt8(GiCombineInt32Low(c2_13, c2_33));
    GI_INT8_t c2_ans1 = GiReinterInt32ToInt8(GiCombineInt32High(c2_0, c2_20));
    GI_INT8_t c2_ans3 = GiReinterInt32ToInt8(GiCombineInt32High(c2_1, c2_21));
    GI_INT8_t c2_ans5 = GiReinterInt32ToInt8(GiCombineInt32High(c2_2, c2_22));
    GI_INT8_t c2_ans7 = GiReinterInt32ToInt8(GiCombineInt32High(c2_3, c2_23));
    GI_INT8_t c2_ans9 = GiReinterInt32ToInt8(GiCombineInt32High(c2_10, c2_30));
    GI_INT8_t c2_ansb = GiReinterInt32ToInt8(GiCombineInt32High(c2_11, c2_31));
    GI_INT8_t c2_ansd = GiReinterInt32ToInt8(GiCombineInt32High(c2_12, c2_32));
    GI_INT8_t c2_ansf = GiReinterInt32ToInt8(GiCombineInt32High(c2_13, c2_33));

    GiStoreZipInt8V3(dst_ptr + 0 * dst_step, c0_ans0, c1_ans0, c2_ans0);
    GiStoreZipInt8V3(dst_ptr + 1 * dst_step, c0_ans1, c1_ans1, c2_ans1);
    GiStoreZipInt8V3(dst_ptr + 2 * dst_step, c0_ans2, c1_ans2, c2_ans2);
    GiStoreZipInt8V3(dst_ptr + 3 * dst_step, c0_ans3, c1_ans3, c2_ans3);
    GiStoreZipInt8V3(dst_ptr + 4 * dst_step, c0_ans4, c1_ans4, c2_ans4);
    GiStoreZipInt8V3(dst_ptr + 5 * dst_step, c0_ans5, c1_ans5, c2_ans5);
    GiStoreZipInt8V3(dst_ptr + 6 * dst_step, c0_ans6, c1_ans6, c2_ans6);
    GiStoreZipInt8V3(dst_ptr + 7 * dst_step, c0_ans7, c1_ans7, c2_ans7);
    GiStoreZipInt8V3(dst_ptr + 8 * dst_step, c0_ans8, c1_ans8, c2_ans8);
    GiStoreZipInt8V3(dst_ptr + 9 * dst_step, c0_ans9, c1_ans9, c2_ans9);
    GiStoreZipInt8V3(dst_ptr + 10 * dst_step, c0_ansa, c1_ansa, c2_ansa);
    GiStoreZipInt8V3(dst_ptr + 11 * dst_step, c0_ansb, c1_ansb, c2_ansb);
    GiStoreZipInt8V3(dst_ptr + 12 * dst_step, c0_ansc, c1_ansc, c2_ansc);
    GiStoreZipInt8V3(dst_ptr + 13 * dst_step, c0_ansd, c1_ansd, c2_ansd);
    GiStoreZipInt8V3(dst_ptr + 14 * dst_step, c0_anse, c1_anse, c2_anse);
    GiStoreZipInt8V3(dst_ptr + 15 * dst_step, c0_ansf, c1_ansf, c2_ansf);
}
    )";
}

std::string trans_4x8_u32_contig_src() {
    return R"(

static inline void trans_4x8_u32_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    int32_t* src_ptr = (int32_t*)src;
    int32_t* dst_ptr = (int32_t*)dst;
    GI_INT32_t src0 = GiLoadInt32(src_ptr + 0 * 4);  // A0A1A2A3
    GI_INT32_t src1 = GiLoadInt32(src_ptr + 1 * 4);  // B0B1B2B3
    GI_INT32_t src2 = GiLoadInt32(src_ptr + 2 * 4);  // C0C1C2C3
    GI_INT32_t src3 = GiLoadInt32(src_ptr + 3 * 4);  // D0D1D2D3
    GI_INT32_t src4 = GiLoadInt32(src_ptr + 4 * 4);  // E0E1E2E3
    GI_INT32_t src5 = GiLoadInt32(src_ptr + 5 * 4);  // F0F1F2F3
    GI_INT32_t src6 = GiLoadInt32(src_ptr + 6 * 4);  // G0G1G2G3
    GI_INT32_t src7 = GiLoadInt32(src_ptr + 7 * 4);  // H0H1H2H3

    GI_INT32_t tmp = src0;
    src0 = GiZipV0Int32(src0, src1); 
    src1 = GiZipV1Int32(tmp, src1);
    tmp = src2;
    src2 = GiZipV0Int32(src2, src3);
    src3 = GiZipV1Int32(tmp, src3);
    tmp = src4;  
    src4 = GiZipV0Int32(src4, src5);  
    src5 = GiZipV1Int32(tmp, src5);
    tmp = src6;  
    src6 = GiZipV0Int32(src6, src7);  
    src7 = GiZipV1Int32(tmp, src7);

    tmp = src0;
    src0 = GiCombineInt32Low(src0, src2); 
    src2 = GiCombineInt32High(tmp, src2);
    tmp = src1; 
    src1 = GiCombineInt32Low(src1, src3);
    src3 = GiCombineInt32High(tmp, src3);  
    tmp = src4; 
    src4 = GiCombineInt32Low(src4, src6);  
    src6 = GiCombineInt32High(tmp, src6);  
    tmp = src5; 
    src5 = GiCombineInt32Low(src5, src7);  
    src7 = GiCombineInt32High(tmp, src7);  


    GiStoreInt32(dst_ptr + 0 * dst_step + 0,  src0);
    GiStoreInt32(dst_ptr + 0 * dst_step + 4,  src4);
    GiStoreInt32(dst_ptr + 1 * dst_step + 0,  src2);
    GiStoreInt32(dst_ptr + 1 * dst_step + 4,  src6);
    GiStoreInt32(dst_ptr + 2 * dst_step + 0,  src1);
    GiStoreInt32(dst_ptr + 2 * dst_step + 4,  src5);
    GiStoreInt32(dst_ptr + 3 * dst_step + 0,  src3);
    GiStoreInt32(dst_ptr + 3 * dst_step + 4,  src7);
}
    )";
}

std::string trans_8x4_u32() {
    return R"(

static inline void trans_8x4_u32(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint32_t* src_ptr = (uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    GI_INT32_t src0 = GiLoadInt32(src_ptr + 0 * src_step);  
    GI_INT32_t src1 = GiLoadInt32(src_ptr + 1 * src_step);  
    GI_INT32_t src2 = GiLoadInt32(src_ptr + 2 * src_step);  
    GI_INT32_t src3 = GiLoadInt32(src_ptr + 3 * src_step);  
    GI_INT32_t src4 = GiLoadInt32(src_ptr + 0 * src_step + 4);  
    GI_INT32_t src5 = GiLoadInt32(src_ptr + 1 * src_step + 4);  
    GI_INT32_t src6 = GiLoadInt32(src_ptr + 2 * src_step + 4);  
    GI_INT32_t src7 = GiLoadInt32(src_ptr + 3 * src_step + 4);  
    GI_INT32_t tmp = src0;
    src0 = GiZipV0Int32(src0, src1); 
    src1 = GiZipV1Int32(tmp, src1);
    tmp = src2;
    src2 = GiZipV0Int32(src2, src3);
    src3 = GiZipV1Int32(tmp, src3);
    tmp = src4;  
    src4 = GiZipV0Int32(src4, src5);  
    src5 = GiZipV1Int32(tmp, src5);
    tmp = src6;  
    src6 = GiZipV0Int32(src6, src7);  
    src7 = GiZipV1Int32(tmp, src7);
    tmp = src0;
    src0 = GiCombineInt32Low(src0, src2); 
    src2 = GiCombineInt32High(tmp, src2);
    tmp = src1;
    src1 = GiCombineInt32Low(src1, src3);
    src3 = GiCombineInt32High(tmp, src3);  
    tmp = src4;
    src4 = GiCombineInt32Low(src4, src6);  
    src6 = GiCombineInt32High(tmp, src6);  

    tmp = src5;
    src5 = GiCombineInt32Low(src5, src7);  
    src7 = GiCombineInt32High(tmp, src7);  
    GiStoreInt32(dst_ptr + 0 * dst_step, src0);
    GiStoreInt32(dst_ptr + 1 * dst_step, src2);
    GiStoreInt32(dst_ptr + 2 * dst_step, src1);
    GiStoreInt32(dst_ptr + 3 * dst_step, src3);
    GiStoreInt32(dst_ptr + 4 * dst_step, src4);
    GiStoreInt32(dst_ptr + 5 * dst_step, src6);
    GiStoreInt32(dst_ptr + 6 * dst_step, src5);
    GiStoreInt32(dst_ptr + 7 * dst_step, src7);
}
    )";
}

std::string transpose_c1_m3_contig() {
    std::stringstream ss;
    ss << trans_3x64_i8_contig_src();
    ss << trans_3x32_i8_contig_src();
    ss << R"(
static inline void transpose_c1_m3_contig(uint8_t* src, uint8_t* dst, int m, int n){
    const int block_m = 3;
    const int block_n = 64;
    const int block_n2 = 32;
    int n_end = n / block_n * block_n;
    int n_remain = n - n_end;
    const int src_step = m;
    for(int n_idx = 0; n_idx < n_end; n_idx += block_n){
        uint8_t* dst_ptr = dst + 0 * n + n_idx;
        uint8_t* src_ptr = src + n_idx * src_step + 0;
        trans_3x64_i8_contig_src(src_ptr, dst_ptr, n);
    }
    if(n_remain >= 32){
        uint8_t* dst_ptr = dst + n_end;
        uint8_t* src_ptr = src + n_end * src_step;
        n_end += 32;
        n_remain -= 32;
        trans_3x32_i8_contig_src(src_ptr, dst_ptr, n);
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + n_end;
        uint8_t* src_ptr = src + n_end * src_step;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, 1, src_step, n);
    }
}
    )";
    return ss.str();
}

std::string transpose_c1_m4_contig(std::string spec) {
    std::stringstream ss;
    if (spec == "uint8_t") {
        ss << trans_4x16_i8_contig_src();
        ss << R"(

static inline void transpose_c1_m4_contig(uint8_t* src, uint8_t* dst, int m, int n){
    const int block_m = 4;
    const int block_n = 16;
    int n_end = n / block_n * block_n;
    int n_remain = n - n_end;
    const int src_step = m;
    for(int n_idx = 0; n_idx < n_end; n_idx += block_n){
        for(int m_idx = 0; m_idx < m; m_idx += block_m){
            uint8_t* dst_ptr = dst + m_idx * n + n_idx;
            uint8_t* src_ptr = src + n_idx * src_step + m_idx;
            trans_4x16_i8_contig_src(src_ptr, dst_ptr, n);
        }
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + n_end;
        uint8_t* src_ptr = src + n_end * src_step;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, 1, src_step, n);
    }
}
    )";
    } else if (spec == "uint32_t") {
        ss << trans_4x8_u32_contig_src();
        ss << R"(
static inline void transpose_c1_m4_contig(uint32_t* src, uint32_t* dst, int m, int n){
    const int block_m = 4;
    const int block_n = 8;
    int n_end = n / block_n * block_n;
    int n_remain = n - n_end;
    const int src_step = m;
    for(int n_idx = 0; n_idx < n_end; n_idx += block_n){
        for(int m_idx = 0; m_idx < m; m_idx += block_m){
            uint32_t* dst_ptr = dst + m_idx * n + n_idx;
            uint32_t* src_ptr = src + n_idx * src_step + m_idx;
            trans_4x8_u32_contig_src(src_ptr, dst_ptr, n);
        }
    }
    if(n_remain > 0){
        uint32_t* dst_ptr = dst + n_end;
        uint32_t* src_ptr = src + n_end * src_step;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, 1, src_step, n);
    }
}
    )";
    }
    return ss.str();
}

std::string transpose_c1(std::string spec) {
    std::stringstream ss;
    if (spec == "uint8_t") {
        ss << transpose_c1_m3_contig();
        ss << transpose_c1_m4_contig(spec);
        ss << trans_16x16_i8();
        ss << R"(
static inline void transpose_c1(uint8_t* src, uint8_t* dst, int m, int n, int c, int src_step, int dst_step){
    if(m == 3 && src_step == m && dst_step == n){
        transpose_c1_m3_contig(src, dst, m, n);
        return;
    }else if(m == 4 && src_step == m && dst_step == n){
        transpose_c1_m4_contig(src, dst, m, n);
        return;
    }
    const int block = 16;
    int m_end = m / block * block;
    int m_remain = m - m_end;
    int n_end = n / block * block;
    int n_remain = n - n_end;
    for(int n_idx = 0; n_idx < n_end; n_idx += block){
        for(int m_idx = 0; m_idx < m_end; m_idx += block){
            uint8_t* dst_ptr = dst + m_idx * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_idx * c;
            trans_16x16_i8(src_ptr, dst_ptr, src_step, dst_step);
        }
        if(m_remain > 0){
            uint8_t* dst_ptr = dst + m_end * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_end * c;
            transpose_naive(src_ptr, dst_ptr, m_remain, block, c, src_step, dst_step);
        }
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + 0 * dst_step + n_end * c;
        uint8_t* src_ptr = src + n_end * src_step + 0 * c;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, c, src_step, dst_step);
    }
}
    )";

    } else if (spec == "uint32_t") {
        ss << transpose_c1_m4_contig(spec);
        ss << trans_8x4_u32();
        ss << R"(

static inline void transpose_c1(uint32_t* src, uint32_t* dst, int m, int n, int c, int src_step, int dst_step){
    if(m == 4 && src_step == m && dst_step == n){
        transpose_c1_m4_contig(src, dst, m, n);
        return;
    }
    const int block_m = 8;
    const int block_n = 4;
    int m_end = m / block_m * block_m;
    int m_remain = m - m_end;
    int n_end = n / block_n * block_n;
    int n_remain = n - n_end;
    
    for(int n_idx = 0; n_idx < n_end; n_idx += block_n){
        for(int m_idx = 0; m_idx < m_end; m_idx += block_m){
            uint32_t* dst_ptr = dst + m_idx * dst_step + n_idx;
            uint32_t* src_ptr = src + n_idx * src_step + m_idx;
            trans_8x4_u32(src_ptr, dst_ptr, src_step, dst_step);
        }
        if(m_remain > 0){
            uint32_t* dst_ptr = dst + m_end * dst_step + n_idx;
            uint32_t* src_ptr = src + n_idx * src_step + m_end;
            transpose_naive(src_ptr, dst_ptr, m_remain, block_n, 1, src_step, dst_step);
        }
    }
    if(n_remain > 0){
        uint32_t* dst_ptr = dst + 0 * dst_step + n_end;
        uint32_t* src_ptr = src + n_end * src_step + 0;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, 1, src_step, dst_step);
    }
}
    )";
    }
    return ss.str();
}

std::string transpose_c2() {
    std::stringstream ss;
    ss << trans_8x8_i16();
    ss << R"(
static inline void transpose_c2(uint8_t* src, uint8_t* dst, int m, int n, int c, int src_step, int dst_step){
    const int block = 8;
    int m_end = m / block * block;
    int m_remain = m - m_end;
    int n_end = n / block * block;
    int n_remain = n - n_end;
    int i16_src_step = src_step / 2;
    for(int n_idx = 0; n_idx < n_end; n_idx += block){
        for(int m_idx = 0; m_idx < m_end; m_idx += block){
            uint8_t* dst_ptr = dst + m_idx * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_idx * c;
            trans_8x8_i16(src_ptr, dst_ptr, i16_src_step , dst_step / 2);
        }
        if(m_remain > 0){
            uint8_t* dst_ptr = dst + m_end * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_end * c;
            transpose_naive(src_ptr, dst_ptr, m_remain, block, c, src_step, dst_step);
        }
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + 0 * dst_step + n_end * c;
        uint8_t* src_ptr = src + n_end * src_step + 0 * c;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, c, src_step, dst_step);
    }
}
    )";
    return ss.str();
}

std::string transpose_c3() {
    std::stringstream ss;
    ss << trans_16x16_i8x3();
    ss << R"(

static inline void transpose_c3(uint8_t* src, uint8_t* dst, int m, int n, int c, int src_step, int dst_step){
    const int block_m = 16, block_n = 16;
    int m_end = m / block_m * block_m;
    int m_remain = m - m_end;
    int n_end = n / block_n * block_n;
    int n_remain = n - n_end;
    for(int n_idx = 0; n_idx < n_end; n_idx += block_n){
        for(int m_idx = 0; m_idx < m_end; m_idx += block_m){
            uint8_t* dst_ptr = dst + m_idx * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_idx * c;
            trans_16x16_i8x3(src_ptr, dst_ptr, src_step, dst_step);
        }
        if(m_remain > 0){
            uint8_t* dst_ptr = dst + m_end * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_end * c;
            transpose_naive(src_ptr, dst_ptr, m_remain, block_n, c, src_step, dst_step);
        }
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + 0 * dst_step + n_end * c;
        uint8_t* src_ptr = src + n_end * src_step + 0 * c;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, c, src_step, dst_step);
    }
}
    )";

    return ss.str();
}
std::string gen_transpose_i8() {
    std::stringstream ss;
    ss << transpose_c1("uint8_t");
    ss << transpose_c2();
    ss << transpose_c3();
    ss << R"(
void fast_transpose_impl_8(void* src, void* dst, int m, int n, int c, int src_step, int dst_step){
    uint8_t* src_base_ptr = src;
    uint8_t* dst_base_ptr = dst;
    if(c == 1) {
        transpose_c1(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    }else if(c == 2) {
        transpose_c2(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    } else if(c == 3) {
        transpose_c3(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    }else{
        transpose_naive(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    }
}


    )";

    return ss.str();
}

std::string gen_transpose_u32() {
    std::stringstream ss;
    ss << transpose_c1("uint32_t");
    ss << R"(
void fast_transpose_impl_32(void* src, void* dst, int m, int n, int c, int src_step, int dst_step){
    uint32_t* src_base_ptr = src;
    uint32_t* dst_base_ptr = dst;
    if(c == 1) {
        transpose_c1(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    }else{
        transpose_naive(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
    }
    return ;
}
        )";
    return ss.str();
}

std::string gen_transpose(int type_size) {
    std::stringstream ss;
    if (type_size == 1) {
        ss << transpose_naive("uint8_t");
        ss << gen_transpose_i8();

        return ss.str();
    } else if (type_size == 4) {
        ss << transpose_naive("uint32_t");
        ss << gen_transpose_u32();
        return ss.str();
    } else {
        CC_ABORT << "not support type size " << type_size << "\n";
    }
    return "";
}
class CommonTransposeKernel : public InternalKernelFunc {
public:
    std::string GetKernelSymbol(TContext* ctx) const override {
        int type_size = ctx->haveAttr("type_size") ? ctx->getAttrInt("type_size") : 0;
        return "Common_Internal_Transpose_" + std::to_string(type_size * 8);
    };

    std::string GetKernelSignature(TContext* ctx) const override {
        int type_size = ctx->haveAttr("type_size") ? ctx->getAttrInt("type_size") : 0;
        std::stringstream ss;
        ss << "void fast_transpose_impl_";
        ss << std::to_string(type_size * 8);
        ss << "(void* src, void* dst, int m, int n, int c, int src_step, int "
              "dst_step);";
        return ss.str();
    };

    std::string GetKernelBody(TContext* ctx) const override {
        std::stringstream writer;
        int type_size = ctx->haveAttr("type_size") ? ctx->getAttrInt("type_size") : 0;
        writer << R"(
            #include "gi_int.h"
        )";
        writer << gen_transpose(type_size);
        return writer.str();
    };
};
}  // namespace
}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc