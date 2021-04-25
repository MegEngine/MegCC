/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/Transpose.h
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

namespace megcc {
namespace KernelGen {
namespace Armv7 {
namespace {
std::string gen_transpose_u8() {
    return R"(
static inline void transpose_naive(uint8_t* src, uint8_t* dst, int m, int n, int ch, int src_stride, int dst_stride){
    for(int row_id = 0; row_id < m; ++row_id)
    for(int col_id = 0; col_id < n; ++col_id){
        uint8_t * dst_ptr = dst + row_id * dst_stride + col_id * ch;
        uint8_t * src_ptr = src + col_id * src_stride + row_id * ch;
        for(int channel_id = 0 ; channel_id < ch; ++channel_id){
            dst_ptr[channel_id] = src_ptr[channel_id];
        }
    }
}

static inline void trans_3x32_u8_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    uint8_t* src_ptr = (uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;
    uint8x16x3_t src0 = vld3q_u8(src_ptr + 0 * 16);  // A0B0C0D0E0F0G0H0I0J0K0L0M0N0O0P0
    uint8x16x3_t src1 = vld3q_u8(src_ptr + 3 * 16);  // A0B0C0D0E0F0G0H0I0J0K0L0M0N0O0P0

    vst1q_u8(dst_ptr + 0 * dst_step + 0,  src0.val[0]);
    vst1q_u8(dst_ptr + 0 * dst_step + 16, src1.val[0]);
    vst1q_u8(dst_ptr + 1 * dst_step + 0,  src0.val[1]);
    vst1q_u8(dst_ptr + 1 * dst_step + 16, src1.val[1]);
    vst1q_u8(dst_ptr + 2 * dst_step + 0,  src0.val[2]);
    vst1q_u8(dst_ptr + 2 * dst_step + 16, src1.val[2]);
}

static inline void trans_3x64_u8_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    uint8_t* src_ptr = (uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;
    uint8x16x3_t src0 = vld3q_u8(src_ptr + 0 * 0 * 16);  // A0B0C0D0E0F0G0H0I0J0K0L0M0N0O0P0
    uint8x16x3_t src1 = vld3q_u8(src_ptr + 1 * 3 * 16);  // A0B0C0D0E0F0G0H0I0J0K0L0M0N0O0P0
    uint8x16x3_t src2 = vld3q_u8(src_ptr + 2 * 3 * 16);
    uint8x16x3_t src3 = vld3q_u8(src_ptr + 3 * 3 * 16);

    vst1q_u8(dst_ptr + 0 * dst_step + 0 * 16, src0.val[0]);
    vst1q_u8(dst_ptr + 0 * dst_step + 1 * 16, src1.val[0]);
    vst1q_u8(dst_ptr + 0 * dst_step + 2 * 16, src2.val[0]);
    vst1q_u8(dst_ptr + 0 * dst_step + 3 * 16, src3.val[0]);
    vst1q_u8(dst_ptr + 1 * dst_step + 0 * 16, src0.val[1]);
    vst1q_u8(dst_ptr + 1 * dst_step + 1 * 16, src1.val[1]);
    vst1q_u8(dst_ptr + 1 * dst_step + 2 * 16, src2.val[1]);
    vst1q_u8(dst_ptr + 1 * dst_step + 3 * 16, src3.val[1]);
    vst1q_u8(dst_ptr + 2 * dst_step + 0 * 16, src0.val[2]);
    vst1q_u8(dst_ptr + 2 * dst_step + 1 * 16, src1.val[2]);
    vst1q_u8(dst_ptr + 2 * dst_step + 2 * 16, src2.val[2]);
    vst1q_u8(dst_ptr + 2 * dst_step + 3 * 16, src3.val[2]);
}

static inline void trans_16x16_u8(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    // 16x16
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
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31");
}

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
        trans_3x64_u8_contig_src(src_ptr, dst_ptr, n);
    }
    if(n_remain >= 32){
        uint8_t* dst_ptr = dst + n_end;
        uint8_t* src_ptr = src + n_end * src_step;
        n_end += 32;
        n_remain -= 32;
        trans_3x32_u8_contig_src(src_ptr, dst_ptr, n);
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + n_end;
        uint8_t* src_ptr = src + n_end * src_step;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, 1, src_step, n);
    }
}

static inline void transpose_c1(uint8_t* src, uint8_t* dst, int m, int n, int c, int src_step, int dst_step){
    if(m == 3 && src_step == m && dst_step == n){
        transpose_c1_m3_contig(src, dst, m, n);
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
            trans_16x16_u8(src_ptr, dst_ptr, src_step, dst_step);
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
static inline void fast_transpose_impl_8(void* src, void* dst, int m, int n, int c, int src_step, int dst_step){
    uint8_t* src_base_ptr = src;
    uint8_t* dst_base_ptr = dst;
    if(c == 1) {
        transpose_c1(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    }
    else{
        for(int row_id = 0; row_id < n; ++row_id)
            for(int col_id = 0; col_id < m; ++col_id){
                uint8_t * src_ptr = src_base_ptr + row_id * src_step + col_id * c;
                uint8_t * dst_ptr = dst_base_ptr + col_id * dst_step + row_id * c;
                for(int channel_id = 0 ; channel_id < c; ++channel_id){
                    dst_ptr[channel_id] = src_ptr[channel_id];
                }
            }
        return ;
    }
}
    )";
}
std::string gen_transpose_u32() {
    return R"(

static inline void transpose_naive(uint32_t* src, uint32_t* dst, int m, int n, int ch, int src_stride, int dst_stride){
    for(int row_id = 0; row_id < m; ++row_id)
    for(int col_id = 0; col_id < n; ++col_id){
        uint32_t * dst_ptr = dst + row_id * dst_stride + col_id * ch;
        uint32_t * src_ptr = src + col_id * src_stride + row_id * ch;
        for(int channel_id = 0 ; channel_id < ch; ++channel_id){
            dst_ptr[channel_id] = src_ptr[channel_id];
        }
    }
}

static inline void trans_4x8_u32_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    uint32_t* src_ptr = (uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    uint32x4_t src0 = vld1q_u32(src_ptr + 0 * 4);  // A0A1A2A3
    uint32x4_t src1 = vld1q_u32(src_ptr + 1 * 4);  // B0B1B2B3
    uint32x4_t src2 = vld1q_u32(src_ptr + 2 * 4);  // C0C1C2C3
    uint32x4_t src3 = vld1q_u32(src_ptr + 3 * 4);  // D0D1D2D3
    uint32x4_t src4 = vld1q_u32(src_ptr + 4 * 4);  // E0E1E2E3
    uint32x4_t src5 = vld1q_u32(src_ptr + 5 * 4);  // F0F1F2F3
    uint32x4_t src6 = vld1q_u32(src_ptr + 6 * 4);  // G0G1G2G3
    uint32x4_t src7 = vld1q_u32(src_ptr + 7 * 4);  // H0H1H2H3

    uint32x4x2_t ab0123 = vzipq_u32(src0, src1);
    uint32x4x2_t cd0123 = vzipq_u32(src2, src3);
    uint32x4x2_t ef0123 = vzipq_u32(src4, src5);
    uint32x4x2_t gh0123 = vzipq_u32(src6, src7);

    uint32x4_t a0b0c0d0 = vcombine_u32(vget_low_u32 (ab0123.val[0]), vget_low_u32 (cd0123.val[0]));    
    uint32x4_t a1b1c1d1 = vcombine_u32(vget_high_u32(ab0123.val[0]), vget_high_u32(cd0123.val[0])); 
    uint32x4_t a2b2c2d2 = vcombine_u32(vget_low_u32 (ab0123.val[1]), vget_low_u32 (cd0123.val[1])); 
    uint32x4_t a3b3c3d3 = vcombine_u32(vget_high_u32(ab0123.val[1]), vget_high_u32(cd0123.val[1]));
    uint32x4_t e0f0g0h0 = vcombine_u32(vget_low_u32 (ef0123.val[0]), vget_low_u32 (gh0123.val[0]));    
    uint32x4_t e1f1g1h1 = vcombine_u32(vget_high_u32(ef0123.val[0]), vget_high_u32(gh0123.val[0])); 
    uint32x4_t e2f2g2h2 = vcombine_u32(vget_low_u32 (ef0123.val[1]), vget_low_u32 (gh0123.val[1])); 
    uint32x4_t e3f3g3h3 = vcombine_u32(vget_high_u32(ef0123.val[1]), vget_high_u32(gh0123.val[1])); 

    vst1q_u32(dst_ptr + 0 * dst_step + 0,  a0b0c0d0);
    vst1q_u32(dst_ptr + 0 * dst_step + 4,  e0f0g0h0);
    vst1q_u32(dst_ptr + 1 * dst_step + 0,  a1b1c1d1);
    vst1q_u32(dst_ptr + 1 * dst_step + 4,  e1f1g1h1);
    vst1q_u32(dst_ptr + 2 * dst_step + 0,  a2b2c2d2);
    vst1q_u32(dst_ptr + 2 * dst_step + 4,  e2f2g2h2);
    vst1q_u32(dst_ptr + 3 * dst_step + 0,  a3b3c3d3);
    vst1q_u32(dst_ptr + 3 * dst_step + 4,  e3f3g3h3);
}

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

static inline void trans_8x4_u32(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint32_t* src_ptr = (uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    uint32x4_t src0 = vld1q_u32(src_ptr + 0 * src_step);  
    uint32x4_t src1 = vld1q_u32(src_ptr + 1 * src_step);  
    uint32x4_t src2 = vld1q_u32(src_ptr + 2 * src_step);  
    uint32x4_t src3 = vld1q_u32(src_ptr + 3 * src_step);  
    uint32x4_t src4 = vld1q_u32(src_ptr + 0 * src_step + 4);  
    uint32x4_t src5 = vld1q_u32(src_ptr + 1 * src_step + 4);  
    uint32x4_t src6 = vld1q_u32(src_ptr + 2 * src_step + 4);  
    uint32x4_t src7 = vld1q_u32(src_ptr + 3 * src_step + 4);  

    uint32x4x2_t ab = vzipq_u32(src0, src1);  
    uint32x4x2_t cd = vzipq_u32(src2, src3);  
    uint32x4x2_t ef = vzipq_u32(src4, src5);  
    uint32x4x2_t gh = vzipq_u32(src6, src7);  

    uint32x4_t abcd_0 = vcombine_u32(vget_low_u32 (ab.val[0]), vget_low_u32 (cd.val[0]));
    uint32x4_t abcd_1 = vcombine_u32(vget_high_u32(ab.val[0]), vget_high_u32(cd.val[0]));
    uint32x4_t abcd_2 = vcombine_u32(vget_low_u32 (ab.val[1]), vget_low_u32 (cd.val[1]));
    uint32x4_t abcd_3 = vcombine_u32(vget_high_u32(ab.val[1]), vget_high_u32(cd.val[1]));
    uint32x4_t abcd_4 = vcombine_u32(vget_low_u32 (ef.val[0]), vget_low_u32 (gh.val[0]));
    uint32x4_t abcd_5 = vcombine_u32(vget_high_u32(ef.val[0]), vget_high_u32(gh.val[0]));
    uint32x4_t abcd_6 = vcombine_u32(vget_low_u32 (ef.val[1]), vget_low_u32 (gh.val[1]));
    uint32x4_t abcd_7 = vcombine_u32(vget_high_u32(ef.val[1]), vget_high_u32(gh.val[1]));

    vst1q_u32(dst_ptr + 0 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 1 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 2 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 3 * dst_step, abcd_3);
    vst1q_u32(dst_ptr + 4 * dst_step, abcd_4);
    vst1q_u32(dst_ptr + 5 * dst_step, abcd_5);
    vst1q_u32(dst_ptr + 6 * dst_step, abcd_6);
    vst1q_u32(dst_ptr + 7 * dst_step, abcd_7);
}


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
static inline void fast_transpose_impl_32(void* src, void* dst, int m, int n, int c, int src_step, int dst_step){
    uint32_t* src_base_ptr = src;
    uint32_t* dst_base_ptr = dst;
    if(c == 1) {
        transpose_c1(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    }else{
        for(int row_id = 0; row_id < n; ++row_id)
            for(int col_id = 0; col_id < m; ++col_id){
                uint32_t * src_ptr = src_base_ptr + row_id * src_step + col_id * c;
                uint32_t * dst_ptr = dst_base_ptr + col_id * dst_step + row_id * c;
                for(int channel_id = 0 ; channel_id < c; ++channel_id){
                    dst_ptr[channel_id] = src_ptr[channel_id];
                }
            }
    }
    return ;
}
        )";
}
std::string gen_transpose(int size) {
    if (size == 1) {
        return gen_transpose_u8();
    } else if (size == 4) {
        return gen_transpose_u32();
    } else {
        CC_ABORT << "not support type size " << size << "\n";
    }
    return "";
}

}  // namespace
}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc