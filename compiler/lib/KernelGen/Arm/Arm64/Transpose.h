/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/Transpose.h
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
namespace Arm64 {
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
static inline void trans_16x4_u8(const void* src, void* dst, const size_t src_step,
                    const size_t dst_step) {
    uint8_t* src_ptr = (uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;
    uint8x16_t src0 = vld1q_u8(src_ptr + 0 * src_step);  // A0A1A2A3A4A5A6A7A8A9A10A11A12A13A14A15
    uint8x16_t src1 = vld1q_u8(src_ptr + 1 * src_step);  // B0B1B2B3B4B5B6B7B8B9B10B11B12B13B14B15
    uint8x16_t src2 = vld1q_u8(src_ptr + 2 * src_step);  // C0C1C2C3C4C5C6C7C8C9C10C11C12C13C14C15
    uint8x16_t src3 = vld1q_u8(src_ptr + 3 * src_step);  // D0D1D2D3D4D5D6D7D8D9D10D11D12D13D14D15
    uint16x8_t ab_low =  vreinterpretq_u16_u8(vzip1q_u8(src0, src1));   // A0B0A1B1A2B2A3B3A4B4A5B5A6B6A7B7
    uint16x8_t ab_high = vreinterpretq_u16_u8(vzip2q_u8(src0, src1));  // A8B8A9B9A10B10A11B11A12B12A13B13A14B14A15B15
    uint16x8_t cd_low =  vreinterpretq_u16_u8(vzip1q_u8(src2, src3));   // C0D0C1D1C2D2C3D3C4D4C5D5C6D6C7D7
    uint16x8_t cd_high = vreinterpretq_u16_u8(vzip2q_u8(src2, src3));  // C8D8C9D9C10D10C11D11C12D12C13D13C14D14C15D15
    uint32x4_t abcd_0 = vreinterpretq_u32_u16(vzip1q_u16(ab_low, cd_low));
    uint32x4_t abcd_1 = vreinterpretq_u32_u16(vzip2q_u16(ab_low, cd_low));
    uint32x4_t abcd_2 = vreinterpretq_u32_u16(vzip1q_u16(ab_high, cd_high));
    uint32x4_t abcd_3 = vreinterpretq_u32_u16(vzip2q_u16(ab_high, cd_high));

    vst1q_lane_u32(dst_ptr + 0 * dst_step,  abcd_0, 0);
    vst1q_lane_u32(dst_ptr + 1 * dst_step,  abcd_0, 1);
    vst1q_lane_u32(dst_ptr + 2 * dst_step,  abcd_0, 2);
    vst1q_lane_u32(dst_ptr + 3 * dst_step,  abcd_0, 3);
    vst1q_lane_u32(dst_ptr + 4 * dst_step,  abcd_1, 0);
    vst1q_lane_u32(dst_ptr + 5 * dst_step,  abcd_1, 1);
    vst1q_lane_u32(dst_ptr + 6 * dst_step,  abcd_1, 2);
    vst1q_lane_u32(dst_ptr + 7 * dst_step,  abcd_1, 3);
    vst1q_lane_u32(dst_ptr + 8 * dst_step,  abcd_2, 0);
    vst1q_lane_u32(dst_ptr + 9 * dst_step,  abcd_2, 1);
    vst1q_lane_u32(dst_ptr + 10 * dst_step, abcd_2, 2);
    vst1q_lane_u32(dst_ptr + 11 * dst_step, abcd_2, 3);
    vst1q_lane_u32(dst_ptr + 12 * dst_step, abcd_3, 0);
    vst1q_lane_u32(dst_ptr + 13 * dst_step, abcd_3, 1);
    vst1q_lane_u32(dst_ptr + 14 * dst_step, abcd_3, 2);
    vst1q_lane_u32(dst_ptr + 15 * dst_step, abcd_3, 3);
}
static inline void trans_4x16_u8_contig_src(const void* src, void* dst,
                    const size_t dst_step) {
    static const uint8x16_t idx = {0, 4, 8, 12, 1, 5, 9, 13,
                                   2, 6, 10,14, 3, 7, 11, 15};
    uint8_t* src_ptr = (uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;
    uint8x16_t src0 = vld1q_u8(src_ptr + 0 * 16);  // A0A1A2A3B0B1B2B3C0C1C2C3D0D1D2D3
    uint8x16_t src1 = vld1q_u8(src_ptr + 1 * 16);  // E0E1E2E3F0F1F2F3G0G1G2G3H0H1H2H3
    uint8x16_t src2 = vld1q_u8(src_ptr + 2 * 16);  // I0I1I2I3J0J1J2J3K0K1K2K3L0L1L2L3
    uint8x16_t src3 = vld1q_u8(src_ptr + 3 * 16);  // M0M1M2M3N0N1N2N3O0O1O2O3P0P1P2P3

    uint32x4_t n048c = vreinterpretq_u32_u8(vqtbl1q_u8(src0, idx));     //A0B0C0D0A1B1C1D1A2B2C2D2A3B3C3D3
    uint32x4_t n159d = vreinterpretq_u32_u8(vqtbl1q_u8(src1, idx));     //E0F0G0H0E1F1G1H1E2F2G2H2E3F3G3H3
    uint32x4_t n26ae = vreinterpretq_u32_u8(vqtbl1q_u8(src2, idx));     //I0J0K0L0I1J1K1L1I2J2K2L2I3J3K3L3
    uint32x4_t n37bf = vreinterpretq_u32_u8(vqtbl1q_u8(src3, idx));     //M0N0O0P0M1N1O1P1M2N2O2P2M3N3O3P3

    uint64x2_t n0145 = vreinterpretq_u64_u32(vzip1q_u32(n048c, n159d));
    uint64x2_t n89cd = vreinterpretq_u64_u32(vzip2q_u32(n048c, n159d));
    uint64x2_t n2367 = vreinterpretq_u64_u32(vzip1q_u32(n26ae, n37bf));
    uint64x2_t nabef = vreinterpretq_u64_u32(vzip2q_u32(n26ae, n37bf));

    uint64x2_t n0123 = vzip1q_u64(n0145, n2367);
    uint64x2_t n4567 = vzip2q_u64(n0145, n2367);
    uint64x2_t n89ab = vzip1q_u64(n89cd, nabef);
    uint64x2_t ncdef = vzip2q_u64(n89cd, nabef);

    vst1q_u8(dst_ptr + 0 * dst_step,  n0123);
    vst1q_u8(dst_ptr + 1 * dst_step,  n4567);
    vst1q_u8(dst_ptr + 2 * dst_step,  n89ab);
    vst1q_u8(dst_ptr + 3 * dst_step,  ncdef);
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
static inline void trans_16x16_u8(const void* src, void* dst, const size_t src_step,
                    const size_t dst_step) {
    asm volatile(
        "\n"
        "ld1 {v0.16b}, [%[src]], %[src_step] \n"
        "ld1 {v1.16b}, [%[src]], %[src_step] \n"
        "ld1 {v2.16b}, [%[src]], %[src_step] \n"
        "ld1 {v3.16b}, [%[src]], %[src_step] \n"
        "ld1 {v4.16b}, [%[src]], %[src_step] \n"
        "ld1 {v5.16b}, [%[src]], %[src_step] \n"
        "ld1 {v6.16b}, [%[src]], %[src_step] \n"
        "ld1 {v7.16b}, [%[src]], %[src_step] \n"
        "ld1 {v8.16b}, [%[src]], %[src_step] \n"
        "ld1 {v9.16b}, [%[src]], %[src_step] \n"
        "ld1 {v10.16b}, [%[src]], %[src_step] \n"
        "ld1 {v11.16b}, [%[src]], %[src_step] \n"
        "ld1 {v12.16b}, [%[src]], %[src_step] \n"
        "ld1 {v13.16b}, [%[src]], %[src_step] \n"
        "ld1 {v14.16b}, [%[src]], %[src_step] \n"
        "ld1 {v15.16b}, [%[src]], %[src_step] \n"
        "trn1 v16.16b, v0.16b, v1.16b \n"
        "trn2 v17.16b, v0.16b, v1.16b \n"
        "trn1 v18.16b, v2.16b, v3.16b \n"
        "trn2 v19.16b, v2.16b, v3.16b \n"
        "trn1 v20.16b, v4.16b, v5.16b \n"
        "trn2 v21.16b, v4.16b, v5.16b \n"
        "trn1 v22.16b, v6.16b, v7.16b \n"
        "trn2 v23.16b, v6.16b, v7.16b \n"
        "trn1 v24.16b, v8.16b, v9.16b \n"
        "trn2 v25.16b, v8.16b, v9.16b \n"
        "trn1 v26.16b, v10.16b, v11.16b \n"
        "trn2 v27.16b, v10.16b, v11.16b \n"
        "trn1 v28.16b, v12.16b, v13.16b \n"
        "trn2 v29.16b, v12.16b, v13.16b \n"
        "trn1 v30.16b, v14.16b, v15.16b \n"
        "trn2 v31.16b, v14.16b, v15.16b \n"
        "trn1 v0.8h, v16.8h, v18.8h \n"
        "trn2 v2.8h, v16.8h, v18.8h \n"
        "trn1 v4.8h, v20.8h, v22.8h \n"
        "trn2 v6.8h, v20.8h, v22.8h \n"
        "trn1 v8.8h, v24.8h, v26.8h \n"
        "trn2 v10.8h, v24.8h, v26.8h \n"
        "trn1 v12.8h, v28.8h, v30.8h \n"
        "trn2 v14.8h, v28.8h, v30.8h \n"
        "trn1 v1.8h, v17.8h, v19.8h \n"
        "trn2 v3.8h, v17.8h, v19.8h \n"
        "trn1 v5.8h, v21.8h, v23.8h \n"
        "trn2 v7.8h, v21.8h, v23.8h \n"
        "trn1 v9.8h, v25.8h, v27.8h \n"
        "trn2 v11.8h, v25.8h, v27.8h \n"
        "trn1 v13.8h, v29.8h, v31.8h \n"
        "trn2 v15.8h, v29.8h, v31.8h \n"
        "trn1 v16.4s, v0.4s, v4.4s \n"
        "trn2 v20.4s, v0.4s, v4.4s \n"
        "trn1 v24.4s, v8.4s, v12.4s \n"
        "trn2 v28.4s, v8.4s, v12.4s \n"
        "trn1 v17.4s, v1.4s, v5.4s \n"
        "trn2 v21.4s, v1.4s, v5.4s \n"
        "trn1 v25.4s, v9.4s, v13.4s \n"
        "trn2 v29.4s, v9.4s, v13.4s \n"
        "trn1 v18.4s, v2.4s, v6.4s \n"
        "trn2 v22.4s, v2.4s, v6.4s \n"
        "trn1 v26.4s, v10.4s, v14.4s \n"
        "trn2 v30.4s, v10.4s, v14.4s \n"
        "trn1 v19.4s, v3.4s, v7.4s \n"
        "trn2 v23.4s, v3.4s, v7.4s \n"
        "trn1 v27.4s, v11.4s, v15.4s \n"
        "trn2 v31.4s, v11.4s, v15.4s \n"
        "trn1 v0.2d, v16.2d, v24.2d \n"
        "trn2 v8.2d, v16.2d, v24.2d \n"
        "trn1 v1.2d, v17.2d, v25.2d \n"
        "trn2 v9.2d, v17.2d, v25.2d \n"
        "trn1 v2.2d, v18.2d, v26.2d \n"
        "trn2 v10.2d, v18.2d, v26.2d \n"
        "trn1 v3.2d, v19.2d, v27.2d \n"
        "trn2 v11.2d, v19.2d, v27.2d \n"
        "trn1 v4.2d, v20.2d, v28.2d \n"
        "trn2 v12.2d, v20.2d, v28.2d \n"
        "trn1 v5.2d, v21.2d, v29.2d \n"
        "trn2 v13.2d, v21.2d, v29.2d \n"
        "trn1 v6.2d, v22.2d, v30.2d \n"
        "trn2 v14.2d, v22.2d, v30.2d \n"
        "trn1 v7.2d, v23.2d, v31.2d \n"
        "trn2 v15.2d, v23.2d, v31.2d \n"
        "st1 {v0.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v1.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v2.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v3.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v4.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v5.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v6.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v7.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v8.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v9.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v10.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v11.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v12.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v13.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v14.16b}, [%[dst]], %[dst_step] \n"
        "st1 {v15.16b}, [%[dst]], %[dst_step] \n"
        :
        [src] "+r" (src),
        [dst] "+r" (dst)
        :
        [src_step] "r" (src_step),
        [dst_step] "r" (dst_step)
        :
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
        "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
        "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30",
        "d31");
}
static inline void trans_8x8_u16(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint16_t* src_ptr = (uint16_t*)src;
    uint16_t* dst_ptr = (uint16_t*)dst;
    uint16x8_t src0 = vld1q_u16(src_ptr + 0 * src_step);  // A0A1A2A3A4A5A6A7
    uint16x8_t src1 = vld1q_u16(src_ptr + 1 * src_step);  // B0B1B2B3B4B5B6B7
    uint16x8_t src2 = vld1q_u16(src_ptr + 2 * src_step);  // C0C1C2C3C4C5C6C7
    uint16x8_t src3 = vld1q_u16(src_ptr + 3 * src_step);  // D0D1D2D3D4D5D6D7
    uint16x8_t src4 = vld1q_u16(src_ptr + 4 * src_step);  // E0E1E2E3E4E5E6E7
    uint16x8_t src5 = vld1q_u16(src_ptr + 5 * src_step);  // F0F1F2F3F4F5F6F7
    uint16x8_t src6 = vld1q_u16(src_ptr + 6 * src_step);  // G0G1G2G3G4G5G6G7
    uint16x8_t src7 = vld1q_u16(src_ptr + 7 * src_step);  // H0H1H2H3H4H5H6H7

    uint16x8_t ab_low = vzip1q_u16(src0, src1);   // A0B0A1B1A2B2A3B3
    uint16x8_t ab_high = vzip2q_u16(src0, src1);  // A4B4A5B5A6B6A7B7
    uint16x8_t cd_low = vzip1q_u16(src2, src3);   // C0D0C1D1C2D2C3D3
    uint16x8_t cd_high = vzip2q_u16(src2, src3);  // C4D4C5D5C6D6C7D7
    uint16x8_t ef_low = vzip1q_u16(src4, src5);   // E0F0E1F1E2F2E3F3
    uint16x8_t ef_high = vzip2q_u16(src4, src5);  // E4F4E5F5E6F6E7F7
    uint16x8_t gh_low = vzip1q_u16(src6, src7);   // G0H0G1H1G2H2G3H3
    uint16x8_t gh_high = vzip2q_u16(src6, src7);  // G4H4G5H5G6H6G7H7

    uint16x8_t abcd_0 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ab_low),
            vreinterpretq_u32_u16(cd_low)));  // A0B0C0D0A1B1C1D1
    uint16x8_t abcd_2 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ab_low),
            vreinterpretq_u32_u16(cd_low)));  // A2B2C2D2A3B3C3D3
    uint16x8_t abcd_4 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ab_high),
            vreinterpretq_u32_u16(cd_high)));  // A4B4C4D4A5B5C5D5
    uint16x8_t abcd_6 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ab_high),
            vreinterpretq_u32_u16(cd_high)));  // A6B6C6D6A7B7C7D7
    uint16x8_t efgh_0 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ef_low),
            vreinterpretq_u32_u16(gh_low)));  // E0F0G0H0E1F1G1H1
    uint16x8_t efgh_2 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ef_low),
            vreinterpretq_u32_u16(gh_low)));  // E2F2G2H2E3F3G3H3
    uint16x8_t efgh_4 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ef_high),
            vreinterpretq_u32_u16(gh_high)));  // E4F4G4H4E5F5G5H5
    uint16x8_t efgh_6 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ef_high),
            vreinterpretq_u32_u16(gh_high)));  // E6F6G6H6E7F7G7H7

    uint16x8_t row_0 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_0), vreinterpretq_u64_u16(efgh_0)));
    uint16x8_t row_1 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_0), vreinterpretq_u64_u16(efgh_0)));
    uint16x8_t row_2 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_2), vreinterpretq_u64_u16(efgh_2)));
    uint16x8_t row_3 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_2), vreinterpretq_u64_u16(efgh_2)));
    uint16x8_t row_4 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_4), vreinterpretq_u64_u16(efgh_4)));
    uint16x8_t row_5 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_4), vreinterpretq_u64_u16(efgh_4)));
    uint16x8_t row_6 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_6), vreinterpretq_u64_u16(efgh_6)));
    uint16x8_t row_7 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_6), vreinterpretq_u64_u16(efgh_6)));

    vst1q_u16(dst_ptr + 0 * dst_step, row_0);
    vst1q_u16(dst_ptr + 1 * dst_step, row_1);
    vst1q_u16(dst_ptr + 2 * dst_step, row_2);
    vst1q_u16(dst_ptr + 3 * dst_step, row_3);
    vst1q_u16(dst_ptr + 4 * dst_step, row_4);
    vst1q_u16(dst_ptr + 5 * dst_step, row_5);
    vst1q_u16(dst_ptr + 6 * dst_step, row_6);
    vst1q_u16(dst_ptr + 7 * dst_step, row_7);
}

static inline void trans_8x8_u8x3(const void* src, void* dst, const size_t src_step,
                      const size_t dst_step) {
    uint8_t* src_ptr = (uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;
    uint8x8x3_t src0 = vld3_u8(src_ptr + 0 * src_step);  // A0A1A2A3A4A5A6A7
    uint8x8x3_t src1 = vld3_u8(src_ptr + 1 * src_step);  // B0B1B2B3B4B5B6B7
    uint8x8x3_t src2 = vld3_u8(src_ptr + 2 * src_step);  // C0C1C2C3C4C5C6C7
    uint8x8x3_t src3 = vld3_u8(src_ptr + 3 * src_step);  // D0D1D2D3D4D5D6D7
    uint8x8x3_t src4 = vld3_u8(src_ptr + 4 * src_step);  // E0E1E2E3E4E5E6E7
    uint8x8x3_t src5 = vld3_u8(src_ptr + 5 * src_step);  // F0F1F2F3F4F5F6F7
    uint8x8x3_t src6 = vld3_u8(src_ptr + 6 * src_step);  // G0G1G2G3G4G5G6G7
    uint8x8x3_t src7 = vld3_u8(src_ptr + 7 * src_step);  // H0H1H2H3H4H5H6H7

    uint8x16_t c0_ab = vzip1q_u8(vcombine_u8(src0.val[0],src0.val[0]), vcombine_u8(src1.val[0],src1.val[0]));   // A0B0A1B1A2B2A3B3A4B4A5B5A6B6A7B7
    uint8x16_t c0_cd = vzip1q_u8(vcombine_u8(src2.val[0],src2.val[0]), vcombine_u8(src3.val[0],src3.val[0]));   // C0D0C1D1C2D2C3D3C4D4C5D5C6D6C7D7
    uint8x16_t c0_ef = vzip1q_u8(vcombine_u8(src4.val[0],src4.val[0]), vcombine_u8(src5.val[0],src5.val[0]));   // E0F0E1F1E2F2E3F3E4F4E5F5E6F6E7F7
    uint8x16_t c0_gh = vzip1q_u8(vcombine_u8(src6.val[0],src6.val[0]), vcombine_u8(src7.val[0],src7.val[0]));   // G0H0G1H1G2H2G3H3G4H4G5H5G6H6G7H7

    uint8x16_t c1_ab = vzip1q_u8(vcombine_u8(src0.val[1],src0.val[1]), vcombine_u8(src1.val[1],src1.val[1]));   // A0B0A1B1A2B2A3B3A4B4A5B5A6B6A7B7
    uint8x16_t c1_cd = vzip1q_u8(vcombine_u8(src2.val[1],src2.val[1]), vcombine_u8(src3.val[1],src3.val[1]));   // C0D0C1D1C2D2C3D3C4D4C5D5C6D6C7D7
    uint8x16_t c1_ef = vzip1q_u8(vcombine_u8(src4.val[1],src4.val[1]), vcombine_u8(src5.val[1],src5.val[1]));   // E0F0E1F1E2F2E3F3E4F4E5F5E6F6E7F7
    uint8x16_t c1_gh = vzip1q_u8(vcombine_u8(src6.val[1],src6.val[1]), vcombine_u8(src7.val[1],src7.val[1]));   // G0H0G1H1G2H2G3H3G4H4G5H5G6H6G7H7

    uint8x16_t c2_ab = vzip1q_u8(vcombine_u8(src0.val[2],src0.val[2]), vcombine_u8(src1.val[2],src1.val[2]));   // A0B0A1B1A2B2A3B3A4B4A5B5A6B6A7B7
    uint8x16_t c2_cd = vzip1q_u8(vcombine_u8(src2.val[2],src2.val[2]), vcombine_u8(src3.val[2],src3.val[2]));   // C0D0C1D1C2D2C3D3C4D4C5D5C6D6C7D7
    uint8x16_t c2_ef = vzip1q_u8(vcombine_u8(src4.val[2],src4.val[2]), vcombine_u8(src5.val[2],src5.val[2]));   // E0F0E1F1E2F2E3F3E4F4E5F5E6F6E7F7
    uint8x16_t c2_gh = vzip1q_u8(vcombine_u8(src6.val[2],src6.val[2]), vcombine_u8(src7.val[2],src7.val[2]));   // G0H0G1H1G2H2G3H3G4H4G5H5G6H6G7H7

    uint8x16_t c0_abcd_low  = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(c0_ab), vreinterpretq_u16_u8(c0_cd)));   // A0B0C0D0A1B1C1D1A2B2C2D2A3B3C3D3
    uint8x16_t c0_abcd_high = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(c0_ab), vreinterpretq_u16_u8(c0_cd)));   // A4B4C4D4A5B5C5D5A6B6C6D6A7B7C7D7
    uint8x16_t c0_efgh_low  = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(c0_ef), vreinterpretq_u16_u8(c0_gh)));   // E0F0G0H0E1F1G1H1E2F2G2H2E3F3G3H3
    uint8x16_t c0_efgh_high = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(c0_ef), vreinterpretq_u16_u8(c0_gh)));   // E4F4G4H4E5F5G5H5E6F6G6H6E7F7G7H7

    uint8x16_t c1_abcd_low  = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(c1_ab), vreinterpretq_u16_u8(c1_cd)));   // A0B0C0D0A1B1C1D1A2B2C2D2A3B3C3D3
    uint8x16_t c1_abcd_high = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(c1_ab), vreinterpretq_u16_u8(c1_cd)));   // A4B4C4D4A5B5C5D5A6B6C6D6A7B7C7D7
    uint8x16_t c1_efgh_low  = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(c1_ef), vreinterpretq_u16_u8(c1_gh)));   // E0F0G0H0E1F1G1H1E2F2G2H2E3F3G3H3
    uint8x16_t c1_efgh_high = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(c1_ef), vreinterpretq_u16_u8(c1_gh)));   // E4F4G4H4E5F5G5H5E6F6G6H6E7F7G7H7

    uint8x16_t c2_abcd_low  = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(c2_ab), vreinterpretq_u16_u8(c2_cd)));   // A0B0C0D0A1B1C1D1A2B2C2D2A3B3C3D3
    uint8x16_t c2_abcd_high = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(c2_ab), vreinterpretq_u16_u8(c2_cd)));   // A4B4C4D4A5B5C5D5A6B6C6D6A7B7C7D7
    uint8x16_t c2_efgh_low  = vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(c2_ef), vreinterpretq_u16_u8(c2_gh)));   // E0F0G0H0E1F1G1H1E2F2G2H2E3F3G3H3
    uint8x16_t c2_efgh_high = vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(c2_ef), vreinterpretq_u16_u8(c2_gh)));   // E4F4G4H4E5F5G5H5E6F6G6H6E7F7G7H7

    uint8x16_t c0_0  = vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(c0_abcd_low ), vreinterpretq_u32_u8(c0_efgh_low ))); 
    uint8x16_t c0_1  = vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(c0_abcd_low ), vreinterpretq_u32_u8(c0_efgh_low ))); 
    uint8x16_t c0_2  = vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(c0_abcd_high), vreinterpretq_u32_u8(c0_efgh_high))); 
    uint8x16_t c0_3  = vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(c0_abcd_high), vreinterpretq_u32_u8(c0_efgh_high)));

    uint8x16_t c1_0  = vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(c1_abcd_low ), vreinterpretq_u32_u8(c1_efgh_low )));   
    uint8x16_t c1_1  = vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(c1_abcd_low ), vreinterpretq_u32_u8(c1_efgh_low )));   
    uint8x16_t c1_2  = vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(c1_abcd_high), vreinterpretq_u32_u8(c1_efgh_high)));   
    uint8x16_t c1_3  = vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(c1_abcd_high), vreinterpretq_u32_u8(c1_efgh_high)));  

    uint8x16_t c2_0  = vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(c2_abcd_low ), vreinterpretq_u32_u8(c2_efgh_low )));   
    uint8x16_t c2_1  = vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(c2_abcd_low ), vreinterpretq_u32_u8(c2_efgh_low )));   
    uint8x16_t c2_2  = vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(c2_abcd_high), vreinterpretq_u32_u8(c2_efgh_high)));   
    uint8x16_t c2_3  = vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(c2_abcd_high), vreinterpretq_u32_u8(c2_efgh_high)));

    uint8x8x3_t res_0;
    res_0.val[0] = vget_low_u8(c0_0);
    res_0.val[1] = vget_low_u8(c1_0);
    res_0.val[2] = vget_low_u8(c2_0);
    uint8x8x3_t res_1;
    res_1.val[0] = vget_high_u8(c0_0);
    res_1.val[1] = vget_high_u8(c1_0);
    res_1.val[2] = vget_high_u8(c2_0);
    uint8x8x3_t res_2;
    res_2.val[0] = vget_low_u8(c0_1);
    res_2.val[1] = vget_low_u8(c1_1);
    res_2.val[2] = vget_low_u8(c2_1);
    uint8x8x3_t res_3;
    res_3.val[0] = vget_high_u8(c0_1);
    res_3.val[1] = vget_high_u8(c1_1);
    res_3.val[2] = vget_high_u8(c2_1);
    uint8x8x3_t res_4;
    res_4.val[0] = vget_low_u8(c0_2);
    res_4.val[1] = vget_low_u8(c1_2);
    res_4.val[2] = vget_low_u8(c2_2);
    uint8x8x3_t res_5;
    res_5.val[0] = vget_high_u8(c0_2);
    res_5.val[1] = vget_high_u8(c1_2);
    res_5.val[2] = vget_high_u8(c2_2);
    uint8x8x3_t res_6;
    res_6.val[0] = vget_low_u8(c0_3);
    res_6.val[1] = vget_low_u8(c1_3);
    res_6.val[2] = vget_low_u8(c2_3);
    uint8x8x3_t res_7;
    res_7.val[0] = vget_high_u8(c0_3);
    res_7.val[1] = vget_high_u8(c1_3);
    res_7.val[2] = vget_high_u8(c2_3);
    vst3_u8(dst_ptr + 0 * dst_step, res_0);
    vst3_u8(dst_ptr + 1 * dst_step, res_1);
    vst3_u8(dst_ptr + 2 * dst_step, res_2);
    vst3_u8(dst_ptr + 3 * dst_step, res_3);
    vst3_u8(dst_ptr + 4 * dst_step, res_4);
    vst3_u8(dst_ptr + 5 * dst_step, res_5);
    vst3_u8(dst_ptr + 6 * dst_step, res_6);
    vst3_u8(dst_ptr + 7 * dst_step, res_7);
}
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
            trans_4x16_u8_contig_src(src_ptr, dst_ptr, n);
        }
    }
    if(n_remain > 0){
        uint8_t* dst_ptr = dst + n_end;
        uint8_t* src_ptr = src + n_end * src_step;
        transpose_naive(src_ptr, dst_ptr, m, n_remain, 1, src_step, n);
    }
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
    if(m == 4 && src_step == m && dst_step == n){
        transpose_c1_m4_contig(src, dst, m, n);
        return;
    }else if(m == 3 && src_step == m && dst_step == n){
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

static inline void transpose_c2(uint8_t* src, uint8_t* dst, int m, int n, int c, int src_step, int dst_step){
    const int block = 8;
    int m_end = m / block * block;
    int m_remain = m - m_end;
    int n_end = n / block * block;
    int n_remain = n - n_end;
    int u16_step = src_step / 2;
    for(int n_idx = 0; n_idx < n_end; n_idx += block){
        for(int m_idx = 0; m_idx < m_end; m_idx += block){
            uint8_t* dst_ptr = dst + m_idx * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_idx * c;
            trans_8x8_u16(src_ptr, dst_ptr, u16_step , n);
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
static inline void transpose_c3(uint8_t* src, uint8_t* dst, int m, int n, int c, int src_step, int dst_step){
    const int block = 8;
    int m_end = m / block * block;
    int m_remain = m - m_end;
    int n_end = n / block * block;
    int n_remain = n - n_end;
    for(int n_idx = 0; n_idx < n_end; n_idx += block){
        for(int m_idx = 0; m_idx < m_end; m_idx += block){
            uint8_t* dst_ptr = dst + m_idx * dst_step + n_idx * c;
            uint8_t* src_ptr = src + n_idx * src_step + m_idx * c;
            trans_8x8_u8x3(src_ptr, dst_ptr, src_step, dst_step);
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
    } else if(c == 2) {
        transpose_c2(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    } else if(c == 3) {
        transpose_c3(src_base_ptr, dst_base_ptr, m, n, c, src_step, dst_step);
        return ;
    } else {
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
static inline void trans_8x8_u32(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint32_t* src_ptr = (uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    uint32x4x2_t src0 = vld1q_u32_x2(src_ptr + 0 * src_step);  // A0A1A2A3
    uint32x4x2_t src1 = vld1q_u32_x2(src_ptr + 1 * src_step);  // B0B1B2B3
    uint32x4x2_t src2 = vld1q_u32_x2(src_ptr + 2 * src_step);  // C0C1C2C3
    uint32x4x2_t src3 = vld1q_u32_x2(src_ptr + 3 * src_step);  // D0D1D2D3
    uint32x4x2_t src4 = vld1q_u32_x2(src_ptr + 4 * src_step);  // E0E1E2E3
    uint32x4x2_t src5 = vld1q_u32_x2(src_ptr + 5 * src_step);  // F0F1F2F3
    uint32x4x2_t src6 = vld1q_u32_x2(src_ptr + 6 * src_step);  // G0G1G2G3
    uint32x4x2_t src7 = vld1q_u32_x2(src_ptr + 7 * src_step);  // H0H1H2H3

    uint32x4_t ab_low = vzip1q_u32(src0.val[0], src1.val[0]);   // A0B0A1B1
    uint32x4_t ab_high = vzip2q_u32(src0.val[0], src1.val[0]);  // A2B2A3B3
    uint32x4_t cd_low = vzip1q_u32(src2.val[0], src3.val[0]);   // C0D0C1D1
    uint32x4_t cd_high = vzip2q_u32(src2.val[0], src3.val[0]);  // C2D2C3D3
    uint32x4_t ef_low = vzip1q_u32(src4.val[0], src5.val[0]);   // E0F0E1F1
    uint32x4_t ef_high = vzip2q_u32(src4.val[0], src5.val[0]);  // E2F2E3F3
    uint32x4_t gh_low = vzip1q_u32(src6.val[0], src7.val[0]);   // G0H0G1H1
    uint32x4_t gh_high = vzip2q_u32(src6.val[0], src7.val[0]);  // G2H2G3H3

    uint32x4_t abcd_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A0B0C0D0
    uint32x4_t abcd_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A1B1C1D1
    uint32x4_t abcd_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A2B2C2D2
    uint32x4_t abcd_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A3B3C3D3
    uint32x4_t efgh_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E0F0G0H0
    uint32x4_t efgh_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E1F1G1H1
    uint32x4_t efgh_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E2F2G2H2
    uint32x4_t efgh_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E3F3G3H3

    vst1q_u32(dst_ptr + 0 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 0 * dst_step + 4, efgh_0);
    vst1q_u32(dst_ptr + 1 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 1 * dst_step + 4, efgh_1);
    vst1q_u32(dst_ptr + 2 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 2 * dst_step + 4, efgh_2);
    vst1q_u32(dst_ptr + 3 * dst_step, abcd_3);
    vst1q_u32(dst_ptr + 3 * dst_step + 4, efgh_3);

    ab_low = vzip1q_u32(src0.val[1], src1.val[1]);   // A0B0A1B1
    ab_high = vzip2q_u32(src0.val[1], src1.val[1]);  // A2B2A3B3
    cd_low = vzip1q_u32(src2.val[1], src3.val[1]);   // C0D0C1D1
    cd_high = vzip2q_u32(src2.val[1], src3.val[1]);  // C2D2C3D3
    ef_low = vzip1q_u32(src4.val[1], src5.val[1]);   // E0F0E1F1
    ef_high = vzip2q_u32(src4.val[1], src5.val[1]);  // E2F2E3F3
    gh_low = vzip1q_u32(src6.val[1], src7.val[1]);   // G0H0G1H1
    gh_high = vzip2q_u32(src6.val[1], src7.val[1]);  // G2H2G3H3

    abcd_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A0B0C0D0
    abcd_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A1B1C1D1
    abcd_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A2B2C2D2
    abcd_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A3B3C3D3
    efgh_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E0F0G0H0
    efgh_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E1F1G1H1
    efgh_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E2F2G2H2
    efgh_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E3F3G3H3

    vst1q_u32(dst_ptr + 4 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 4 * dst_step + 4, efgh_0);
    vst1q_u32(dst_ptr + 5 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 5 * dst_step + 4, efgh_1);
    vst1q_u32(dst_ptr + 6 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 6 * dst_step + 4, efgh_2);
    vst1q_u32(dst_ptr + 7 * dst_step, abcd_3);
    vst1q_u32(dst_ptr + 7 * dst_step + 4, efgh_3);
}

static inline void trans_8x4_u32(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint32_t* src_ptr = (uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    uint32x4x2_t src0 = vld1q_u32_x2(src_ptr + 0 * src_step);  // A0A1A2A3
    uint32x4x2_t src1 = vld1q_u32_x2(src_ptr + 1 * src_step);  // B0B1B2B3
    uint32x4x2_t src2 = vld1q_u32_x2(src_ptr + 2 * src_step);  // C0C1C2C3
    uint32x4x2_t src3 = vld1q_u32_x2(src_ptr + 3 * src_step);  // D0D1D2D3

    uint32x4_t ab_low = vzip1q_u32(src0.val[0], src1.val[0]);   // A0B0A1B1
    uint32x4_t ab_high = vzip2q_u32(src0.val[0], src1.val[0]);  // A2B2A3B3
    uint32x4_t cd_low = vzip1q_u32(src2.val[0], src3.val[0]);   // C0D0C1D1
    uint32x4_t cd_high = vzip2q_u32(src2.val[0], src3.val[0]);  // C2D2C3D3


    uint32x4_t abcd_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A0B0C0D0
    uint32x4_t abcd_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A1B1C1D1
    uint32x4_t abcd_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A2B2C2D2
    uint32x4_t abcd_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A3B3C3D3

    vst1q_u32(dst_ptr + 0 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 1 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 2 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 3 * dst_step, abcd_3);

    ab_low = vzip1q_u32(src0.val[1], src1.val[1]);   // A0B0A1B1
    ab_high = vzip2q_u32(src0.val[1], src1.val[1]);  // A2B2A3B3
    cd_low = vzip1q_u32(src2.val[1], src3.val[1]);   // C0D0C1D1
    cd_high = vzip2q_u32(src2.val[1], src3.val[1]);  // C2D2C3D3

    abcd_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A0B0C0D0
    abcd_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A1B1C1D1
    abcd_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A2B2C2D2
    abcd_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A3B3C3D3

    vst1q_u32(dst_ptr + 4 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 5 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 6 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 7 * dst_step, abcd_3);
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

    uint64x2_t a0b0a1b1 = vreinterpretq_u64_u32(vzip1q_u32(src0, src1));
    uint64x2_t a2b2a3b3 = vreinterpretq_u64_u32(vzip2q_u32(src0, src1));
    uint64x2_t c0d0c1d1 = vreinterpretq_u64_u32(vzip1q_u32(src2, src3));
    uint64x2_t c2d2c3d3 = vreinterpretq_u64_u32(vzip2q_u32(src2, src3));
    uint64x2_t e0f0e1f1 = vreinterpretq_u64_u32(vzip1q_u32(src4, src5));
    uint64x2_t e2f2e3f3 = vreinterpretq_u64_u32(vzip2q_u32(src4, src5));
    uint64x2_t g0h0g1h1 = vreinterpretq_u64_u32(vzip1q_u32(src6, src7));
    uint64x2_t g2h2g3h3 = vreinterpretq_u64_u32(vzip2q_u32(src6, src7));

    uint64x2_t a0b0c0d0 = vzip1q_u64(a0b0a1b1, c0d0c1d1);
    uint64x2_t a1b1c1d1 = vzip2q_u64(a0b0a1b1, c0d0c1d1);
    uint64x2_t a2b2c2d2 = vzip1q_u64(a2b2a3b3, c2d2c3d3);
    uint64x2_t a3b3c3d3 = vzip2q_u64(a2b2a3b3, c2d2c3d3);
    uint64x2_t e0f0g0h0 = vzip1q_u64(e0f0e1f1, g0h0g1h1);
    uint64x2_t e1f1g1h1 = vzip2q_u64(e0f0e1f1, g0h0g1h1);
    uint64x2_t e2f2g2h2 = vzip1q_u64(e2f2e3f3, g2h2g3h3);
    uint64x2_t e3f3g3h3 = vzip2q_u64(e2f2e3f3, g2h2g3h3);

    vst1q_u32(dst_ptr + 0 * dst_step + 0,  a0b0c0d0);
    vst1q_u32(dst_ptr + 0 * dst_step + 4,  e0f0g0h0);
    vst1q_u32(dst_ptr + 1 * dst_step + 0,  a1b1c1d1);
    vst1q_u32(dst_ptr + 1 * dst_step + 4,  e1f1g1h1);
    vst1q_u32(dst_ptr + 2 * dst_step + 0,  a2b2c2d2);
    vst1q_u32(dst_ptr + 2 * dst_step + 4,  e2f2g2h2);
    vst1q_u32(dst_ptr + 3 * dst_step + 0,  a3b3c3d3);
    vst1q_u32(dst_ptr + 3 * dst_step + 4,  e3f3g3h3);
}

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
static inline void transpose_c1(uint32_t* src, uint32_t* dst, int m, int n, int c, int src_step, int dst_step){
    if(m == 4 && src_step == m && dst_step == n){
        transpose_c1_m4_contig(src, dst, m, n);
        return;
    }
    const int block = 8;
    int m_end = m / block * block;
    int m_remain = m - m_end;
    int n_end = n / block * block;
    int n_remain = n - n_end;
    const int block_n_2 = 4;
    int n_end2 = n / block_n_2 * block_n_2;
    int n_remain_2 = n - n_end2;
    for(int n_idx = 0; n_idx < n_end; n_idx += block){
        for(int m_idx = 0; m_idx < m_end; m_idx += block){
            uint32_t* dst_ptr = dst + m_idx * dst_step + n_idx;
            uint32_t* src_ptr = src + n_idx * src_step + m_idx;
            trans_8x8_u32(src_ptr, dst_ptr, src_step, dst_step);
        }
        if(m_remain > 0){
            uint32_t* dst_ptr = dst + m_end * dst_step + n_idx;
            uint32_t* src_ptr = src + n_idx * src_step + m_end;
            transpose_naive(src_ptr, dst_ptr, m_remain, block, 1, src_step, dst_step);
        }
    }
    if(n_remain > 0){
        for(int n_idx = n_end; n_idx < n_end2; n_idx += block_n_2){
            for(int m_idx = 0; m_idx < m_end; m_idx += block){
                uint32_t* dst_ptr = dst + m_idx * dst_step + n_idx;
                uint32_t* src_ptr = src + n_idx * src_step + m_idx;
                trans_8x4_u32(src_ptr, dst_ptr, src_step, dst_step);
            }
            if(m_remain > 0){
                uint32_t* dst_ptr = dst + m_end * dst_step + n_idx;
                uint32_t* src_ptr = src + n_idx * src_step + m_end;
                transpose_naive(src_ptr, dst_ptr, m_remain, block_n_2, 1, src_step, dst_step);
            }
        }
        if(n_remain_2 > 0){
            uint32_t* dst_ptr = dst + 0 * dst_step + n_end2;
            uint32_t* src_ptr = src + n_end2 * src_step + 0;
            transpose_naive(src_ptr, dst_ptr, m, n_remain_2, 1, src_step, dst_step);
        }
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

std::string gen_transpose(int data_size) {
    if (data_size == 1) {
        return gen_transpose_u8();
    } else if (data_size == 4) {
        return gen_transpose_u32();
    } else {
        CC_ABORT << "not support type size " << data_size << "\n";
    }
    return "";
}

}  // namespace
}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc