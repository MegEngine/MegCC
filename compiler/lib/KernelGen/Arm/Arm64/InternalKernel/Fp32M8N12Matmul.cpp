/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/InternalKernel/Fp32M8N12Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string>
#include "../../../Utils/StringTemplate.h"
#include "Arm/Arm64/Activation.h"
#include "Arm/ArmCommon/ElemwiseHelper/ElemwiseHelper.h"
#include "Arm/ArmCommon/MatmulCommon.h"
#include "Arm/ArmCommon/common_asm_utils.h"
#include "InternalKernel.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;
namespace {
std::string utilsFunc(void) {
    return R"(
static inline size_t min(size_t a,size_t b){
    return a < b ? a : b;
}
    )";
}

std::string prefetch(void) {
    return R"(
        #define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_2x_f32() +
           KernelGen::ArmCommon::gen_common_prefetch_3x_f32();
}
std::string transpose_8x4_1_s(void) {
    return R"(
static inline void transpose_8x4_1_s(const float* inptr0, const float* inptr1,
                                     const float* inptr2, const float* inptr3,
                                     const float* inptr4, const float* inptr5,
                                     const float* inptr6, const float* inptr7,
                                     float* outptr){
    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], 16\n"  // A0A1A2A3
            "ld1 {v1.4s},  [%[inptr1]], 16\n"  // B0B1B2B3
            "ld1 {v2.4s},  [%[inptr2]], 16\n"  // C0C1C2C3
            "ld1 {v3.4s},  [%[inptr3]], 16\n"  // D0D1D2D3
            "ld1 {v4.4s},  [%[inptr4]], 16\n"  // E0E1E2E3
            "ld1 {v5.4s},  [%[inptr5]], 16\n"  // F0F1F2F3
            "ld1 {v6.4s},  [%[inptr6]], 16\n"  // G0G1G2G3
            "ld1 {v7.4s},  [%[inptr7]], 16\n"  // H0H1H2H3

            "zip1 v8.4s, v0.4s, v1.4s\n"   // A0B0A1B1
            "zip2 v9.4s, v0.4s, v1.4s\n"   // A2B2A3B3
            "zip1 v10.4s, v2.4s, v3.4s\n"  // C0D0C1D1
            "zip2 v11.4s, v2.4s, v3.4s\n"  // C2D2C3D3
            "zip1 v12.4s, v4.4s, v5.4s\n"  // E0F0E1F1
            "zip2 v13.4s, v4.4s, v5.4s\n"  // E2F2E3F3
            "zip1 v14.4s, v6.4s, v7.4s\n"  // G0H0G1H1
            "zip2 v15.4s, v6.4s, v7.4s\n"  // G2H2G3H3

            "zip1 v0.2d, v8.2d, v10.2d\n"  // A0B0C0D0
            "zip2 v2.2d, v8.2d, v10.2d\n"  // A1B1C1D1

            "zip1 v4.2d, v9.2d, v11.2d\n"  // A2B2C2D2
            "zip2 v6.2d, v9.2d, v11.2d\n"  // A3B3C3D3

            "zip1 v1.2d, v12.2d, v14.2d\n"  // E0F0G0H0
            "zip2 v3.2d, v12.2d, v14.2d\n"  // E1F1G1H1

            "zip1 v5.2d, v13.2d, v15.2d\n"  // E2F2G2H2
            "zip2 v7.2d, v13.2d, v15.2d\n"  // E3F3G3H3

            "st1 {v0.4s,v1.4s,v2.4s,v3.4s},  [%[outptr]], #64\n"
            "st1 {v4.4s,v5.4s,v6.4s,v7.4s},  [%[outptr]], #64\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "memory");
}
    )";
}

std::string interleave(void) {
    return KernelGen::ArmCommon::gen_common_interleve_f32() + R"(

static inline void interleave_4x12_1_s(const float* inptr0, const float* inptr1,
                                       const float* inptr2, const float* inptr3,
                                       float* outptr) {
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s}, [%[inptr0]], #48\n"
            "ld1 {v4.4s, v5.4s, v6.4s}, [%[inptr1]], #48\n"
            "ld1 {v8.4s, v9.4s, v10.4s}, [%[inptr2]], #48\n"
            "ld1 {v12.4s, v13.4s, v14.4s}, [%[inptr3]], #48\n"
            "st1 {v0.4s, v1.4s, v2.4s}, [%[outptr]], #48\n"
            "st1 {v4.4s, v5.4s, v6.4s}, [%[outptr]], #48\n"
            "st1 {v8.4s, v9.4s, v10.4s}, [%[outptr]], #48\n"
            "st1 {v12.4s, v13.4s, v14.4s}, [%[outptr]], #48\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v4", "v5", "v6", "v8", "v9", "v10", "v12",
              "v13", "v14", "cc", "memory");
}

static inline void interleave_4x4_1_s(const float* inptr0, const float* inptr1,
                                      const float* inptr2, const float* inptr3,
                                      float* outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"
            "ld1 {v1.4s}, [%[inptr1]], #16\n"
            "ld1 {v2.4s}, [%[inptr2]], #16\n"
            "ld1 {v3.4s}, [%[inptr3]], #16\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[outptr]], #64\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "cc", "memory");
}

static inline void interleave_1x12_1_s(const float* inptr0, float* outptr) {
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s}, [%[inptr0]], #48\n"
            "st1 {v0.4s, v1.4s, v2.4s}, [%[outptr]], #48\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "cc", "memory");
}

static inline void interleave_1x4_1_s(const float* inptr0, float* outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"
            "st1 {v0.4s}, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "cc", "memory");
}

static inline void interleave_4x8_1_s(const float* inptr0, const float* inptr1,
                                      const float* inptr2, const float* inptr3,
                                      float* outptr) {
    asm volatile(
            "ld1 {v0.4s, v1.4s}, [%[inptr0]], #32\n"
            "ld1 {v2.4s, v3.4s}, [%[inptr1]], #32\n"
            "ld1 {v4.4s, v5.4s}, [%[inptr2]], #32\n"
            "ld1 {v6.4s, v7.4s}, [%[inptr3]], #32\n"
            "st1 {v0.4s, v1.4s}, [%[outptr]], #32\n"
            "st1 {v2.4s, v3.4s}, [%[outptr]], #32\n"
            "st1 {v4.4s, v5.4s}, [%[outptr]], #32\n"
            "st1 {v6.4s, v7.4s}, [%[outptr]], #32\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
}

static inline void interleave_1x8_1_s(const float* inptr0, float* outptr) {
    asm volatile(
            "ld1 {v0.4s, v1.4s}, [%[inptr0]], #32\n"
            "st1 {v0.4s, v1.4s}, [%[outptr]], #32\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "cc", "memory");
}

static inline void transpose_12x4_1_s(const float* inptr0, const float* inptr1,
                                      const float* inptr2, const float* inptr3,
                                      const float* inptr4, const float* inptr5,
                                      const float* inptr6, const float* inptr7,
                                      const float* inptr8, const float* inptr9,
                                      const float* inptr10, const float* inptr11,
                                      float* outptr) {
    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], 16\n"    // A0A1A2A3
            "ld1 {v1.4s},  [%[inptr1]], 16\n"    // B0B1B2B3
            "ld1 {v2.4s},  [%[inptr2]], 16\n"    // C0C1C2C3
            "ld1 {v3.4s},  [%[inptr3]], 16\n"    // D0D1D2D3
            "ld1 {v4.4s},  [%[inptr4]], 16\n"    // E0E1E2E3
            "ld1 {v5.4s},  [%[inptr5]], 16\n"    // F0F1F2F3
            "ld1 {v6.4s},  [%[inptr6]], 16\n"    // G0G1G2G3
            "ld1 {v7.4s},  [%[inptr7]], 16\n"    // H0H1H2H3
            "ld1 {v16.4s},  [%[inptr8]], 16\n"   // I0I1I2I3
            "ld1 {v17.4s},  [%[inptr9]], 16\n"   // J0J1J2J3
            "ld1 {v18.4s},  [%[inptr10]], 16\n"  // K0K1K2K3
            "ld1 {v19.4s},  [%[inptr11]], 16\n"  // L0L1L2L3

            "zip1 v8.4s, v0.4s, v1.4s\n"   // A0B0A1B1
            "zip2 v9.4s, v0.4s, v1.4s\n"   // A2B2A3B3
            "zip1 v10.4s, v2.4s, v3.4s\n"  // C0D0C1D1
            "zip2 v11.4s, v2.4s, v3.4s\n"  // C2D2C3D3

            "zip1 v12.4s, v4.4s, v5.4s\n"  // E0F0E1F1
            "zip2 v13.4s, v4.4s, v5.4s\n"  // E2F2E3F3
            "zip1 v14.4s, v6.4s, v7.4s\n"  // G0H0G1H1
            "zip2 v15.4s, v6.4s, v7.4s\n"  // G2H2G3H3

            "zip1 v20.4s, v16.4s, v17.4s\n"  // I0J0I1J1
            "zip2 v21.4s, v16.4s, v17.4s\n"  // I2J2I3J3
            "zip1 v22.4s, v18.4s, v19.4s\n"  // K0L0K1L1
            "zip2 v23.4s, v18.4s, v19.4s\n"  // K2L2K3L3

            "zip1 v0.2d, v8.2d, v10.2d\n"  // A0B0C0D0
            "zip2 v3.2d, v8.2d, v10.2d\n"  // A1B1C1D1

            "zip1 v6.2d, v9.2d, v11.2d\n"   // A2B2C2D2
            "zip2 v24.2d, v9.2d, v11.2d\n"  // A3B3C3D3

            "zip1 v1.2d, v12.2d, v14.2d\n"  // E0F0G0H0
            "zip2 v4.2d, v12.2d, v14.2d\n"  // E1F1G1H1

            "zip1 v7.2d, v13.2d, v15.2d\n"   // E2F2G2H2
            "zip2 v25.2d, v13.2d, v15.2d\n"  // E3F3G3H3

            "zip1 v2.2d, v20.2d, v22.2d\n"  // I0J0K0L0
            "zip2 v5.2d, v20.2d, v22.2d\n"  // I1J1K1L1

            "zip1 v8.2d, v21.2d, v23.2d\n"   // I2J2K2L2
            "zip2 v26.2d, v21.2d, v23.2d\n"  // I3J3K3L3

            "st1 {v0.4s,v1.4s,v2.4s},  [%[outptr]], #48\n"
            "st1 {v3.4s,v4.4s,v5.4s},  [%[outptr]], #48\n"
            "st1 {v6.4s,v7.4s,v8.4s},  [%[outptr]], #48\n"
            "st1 {v24.4s,v25.4s,v26.4s},  [%[outptr]], #48\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [inptr8] "+r"(inptr8),
            [inptr9] "+r"(inptr9), [inptr10] "+r"(inptr10),
            [inptr11] "+r"(inptr11), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "memory");
}
    )";
}

std::string transpose_4x4_1_s(void) {
    return R"(
static inline void transpose_4x4_1_s(const float* inptr0, const float* inptr1,
                                     const float* inptr2, const float* inptr3,
                                     float* outptr, int stride) {
    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], 16\n"  // A0A1A2A3
            "ld1 {v1.4s},  [%[inptr1]], 16\n"  // B0B1B2B3
            "ld1 {v2.4s},  [%[inptr2]], 16\n"  // C0C1C2C3
            "ld1 {v3.4s},  [%[inptr3]], 16\n"  // D0D1D2D3

            "zip1 v4.4s, v0.4s, v1.4s\n"
            "zip1 v5.4s, v2.4s, v3.4s\n"
            "zip2 v6.4s, v0.4s, v1.4s\n"
            "zip2 v7.4s, v2.4s, v3.4s\n"

            "zip1 v8.2d, v4.2d, v5.2d\n"
            "zip1 v9.2d, v6.2d, v7.2d\n"
            "zip2 v10.2d, v4.2d, v5.2d\n"
            "zip2 v11.2d, v6.2d, v7.2d\n"

            "st1 {v8.4s},  [%[outptr]], %x[stride]\n"
            "st1 {v10.4s},  [%[outptr]], %x[stride]\n"
            "st1 {v9.4s},  [%[outptr]], %x[stride]\n"
            "st1 {v11.4s},  [%[outptr]], %x[stride]\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "memory");
}

    )";
}

std::string pack_A_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_n" + MatmulInternal::GenPackACall(ctx) +
           R"({
        float zerobuff[8];
        memset(zerobuff, 0, sizeof(float) * 8);
        const int PACK_SIZE_32 = 4 * 8;
        const int PACK_SIZE_16 = 4 * 4;
        int y = y0;
        for (; y + 7 < ymax; y += 8) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            const float* inptr4 = inptr3 + ldin;
            const float* inptr5 = inptr4 + ldin;
            const float* inptr6 = inptr5 + ldin;
            const float* inptr7 = inptr6 + ldin;
            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);
            prefetch_2x(inptr4);
            prefetch_2x(inptr5);
            prefetch_2x(inptr6);
            prefetch_2x(inptr7);
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_8x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4,
                                  inptr5, inptr6, inptr7, outptr);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                inptr4 += 4;inptr5 += 4;inptr6 += 4;inptr7 += 4;
                outptr += PACK_SIZE_32;
            }
            for (; x > 0; x--) {
                *outptr++ = *inptr0++;
                *outptr++ = *inptr1++;
                *outptr++ = *inptr2++;
                *outptr++ = *inptr3++;
                *outptr++ = *inptr4++;
                *outptr++ = *inptr5++;
                *outptr++ = *inptr6++;
                *outptr++ = *inptr7++;
            }
        }

        for (; y < ymax; y += 4) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);

            int K = (kmax - k0);
            for (; K > 3; K -= 4) {
                if ((y + 3) >= ymax) {
                    switch ((y + 3) - ymax) {
                        /* Everything falls through in here */
                        case 2:
                            inptr1 = zerobuff;
                            
                        case 1:
                            inptr2 = zerobuff;
                            
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:;
                    }
                }

                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr,16);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                outptr += PACK_SIZE_16;
            }

            if (K > 0) {
                if (y + 3 >= ymax) {
                    switch (y + 3 - ymax) {
                        case 2:
                            inptr1 = zerobuff;
                        case 1:
                            inptr2 = zerobuff;
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:;
                    }
                }
                interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K,0);
                outptr += K * 4;
            }
        }
    }        
    )";
}

std::string pack_A_t(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_t" + MatmulInternal::GenPackACall(ctx) +
           R"( {
        int x0 = y0;
        int xmax = ymax;
        int ksize = kmax - k0;
        int ksize8 = (ksize << 3);
        int ksize4 = (ksize << 2);
        float* outptr_base = outptr;
        float* outptr_base4 = outptr_base + (xmax - x0) / 8 * ksize8;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* tmp_inptr = inptr + k * ldin + x0;
            const float* tmp_inptr1 = tmp_inptr + ldin;
            const float* tmp_inptr2 = tmp_inptr1 + ldin;
            const float* tmp_inptr3 = tmp_inptr2 + ldin;

            prefetch_3x(tmp_inptr);
            prefetch_3x(tmp_inptr1);
            prefetch_3x(tmp_inptr2);
            prefetch_3x(tmp_inptr3);

            int x = x0;
            float* tmp_outptr = outptr_base;
            for (; x + 8 <= xmax; x += 8) {
                float* outptr_interleave = tmp_outptr;
                interleave_4x8_1_s(tmp_inptr, tmp_inptr1, tmp_inptr2, tmp_inptr3,
                                   outptr_interleave);
                tmp_inptr += 8;
                tmp_inptr1 += 8;
                tmp_inptr2 += 8;
                tmp_inptr3 += 8;
                tmp_outptr += ksize8;
            }
            tmp_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = tmp_outptr;
                interleave_4x4_1_s(tmp_inptr, tmp_inptr1, tmp_inptr2, tmp_inptr3,
                                   outptr_interleave);
                tmp_inptr += 4;
                tmp_inptr1 += 4;
                tmp_inptr2 += 4;
                tmp_inptr3 += 4;  
                tmp_outptr += ksize4;
            }
            if (x < xmax) {
                interleave_4(tmp_inptr, tmp_inptr1, tmp_inptr2, tmp_inptr3, tmp_outptr, 4,
                             xmax - x,0);
            }
            outptr_base += 4 * 8;
            outptr_base4 += 4 * 4;
        }

        for (; k < kmax; k++) {
            const float* tmp_inptr = inptr + k * ldin + x0;
            prefetch_3x(tmp_inptr);
            int x = x0;
            float* tmp_outptr = outptr_base;
            for (; x + 8 <= xmax; x += 8) {
                float* outptr_interleave = tmp_outptr;
                interleave_1x8_1_s(tmp_inptr, outptr_interleave);
                tmp_inptr += 8;
                tmp_outptr += ksize8;
            }
            tmp_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = tmp_outptr;
                interleave_1x4_1_s(tmp_inptr, outptr_interleave);
                tmp_inptr += 4;
                tmp_outptr += ksize4;
            }
            if (x < xmax) {
                interleave_1(tmp_inptr, tmp_outptr, 4, xmax - x,0);
            }
            outptr_base += 8;
            outptr_base4 += 4;
        }
    }
    )";
}

std::string pack_B_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_n" + MatmulInternal::GenPackBCall(ctx) +
           R"( {
        int ksize = kmax - k0;
        int ksize12 = ksize * 12;
        int ksize4 = (ksize << 2);
        float* outptr_base = outptr;
        float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* tmp_inptr = inptr + k * ldin + x0;
            const float* tmp_inptr1 = tmp_inptr + ldin;
            const float* tmp_inptr2 = tmp_inptr1 + ldin;
            const float* tmp_inptr3 = tmp_inptr2 + ldin;

            prefetch_3x(tmp_inptr);
            prefetch_3x(tmp_inptr1);
            prefetch_3x(tmp_inptr2);
            prefetch_3x(tmp_inptr3);

            int x = x0;
            float* tmp_outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                float* outptr_interleave = tmp_outptr;
                interleave_4x12_1_s(tmp_inptr, tmp_inptr1, tmp_inptr2, tmp_inptr3,
                                    outptr_interleave);
                tmp_inptr += 12;
                tmp_inptr1+=12;
                tmp_inptr2+=12;
                tmp_inptr3+=12;
                tmp_outptr += ksize12;
            }
            tmp_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = tmp_outptr;
                interleave_4x4_1_s(tmp_inptr, tmp_inptr1, tmp_inptr2, tmp_inptr3,
                                   outptr_interleave);
                tmp_inptr += 4;
                tmp_inptr1 += 4;
                tmp_inptr2 += 4;
                tmp_inptr3 += 4;                   
                tmp_outptr += ksize4;
            }
            if (x < xmax) {
                interleave_4(tmp_inptr, tmp_inptr1, tmp_inptr2, tmp_inptr3, tmp_outptr, 4,
                             xmax - x, 0);                
            }
            outptr_base += 12 * 4;
            outptr_base4 += 4 * 4;
        }

        for (; k < kmax; k++) {
            const float* tmp_inptr = inptr + k * ldin + x0;
            prefetch_3x(tmp_inptr);
            int x = x0;
            float* tmp_outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                float* outptr_interleave = tmp_outptr;
                interleave_1x12_1_s(tmp_inptr, outptr_interleave);
                tmp_inptr += 12;
                tmp_outptr += ksize12;
            }
            tmp_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = tmp_outptr;
                interleave_1x4_1_s(tmp_inptr, outptr_interleave);
                tmp_inptr += 4;
                tmp_outptr += ksize4;
            }
            if (x < xmax) {
                interleave_1(tmp_inptr, tmp_outptr, 4, xmax - x,0);
            }
            outptr_base += 12;
            outptr_base4 += 4;
        }
    }
    )";
}

std::string pack_B_t(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_t" + MatmulInternal::GenPackBCall(ctx) +
           R"(
     {
        int y0 = x0;
        int ymax = xmax;
        float zerobuff[12];
        memset(zerobuff, 0, sizeof(float) * 12);
        int y = y0;
        for (; y + 12 <= ymax; y += 12) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            const float* inptr4 = inptr3 + ldin;
            const float* inptr5 = inptr4 + ldin;
            const float* inptr6 = inptr5 + ldin;
            const float* inptr7 = inptr6 + ldin;
            const float* inptr8 = inptr7 + ldin;
            const float* inptr9 = inptr8 + ldin;
            const float* inptr10 = inptr9 + ldin;
            const float* inptr11 = inptr10 + ldin;
            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);
            prefetch_2x(inptr4);
            prefetch_2x(inptr5);
            prefetch_2x(inptr6);
            prefetch_2x(inptr7);
            prefetch_2x(inptr8);
            prefetch_2x(inptr9);
            prefetch_2x(inptr10);
            prefetch_2x(inptr11);
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_12x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4,
                                   inptr5, inptr6, inptr7, inptr8, inptr9,
                                   inptr10, inptr11, outptr);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                inptr4 += 4;inptr5 += 4;inptr6 += 4;inptr7 += 4;
                inptr8 += 4;inptr9 += 4;inptr10 += 4;inptr11 += 4;
                outptr += 48;
            }
            for (; x > 0; x--) {
                *outptr++ = *inptr0++;
                *outptr++ = *inptr1++;
                *outptr++ = *inptr2++;
                *outptr++ = *inptr3++;
                *outptr++ = *inptr4++;
                *outptr++ = *inptr5++;
                *outptr++ = *inptr6++;
                *outptr++ = *inptr7++;
                *outptr++ = *inptr8++;
                *outptr++ = *inptr9++;
                *outptr++ = *inptr10++;
                *outptr++ = *inptr11++;
            }
        }

        for (; y < ymax; y += 4) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);

            /* Cope with ragged cases by copying from a buffer of zeroes instead
             */
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                if ((y + 3) >= ymax) {
                    switch ((y + 3) - ymax) {
                        /* Everything falls through in here */
                        case 2:
                            inptr1 = zerobuff;
                            
                        case 1:
                            inptr2 = zerobuff;
                            
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:;
                            
                    }
                }

                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr,16);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                outptr += 16;
            }

            if (x > 0) {
                if ((y + 3) >= ymax) {
                    switch ((y + 3) - ymax) {
                        /* Everything falls through in here */
                        case 2:
                            inptr1 = zerobuff;
                            
                        case 1:
                            inptr2 = zerobuff;
                            
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:;
                            
                    }
                }
                interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, x,0);
                inptr0 += x;inptr1 += x;inptr2 += x;inptr3 += x;
                outptr += 4*x;
            }
        }
    }
    )";
}

std::string kern8x12(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    ss << R"(
static inline void kern_8x12(const float* packA, const float* packB, int K,
                          float* output, int LDC, const float* bias_ptr) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = (float*)(output);

// clang-format off

    // clang-format on

        asm volatile(
                "add x1, x0, %x[LDC]\n"
                "add x2, x1, %x[LDC]\n"
                "add x3, x2, %x[LDC]\n"
                "add x4, x3, %x[LDC]\n"
                "add x5, x4, %x[LDC]\n"
                "add x6, x5, %x[LDC]\n"
                "add x7, x6, %x[LDC]\n"

)";
    if (with_bias) {
        ss << R"(
                "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
                "dup  v8.4s, v6.s[0]          \n"
                "dup  v9.4s, v6.s[0]          \n"
                "dup v10.4s, v6.s[0]          \n"
                "ld1 {v7.4s}, [%[bias_ptr]], #16\n"
                "dup v11.4s, v6.s[1]          \n"
                "dup v12.4s, v6.s[1]          \n"
                "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
                "dup v13.4s, v6.s[1]          \n"
                "dup v14.4s, v6.s[2]          \n"
                "dup v15.4s, v6.s[2]          \n"
                "dup v16.4s, v6.s[2]          \n"
                "dup v17.4s, v6.s[3]          \n"
                "ld1 {v4.4s}, [%[b_ptr]], 16\n"
                "dup v18.4s, v6.s[3]          \n"
                "dup v19.4s, v6.s[3]          \n"  
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"              
                "dup v20.4s, v7.s[0]          \n"
                "dup v21.4s, v7.s[0]          \n"
                "dup v22.4s, v7.s[0]          \n"
                "dup v23.4s, v7.s[1]          \n"
                "dup v24.4s, v7.s[1]          \n"
                "dup v25.4s, v7.s[1]          \n"
                "dup v26.4s, v7.s[2]          \n"
                "dup v27.4s, v7.s[2]          \n"
                "dup v28.4s, v7.s[2]          \n"
                "dup v29.4s, v7.s[3]          \n"
                "dup v30.4s, v7.s[3]          \n"
                "dup v31.4s, v7.s[3]          \n"

            )";
    } else {
        ss << R"(
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "eor v16.16b, v16.16b, v16.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "eor v18.16b, v18.16b, v18.16b\n"
                "eor v19.16b, v19.16b, v19.16b\n"
                "eor v20.16b, v20.16b, v20.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "eor v21.16b, v21.16b, v21.16b\n"
                "eor v22.16b, v22.16b, v22.16b\n"
                "eor v23.16b, v23.16b, v23.16b\n"
                "eor v24.16b, v24.16b, v24.16b\n"
                "eor v25.16b, v25.16b, v25.16b\n"
                "eor v26.16b, v26.16b, v26.16b\n"
                "eor v27.16b, v27.16b, v27.16b\n"
                "eor v28.16b, v28.16b, v28.16b\n"
                "eor v29.16b, v29.16b, v29.16b\n"
                "eor v30.16b, v30.16b, v30.16b\n"
                "eor v31.16b, v31.16b, v31.16b\n"
            )";
    }
    std::string body_temp = R"(
                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "prfm pldl1keep, [%[a_ptr], #64]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "prfm pldl1keep, [%[b_ptr], #64]\n"
                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "fmla v31.4s, v4.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "fmla v9.4s,  v6.4s, v0.s[0]\n"
                "fmla v10.4s, v7.4s, v0.s[0]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "fmla v12.4s, v6.4s, v0.s[1]\n"
                "fmla v13.4s, v7.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "fmla v15.4s, v6.4s, v0.s[2]\n"
                "fmla v16.4s, v7.4s, v0.s[2]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "fmla v18.4s, v6.4s, v0.s[3]\n"
                "fmla v19.4s, v7.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "fmla v21.4s, v6.4s, v1.s[0]\n"
                "fmla v22.4s, v7.4s, v1.s[0]\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v24.4s, v6.4s, v1.s[1]\n"
                "fmla v25.4s, v7.4s, v1.s[1]\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "fmla v27.4s, v6.4s, v1.s[2]\n"
                "fmla v28.4s, v7.4s, v1.s[2]\n"
                "fmla v29.4s, v5.4s, v1.s[3]\n"
                "fmla v30.4s, v6.4s, v1.s[3]\n"
                "prfm pldl1keep, [%[b_ptr], #64]\n"
                "fmla v31.4s, v7.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "fmla v31.4s, v4.4s, v1.s[3]\n"

                "eor v2.16b, v2.16b, v2.16b\n"
                "fmla v8.4s,  v5.4s, v0.s[0]\n"                
                "fmla v9.4s,  v6.4s, v0.s[0]\n"
                "fmla v10.4s, v7.4s, v0.s[0]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                ${GenAsmFloat(v8, v2)}
                "fmla v12.4s, v6.4s, v0.s[1]\n"
                ${GenAsmFloat(v9, v2)}
                "fmla v13.4s, v7.4s, v0.s[1]\n"
                ${GenAsmFloat(v10, v2)}
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                ${GenAsmFloat(v11, v2)}
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                ${GenAsmFloat(v12, v2)}
                "fmla v15.4s, v6.4s, v0.s[2]\n"
                ${GenAsmFloat(v13, v2)}
                "fmla v16.4s, v7.4s, v0.s[2]\n"
                ${GenAsmFloat(v14, v2)}
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                ${GenAsmFloat(v15, v2)}
                "st1 {v8.4s, v9.4s, v10.4s}, [x0]\n"
                ${GenAsmFloat(v16, v2)}
                "fmla v18.4s, v6.4s, v0.s[3]\n"
                ${GenAsmFloat(v17, v2)}
                "fmla v19.4s, v7.4s, v0.s[3]\n"
                ${GenAsmFloat(v18, v2)}
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                ${GenAsmFloat(v19, v2)}
                "fmla v21.4s, v6.4s, v1.s[0]\n"
                ${GenAsmFloat(v20, v2)}
                "st1 {v11.4s, v12.4s, v13.4s}, [x1]\n"
                ${GenAsmFloat(v21, v2)}
                "fmla v22.4s, v7.4s, v1.s[0]\n"
                ${GenAsmFloat(v22, v2)}
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                ${GenAsmFloat(v23, v2)}
                "fmla v24.4s, v6.4s, v1.s[1]\n"
                ${GenAsmFloat(v24, v2)}
                "fmla v25.4s, v7.4s, v1.s[1]\n"
                "st1 {v14.4s, v15.4s, v16.4s}, [x2]\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                ${GenAsmFloat(v25, v2)}
                "fmla v27.4s, v6.4s, v1.s[2]\n"
                ${GenAsmFloat(v26, v2)}
                "st1 {v17.4s, v18.4s, v19.4s}, [x3]\n"
                "fmla v28.4s, v7.4s, v1.s[2]\n"
                ${GenAsmFloat(v27, v2)}
                "fmla v29.4s, v5.4s, v1.s[3]\n"
                "st1 {v20.4s, v21.4s, v22.4s}, [x4]\n"
                ${GenAsmFloat(v28, v2)}
                "fmla v30.4s, v6.4s, v1.s[3]\n"
                "st1 {v23.4s, v24.4s, v25.4s}, [x5]\n"
                ${GenAsmFloat(v29, v2)}
                "fmla v31.4s, v7.4s, v1.s[3]\n"
                ${GenAsmFloat(v30, v31, v2)}                                                
                "st1 {v26.4s, v27.4s, v28.4s}, [x6]\n"
                "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "eor v7.16b, v7.16b, v7.16b\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                ${GenAsmFloat(v8, v7)}
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                ${GenAsmFloat(v9, v7)}
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                ${GenAsmFloat(v10, v7)}
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"               
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                ${GenAsmFloat(v11, v7)}                
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                ${GenAsmFloat(v12, v7)}
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                ${GenAsmFloat(v13, v7)}
                "st1 {v8.4s, v9.4s, v10.4s}, [x0]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                ${GenAsmFloat(v14, v7)}
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                ${GenAsmFloat(v15, v7)}
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                ${GenAsmFloat(v16, v7)}
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                ${GenAsmFloat(v17, v7)}
                "st1 {v11.4s, v12.4s, v13.4s}, [x1]\n"
                "fmla v21.4s, v3.4s, v1.s[0]\n"
                ${GenAsmFloat(v18, v7)}
                "fmla v22.4s, v4.4s, v1.s[0]\n"
                ${GenAsmFloat(v19, v7)}
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                ${GenAsmFloat(v20, v7)}
                "fmla v24.4s, v3.4s, v1.s[1]\n"
                ${GenAsmFloat(v21, v7)}
                "st1 {v14.4s, v15.4s, v16.4s}, [x2]\n"
                "fmla v25.4s, v4.4s, v1.s[1]\n"
                ${GenAsmFloat(v22, v7)}
                "st1 {v17.4s, v18.4s, v19.4s}, [x3]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                ${GenAsmFloat(v23, v7)}
                "fmla v27.4s, v3.4s, v1.s[2]\n"
                ${GenAsmFloat(v24, v7)}
                "st1 {v20.4s, v21.4s, v22.4s}, [x4]\n"
                "fmla v28.4s, v4.4s, v1.s[2]\n"
                ${GenAsmFloat(v25, v7)}
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                ${GenAsmFloat(v26, v7)}
                "st1 {v23.4s, v24.4s, v25.4s}, [x5]\n"
                ${GenAsmFloat(v27, v7)}
                "fmla v30.4s, v3.4s, v1.s[3]\n"
                ${GenAsmFloat(v28, v7)}
                "fmla v31.4s, v4.4s, v1.s[3]\n"
                "st1 {v26.4s, v27.4s, v28.4s}, [x6]\n"
                ${GenAsmFloat(v29, v30, v31, v7)}                
                "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"

                "6:\n"

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), 
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31", "x1", "x2", "x3", "x4", "x5",
                  "x6", "x7", "cc", "memory");


    }
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("GenAsmFloat",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenAsmFloat(args);
                         })
                    .render(body_temp);
    return ss.str();
}

std::string kern8x4(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    ss << R"(
    static void kern_8x4(const float* packA, const float* packB, int K,
                         float* output, int LDC, int n_remain, const float* bias_ptr) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = (float*)(output);

// clang-format off
#define STORE_LINE(v0, n)               \
    "cmp %w[n_remain], #4\n"            \
    "blt 103" n "f\n"                   \
    "st1 {v" v0 ".4s}, [x" n " ],#16\n" \
    "b 104" n "f\n"                     \
    "103" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[0], [x" n "],#4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[1], [x" n "],#4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[2], [x" n "],#4\n" \
    "104" n ":\n"                       \

#define STORE_C                  \
    STORE_LINE("8", "0")         \
    STORE_LINE("11", "1")        \
    STORE_LINE("14", "2")        \
    STORE_LINE("17", "3")        \
    STORE_LINE("20", "4")        \
    STORE_LINE("23", "5")        \
    STORE_LINE("26", "6")        \
    STORE_LINE("29", "7") \
    // clang-format on

        asm volatile(
                "add x1, x0, %x[LDC]\n"
                "add x2, x1, %x[LDC]\n"
                "add x3, x2, %x[LDC]\n"
                "add x4, x3, %x[LDC]\n"
                "add x5, x4, %x[LDC]\n"
                "add x6, x5, %x[LDC]\n"
                "add x7, x6, %x[LDC]\n"
)";
    if (with_bias) {
        ss << R"(
                "ld1 {v6.4s, v7.4s}, [%[bias_ptr]], #32\n"
                "dup  v8.4s,  v6.s[0]          \n"
                "dup v11.4s,  v6.s[1]          \n"
                "dup v14.4s,  v6.s[2]          \n"
                "dup v17.4s,  v6.s[3]          \n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "dup v20.4s,  v7.s[0]          \n"
                "dup v23.4s,  v7.s[1]          \n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "dup v26.4s,   v7.s[2]          \n"
                "dup v29.4s,   v7.s[3]          \n"

        )";
    } else {
        ss << R"(
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "eor v20.16b, v20.16b, v20.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "eor v23.16b, v23.16b, v23.16b\n"
                "eor v26.16b, v26.16b, v26.16b\n"
                "eor v29.16b, v29.16b, v29.16b\n"
        )";
    }
    std::string body_temp = R"(                                                
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "fmla v29.4s, v5.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "eor v6.16b, v6.16b, v6.16b\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"                
                "fmla v14.4s, v2.4s, v0.s[2]\n"                
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"                
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                ${GenAsmFloat(v8, v6)}
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                ${GenAsmFloat(v11, v6)}
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                ${GenAsmFloat(v14, v6)}
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                ${GenAsmFloat(v17, v6)}
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                ${GenAsmFloat(v20, v6)}
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                ${GenAsmFloat(v23, v6)}
                "fmla v29.4s, v5.4s, v1.s[3]\n"
                ${GenAsmFloat(v26, v6)}
                ${GenAsmFloat(v29, v6)}

                "b 6f\n"

                // odd tail
                "5:\n"
                "eor v6.16b, v6.16b, v6.16b\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                ${GenAsmFloat(v8, v6)}
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                ${GenAsmFloat(v11, v6)}
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                ${GenAsmFloat(v14, v6)}
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                ${GenAsmFloat(v17, v6)}
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                ${GenAsmFloat(v20, v6)}
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                ${GenAsmFloat(v23, v6)}
                ${GenAsmFloat(v26, v6)}
                ${GenAsmFloat(v29, v6)}

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), 
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr),
                  [n_remain] "+r"(n_remain)
                :
                : "v0", "v1", "v2", "v5", "v6", "v7", "v8", "v11", "v14", "v17", "v20",
                  "v23", "v26", "v29", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
                  "cc", "memory");


#undef STORE_LINE
#undef STORE_C
    }
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("GenAsmFloat",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenAsmFloat(args);
                         })
                    .render(body_temp);
    return ss.str();
}

std::string kern4x12(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    ss << R"(
    static void kern_4x12(const float* packA, const float* packB, int K,
                          float* output, int LDC, 
                          int m_remain, const float* bias_ptr) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = output;

// clang-format off


#define STORE_LINE(v0, v1, v2, n)                           \
    "cmp x10, #0 \n"                                        \
    "beq 105f\n"                                            \
    "st1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \
    "subs x10, x10, #1\n"


#define STORE_C                          \
    "mov x10, %x[m_remain]\n"            \
    STORE_LINE("8","9","10", "0")        \
    STORE_LINE("11","12","13", "1")      \
    STORE_LINE("14","15","16", "2")      \
    STORE_LINE("17","18","19", "3")      \
    "105:\n"
        // clang-format on

        asm volatile(
                "add x1, x0, %x[LDC]\n"
                "add x2, x1, %x[LDC]\n"
                "add x3, x2, %x[LDC]\n"
)";
    if (with_bias) {
        ss << R"(
                "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
                "dup  v8.4s,   v6.s[0]          \n"
                "dup  v9.4s,   v6.s[0]          \n"
                "dup v10.4s,  v6.s[0]          \n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "dup v11.4s,  v6.s[1]          \n"
                "dup v12.4s,  v6.s[1]          \n"
                "dup v13.4s,  v6.s[1]          \n"
                "dup v14.4s,  v6.s[2]          \n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "dup v15.4s,  v6.s[2]          \n"
                "dup v16.4s,  v6.s[2]          \n"
                "dup v17.4s,  v6.s[3]          \n"
                "dup v18.4s,  v6.s[3]          \n"
                "dup v19.4s,  v6.s[3]          \n"
            )";
    } else {
        ss << R"(
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"                
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "eor v16.16b, v16.16b, v16.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "eor v18.16b, v18.16b, v18.16b\n"
                "eor v19.16b, v19.16b, v19.16b\n"
            )";
    }

    std::string body_temp = R"(
                
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v9.4s,  v6.4s, v1.s[0]\n"
                "fmla v10.4s, v7.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "fmla v12.4s, v6.4s, v1.s[1]\n"
                "fmla v13.4s, v7.4s, v1.s[1]\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v15.4s, v6.4s, v1.s[2]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v16.4s, v7.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"
                "fmla v18.4s, v6.4s, v1.s[3]\n"
                "fmla v19.4s, v7.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "eor v2.16b, v2.16b, v2.16b\n"
                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v9.4s,  v6.4s, v1.s[0]\n"
                "fmla v10.4s, v7.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                ${GenAsmFloat(v8, v2)}
                "fmla v12.4s, v6.4s, v1.s[1]\n"
                ${GenAsmFloat(v9, v2)}
                "fmla v13.4s, v7.4s, v1.s[1]\n"
                ${GenAsmFloat(v10, v2)}
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                ${GenAsmFloat(v11, v2)}
                "fmla v15.4s, v6.4s, v1.s[2]\n"
                ${GenAsmFloat(v12, v2)}
                "fmla v16.4s, v7.4s, v1.s[2]\n"
                ${GenAsmFloat(v13, v2)}
                "fmla v17.4s, v5.4s, v1.s[3]\n"
                ${GenAsmFloat(v14, v2)}
                "fmla v18.4s, v6.4s, v1.s[3]\n"
                ${GenAsmFloat(v15, v2)}
                "fmla v19.4s, v7.4s, v1.s[3]\n"
                ${GenAsmFloat(v16, v17, v18, v19, v2)}

                "b 6f\n"

                // odd tail
                "5:\n"
                "eor v6.16b, v6.16b, v6.16b\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                ${GenAsmFloat(v8, v6)}
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                ${GenAsmFloat(v9, v6)}
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                ${GenAsmFloat(v10, v6)}
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                ${GenAsmFloat(v11, v6)}
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                ${GenAsmFloat(v12, v6)}
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                ${GenAsmFloat(v13, v6)}
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                ${GenAsmFloat(v14, v6)}
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                ${GenAsmFloat(v15, v6)}
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                ${GenAsmFloat(v16, v17, v18, v19, v6)}

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), 
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr),
                  [m_remain] "+r"(m_remain)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "x1", "x2", "x3", "x10", "cc", "memory");


#undef STORE_LINE
#undef STORE_C
    }
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("GenAsmFloat",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenAsmFloat(args);
                         })
                    .render(body_temp);
    return ss.str();
}

std::string kern4x4(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    ss << R"(
    static void kern_4x4(const float* packA, const float* packB, int K,
                         float* output, int LDC, int m_remain,
                         int n_remain, const float* bias_ptr) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = output;

// clang-format off

#define STORE_LINE(v0, n)               \
    "cmp x10, #0 \n"                    \
    "beq 105f\n"                        \
    "cmp %w[n_remain], #4\n"            \
    "blt 103" n "f\n"                   \
    "st1 {v" v0 ".4s}, [x" n " ], 16\n" \
    "b 104" n "f\n"                     \
    "103" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[0], [x" n "], 4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[1], [x" n "], 4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[2], [x" n "], 4\n" \
    "104" n ":\n"                       \
    "subs x10, x10, #1\n"


#define STORE_C                 \
    "mov x10, %x[m_remain]\n"   \
    STORE_LINE("8", "0")        \
    STORE_LINE("11", "1")       \
    STORE_LINE("14", "2")       \
    STORE_LINE("17", "3")       \
    "105:\n"
        // clang-format on

        asm volatile(
                "add x1, x0, %x[LDC]\n"
                "add x2, x1, %x[LDC]\n"
                "add x3, x2, %x[LDC]\n"
)";
    if (with_bias) {
        ss << R"(
                "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
                "dup  v8.4s,  v6.s[0]          \n"
                "dup v11.4s,  v6.s[1]          \n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "dup v14.4s,  v6.s[2]          \n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "dup v17.4s,  v6.s[3]          \n"
                )";

    } else {
        ss << R"(
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                )";
    }

    std::string body_temp = R"(
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"

                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "eor v6.16b, v6.16b, v6.16b\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"

                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"
                ${GenAsmFloat(v8, v11, v14, v17, v6)}

                "b 6f\n"

                // odd tail
                "5:\n"
                "eor v6.16b, v6.16b, v6.16b\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                ${GenAsmFloat(v8, v11, v14, v17, v6)}

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr),[K] "+r"(K),
                  [LDC] "+r"(LDC), 
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr),
                  [n_remain] "+r"(n_remain), [m_remain] "+r"(m_remain)
                :
                : "v0", "v1", "v2", "v5", "v6", "v8", "v11", "v14", "v17", "x1", "x2",
                  "x3", "x10", "cc", "memory");

#undef STORE_LINE
#undef STORE_C
    }
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("GenAsmFloat",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenAsmFloat(args);
                         })
                    .render(body_temp);
    return ss.str();
}

std::string naked_kern(const std::string& sig, TContext* ctx) {
    std::stringstream ss;
    auto postprocess_pair = gen_postprocess_inline(ctx);
    ss << postprocess_pair.first;
    ss << sig;
    ss << R"({
    size_t m = 0;
    const int K12 = K * 12;
    const int K8 = K * 8;
    const int K4 = K * 4;
    const size_t A_INTERLEAVE = 8;
    const size_t A_INTERLEAVE4 = 4;
    const size_t B_INTERLEAVE = 12;
    for(;m + A_INTERLEAVE <= M;m += A_INTERLEAVE){
        float* output = C + (m * LDC);
        size_t n = 0;
        const float* cur_pack_b = pack_b;
        for(;n + B_INTERLEAVE <= N;n += B_INTERLEAVE){
            kern_8x12(pack_a, cur_pack_b, K, output, LDC, bias_ptr);
            output += B_INTERLEAVE;
            cur_pack_b += K12;
        }

        for (; n < N; n += 4) {
            kern_8x4(pack_a, cur_pack_b, K, output, LDC, 
                                 min(N - n, 4), bias_ptr);
            output += 4;
            cur_pack_b += K4;
        }
        pack_a += K8;
        bias_ptr += A_INTERLEAVE;
    }
    for (; m < M; m += A_INTERLEAVE4) {
        float* output = C + (m * LDC);
        size_t n = 0;
        const float* cur_pack_b = pack_b;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(pack_a, cur_pack_b, K, output, LDC, 
                                  min(M - m, 4), bias_ptr);
            output += B_INTERLEAVE;
            cur_pack_b += K12;
        }

        for (; n < N; n += 4) {
            kern_4x4(pack_a, cur_pack_b, K, output, LDC, 
                                 min(M - m, 4),
                                 min(N - n, 4), bias_ptr);
            output += 4;
            cur_pack_b += K4;
        }
        pack_a += K4;
        bias_ptr += A_INTERLEAVE4;
    }
 )";
    ss << postprocess_pair.second;
    ss << "\n}";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_m = 8;
        int k = kmax - k0;
        int m = ymax - y0;
        int round_m = (m + packed_m - 1) / packed_m * packed_m;
        size_t res = (size_t)k * round_m * sizeof(float);
        return res;
    })";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_n = 12;
        const size_t packed_hw = (xmax - x0 + packed_n - 1) / packed_n * packed_n;
        size_t res = (size_t)(kmax - k0) * packed_hw * sizeof(float);
        return res;
    })";
    return ss.str();
}

}  // namespace

std::vector<KernelObj> MatmulM8N12Kernel::GetDependInternalSymbol(
        TContext* ctx) const {
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    std::vector<KernelObj> depends;
    if (nonline_mode == "SIGMOID") {
        ExpNeonKernel kern;
        depends.emplace_back(kern.GetKernelSymbol(ctx), kern.GetKernelBody(ctx),
                             kern.GetBodyGuardBegin(ctx),
                             kern.GetBodyGuardEnd(ctx));
    }
    return depends;
}

std::string MatmulM8N12Kernel::GetKernelSymbol(TContext* ctx) const {
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string bias_suffix = with_bias ? "_bias" : "";
    std::string act_suffix = "";
    if (ctx->haveAttr("nonlineMode") &&
        ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        act_suffix = "_" + ctx->getAttrStr("nonlineMode");
    }
    return "Arm64_fp32_m8_n12_matmul" + bias_suffix + act_suffix;
}

std::string MatmulM8N12Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto kern_sym = GetKernelSymbol(ctx);
    writer << "#include <arm_neon.h>\n";
    writer << "#include <string.h>\n";
    writer << prefetch();
    writer << transpose_8x4_1_s();
    writer << utilsFunc();
    writer << interleave();
    writer << transpose_4x4_1_s();
    writer << pack_A_n(kern_sym, ctx);
    writer << pack_A_t(kern_sym, ctx);
    writer << pack_B_n(kern_sym, ctx);
    writer << pack_B_t(kern_sym, ctx);
    writer << kern8x12(ctx);
    writer << kern8x4(ctx);
    writer << kern4x12(ctx);
    writer << kern4x4(ctx);
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << naked_kern(GetNakedKernelSignature(ctx), ctx);
    return writer.str();
}

std::string MatmulM8N12Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulM8N12Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
