/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/InternalKernel/Fp32M8N12Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string>
#include "../../../Utils/StringTemplate.h"
#include "Arm/ArmCommon/ElemwiseHelper/ElemwiseHelper.h"
#include "Arm/ArmCommon/MatmulCommon.h"
#include "Arm/ArmCommon/common_asm_utils.h"
#include "Arm/Armv7/Activation.h"
#include "Arm/Armv7/InternalKernel/InternalKernel.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
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
        #define ASM_PREFETCH(address) "PLD " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_2x_f32() +
           KernelGen::ArmCommon::gen_common_prefetch_3x_f32();
}

std::string transpose_4x4_1_s(void) {
    return R"(
static inline void transpose_4x4_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr, int stride) {

    stride -= 8;
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"  // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"  // D0D1D2D3
            "vtrn.32 q0, q1\n"                   // A0B0A2B2 A1B1A3B3
            "vtrn.32 q2, q3\n"                   // C0D0C2D2 C1D1C3D3
            "vst1.32 {d0},  [%[outptr]]!\n"
            "vst1.32 {d4},  [%[outptr]], %[stride]\n"
            "vst1.32 {d2},  [%[outptr]]!\n"
            "vst1.32 {d6},  [%[outptr]], %[stride]\n"
            "vst1.32 {d1},  [%[outptr]]!\n"
            "vst1.32 {d5},  [%[outptr]], %[stride]\n"
            "vst1.32 {d3},  [%[outptr]]!\n"
            "vst1.32 {d7},  [%[outptr]], %[stride]\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
              [inptr3] "+r"(inptr3), [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "memory");
}
    )";
}

std::string interleave(void) {
    return KernelGen::ArmCommon::gen_common_interleve_f32() + R"(

static inline void interleave_4x12_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr) {
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"    // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr0]]!\n"    // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr0]]!\n"    // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr1]]!\n"    // A0A1A2A3
            "vld1.32 {d8, d9},  [%[inptr1]]!\n"    // B0B1B2B3
            "vld1.32 {d10, d11},  [%[inptr1]]!\n"  // C0C1C2C3
            "vld1.32 {d12, d13},  [%[inptr2]]!\n"  // A0A1A2A3
            "vld1.32 {d14, d15},  [%[inptr2]]!\n"  // B0B1B2B3
            "vld1.32 {d16, d17},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.32 {d18, d19},  [%[inptr3]]!\n"  // A0A1A2A3
            "vld1.32 {d20, d21}, [%[inptr3]]!\n"   // B0B1B2B3
            "vld1.32 {d22, d23}, [%[inptr3]]!\n"   // C0C1C2C3

            "vst1.32 {d0, d1},   [%[outptr]]!\n"   // A0B0C0D0
            "vst1.32 {d2, d3},   [%[outptr]]!\n"   // E0F0G0H0
            "vst1.32 {d4, d5},   [%[outptr]]!\n"   // I0J0K0L0
            "vst1.32 {d6, d7},   [%[outptr]]!\n"   // D0D1D2D3
            "vst1.32 {d8, d9},   [%[outptr]]!\n"   // E0E1E2E3
            "vst1.32 {d10, d11}, [%[outptr]]!\n"   // F0F1F2F3
            "vst1.32 {d12, d13}, [%[outptr]]!\n"   // G0G1G2G3
            "vst1.32 {d14, d15}, [%[outptr]]!\n"   // H0H1H2H3
            "vst1.32 {d16, d17}, [%[outptr]]!\n"   // H0H1H2H3
            "vst1.32 {d18, d19}, [%[outptr]]!\n"   // G0G1G2G3
            "vst1.32 {d20, d21},  [%[outptr]]!\n"  // H0H1H2H3
            "vst1.32 {d22, d23},  [%[outptr]]!\n"  // H0H1H2H3
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
              [inptr3] "+r"(inptr3), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "memory");
}

static inline void interleave_1x12_1_s(const float* inptr0, float* outptr) {
    asm volatile(
            "vld1.32 {d0, d1}, [%[inptr0]]!\n"
            "vld1.32 {d2, d3}, [%[inptr0]]!\n"
            "vld1.32 {d4, d5}, [%[inptr0]]!\n"
            "vst1.32 {d0, d1}, [%[outptr]]!\n"
            "vst1.32 {d2, d3}, [%[outptr]]!\n"
            "vst1.32 {d4, d5}, [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "memory");
}

static inline void interleave_4x4_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr) {
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"  // A0A1A2A3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"  // A0A1A2A3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"  // A0A1A2A3

            "vst1.32 {d0, d1},   [%[outptr]]!\n"  // A0B0C0D0
            "vst1.32 {d2, d3},   [%[outptr]]!\n"  // E0F0G0H0
            "vst1.32 {d4, d5},   [%[outptr]]!\n"  // I0J0K0L0
            "vst1.32 {d6, d7},   [%[outptr]]!\n"  // D0D1D2D3
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
              [inptr3] "+r"(inptr3), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "memory");
}

static inline void interleave_1x4_1_s(const float* inptr0, float* outptr) {
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"
            "vst1.32 {d0, d1},  [%[outptr]]\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "memory");
}
)";
}

std::string pack_A_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_n" +
           ArmCommon::MatmulInternal::GenPackACall(ctx) +
           R"({
        float zerobuff[4];
        memset(zerobuff, 0, sizeof(float) * 4);
        int y = y0;
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
                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr, 16);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                outptr+=16;

            }

            if (K > 0) {
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
                interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K, 0);
                outptr+=4*K;
            }
        }
    }
    )";
}

std::string pack_A_t(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_t" +
           ArmCommon::MatmulInternal::GenPackACall(ctx) +
           R"( {
        int x0 = y0;
        int xmax = ymax;
        int ksize = kmax - k0;
        int ksize4 = (ksize << 2);
        float* outptr_base = outptr;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* inptr0 = inptr + k * ldin + x0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_3x(inptr0);
            prefetch_3x(inptr1);
            prefetch_3x(inptr2);
            prefetch_3x(inptr3);

            int x = x0;
            float* access_outptr = outptr_base;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = access_outptr;
                interleave_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                access_outptr += ksize4;
            }

            if (x < xmax) {
                interleave_4(inptr0, inptr1, inptr2, inptr3, access_outptr, 4, xmax - x, 0);
            }

            outptr_base += 4 * 4;
        }

        for (; k < kmax; k++) {
            const float* inptr0 = inptr + k * ldin + x0;
            prefetch_3x(inptr0);
            int x = x0;
            float* access_outptr = outptr_base;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = access_outptr;
                interleave_1x4_1_s(inptr0, outptr_interleave);
                inptr0 += 4;
                access_outptr += ksize4;
            }

            if (x < xmax) {
                interleave_1(inptr0, access_outptr, 4, xmax - x, 0);
                inptr0 += 4;
            }

            outptr_base += 4;
        }
    }
    )";
}

std::string pack_B_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_n" +
           ArmCommon::MatmulInternal::GenPackBCall(ctx) +
           R"( {
        int ksize = kmax - k0;
        int ksize12 = ksize * 12;
        int ksize4 = (ksize << 2);
        float* outptr_base = outptr;
        float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* inptr0 = inptr + k * ldin + x0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_3x(inptr0);
            prefetch_3x(inptr1);
            prefetch_3x(inptr2);
            prefetch_3x(inptr3);

            int x = x0;
            float* access_outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                float* outptr_interleave = access_outptr;
                interleave_4x12_1_s(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
                inptr0 += 12;inptr1 += 12;inptr2 += 12;inptr3 += 12;
                access_outptr += ksize12;
            }
            access_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = access_outptr;
                interleave_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                access_outptr += ksize4;
            }

            if (x < xmax) {
                interleave_4(inptr0, inptr1, inptr2, inptr3, access_outptr, 4, xmax - x, 0);
            }

            outptr_base += 12 * 4;
            outptr_base4 += 4 * 4;
        }

        for (; k < kmax; k++) {
            const float* inptr0 = inptr + k * ldin + x0;
            prefetch_3x(inptr0);
            int x = x0;
            float* access_outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                float* outptr_interleave = access_outptr;
                interleave_1x12_1_s(inptr0, outptr_interleave);
                inptr0 += 12;
                access_outptr += ksize12;
            }
            access_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = access_outptr;
                interleave_1x4_1_s(inptr0, outptr_interleave);
                inptr0 += 4;
                access_outptr += ksize4;
            }

            if (x < xmax) {
                interleave_1(inptr0, access_outptr, 4, xmax - x, 0);
            }

            outptr_base += 12;
            outptr_base4 += 4;
        }
    }
    )";
}

std::string pack_B_t(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_t" +
           ArmCommon::MatmulInternal::GenPackBCall(ctx) +
           R"(
     {
        int y0 = x0;
        int ymax = xmax;
        float* access_outptr = outptr;
        float zerobuff[4];
        memset(zerobuff, 0, sizeof(float) * 4);
        int K12 = 12 * (kmax - k0);

        int y = y0;

        for (; y + 12 <= ymax; y += 12) {
            int yi = y;
            for (; yi < y + 12; yi += 4) {
                const float* inptr0 = inptr + yi * ldin + k0;
                const float* inptr1 = inptr0 + ldin;
                const float* inptr2 = inptr1 + ldin;
                const float* inptr3 = inptr2 + ldin;
                float* outptr_inner = access_outptr + yi - y;

                prefetch_2x(inptr0);
                prefetch_2x(inptr1);
                prefetch_2x(inptr2);
                prefetch_2x(inptr3);

                int x = (kmax - k0);
                for (; x > 3; x -= 4) {
                    transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner, 48);
                    inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                    outptr_inner+=48;
                }
                for (; x > 0; x--) {
                    *outptr_inner++ = *inptr0++;
                    *outptr_inner++ = *inptr1++;
                    *outptr_inner++ = *inptr2++;
                    *outptr_inner++ = *inptr3++;
                    outptr_inner += 8;
                }
            }
            access_outptr += K12;
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

                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, access_outptr, 16);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                access_outptr+=16;
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
                interleave_4(inptr0, inptr1, inptr2, inptr3, access_outptr, 1, x, 0);
                inptr0 += x;inptr1 += x;inptr2 += x;inptr3 += x;
                access_outptr+=4*x;
            }
        }
    }
    )";
}

std::string kern4x12(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
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
    register float* outptr asm("r0") =(float*)(output);



#define STORE_LINE(d0, d1, d2, d3, d4, d5, n)            \
    "cmp r10, #0\n"                                      \
    "beq 101f\n"                                         \
    "mov r9, r" n "\n"                                   \
    "vst1.32 {d" d0 ",d" d1 ",d" d2 ",d" d3 "}, [r9]!\n" \
    "vst1.32 {d" d4 ",d" d5 "}, [r9]\n"                  \
    "subs r10, r10, #1\n"

#define STORE_C                                         \
    "mov r10, %[m_remain]\n"                            \
    STORE_LINE("8", "9", "10", "11", "12", "13", "0")   \
    STORE_LINE("14", "15", "16", "17", "18", "19", "1") \
    STORE_LINE("20", "21", "22", "23", "24", "25", "2") \
    STORE_LINE("26", "27", "28", "29", "30", "31", "3") \
    "101:\n"

    asm volatile(
            "add r1, r0, %[LDC]\n"
            "add r2, r1, %[LDC]\n"
            "add r3, r2, %[LDC]\n"

)";
    if (with_bias) {
        ss << R"(
            "vld1.32 {d6, d7}, [%[bias_ptr]]!\n"
            "vdup.f32 q4, d6[0]\n"
            "vdup.f32 q5, d6[0]\n"
            "vdup.f32 q6, d6[0]\n"
            "vld1.32 {d2, d3}, [%[b_ptr]]!\n"
            "vdup.f32 q7, d6[1]\n"
            "vdup.f32 q8, d6[1]\n"
            "vdup.f32 q9, d6[1]\n"
            "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
            "vdup.f32 q10, d7[0]\n"
            "vdup.f32 q11, d7[0]\n"
            "vdup.f32 q12, d7[0]\n"
            "vdup.f32 q13, d7[1]\n"
            "vdup.f32 q14, d7[1]\n"
            "vdup.f32 q15, d7[1]\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            )";
    } else {
        ss << R"(
                "vld1.32 {d2, d3}, [%[b_ptr]]!\n"
                "veor.32 q4, q4, q4\n"
                "veor.32 q5, q5, q5\n"
                "veor.32 q6, q6, q6\n"
                "veor.32 q7, q7, q7\n"
                "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
                "veor.32 q8, q8, q8\n"
                "veor.32 q9, q9, q9\n"
                "veor.32 q10, q10, q10\n"
                "veor.32 q11, q11, q11\n"
                "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
                "veor.32 q12, q12, q12\n"
                "veor.32 q13, q13, q13\n"
                "veor.32 q14, q14, q14\n"
                "veor.32 q15, q15, q15\n"
            )";
    }

    std::string body_temp = R"(
                
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vld1.32 {d2, d3}, [%[b_ptr]]!\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q15, q3, d1[1]\n"
            

            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vld1.32 {d2, d3}, [%[b_ptr]]!\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q15, q3, d1[1]\n"
            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]]!\n"
            "subs %[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vld1.32 {d2, d3}, [%[b_ptr]]!\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q15, q3, d1[1]\n"

            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "veor.32 q1, q1, q1\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"

            ${GenAsmFloat(q4, q1)}
            "vmla.f32 q5, q2, d0[0]\n"
            ${GenAsmFloat(q5, q1)}
            "vmla.f32 q6, q3, d0[0]\n"
            ${GenAsmFloat(q6, q1)}
            
            ${GenAsmFloat(q7, q1)}
            "vmla.f32 q8, q2, d0[1]\n"
            ${GenAsmFloat(q8, q1)}
            "vmla.f32 q9, q3, d0[1]\n"
            ${GenAsmFloat(q9, q1)}

            ${GenAsmFloat(q10, q1)}
            "vmla.f32 q11, q2, d1[0]\n"
            ${GenAsmFloat(q11, q1)}
            "vmla.f32 q12, q3, d1[0]\n"
            ${GenAsmFloat(q12, q1)}

            ${GenAsmFloat(q13, q1)}
            "vmla.f32 q14, q2, d1[1]\n"
            ${GenAsmFloat(q14, q1)}
            "vmla.f32 q15, q3, d1[1]\n"
            ${GenAsmFloat(q15, q1)}

            "b 6f\n"

            // odd tail
            "5:\n"
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "veor.32 q1, q1, q1\n"

            ${GenAsmFloat(q4, q1)}
            "vmla.f32 q5, q2, d0[0]\n"
            ${GenAsmFloat(q5, q1)}
            "vmla.f32 q6, q3, d0[0]\n"
            ${GenAsmFloat(q6, q1)}
            
            ${GenAsmFloat(q7, q1)}
            "vmla.f32 q8, q2, d0[1]\n"
            ${GenAsmFloat(q8, q1)}
            "vmla.f32 q9, q3, d0[1]\n"
            ${GenAsmFloat(q9, q1)}

            ${GenAsmFloat(q10, q1)}
            "vmla.f32 q11, q2, d1[0]\n"
            ${GenAsmFloat(q11, q1)}
            "vmla.f32 q12, q3, d1[0]\n"
            ${GenAsmFloat(q12, q1)}

            ${GenAsmFloat(q13, q1)}
            "vmla.f32 q14, q2, d1[1]\n"
            ${GenAsmFloat(q14, q1)}
            "vmla.f32 q15, q3, d1[1]\n"
            ${GenAsmFloat(q15, q1)}

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr), [K] "+r"(K), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [m_remain] "+r"(m_remain), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
              "r1", "r2", "r3", "r9", "r10", "cc", "memory");

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
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
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
    register float* outptr asm("r0") = output;

#define STORE_LINE(d0, d1, n)                  \
    "cmp r10, #0 \n"                           \
    "beq 105f\n"                               \
    "cmp %[n_remain], #4\n"                    \
    "blt 103" n "f\n"                          \
    "vst1.32 {d" d0 ", d" d1 "}, [r" n " ]!\n" \
    "b 104" n "f\n"                            \
    "103" n ":\n"                              \
    "cmp %[n_remain], #0\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" d0 "[0]}, [r" n "]!\n"        \
    "cmp %[n_remain], #1\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" d0 "[1]}, [r" n "]!\n"        \
    "cmp %[n_remain], #2\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" d1 "[0]}, [r" n "]!\n"        \
    "104" n ":\n"                              \
    "subs r10, r10, #1\n"


#define STORE_C                 \
    "mov r10, %[m_remain]\n"    \
    STORE_LINE("8", "9", "0")   \
    STORE_LINE("10", "11", "1") \
    STORE_LINE("12", "13", "2") \
    STORE_LINE("14", "15", "3") \
    "105:\n"

    asm volatile(
            "add r1, r0, %[LDC]\n"
            "add r2, r1, %[LDC]\n"
            "add r3, r2, %[LDC]\n"
)";
    if (with_bias) {
        ss << R"(
                "vld1.32 {d6, d7}, [%[bias_ptr]]!\n"
                "vdup.f32 q4, d6[0]\n"
                "vdup.f32 q5, d6[1]\n"
                "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
                "vdup.f32 q6, d7[0]\n"
                "vdup.f32 q7, d7[1]\n"
                "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
                )";

    } else {
        ss << R"(
                "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
                "veor.32 q4, q4, q4\n"
                "veor.32 q5, q5, q5\n"
                "veor.32 q6, q6, q6\n"
                "veor.32 q7, q7, q7\n"
                "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
                )";
    }

    std::string body_temp = R"(
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vld1.32 {d2, d3}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q2, d0[0]\n"
            "vmla.f32 q5, q2, d0[1]\n"
            "vmla.f32 q6, q2, d1[0]\n"
            "vmla.f32 q7, q2, d1[1]\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"

            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q3, d2[0]\n"
            "vmla.f32 q5, q3, d2[1]\n"
            "vmla.f32 q6, q3, d3[0]\n"
            "vmla.f32 q7, q3, d3[1]\n"
            "vld1.32 {d4, d5}, [%[b_ptr]]!\n"

            "subs %[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            
            "vld1.32 {d2, d3}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q2, d0[0]\n"
            "vmla.f32 q5, q2, d0[1]\n"
            "vmla.f32 q6, q2, d1[0]\n"
            "vmla.f32 q7, q2, d1[1]\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "veor.32 q2, q2, q2\n"

            "vmla.f32 q4, q3, d2[0]\n"
            "vmla.f32 q5, q3, d2[1]\n"
            "vmla.f32 q6, q3, d3[0]\n"
            "vmla.f32 q7, q3, d3[1]\n"
            ${GenAsmFloat(q4, q5, q6, q7, q2)}

            "b 6f\n"

            // odd tail
            "5:\n"
            "veor.32 q3, q3, q3\n"
            "vmla.f32 q4, q2, d0[0]\n"
            "vmla.f32 q5, q2, d0[1]\n"
            "vmla.f32 q6, q2, d1[0]\n"
            "vmla.f32 q7, q2, d1[1]\n"
            ${GenAsmFloat(q4, q5, q6, q7, q3)}

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr), [K] "+r"(K), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [m_remain] "+r"(m_remain), [n_remain] "+r"(n_remain),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "r1", "r2", "r3", "r10", "cc", "memory");

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
    std::string post_process_str;
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    if (nonline_mode == "SIGMOID") {
        std::vector<CCOperand> operands;
        operands.resize(2);
        auto ElemwiseImpl =
                std::make_shared<ElemwiseGenUnarySigmoid>("f32", "f32", true);
        auto ImpleGen = [=](std::vector<std::string> strs) {
            return ElemwiseImpl->GenCodeBody(strs);
        };
        std::string post_process_temp = R"(
            if (LDC == N){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                for(int m_idx = 0; m_idx < M; ++m_idx){
                    ${ElemwiseImplName}(C + m_idx * LDC, C + m_idx * LDC, N);
                }
            }
        )";
        post_process_str =
                StringTemplate::StringTemplateArgs()
                        .add("ElemwiseImplName", ElemwiseImpl->GenInlineName())
                        .render(post_process_temp);
        auto InternalKernelFunc = ExpNeonKernel();
        ss << "extern " << InternalKernelFunc.GetKernelSignature(ctx) << ";\n";
        ss << ElemwiseImpl->GenCodeBody({});
    }
    ss << sig;
    ss << R"({
    size_t m = 0;
    const int K12 = K * 12;
    const int K4 = K * 4;
    const size_t A_INTERLEAVE = 4;
    const size_t B_INTERLEAVE = 12;

    for(;m <= M;m += A_INTERLEAVE){
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
        bias_ptr += A_INTERLEAVE;
    }
 )";
    ss << post_process_str;
    ss << "\n}";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_m = 4;
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

std::vector<KernelObj> MatmulM4N12Kernel::GetDependInternalSymbol(TContext* ctx) const {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::vector<KernelObj> depends;
    if (nonline_mode == "SIGMOID") {
        ExpNeonKernel kern;
        depends.emplace_back(
                kern.GetKernelSymbol(ctx), kern.GetKernelBody(ctx),
                kern.GetBodyGuardBegin(ctx), kern.GetBodyGuardEnd(ctx));
    }
    return depends;
}

std::string MatmulM4N12Kernel::GetKernelSymbol(TContext* ctx) const {
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string bias_suffix = with_bias ? "_bias" : "";
    std::string act_suffix = "";
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        act_suffix = "_" + ctx->getAttrStr("nonlineMode");
    }
    return "Armv7_fp32_m4_n12_matmul" + bias_suffix + act_suffix;
}

std::string MatmulM4N12Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto kern_sym = GetKernelSymbol(ctx);
    writer << "#include <arm_neon.h>\n";
    writer << "#include <string.h>\n";
    writer << "#include <math.h>\n";
    writer << prefetch();
    writer << utilsFunc();
    writer << interleave();
    writer << transpose_4x4_1_s();
    writer << pack_A_n(kern_sym, ctx);
    writer << pack_A_t(kern_sym, ctx);
    writer << pack_B_n(kern_sym, ctx);
    writer << pack_B_t(kern_sym, ctx);
    writer << kern4x12(ctx);
    writer << kern4x4(ctx);
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << naked_kern(GetNakedKernelSignature(ctx), ctx);
    return writer.str();
}

std::string MatmulM4N12Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string MatmulM4N12Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
