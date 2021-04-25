/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/InternalKernel/Fp32M4N8K4Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Arm/Armv7/Activation.h"
#include "InternalKernel.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
using namespace ArmCommon;
namespace {

std::string GetKern4x1() {
    std::stringstream writer;
    writer << R"(
void kern_4x1(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB = (LDB - 4) * sizeof(float);
    asm volatile(
            "subs %[K], %[K], #4\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"
            "vld1.32 {d12-d15}, [%[A]]!\n"
            "veor    q8,     q8 \n"
            "veor    q9,     q9 \n"
            "veor    q10,    q10 \n"
            "veor    q11,    q11 \n"

            "vld1.32 {d0-d1}, [%[B]]!\n"

            "vmla.f32 q8, q4, d0[0]\n"
            "vmla.f32 q9, q5, d0[1]\n"

            "beq 2f\n"

            "1:\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"
            "vmla.f32 q10, q6, d1[0]\n"
            "vmla.f32 q11, q7, d1[1]\n"

            "add %[B], %[B], %[LDB]\n"
            "vld1.32 {d0-d1}, [%[B]]!\n"
            "vld1.32 {d12-d15}, [%[A]]!\n"

            "vmla.f32 q8, q4, d0[0]\n"
            "vmla.f32 q9, q5, d0[1]\n"

            "subs %[K], %[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "vmla.f32 q10, q6, d1[0]\n"
            "vmla.f32 q11, q7, d1[1]\n"
            "vadd.f32 q8,  q8, q10\n"
            "vadd.f32 q9,  q9, q11\n"
            "vadd.f32 q8,  q8, q9\n"

            "vst1.32 {d16, d17}, [%[C]]!\n"

            : [A] "+r"(A), [B] "+r"(B), [K] "+r"(K), [C] "+r"(C)
            : [LDB] "r"(LDB)
            : "d0", "d1", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
              "d17", "d18", "d19", "d20", "d21", "d22", "d23", "cc", "memory");
})";
    return writer.str();
}
std::string GetKern4x4() {
    std::stringstream writer;
    writer << R"(
void kern_4x4(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    //! as each load 16 number from B, and pos add 16 * 4, we should minus it
    //! before we add stride
    LDB = (LDB - 16) * sizeof(float);
    asm volatile(
            "subs %[K], %[K], #4\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"
            "vld1.32 {d12-d15}, [%[A]]!\n"

            "vld1.32 {d0-d3}, [%[B]]!\n"
            "vld1.32 {d4-d7}, [%[B]]!\n"

            "vmul.f32 q8, q4, d0[0]\n"
            "vmul.f32 q9, q4, d2[0]\n"
            "vmul.f32 q10, q4, d4[0]\n"
            "vmul.f32 q11, q4, d6[0]\n"

            "vmla.f32 q8, q5, d0[1]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmla.f32 q11, q5, d6[1]\n"

            "beq 2f\n"

            "1:\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"

            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q6, d7[0]\n"

            "add %[B], %[B], %[LDB]\n"

            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vld1.32 {d0-d1}, [%[B]]!\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vld1.32 {d2-d3}, [%[B]]!\n"
            "vmla.f32 q11, q7, d7[1]\n"
            "vld1.32 {d4-d5}, [%[B]]!\n"

            "vmla.f32 q8, q4, d0[0]\n"
            "vld1.32 {d6-d7}, [%[B]]!\n"
            "vmla.f32 q9, q4, d2[0]\n"
            "vmla.f32 q10, q4, d4[0]\n"
            "vmla.f32 q11, q4, d6[0]\n"

            "vld1.32 {d12-d15}, [%[A]]!\n"

            "vmla.f32 q8, q5, d0[1]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmla.f32 q11, q5, d6[1]\n"

            "subs %[K], %[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q6, d7[0]\n"

            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vmla.f32 q11, q7, d7[1]\n"

            "vst1.32 {d16, d17, d18, d19}, [%[C]]!\n"
            "vst1.32 {d20, d21, d22, d23}, [%[C]]!\n"

            : [A] "+r"(A), [B] "+r"(B), [K] "+r"(K), [C] "+r"(C)
            : [LDB] "r"(LDB)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "cc", "memory");
})";
    return writer.str();
}
std::string GetKern4x8() {
    std::stringstream writer;
    writer << R"(
void kern_4x8(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB *= sizeof(float);
    //! as each load 32 number from B, the pos add 32 * 4, we should minus it
    //! before we add stride
    LDB -= 32 * sizeof(float);
    asm volatile(
            "vld1.32 {d8, d9, d10, d11}, [%[A]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[A]]!\n"

            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmul.f32 q8, q4, d0[0]\n"
            "vmla.f32 q8, q5, d0[1]\n"
            "vmul.f32 q9, q4, d2[0]\n"
            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vmul.f32 q10, q4, d4[0]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmul.f32 q11, q4, d6[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q5, d6[1]\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vmla.f32 q11, q6, d7[0]\n"
            "vmla.f32 q11, q7, d7[1]\n"

            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmul.f32 q12, q4, d0[0]\n"
            "vmla.f32 q12, q5, d0[1]\n"
            "vmul.f32 q13, q4, d2[0]\n"
            "vmla.f32 q12, q6, d1[0]\n"
            "vmla.f32 q13, q5, d2[1]\n"
            "vmla.f32 q12, q7, d1[1]\n"
            "vmla.f32 q13, q6, d3[0]\n"
            "vmla.f32 q13, q7, d3[1]\n"
            "vmul.f32 q14, q4, d4[0]\n"
            "vmla.f32 q14, q5, d4[1]\n"
            "vmul.f32 q15, q4, d6[0]\n"
            "vmla.f32 q14, q6, d5[0]\n"
            "vmla.f32 q15, q5, d6[1]\n"
            "vmla.f32 q14, q7, d5[1]\n"
            "vmla.f32 q15, q6, d7[0]\n"
            "vmla.f32 q15, q7, d7[1]\n"

            "add %[B], %[B], %[LDB]\n"
            "subs %[K], %[K], #4\n"
            "cmp %[K], #0\n"
            "beq 2f\n"

            "1:\n"
            "vld1.32 {d8, d9, d10, d11}, [%[A]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[A]]!\n"

            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmla.f32 q8, q4, d0[0]\n"
            "vmla.f32 q8, q5, d0[1]\n"
            "vmla.f32 q9, q4, d2[0]\n"
            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vmla.f32 q10, q4, d4[0]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmla.f32 q11, q4, d6[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q5, d6[1]\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vmla.f32 q11, q6, d7[0]\n"
            "vmla.f32 q11, q7, d7[1]\n"

            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmla.f32 q12, q4, d0[0]\n"
            "vmla.f32 q12, q5, d0[1]\n"
            "vmla.f32 q13, q4, d2[0]\n"
            "vmla.f32 q12, q6, d1[0]\n"
            "vmla.f32 q13, q5, d2[1]\n"
            "vmla.f32 q12, q7, d1[1]\n"
            "vmla.f32 q13, q6, d3[0]\n"
            "vmla.f32 q13, q7, d3[1]\n"
            "vmla.f32 q14, q4, d4[0]\n"
            "vmla.f32 q14, q5, d4[1]\n"
            "vmla.f32 q15, q4, d6[0]\n"
            "vmla.f32 q14, q6, d5[0]\n"
            "vmla.f32 q15, q5, d6[1]\n"
            "vmla.f32 q14, q7, d5[1]\n"
            "vmla.f32 q15, q6, d7[0]\n"
            "vmla.f32 q15, q7, d7[1]\n"

            "add %[B], %[B], %[LDB]\n"
            "subs %[K], %[K], #4\n"
            "cmp %[K], #0\n"
            "bne 1b\n"
            "2:\n"
            "vst1.32 {d16, d17, d18, d19}, [%[C]]!\n"
            "vst1.32 {d20, d21, d22, d23}, [%[C]]!\n"
            "vst1.32 {d24, d25, d26, d27}, [%[C]]!\n"
            "vst1.32 {d28, d29, d30, d31}, [%[C]]!\n"
            : [A] "+r"(A), [B] "+r"(B), [K] "+r"(K), [C] "+r"(C)
            : [LDB] "r"(LDB)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
              "cc", "memory");
})";
    return writer.str();
}

}  // namespace

std::string MatmulM4N8MK4Kernel::GetKernelSymbol(TContext*) const {
    return "Armv7_fp32_m4_n8_k4_matmul";
}

std::string MatmulM4N8MK4Kernel::GetKernelSignature(TContext* ctx) const {
    std::stringstream writer;
    writer << "void " << GetKernelSymbol(ctx) << R"((const float* A, size_t LDA,
                            const float* B, size_t LDB, float* C,
                            size_t LDC, size_t M, size_t N, size_t K))";
    return writer.str();
}

std::string MatmulM4N8MK4Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include <arm_neon.h>\n";
    writer << "#include \"stddef.h\"\n";

    writer << GetKern4x1();
    writer << "\n\n";
    writer << GetKern4x4();
    writer << "\n\n";
    writer << GetKern4x8();
    writer << "\n\n";

    writer << GetKernelSignature(ctx);
    writer << "{\n";
    writer << R"(
    const int MB=4;
    const int KB=4;
    const int NB=8;
    const int NB_HALF=4;
    //! (m/4, k/4, 4, 4) * (k/4, n, 4) = (m/4, n, 4)
    for (size_t m = 0; m < M; m += MB) {
        float* output = C + (m / MB) * LDC;
        const float* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_4x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_4x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB_HALF;
            output += MB * NB_HALF;
            n += 4;
        }
        while (n < N) {
            kern_4x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    })";
    writer << "\n}";
    return writer.str();
}

// vim: syntax=cpp.doxygen
