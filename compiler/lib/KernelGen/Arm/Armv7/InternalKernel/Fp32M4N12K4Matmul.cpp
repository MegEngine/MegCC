/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/InternalKernel/Fp32M8N12K4Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

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
std::string prefetch(void) {
    return R"(
        #define ASM_PREFETCH(address) "PLD " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_3x_f32();
}
std::string transpose_1x12_4_s() {
    return R"(
static inline void transpose_1x12_4_s(const float* inptr0, float* outptr) {
    asm volatile(
            "vld4.32 {d0-d3},  [%[inptr0]]!\n"
            "vld4.32 {d4-d7},  [%[inptr0]]!\n"
            "vld4.32 {d8-d11},  [%[inptr0]]!\n"
            "vld4.32 {d12-d15},  [%[inptr0]]!\n"
            "vld4.32 {d16-d19},  [%[inptr0]]!\n"
            "vld4.32 {d20-d23},  [%[inptr0]]!\n"
            "vswp d1, d4\n"
            "vswp d3, d6\n"
            "vswp d9, d12\n"
            "vswp d11, d14\n"
            "vswp d17, d20\n"
            "vswp d19, d22\n"

            "vst1.32 {d0-d1}, [%[outptr]]! \n"
            "vst1.32 {d8-d9}, [%[outptr]]! \n"
            "vst1.32 {d16-d17}, [%[outptr]]! \n"
            "vst1.32 {d4-d5}, [%[outptr]]! \n"
            "vst1.32 {d12-d13}, [%[outptr]]! \n"
            "vst1.32 {d20-d21}, [%[outptr]]! \n"
            "vst1.32 {d2-d3}, [%[outptr]]! \n"
            "vst1.32 {d10-d11}, [%[outptr]]! \n"
            "vst1.32 {d18-d19}, [%[outptr]]! \n"
            "vst1.32 {d6-d7}, [%[outptr]]! \n"
            "vst1.32 {d14-d15}, [%[outptr]]! \n"
            "vst1.32 {d22-d23}, [%[outptr]]! \n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
              "memory");
}
)";
}

std::string transpose_1x4_4_s() {
    return R"(
static inline void transpose_1x4_4_s(const float* inptr0, float* outptr) {
    asm volatile(
            "vld4.32 {d0-d3},  [%[inptr0]]!\n"
            "vld4.32 {d4-d7},  [%[inptr0]]!\n"
            "vswp d1, d4\n"
            "vswp d3, d6\n"
            "vst1.32 {d0-d1}, [%[outptr]]! \n"
            "vst1.32 {d4-d5}, [%[outptr]]! \n"
            "vst1.32 {d2-d3}, [%[outptr]]! \n"
            "vst1.32 {d6-d7}, [%[outptr]]! \n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "memory");
}
)";
}

static std::string kern_4x12(TContext* ctx) {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::stringstream writer;
    writer << R"(static inline void kern_4x12_bias_relu(const float* packA, const float* packB, int K,
                          float* output, int LDC, const float* bias_ptr) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        float* output0 = output;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()";
    if (with_bias) {
        writer << R"(
            "vld1.32 {d2, d3}, [%[bias_ptr]]!\n"
            "vmov.f32 q4, q1\n"
            "pld [%[output0]]\n"
            "vmov.f32 q5, q1\n"
            "vmov.f32 q6, q1\n"
            "vmov.f32 q7, q1\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vmov.f32 q8, q1\n"
            "vmov.f32 q9, q1\n"
            "vmov.f32 q10, q1\n"
            "vmov.f32 q11, q1\n"
            "vld1.32 {d4-d7}, [%[b_ptr]]!\n"
            "vmov.f32 q12, q1\n"
            "vmov.f32 q13, q1\n"
            "vmov.f32 q14, q1\n"
            "vmov.f32 q15, q1\n")";
    } else {
        writer << R"(
            "veor.32 q4, q4, q4\n"
            "pld [%[output0]]\n"
            "veor.32 q5, q4, q4\n"
            "veor.32 q6, q4, q4\n"
            "veor.32 q7, q4, q4\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "veor.32 q8, q4, q4\n"
            "veor.32 q9, q4, q4\n"
            "veor.32 q10, q4, q4\n"
            "veor.32 q11, q4, q4\n"
            "vld1.32 {d4-d7}, [%[b_ptr]]!\n"
            "veor.32 q12, q4, q4\n"
            "veor.32 q13, q4, q4\n"
            "veor.32 q14, q4, q4\n"
            "veor.32 q15, q4, q4\n")";
    }
    std::string body_temp = R"(
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vmla.f32 q4, q0, d4[0]\n"
            "vmla.f32 q5, q0, d4[1]\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q0, d6[0]\n"
            "vmla.f32 q9, q0, d6[1]\n"
            "vmla.f32 q10, q0, d7[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q11, q0, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q12, q0, d4[0]\n"
            "vmla.f32 q13, q0, d4[1]\n"
            "vmla.f32 q14, q0, d5[0]\n"
            "vmla.f32 q15, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"

            "vmla.f32 q4, q1, d6[0]\n"
            "vmla.f32 q5,  q1, d6[1]\n"
            "vmla.f32 q6, q1, d7[0]\n"
            "vmla.f32 q7, q1, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q1, d4[0]\n"
            "vmla.f32 q9, q1, d4[1]\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vmla.f32 q10, q1, d5[0]\n"
            "vmla.f32 q11, q1, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q12, q1, d6[0]\n"
            "vmla.f32 q13, q1, d6[1]\n"
            "vmla.f32 q14, q1, d7[0]\n"
            "vmla.f32 q15, q1, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "subs %[K], %[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vmla.f32 q4,  q0, d4[0]\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q0, d6[0]\n"
            "vmla.f32 q9, q0, d6[1]\n"
            "vmla.f32 q10, q0, d7[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q11, q0, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q12, q0, d4[0]\n"
            "vmla.f32 q13, q0, d4[1]\n"
            "vmla.f32 q14, q0, d5[0]\n"
            "vmla.f32 q15, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"

            "veor.32 q0, q0, q0\n"
            "vmla.f32 q4, q1, d6[0]\n"
            ${GenAsmFloat(q4, q0)}
            "vmla.f32 q5,  q1, d6[1]\n"
            ${GenAsmFloat(q5, q0)}
            "vmla.f32 q6, q1, d7[0]\n"
            ${GenAsmFloat(q6, q0)}
            "vmla.f32 q7, q1, d7[1]\n"
            ${GenAsmFloat(q7, q0)}
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q1, d4[0]\n"
            ${GenAsmFloat(q8, q0)}
            "vmla.f32 q9, q1, d4[1]\n"
            ${GenAsmFloat(q9, q0)}
            "vst1.32 {d8-d11}, [%[output0]]!\n"
            "vmla.f32 q10, q1, d5[0]\n"
            ${GenAsmFloat(q10, q0)}
            "vmla.f32 q11, q1, d5[1]\n"
            ${GenAsmFloat(q11, q0)}
            "vst1.32 {d12-d15}, [%[output0]]!\n"
            "vmla.f32 q12, q1, d6[0]\n"
            ${GenAsmFloat(q12, q0)}
            "vmla.f32 q13, q1, d6[1]\n"
            ${GenAsmFloat(q13, q0)}
            "vst1.32 {d16-d19}, [%[output0]]!\n"
            "vmla.f32 q14, q1, d7[0]\n"
            ${GenAsmFloat(q14, q0)}
            "vmla.f32 q15, q1, d7[1]\n"
            ${GenAsmFloat(q15, q0)}
            "vst1.32 {d20-d23}, [%[output0]]!\n"
            "vst1.32 {d24-d27}, [%[output0]]!\n"
            "vst1.32 {d28-d31}, [%[output0]]!\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "veor.32 q1, q1, q1\n"
            "vmla.f32 q4, q0, d4[0]\n"
            ${GenAsmFloat(q4, q1)}
            "vmla.f32 q5,  q0, d4[1]\n"
            ${GenAsmFloat(q5, q1)}
            "vmla.f32 q6, q0, d5[0]\n"
            ${GenAsmFloat(q6, q1)}
            "vmla.f32 q7, q0, d5[1]\n"
            ${GenAsmFloat(q7, q1)}
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            ${GenAsmFloat(q7, q1)}
            "vmla.f32 q8, q0, d6[0]\n"
            ${GenAsmFloat(q8, q1)}
            "vst1.32 {d8-d11}, [%[output0]]!\n"
            "vmla.f32 q9, q0, d6[1]\n"
            ${GenAsmFloat(q9, q1)}
            "vmla.f32 q10, q0, d7[0]\n"
            ${GenAsmFloat(q10, q1)}
            "vst1.32 {d12-d15}, [%[output0]]!\n"
            "vmla.f32 q11, q0, d7[1]\n"
            ${GenAsmFloat(q11, q1)}
            "vmla.f32 q12, q0, d4[0]\n"
            ${GenAsmFloat(q12, q1)}
            "vst1.32 {d16-d19}, [%[output0]]!\n"
            "vmla.f32 q13, q0, d4[1]\n"
            ${GenAsmFloat(q13, q1)}
            "vst1.32 {d20-d23}, [%[output0]]!\n"
            "vmla.f32 q14, q0, d5[0]\n"
            ${GenAsmFloat(q14, q1)}
            "vst1.32 {d24-d27}, [%[output0]]!\n"
            "vmla.f32 q15, q0, d5[1]\n"
            ${GenAsmFloat(q15, q1)}
            "vst1.32 {d28-d31}, [%[output0]]!\n"

            "6:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr), 
              [K] "+r"(K), [oddk] "+r"(oddk), [output0] "+r"(output0)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
              "q12", "q13", "q14", "q15", "r1", "cc", "memory");
        }
              )";
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenAsmFloat",
                           [=](std::vector<std::string> args) {
                               return activation_gen->GenAsmFloat(args);
                           })
                      .render(body_temp);
    return writer.str();
}

static std::string kern_4x4(TContext* ctx) {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::stringstream writer;
    writer << R"(
            static inline void kern_4x4_bias_relu(const float* packA, const float* packB, int K,
                                           float* output, int LDC, const float* bias_ptr,
                                           int n_remain) {
                   const float* a_ptr = packA;
    const float* b_ptr = packB;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

#define STORE_C                         \
    "cmp %[n_remain], #4\n"             \
    "blt 21f\n"                         \
    "vst1.32 {d8-d11}, [%[output]]!\n"  \
    "vst1.32 {d12-d15}, [%[output]]!\n" \
    "b 24f\n"                           \
    "21:\n"                             \
    "cmp %[n_remain], #3\n"             \
    "blt 22f\n"                         \
    "vst1.32 {d8-d11}, [%[output]]!\n"  \
    "vst1.32 {d12-d13}, [%[output]]!\n" \
    "b 24f\n"                           \
    "22:\n"                             \
    "cmp %[n_remain], #2\n"             \
    "blt 23f\n"                         \
    "vst1.32 {d8-d11}, [%[output]]!\n"  \
    "b 24f\n"                           \
    "23:\n"                             \
    "vst1.32 {d8-d9}, [%[output]]!\n"   \
    "24:\n"

    asm volatile()";
    if (with_bias) {
        writer << R"(
            "vld1.32 {d2, d3}, [%[bias_ptr]]!\n"
            "vmov.f32 q4, q1\n"
            "pld [%[output]]\n"
            "vmov.f32 q5, q1\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vmov.f32 q6, q1\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmov.f32 q7, q1\n"
         )";
    } else {
        writer << R"(
            "veor.32 q4, q4, q4\n"
            "pld [%[output]]\n"
            "veor.32 q5, q4, q4\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "veor.32 q6, q4, q4\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "veor.32 q7, q4, q4\n"
        )";
    }
    std::string body_temp = R"(
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vmla.f32 q4,  q0, d4[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"

            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q4,  q1, d6[0]\n"
            "subs %[K], %[K], #1\n"
            "vmla.f32 q5,  q1, d6[1]\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vmla.f32 q6, q1, d7[0]\n"
            "vmla.f32 q7, q1, d7[1]\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vmla.f32 q4,  q0, d4[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"

            "veor.32 q0, q0, q0\n"
            "vmla.f32 q4, q1, d6[0]\n"
            ${GenAsmFloat(q4, q0)}
            "vmla.f32 q5, q1, d6[1]\n"
            ${GenAsmFloat(q5, q0)}
            "vmla.f32 q6, q1, d7[0]\n"
            ${GenAsmFloat(q6, q0)}
            "vmla.f32 q7, q1, d7[1]\n"
            ${GenAsmFloat(q7, q0)}
            "b 6f\n"

            // odd tail
            "5:\n"
            "veor.32 q1, q1, q1\n"
            "vmla.f32 q4, q0, d6[0]\n"
            ${GenAsmFloat(q4, q1)}
            "vmla.f32 q5, q0, d6[1]\n"
            ${GenAsmFloat(q5, q1)}
            "vmla.f32 q6, q0, d7[0]\n"
            ${GenAsmFloat(q6, q1)}
            "vmla.f32 q7, q0, d7[1]\n"
            ${GenAsmFloat(q7, q1)}

            "6:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr),
              [K] "+r"(K), [oddk] "+r"(oddk), [output] "+r"(output),
              [n_remain] "+r"(n_remain)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "r1", "cc", "memory");
#undef STORE_C
}
)";
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenAsmFloat",
                           [=](std::vector<std::string> args) {
                               return activation_gen->GenAsmFloat(args);
                           })
                      .render(body_temp);
    return writer.str();
}

std::string gen_pack_a(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    const int PACK_C_SIZE = 4;
    size_t cp_length = (kmax - k0) * PACK_C_SIZE;
    for (int m = y0; m < ymax; m += 4) {
        const float* src = inptr + (m / PACK_C_SIZE) * ldin + k0 * PACK_C_SIZE;
        memcpy(outptr, src, cp_length * sizeof(float));
        outptr += cp_length;
    }
})";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    float tmpbuff[16] = {0.0f};
    const int PACK_C_SIZE = 4;
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = outptr;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr0 = inptr + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
        prefetch_3x(inptr0);

        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            float*  outptr_interleave = access_outptr;
            transpose_1x12_4_s(inptr0, outptr_interleave);
            inptr0 += 48;
            access_outptr += ksize12;
        }
        access_outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            float*  outptr_interleave = access_outptr;
            transpose_1x4_4_s(inptr0, outptr_interleave);
            inptr0 += 16;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            memcpy(tmpbuff, inptr0, sizeof(float) * (xmax - x) * PACK_C_SIZE);
            float*  outptr_interleave = access_outptr;
            const float* tmp_ptr = &tmpbuff[0];
            transpose_1x4_4_s(tmp_ptr, outptr_interleave);
            access_outptr += ksize4;
        }
        outptr_base += 12 * PACK_C_SIZE;
        outptr_base4 += 4 * PACK_C_SIZE;
    }
    })";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)(kmax - k0) * (ymax - y0) * sizeof(float);
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

std::string gen_kernel(
        const std::string& sig, TContext* ctx, const std::string& postprocess_call,
        const std::string& preset_str = "") {
    auto post_process_strs = gen_postprocess_inline(ctx);
    std::string keren_body =
            R"(
    ${kernel_sig}{
        ${preset_str}
        const int m_block = 4;
        const int n_block = 12;
        const int pack_mk = 4;
        const int K12 = K * 12;
        const int K4 = K * 4;
        size_t m = 0;        
        for (; m + m_block <= M; m += m_block) {
            float* output = C + (m / pack_mk * LDC);

            size_t n = 0;
            const float* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_4x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr);
                output += n_block * pack_mk;
                cur_pack_b += K12;
            }

            for (; n < N; n += 4) {                
                kern_4x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, N - n > 4 ? 4 : N - n);
                output += 4 * pack_mk;
                cur_pack_b += K4;
            }
            pack_a += K4;
            bias_ptr += m_block;
        }        
        ${postprocess_call}
    }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("postprocess_call", postprocess_call)
            .add("preset_str", preset_str)
            .add("kernel_sig", sig)
            .render(keren_body);
}

}  // namespace

std::string MatmulM4N12MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Armv7_fp32_m4_n12_k4_matmul";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    return ss.str();
}

std::vector<KernelObj> MatmulM4N12MK4Kernel::GetDependInternalSymbol(
        TContext* ctx) const {
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

std::string MatmulM4N12MK4Kernel::GetKernelBody(TContext* ctx) const {
    auto postprocess_pair = gen_postprocess_inline(ctx);
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <arm_neon.h>\n";
    writer << "#include <math.h>\n";
    writer << transpose_1x12_4_s();
    writer << transpose_1x4_4_s();
    writer << kern_4x12(ctx);
    writer << kern_4x4(ctx);
    writer << prefetch();
    writer << gen_pack_a(GetPackASignature(ctx));
    writer << gen_pack_b(GetPackBSignature(ctx));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << postprocess_pair.first;
    writer << gen_kernel(GetNakedKernelSignature(ctx), ctx, postprocess_pair.second);

    std::string preset_temp = R"(
        size_t pack_a_size = ${packa_workspace_sym}(0, M, 0, K);
        float* pack_a = workspace;
        float* pack_b = workspace + pack_a_size;
        ${packa_sym}(pack_a, A, LDA, 0, M, 0, K);
        ${packb_sym}(pack_b, B, LDB, 0, N, 0, K);
    )";
    std::string preset_str =
            StringTemplate::StringTemplateArgs()
                    .add("packa_workspace_sym", GetPackAWorkspaceSymbol(ctx))
                    .add("packa_sym", GetPackASymbol(ctx))
                    .add("packb_sym", GetPackBSymbol(ctx))
                    .render(preset_temp);
    writer << gen_kernel(
            GetKernelSignature(ctx), ctx, postprocess_pair.second, preset_str);
    return writer.str();
}

std::string MatmulM4N12MK4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string MatmulM4N12MK4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
