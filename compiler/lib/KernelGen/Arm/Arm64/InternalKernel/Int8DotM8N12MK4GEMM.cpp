/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/InternalKernel/Int8DotM8N12MK4GEMM.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Arm/Arm64/Activation.h"
#include "Arm/ArmCommon/MatmulCommon.h"
#include "Arm/ArmCommon/common_asm_utils.h"
#include "InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;

namespace {
std::string interleave_2x4_4_b() {
    return std::string{
            R"(
    static inline void interleave_2x4_4_b(const int8_t* inptr0, const int8_t* inptr1,
                                      int8_t* outptr) {
        asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"
            "ld1 {v1.4s}, [%[inptr1]], #16\n"
            "stp q0, q1, [%[outptr]]\n"
            : [ inptr0 ] "+r"(inptr0), [ inptr1 ] "+r"(inptr1),
              [ outptr ] "+r"(outptr)
            :
            : "v0", "v1", "memory");
    })"};
}

std::string interleave_1x4_4_b() {
    return std::string{
            R"(
    static inline void interleave_1x4_4_b(const int8_t* inptr0, int8_t* outptr) {
        asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"
            "st1 {v0.4s}, [%[outptr]]\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "memory");
    })"};
}

std::string prefetch() {
    return R"(
        #define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_2x_f32() +
           KernelGen::ArmCommon::gen_common_prefetch_3x_f32();
}

std::string transpose_1x12() {
    return std::string{R"(
static inline void transpose_1x12_4_s(const float* inptr0, float* outptr) {

    asm volatile(
            "ld4 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[inptr0]], #64\n"
            "ld4 {v4.4s, v5.4s, v6.4s, v7.4s},  [%[inptr0]], #64\n"
            "ld4 {v8.4s, v9.4s, v10.4s, v11.4s},[%[inptr0]], #64\n"

            "stp q0, q4, [%[outptr]] \n"
            "stp q8, q1, [%[outptr], #32] \n"
            "stp q5, q9, [%[outptr], #64] \n"
            "stp q2, q6, [%[outptr], #96] \n"
            "stp q10, q3, [%[outptr], #128] \n"
            "stp q7, q11, [%[outptr], #160] \n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "memory");
    };
)"};
}

static std::string kern_8x12(TContext* ctx, const std::string& dst_specifier,
                             const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    //! kern_8x12
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
    // Overview of register layout:
    //
    // A 12x4 cell of Rhs is stored in 8bit in q2-q4.
    // A 8x4x2 cell of Lhs is stored in 8bit in q0-q1,q5-q6
    // A 8x12 block of accumulators is stored in 32bit in q8--q31.
    //
    //                              +------------+------------+------------+
    //                              |    v2[0-16]|    v3[0-16]|    v4[0-16]|
    //                              |------------|------------|------------|
    //                              |    v5[0-16]|    v6[0-16]|    v7[0-16]|
    //                         Rhs  +------------+------------+------------+
    //
    //                              |            |            |            |
    //
    //    Lhs                       |            |            |            |
    //
    //  +--------+--------+ - - - - +------------+------------+------------+
    //  |v0[0-16]|        |         | v8 v9v10v11|v16v17v18v19|v24v25v26v27|
    //  |v1[0-16]|        |         |v12v13v14v15|v20v21v22v23|v28v29v30v31|
    //  +--------+--------+ - - - - +------------+------------+------------+
    //
    //                            Accumulator
    __attribute__((target("dotprod")))
    static inline void kern_8x12_bias_relu(const int8_t* packA, const int8_t* packB,
                          int K, ${dst_specifier}* output, int LDC,
                          const int32_t* bias_ptr, float scale) {
        K /= 4;
        const int8_t* a_ptr = packA;
        const int8_t* b_ptr = packB;
        ${dst_specifier}* output0 = output;
        ${dst_specifier}* output1 = output0 + LDC;
        float* scale_ptr = &scale;
        const float inv_6 = 1.f / 6.f;
        const float* inv_6_ptr = &inv_6;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()");
    //! if convolution with bias
    if (with_bias) {
        writer << R"(
            "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
            "mov    v8.16b, v6.16b             \n"
            "mov    v9.16b, v6.16b             \n"
            "mov    v10.16b, v6.16b            \n"
            "prfm pstl1keep, [%[output0]]\n"
            "mov    v11.16b, v6.16b            \n"
            "mov    v12.16b, v6.16b            \n"
            "mov    v13.16b, v6.16b            \n"
            "prfm pstl1keep, [%[output1]]\n"
            "mov    v14.16b, v6.16b            \n"
            "mov    v15.16b, v6.16b            \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "mov    v16.16b, v6.16b            \n"
            "ld1 {v7.4s}, [%[bias_ptr]], #16\n"
            "mov    v17.16b, v6.16b            \n"
            "mov    v18.16b, v6.16b            \n"
            "mov    v19.16b, v6.16b            \n"
            "mov    v20.16b, v7.16b            \n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "mov    v21.16b, v7.16b            \n"
            "mov    v22.16b, v7.16b            \n"
            "mov    v23.16b, v7.16b            \n"
            "ld1 {v4.4s}, [%[b_ptr]], #16\n"
            "mov    v24.16b, v7.16b            \n"
            "mov    v25.16b, v7.16b            \n"
            "mov    v26.16b, v7.16b            \n"
            "mov    v27.16b, v7.16b            \n"
            "mov    v28.16b, v7.16b            \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
            "mov    v29.16b, v7.16b            \n"
            "mov    v30.16b, v7.16b            \n"
            "mov    v31.16b, v7.16b            \n")";

        //! if convolution without bias
    } else {
        writer << R"(
            "eor  v8.16b, v8.16b, v8.16b     \n"
            "eor  v9.16b, v9.16b, v9.16b     \n"
            "eor  v10.16b, v10.16b, v10.16b  \n"
            "prfm pstl1keep, [%[output0]]    \n"
            "eor  v11.16b, v11.16b, v11.16b  \n"
            "eor  v12.16b, v12.16b, v12.16b  \n"
            "eor  v13.16b, v13.16b, v13.16b  \n"
            "prfm pstl1keep, [%[output1]]    \n"
            "eor  v14.16b, v14.16b, v14.16b  \n"
            "eor  v15.16b, v15.16b, v15.16b  \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16    \n"
            "eor  v16.16b, v16.16b, v16.16b  \n"
            "eor  v17.16b, v17.16b, v17.16b  \n"
            "eor  v18.16b, v18.16b, v18.16b  \n"
            "eor  v19.16b, v19.16b, v19.16b  \n"
            "eor  v20.16b, v20.16b, v20.16b  \n"
            "ld1 {v3.4s}, [%[b_ptr]], #16    \n"
            "eor  v21.16b, v21.16b, v21.16b  \n"
            "eor  v22.16b, v22.16b, v22.16b  \n"
            "eor  v23.16b, v23.16b, v23.16b  \n"
            "ld1 {v4.4s}, [%[b_ptr]], #16    \n"
            "eor  v24.16b, v24.16b, v24.16b  \n"
            "eor  v25.16b, v25.16b, v25.16b  \n"
            "eor  v26.16b, v26.16b, v26.16b  \n"
            "eor  v27.16b, v27.16b, v27.16b  \n"
            "eor  v28.16b, v28.16b, v28.16b  \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16    \n"
            "eor  v29.16b, v29.16b, v29.16b  \n"
            "eor  v30.16b, v30.16b, v30.16b  \n"
            "eor  v31.16b, v31.16b, v31.16b  \n")";
    }
    writer << R"(
            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v12.4s, v0.16b, v3.4b[0]\n"
            "sdot v13.4s, v0.16b, v3.4b[1]\n"
            "ld1 {v6.4s}, [%[b_ptr]], #16\n"
            "sdot v14.4s, v0.16b, v3.4b[2]\n"
            "sdot v15.4s, v0.16b, v3.4b[3]\n"
            "ld1 {v7.4s}, [%[b_ptr]], #16\n"
            "sdot v16.4s, v0.16b, v4.4b[0]\n"
            "sdot v17.4s, v0.16b, v4.4b[1]\n"
            "sdot v18.4s, v0.16b, v4.4b[2]\n"
            "sdot v19.4s, v0.16b, v4.4b[3]\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v20.4s, v1.16b, v2.4b[0]\n"
            "sdot v21.4s, v1.16b, v2.4b[1]\n"
            "sdot v22.4s, v1.16b, v2.4b[2]\n"
            "sdot v23.4s, v1.16b, v2.4b[3]\n"
            "sdot v24.4s, v1.16b, v3.4b[0]\n"
            "sdot v25.4s, v1.16b, v3.4b[1]\n"
            "sdot v26.4s, v1.16b, v3.4b[2]\n"
            "sdot v27.4s, v1.16b, v3.4b[3]\n"
            "sdot v28.4s, v1.16b, v4.4b[0]\n"
            "sdot v29.4s, v1.16b, v4.4b[1]\n"
            "sdot v30.4s, v1.16b, v4.4b[2]\n"
            "sdot v31.4s, v1.16b, v4.4b[3]\n"

            "sdot v8.4s,  v0.16b, v5.4b[0]\n"
            "sdot v9.4s,  v0.16b, v5.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v10.4s, v0.16b, v5.4b[2]\n"
            "sdot v11.4s, v0.16b, v5.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "sdot v12.4s, v0.16b, v6.4b[0]\n"
            "sdot v13.4s, v0.16b, v6.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], 16\n"
            "sdot v14.4s, v0.16b, v6.4b[2]\n"
            "sdot v15.4s, v0.16b, v6.4b[3]\n"
            "ld1 {v4.4s}, [%[b_ptr]], 16\n"
            "sdot v16.4s, v0.16b, v7.4b[0]\n"
            "sdot v17.4s, v0.16b, v7.4b[1]\n"
            "sdot v18.4s, v0.16b, v7.4b[2]\n"
            "sdot v19.4s, v0.16b, v7.4b[3]\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v20.4s, v1.16b, v5.4b[0]\n"
            "sdot v21.4s, v1.16b, v5.4b[1]\n"
            "sdot v22.4s, v1.16b, v5.4b[2]\n"
            "sdot v23.4s, v1.16b, v5.4b[3]\n"
            "sdot v24.4s, v1.16b, v6.4b[0]\n"
            "subs %w[K], %w[K], #1\n"
            "sdot v25.4s, v1.16b, v6.4b[1]\n"
            "sdot v26.4s, v1.16b, v6.4b[2]\n"
            "sdot v27.4s, v1.16b, v6.4b[3]\n"
            "sdot v28.4s, v1.16b, v7.4b[0]\n"
            "sdot v29.4s, v1.16b, v7.4b[1]\n"
            "sdot v30.4s, v1.16b, v7.4b[2]\n"
            "sdot v31.4s, v1.16b, v7.4b[3]\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
)";
    std::string tail_temp = R"(
            // Even tail
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "sdot v12.4s, v0.16b, v3.4b[0]\n"
            "sdot v13.4s, v0.16b, v3.4b[1]\n"
            "sdot v14.4s, v0.16b, v3.4b[2]\n"
            "sdot v15.4s, v0.16b, v3.4b[3]\n"
            "sdot v16.4s, v0.16b, v4.4b[0]\n"
            "sdot v17.4s, v0.16b, v4.4b[1]\n"
            "sdot v18.4s, v0.16b, v4.4b[2]\n"
            "sdot v19.4s, v0.16b, v4.4b[3]\n"

            "sdot v20.4s, v1.16b, v2.4b[0]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v21.4s, v1.16b, v2.4b[1]\n"
            "sdot v22.4s, v1.16b, v2.4b[2]\n"
            "sdot v23.4s, v1.16b, v2.4b[3]\n"
            "sdot v24.4s, v1.16b, v3.4b[0]\n"
            "sdot v25.4s, v1.16b, v3.4b[1]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v26.4s, v1.16b, v3.4b[2]\n"
            "sdot v27.4s, v1.16b, v3.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v28.4s, v1.16b, v4.4b[0]\n"

            ${gen_postprocess_reg_init}

            "sdot v29.4s, v1.16b, v4.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v30.4s, v1.16b, v4.4b[2]\n"
            "sdot v31.4s, v1.16b, v4.4b[3]\n"

            "sdot v8.4s,  v0.16b, v5.4b[0]\n"
            "sdot v9.4s,  v0.16b, v5.4b[1]\n"
            "sdot v10.4s, v0.16b, v5.4b[2]\n"
            "sdot v11.4s, v0.16b, v5.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v12.4s, v0.16b, v2.4b[0]\n"
            
            "sdot v13.4s, v0.16b, v2.4b[1]\n"

            "sdot v14.4s, v0.16b, v2.4b[2]\n"
            
            "sdot v15.4s, v0.16b, v2.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}

            "sdot v16.4s, v0.16b, v3.4b[0]\n"
            
            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            
            
            "sdot v17.4s, v0.16b, v3.4b[1]\n"
            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            "sdot v18.4s, v0.16b, v3.4b[2]\n"
            
            ${GenAsmGenAsmQuantStore(v11, output0, 12)}
            "sdot v19.4s, v0.16b, v3.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v12, output0, 16)}

            "sdot v20.4s, v1.16b, v5.4b[0]\n"
            
            ${GenAsmGenAsmQuantStore(v13, output0, 20)}
            "sdot v21.4s, v1.16b, v5.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v14, output0, 24)}
            "sdot v22.4s, v1.16b, v5.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v15, output0, 28)}
            "sdot v23.4s, v1.16b, v5.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v16, output0, 32)}
            "sdot v24.4s, v1.16b, v2.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v17, output0, 36)}
            "sdot v25.4s, v1.16b, v2.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v18, output0, 40)}
            "sdot v26.4s, v1.16b, v2.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v19, output0, 44)}
            "sdot v27.4s, v1.16b, v2.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v20, output1, 0)}
            "sdot v28.4s, v1.16b, v3.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v21, output1, 4)}
            "sdot v29.4s, v1.16b, v3.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v22, output1, 8)}
            "sdot v30.4s, v1.16b, v3.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v23, output1, 12)}
            "sdot v31.4s, v1.16b, v3.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v24, output1, 16)}
            ${GenAsmGenAsmQuantStore(v25, output1, 20)}
            ${GenAsmGenAsmQuantStore(v26, output1, 24)}
            ${GenAsmGenAsmQuantStore(v27, output1, 28)}
            ${GenAsmGenAsmQuantStore(v28, output1, 32)}
            ${GenAsmGenAsmQuantStore(v29, output1, 36)}
            ${GenAsmGenAsmQuantStore(v30, output1, 40)}
            ${GenAsmGenAsmQuantStore(v31, output1, 44)}
            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            ${gen_postprocess_reg_init}
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"

            "sdot v12.4s, v0.16b, v3.4b[0]\n"

            "sdot v13.4s, v0.16b, v3.4b[1]\n"

            "sdot v14.4s, v0.16b, v3.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            "sdot v15.4s, v0.16b, v3.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            "sdot v16.4s, v0.16b, v4.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            "sdot v17.4s, v0.16b, v4.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v11, output0, 12)}
            "sdot v18.4s, v0.16b, v4.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v12, output0, 16)}
            "sdot v19.4s, v0.16b, v4.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v13, output0, 20)}

            "sdot v20.4s, v1.16b, v2.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v14, output0, 24)}
            "sdot v21.4s, v1.16b, v2.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v15, output0, 28)}
            "sdot v22.4s, v1.16b, v2.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v16, output0, 32)}
            "sdot v23.4s, v1.16b, v2.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v17, output0, 36)}
            "sdot v24.4s, v1.16b, v3.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v18, output0, 40)}
            "sdot v25.4s, v1.16b, v3.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v19, output0, 44)}
            "sdot v26.4s, v1.16b, v3.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v20, output1, 0)}
            "sdot v27.4s, v1.16b, v3.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v21, output1, 4)}
            "sdot v28.4s, v1.16b, v4.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v22, output1, 8)}
            "sdot v29.4s, v1.16b, v4.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v23, output1, 12)}
            "sdot v30.4s, v1.16b, v4.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v24, output1, 16)}
            ${GenAsmGenAsmQuantStore(v25, output1, 20)}
            ${GenAsmGenAsmQuantStore(v26, output1, 24)}
            ${GenAsmGenAsmQuantStore(v27, output1, 28)}
            "sdot v31.4s, v1.16b, v4.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v28, output1, 32)}
            ${GenAsmGenAsmQuantStore(v29, output1, 36)}
            ${GenAsmGenAsmQuantStore(v30, output1, 40)}
            ${GenAsmGenAsmQuantStore(v31, output1, 44)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1), [scale_ptr] "+r" (scale_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
        })";

    std::string tail_many_reg_temp = R"(
            // Even tail
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "sdot v12.4s, v0.16b, v3.4b[0]\n"
            "sdot v13.4s, v0.16b, v3.4b[1]\n"
            "sdot v14.4s, v0.16b, v3.4b[2]\n"
            "sdot v15.4s, v0.16b, v3.4b[3]\n"
            "sdot v16.4s, v0.16b, v4.4b[0]\n"
            "sdot v17.4s, v0.16b, v4.4b[1]\n"
            "sdot v18.4s, v0.16b, v4.4b[2]\n"
            "sdot v19.4s, v0.16b, v4.4b[3]\n"
            "sdot v20.4s, v1.16b, v2.4b[0]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v21.4s, v1.16b, v2.4b[1]\n"
            "sdot v22.4s, v1.16b, v2.4b[2]\n"
            "sdot v23.4s, v1.16b, v2.4b[3]\n"
            "sdot v24.4s, v1.16b, v3.4b[0]\n"
            "sdot v25.4s, v1.16b, v3.4b[1]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v26.4s, v1.16b, v3.4b[2]\n"
            "sdot v27.4s, v1.16b, v3.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v28.4s, v1.16b, v4.4b[0]\n"
            "sdot v29.4s, v1.16b, v4.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v30.4s, v1.16b, v4.4b[2]\n"
            "sdot v31.4s, v1.16b, v4.4b[3]\n"
            "sdot v8.4s,  v0.16b, v5.4b[0]\n"
            "sdot v9.4s,  v0.16b, v5.4b[1]\n"
            "sdot v10.4s, v0.16b, v5.4b[2]\n"
            "sdot v11.4s, v0.16b, v5.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v12.4s, v0.16b, v2.4b[0]\n"            
            "sdot v13.4s, v0.16b, v2.4b[1]\n"
            "sdot v14.4s, v0.16b, v2.4b[2]\n"            
            "sdot v15.4s, v0.16b, v2.4b[3]\n"                        
            "sdot v16.4s, v0.16b, v3.4b[0]\n"                                    
            "sdot v17.4s, v0.16b, v3.4b[1]\n"            
            "sdot v18.4s, v0.16b, v3.4b[2]\n"            
            "sdot v19.4s, v0.16b, v3.4b[3]\n"
            "sdot v20.4s, v1.16b, v5.4b[0]\n"            
            "sdot v21.4s, v1.16b, v5.4b[1]\n"            
            "sdot v22.4s, v1.16b, v5.4b[2]\n"            
            "sdot v23.4s, v1.16b, v5.4b[3]\n"            
            "sdot v24.4s, v1.16b, v2.4b[0]\n"            
            "sdot v25.4s, v1.16b, v2.4b[1]\n"            
            "sdot v26.4s, v1.16b, v2.4b[2]\n"            
            "sdot v27.4s, v1.16b, v2.4b[3]\n"            
            "sdot v28.4s, v1.16b, v3.4b[0]\n"            
            "sdot v29.4s, v1.16b, v3.4b[1]\n"            
            "sdot v30.4s, v1.16b, v3.4b[2]\n"            
            "sdot v31.4s, v1.16b, v3.4b[3]\n"
            ${gen_postprocess_reg_init}
            ${GenAsmGenAsmQuantStore(v8, v9, output0, 0)}
            ${GenAsmGenAsmQuantStore(v10, v11, output0, 8)}
            ${GenAsmGenAsmQuantStore(v12, v13, output0, 16)}
            ${GenAsmGenAsmQuantStore(v14, v15, output0, 24)}
            ${GenAsmGenAsmQuantStore(v16, v17, output0, 32)}
            ${GenAsmGenAsmQuantStore(v18, v19, output0, 40)}
            ${GenAsmGenAsmQuantStore(v20, v21, output1, 0)}
            ${GenAsmGenAsmQuantStore(v22, v23, output1, 8)}
            ${GenAsmGenAsmQuantStore(v24, v25, output1, 16)}
            ${GenAsmGenAsmQuantStore(v26, v27, output1, 24)}
            ${GenAsmGenAsmQuantStore(v28, v29, output1, 32)}
            ${GenAsmGenAsmQuantStore(v30, v31, output1, 40)}

            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "sdot v12.4s, v0.16b, v3.4b[0]\n"
            "sdot v13.4s, v0.16b, v3.4b[1]\n"
            "sdot v14.4s, v0.16b, v3.4b[2]\n"            
            "sdot v15.4s, v0.16b, v3.4b[3]\n"            
            "sdot v16.4s, v0.16b, v4.4b[0]\n"            
            "sdot v17.4s, v0.16b, v4.4b[1]\n"            
            "sdot v18.4s, v0.16b, v4.4b[2]\n"            
            "sdot v19.4s, v0.16b, v4.4b[3]\n"            
            "sdot v20.4s, v1.16b, v2.4b[0]\n"            
            "sdot v21.4s, v1.16b, v2.4b[1]\n"            
            "sdot v22.4s, v1.16b, v2.4b[2]\n"            
            "sdot v23.4s, v1.16b, v2.4b[3]\n"            
            "sdot v24.4s, v1.16b, v3.4b[0]\n"            
            "sdot v25.4s, v1.16b, v3.4b[1]\n"            
            "sdot v26.4s, v1.16b, v3.4b[2]\n"            
            "sdot v27.4s, v1.16b, v3.4b[3]\n"            
            "sdot v28.4s, v1.16b, v4.4b[0]\n"            
            "sdot v29.4s, v1.16b, v4.4b[1]\n"            
            "sdot v30.4s, v1.16b, v4.4b[2]\n"
            "sdot v31.4s, v1.16b, v4.4b[3]\n"

            ${gen_postprocess_reg_init}
            ${GenAsmGenAsmQuantStore(v8, v9, output0, 0)}
            ${GenAsmGenAsmQuantStore(v10, v11, output0, 8)}
            ${GenAsmGenAsmQuantStore(v12, v13, output0, 16)}
            ${GenAsmGenAsmQuantStore(v14, v15, output0, 24)}
            ${GenAsmGenAsmQuantStore(v16, v17, output0, 32)}
            ${GenAsmGenAsmQuantStore(v18, v19, output0, 40)}
            ${GenAsmGenAsmQuantStore(v20, v21, output1, 0)}
            ${GenAsmGenAsmQuantStore(v22, v23, output1, 8)}
            ${GenAsmGenAsmQuantStore(v24, v25, output1, 16)}
            ${GenAsmGenAsmQuantStore(v26, v27, output1, 24)}
            ${GenAsmGenAsmQuantStore(v28, v29, output1, 32)}
            ${GenAsmGenAsmQuantStore(v30, v31, output1, 40)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1), [scale_ptr] "+r" (scale_ptr), [inv6_ptr] "+r" (inv_6_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
        })";
    if (nonline_mode == "H_SWISH") {
        std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
                {"v0", "v1", "v2", "v3", "v7"}, nonline_mode,
                {"inv6_ptr", "scale_ptr"});
        writer << StringTemplate::StringTemplateArgs()
                          .add("dst_specifier", dst_specifier)
                          .add("GenAsmGenAsmQuantStore",
                               [=](std::vector<std::string> args) {
                                   CC_ASSERT(args.size() == 4);
                                   return activation_gen->GenAsmQuantStore(
                                           {args[0], args[1]}, "v7", args[2],
                                           std::stoi(args[3]), dst_specifier,
                                           {"v0", "v1", "v2", "v3", "v4", "v5"},
                                           nonline_mode);
                               })
                          .add("gen_postprocess_reg_init", postprocess_reg_init)
                          .render(tail_many_reg_temp);
    } else {
        std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
                {"v7", "v6"}, nonline_mode, {"scale_ptr"});
        writer << StringTemplate::StringTemplateArgs()
                          .add("dst_specifier", dst_specifier)
                          .add("GenAsmGenAsmQuantStore",
                               [=](std::vector<std::string> args) {
                                   CC_ASSERT(args.size() == 3);
                                   return activation_gen->GenAsmQuantStore(
                                           {args[0]}, "v6", args[1],
                                           std::stoi(args[2]), dst_specifier,
                                           {"v7"}, nonline_mode);
                               })
                          .add("gen_postprocess_reg_init", postprocess_reg_init)
                          .render(tail_temp);
    }
    return writer.str();
}

static std::string kern_4x12(TContext* ctx, const std::string& dst_specifier,
                             const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
        __attribute__((target("dotprod")))
        static inline void kern_4x12_bias_relu(const int8_t* packA, const int8_t* packB, int K,
                          ${dst_specifier}* output, int LDC, const int32_t* bias_ptr, float scale) {
        K /= 4;
        const int8_t* a_ptr = packA;
        const int8_t* b_ptr = packB;
        ${dst_specifier}* output0 = output;
        

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;
        float* scale_ptr = &scale;
        const float inv_6 = 1.f / 6.f;
        const float* inv_6_ptr = &inv_6;

        asm volatile()");
    //! if convolution with bias
    if (with_bias) {
        writer << R"(
                "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
                "mov    v8.16b, v6.16b             \n"
                "mov    v9.16b, v6.16b             \n"
                "mov    v10.16b, v6.16b            \n"
                "prfm pstl1keep, [%[output0]]\n"
                "mov    v11.16b, v6.16b            \n"
                "mov    v12.16b, v6.16b            \n"
                "mov    v13.16b, v6.16b            \n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], #48\n"
                "mov    v14.16b, v6.16b            \n"
                "mov    v15.16b, v6.16b            \n"
                "mov    v16.16b, v6.16b            \n"
                "mov    v17.16b, v6.16b            \n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "mov    v18.16b, v6.16b            \n"
                "mov    v19.16b, v6.16b            \n"
        )";
    } else {
        //! if convolution without bias
        writer << R"(
                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "prfm pstl1keep, [%[output0]]\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], #48\n"
                "eor v16.16b, v16.16b, v16.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "eor v18.16b, v18.16b, v18.16b\n"
                "eor v19.16b, v19.16b, v19.16b\n")";
    }
    std::string body_temp = R"(

               "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "sdot v8.4s,  v0.16b, v2.4b[0]\n"
                "sdot v9.4s,  v0.16b, v2.4b[1]\n"
                "ld1 {v1.16b}, [%[a_ptr]], 16\n"
                "sdot v10.4s, v0.16b, v2.4b[2]\n"
                "sdot v11.4s, v0.16b, v2.4b[3]\n"
                "sdot v12.4s, v0.16b, v3.4b[0]\n"
                "sdot v13.4s, v0.16b, v3.4b[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], #48\n"
                "sdot v14.4s, v0.16b, v3.4b[2]\n"
                "sdot v15.4s, v0.16b, v3.4b[3]\n"
                "sdot v16.4s, v0.16b, v4.4b[0]\n"
                "sdot v17.4s, v0.16b, v4.4b[1]\n"
                "sdot v18.4s, v0.16b, v4.4b[2]\n"
                "sdot v19.4s, v0.16b, v4.4b[3]\n"

                "sdot v8.4s,  v1.16b, v5.4b[0]\n"
                "sdot v9.4s,  v1.16b, v5.4b[1]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "sdot v10.4s, v1.16b, v5.4b[2]\n"
                "sdot v11.4s, v1.16b, v5.4b[3]\n"
                "ld1 {v0.16b}, [%[a_ptr]], 16\n"
                "sdot v12.4s, v1.16b, v6.4b[0]\n"
                "sdot v13.4s, v1.16b, v6.4b[1]\n"
                "subs %w[K], %w[K], #1\n"
                "sdot v14.4s, v1.16b, v6.4b[2]\n"
                "sdot v15.4s, v1.16b, v6.4b[3]\n"
                "sdot v16.4s, v1.16b, v7.4b[0]\n"
                "sdot v17.4s, v1.16b, v7.4b[1]\n"
                "sdot v18.4s, v1.16b, v7.4b[2]\n"
                "sdot v19.4s, v1.16b, v7.4b[3]\n"
                "bne 3b\n"

                "4:\n"        
                ${gen_postprocess_reg_init}       
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "sdot v8.4s,  v0.16b, v2.4b[0]\n"
                "sdot v9.4s,  v0.16b, v2.4b[1]\n"
                "ld1 {v1.16b}, [%[a_ptr]], 16\n"
                "sdot v10.4s, v0.16b, v2.4b[2]\n"
                "sdot v11.4s, v0.16b, v2.4b[3]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], #48\n"
                "sdot v12.4s, v0.16b, v3.4b[0]\n"
                "sdot v13.4s, v0.16b, v3.4b[1]\n"
                "sdot v14.4s, v0.16b, v3.4b[2]\n"
                "sdot v15.4s, v0.16b, v3.4b[3]\n"
                "sdot v16.4s, v0.16b, v4.4b[0]\n"
                "sdot v17.4s, v0.16b, v4.4b[1]\n"
                "sdot v18.4s, v0.16b, v4.4b[2]\n"
                "sdot v19.4s, v0.16b, v4.4b[3]\n"

                "sdot v8.4s,  v1.16b, v5.4b[0]\n"
                "sdot v9.4s,  v1.16b, v5.4b[1]\n"
                "sdot v10.4s, v1.16b, v5.4b[2]\n"
                "sdot v11.4s, v1.16b, v5.4b[3]\n"
                "ld1 {v0.16b}, [%[a_ptr]], 16\n"
                "sdot v12.4s, v1.16b, v6.4b[0]\n"
                ${GenAsmGenAsmQuantStore(v8, v9, output0, 0)}
                "sdot v13.4s, v1.16b, v6.4b[1]\n"
                
                "sdot v14.4s, v1.16b, v6.4b[2]\n"
                ${GenAsmGenAsmQuantStore(v10, v11, output0, 8)}
                "sdot v15.4s, v1.16b, v6.4b[3]\n"
                "sdot v16.4s, v1.16b, v7.4b[0]\n"
                ${GenAsmGenAsmQuantStore(v12, v13, output0, 16)}
                "sdot v17.4s, v1.16b, v7.4b[1]\n"
                "sdot v18.4s, v1.16b, v7.4b[2]\n"
                ${GenAsmGenAsmQuantStore(v14, v15, output0, 24)}
                "sdot v19.4s, v1.16b, v7.4b[3]\n"
                ${GenAsmGenAsmQuantStore(v16, v17, output0, 32)}
                ${GenAsmGenAsmQuantStore(v18, v19, output0, 40)}

                "b 6f\n"

                // odd tail
                "5:\n"
                "sdot v8.4s,  v0.16b, v2.4b[0]\n"
                "sdot v9.4s,  v0.16b, v2.4b[1]\n"
                "sdot v10.4s, v0.16b, v2.4b[2]\n"
                "sdot v11.4s, v0.16b, v2.4b[3]\n"
                "sdot v12.4s, v0.16b, v3.4b[0]\n"
                "sdot v13.4s, v0.16b, v3.4b[1]\n"
                "sdot v14.4s, v0.16b, v3.4b[2]\n"
                ${GenAsmGenAsmQuantStore(v8, v9, output0, 0)}
                "sdot v15.4s, v0.16b, v3.4b[3]\n"
                "sdot v16.4s, v0.16b, v4.4b[0]\n"
                ${GenAsmGenAsmQuantStore(v10, v11, output0, 8)}
                "sdot v17.4s, v0.16b, v4.4b[1]\n"
                "sdot v18.4s, v0.16b, v4.4b[2]\n"
                ${GenAsmGenAsmQuantStore(v12, v13, output0, 16)}
                "sdot v19.4s, v0.16b, v4.4b[3]\n"
                ${GenAsmGenAsmQuantStore(v14, v15, output0, 24)}
                ${GenAsmGenAsmQuantStore(v16, v17, output0, 32)}
                ${GenAsmGenAsmQuantStore(v18, v19, output0, 40)}

                "6:\n"
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [oddk] "+r"(oddk), [bias_ptr] "+r"(bias_ptr),
                  [output0] "+r"(output0), [scale_ptr] "+r" (scale_ptr), [inv6_ptr] "+r" (inv_6_ptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "x1", "cc", "memory");
    }
    )";

    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v20", "v21", "v22", "v23", "v27"}, nonline_mode,
            {"inv6_ptr", "scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 4);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0], args[1]}, "v27", args[2],
                                       std::stoi(args[3]), dst_specifier,
                                       {"v20", "v21", "v22", "v23", "v24",
                                        "v25"},
                                       nonline_mode);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(body_temp);
    return writer.str();
}

static std::string kern_8x4(TContext* ctx, const std::string& dst_specifier,
                            const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string store_str = "STORE_C";
    if (dst_specifier == "int8_t") {
        store_str = "STORE_C_QUANT";
    }

    std::stringstream writer;
    //! kern_8x4
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
        
        // Overview of register layout:
        //
        // A 1x12 cell of Rhs is stored in 32bit in v2-v7
        // A 8x1 cell of Lhs is stored in 32bit in (v0-v1)
        // A 8x12 block of accumulators is stored in 32bit in v8-v31.
        //
        //                 +--------+
        //                 | v2[0-3]|
        //                 | v3[0-3]|
        //           Rhs   +--------+
        //
        //                 |        |
        //
        //    Lhs          |        |
        //
        //  +--+   ---  -  +--------+
        //  |v0|           | v8[0-3]|
        //  |v0|           |v11[0-3]|
        //  |v0|           |v14[0-3]|
        //  |v0|           |v17[0-3]|
        //  |v1|           |v20[0-3]|
        //  |v1|           |v23[0-3]|
        //  |v1|           |v26[0-3]|
        //  |v1|           |v29[0-3]|
        //  +--+   ---  -  +--------+
        //
        //
        __attribute__((target("dotprod")))
        static inline void kern_8x4_bias_relu(const int8_t* packA, const int8_t* packB, int K,
                            ${dst_specifier}* output, int LDC, const int32_t* bias_ptr, int n_remain, float scale) {
            K /= 4;
            const int8_t* a_ptr = packA;
            const int8_t* b_ptr = packB;
            ${dst_specifier}* output0 = output;
            ${dst_specifier}* output1 = output0 + LDC;
            float* scale_ptr = &scale;
            const float inv_6 = 1.f / 6.f;
            const float* inv_6_ptr = &inv_6;
            
            int oddk = (K & 1);
            K = ((K + 1) / 2) - 1;

            //clang-format off
        #define STORE_C                                           \
            "cmp %w[n_remain], #4\n"                              \
            "blt 21f\n"                                           \
            "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n"  \
            "st1 {v12.4s, v13.4s, v14.4s, v15.4s},[%[output1]]\n" \
            "b 24f\n"                                             \
            "21:\n"                                               \
            "cmp %w[n_remain], #3\n"                              \
            "blt 22f\n"                                           \
            "st1 {v8.4s, v9.4s, v10.4s}, [%[output0]]\n"          \
            "st1 {v12.4s, v13.4s, v14.4s},[%[output1]]\n"         \
            "b 23f\n"                                             \
            "22:\n"                                               \
            "cmp %w[n_remain], #2\n"                              \
            "blt 23f\n"                                           \
            "st1 {v8.4s, v9.4s}, [%[output0]]\n"                  \
            "st1 {v12.4s, v13.4s},[%[output1]]\n"                 \
            "b 24f\n"                                             \
            "23:\n"                                               \
            "st1 {v8.4s}, [%[output0]]\n"                         \
            "st1 {v12.4s},[%[output1]]\n"                         \
            "24:\n"

        #define STORE_C_QUANT                                     \
            "cmp %w[n_remain], #4\n"                              \
            "blt 21f\n"                                           \
            "str s8,  [%[output0], #0]\n"                         \
            "str s9,  [%[output0], #4]\n"                         \
            "str s10, [%[output0], #8]\n"                         \
            "str s11, [%[output0], #12]\n"                        \
            "str s12, [%[output1], #0]\n"                         \
            "str s13, [%[output1], #4]\n"                         \
            "str s14, [%[output1], #8]\n"                         \
            "str s15, [%[output1], #12]\n"                        \
            "b 24f\n"                                             \
            "21:\n"                                               \
            "cmp %w[n_remain], #3\n"                              \
            "blt 22f\n"                                           \
            "str s8,  [%[output0], #0]\n"                         \
            "str s9,  [%[output0], #4]\n"                         \
            "str s10, [%[output0], #8]\n"                         \
            "str s12, [%[output1], #0]\n"                         \
            "str s13, [%[output1], #4]\n"                         \
            "str s14, [%[output1], #8]\n"                         \
            "b 23f\n"                                             \
            "22:\n"                                               \
            "cmp %w[n_remain], #2\n"                              \
            "blt 23f\n"                                           \
            "str s8,  [%[output0], #0]\n"                         \
            "str s9,  [%[output0], #4]\n"                         \
            "str s12, [%[output1], #0]\n"                         \
            "str s13, [%[output1], #4]\n"                         \
            "b 24f\n"                                             \
            "23:\n"                                               \
            "str s8,  [%[output0], #0]\n"                         \
            "str s12, [%[output1], #0]\n"                         \
            "24:\n"

            //clang-format on

            asm volatile()");
    if (with_bias) {
        writer << R"(
            "ld1 {v30.4s}, [%[bias_ptr]], #16\n"
            "ld1 {v31.4s}, [%[bias_ptr]], #16\n"
            "mov v8.16b, v30.16b            \n"
            "mov v9.16b, v30.16b            \n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "mov v10.16b, v30.16b            \n"
            "prfm pstl1keep, [%[output0]]\n"
            "mov v11.16b, v30.16b            \n"
            "mov v12.16b, v31.16b            \n"
            "prfm pstl1keep, [%[output1]]\n"
            "mov v13.16b, v31.16b            \n"
            "mov v14.16b, v31.16b            \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "mov v15.16b, v31.16b            \n")";
    } else {
        writer << R"(
            "eor    v8.16b, v8.16b, v8.16b   \n"
            "eor    v9.16b, v9.16b, v9.16b   \n"
            "ld1 {v0.16b}, [%[a_ptr]], #16    \n"
            "eor    v10.16b, v10.16b, v10.16b\n"
            "prfm pstl1keep, [%[output0]]    \n"
            "eor    v11.16b, v11.16b, v11.16b\n"
            "eor    v12.16b, v12.16b, v12.16b\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16    \n"
            "eor    v13.16b, v13.16b, v13.16b\n"
            "prfm pstl1keep, [%[output1]]    \n"
            "eor    v14.16b, v14.16b, v14.16b\n"
            "eor    v15.16b, v15.16b, v15.16b\n")";
    }

    std::string body_temp = R"(
            ${gen_postprocess_reg_init}

            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "sdot v12.4s, v1.16b, v2.4b[0]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v13.4s, v1.16b, v2.4b[1]\n"
            "sdot v14.4s, v1.16b, v2.4b[2]\n"
            "sdot v15.4s, v1.16b, v2.4b[3]\n"

            "sdot v8.4s,  v0.16b, v3.4b[0]\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "sdot v9.4s,  v0.16b, v3.4b[1]\n"
            "sdot v10.4s, v0.16b, v3.4b[2]\n"
            "sdot v11.4s, v0.16b, v3.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v12.4s, v1.16b, v3.4b[0]\n"
            "subs %w[K], %w[K], #1\n"
            "sdot v13.4s, v1.16b, v3.4b[1]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v14.4s, v1.16b, v3.4b[2]\n"
            "sdot v15.4s, v1.16b, v3.4b[3]\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "sdot v12.4s, v1.16b, v2.4b[0]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v13.4s, v1.16b, v2.4b[1]\n"
            "sdot v14.4s, v1.16b, v2.4b[2]\n"
            "sdot v15.4s, v1.16b, v2.4b[3]\n"

            "sdot v8.4s,  v0.16b, v3.4b[0]\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "sdot v9.4s,  v0.16b, v3.4b[1]\n"
            "sdot v10.4s, v0.16b, v3.4b[2]\n"
            "sdot v11.4s, v0.16b, v3.4b[3]\n"
            "sdot v12.4s, v1.16b, v3.4b[0]\n"
            "sdot v13.4s, v1.16b, v3.4b[1]\n"
            "sdot v14.4s, v1.16b, v3.4b[2]\n"
            "sdot v15.4s, v1.16b, v3.4b[3]\n"
            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v0.16b, v2.4b[0]\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "sdot v9.4s,  v0.16b, v2.4b[1]\n"
            "sdot v10.4s, v0.16b, v2.4b[2]\n"
            "sdot v11.4s, v0.16b, v2.4b[3]\n"
            "sdot v12.4s, v1.16b, v2.4b[0]\n"
            "sdot v13.4s, v1.16b, v2.4b[1]\n"
            "sdot v14.4s, v1.16b, v2.4b[2]\n"
            "sdot v15.4s, v1.16b, v2.4b[3]\n"

            "6:\n"
            ${GenAsmGenAsmQuantStore(v8, v9)}
            ${GenAsmGenAsmQuantStore(v10, v11)}
            ${GenAsmGenAsmQuantStore(v12, v13)}
            ${GenAsmGenAsmQuantStore(v14, v15)}

            ${gen_store}
            

            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1),
              [ n_remain ] "+r"(n_remain), [scale_ptr] "+r" (scale_ptr), [inv6_ptr] "+r" (inv_6_ptr)
            :
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13",
              "v14", "v15", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "cc", "memory");
            #undef STORE_C
            #undef STORE_C_QUANT
    })";

    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v20", "v21", "v22", "v23", "v27"}, nonline_mode,
            {"inv6_ptr", "scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("gen_store", store_str)
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 2);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0], args[1]}, "v27", "None", 0,
                                       dst_specifier,
                                       {"v20", "v21", "v22", "v23", "v24",
                                        "v25"},
                                       nonline_mode, false);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(body_temp);
    return writer.str();
}

static std::string kern_4x4(TContext* ctx, const std::string& dst_specifier,
                            const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string store_str = "STORE_C";
    if (dst_specifier == "int8_t") {
        store_str = "STORE_C_QUANT";
    }
    std::stringstream writer;
    //! kern_4x4
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
        __attribute__((target("dotprod")))
            // Overview of register layout:
            //
            // A 2x4 cell of Rhs is stored in 32bit in v2 - v3
            // A 4x2 cell of Lhs is stored in 32bit in v0 - v1
            // A 4x4 block of accumulators is stored in 32bit in v4-v6
            //
            //                 +--------+
            //                 | v2[0-3]|
            //                 | v5[0-3]|
            //           Rhs   +--------+
            //
            //                 |        |
            //
            //    Lhs          |        |
            //
            //  +--+   ---  -  +--------+
            //  |v0|           | v8[0-3]|
            //  |v0|           |v11[0-3]|
            //  |v0|           |v14[0-3]|
            //  |v0|           |v17[0-3]|
            //  +--+   ---  -  +--------+
            //
            //                        Accumulator
            static inline void kern_4x4_bias_relu(const int8_t* packA, const int8_t* packB, int K,
                                           ${dst_specifier}* output, int LDC, const int32_t* bias_ptr,
                                           int n_remain, float scale) {
                K /= 4;
                const int8_t* a_ptr = packA;
                const int8_t* b_ptr = packB;
                ${dst_specifier}* output0 = output;
                float* scale_ptr = &scale;
                const float inv_6 = 1.f / 6.f;
                const float* inv_6_ptr = &inv_6;

                int oddk = (K & 1);
                K = ((K + 1) / 2) - 1;

                //clang-format off
            #define STORE_C                                          \
                "cmp %w[n_remain], #4\n"                             \
                "blt 21f\n"                                          \
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n" \
                "b 24f\n"                                            \
                "21:\n"                                              \
                "cmp %w[n_remain], #3\n"                             \
                "blt 22f\n"                                          \
                "st1 {v8.4s, v9.4s, v10.4s}, [%[output0]]\n"         \
                "b 24f\n"                                            \
                "22:\n"                                              \
                "cmp %w[n_remain], #2\n"                             \
                "blt 23f\n"                                          \
                "st1 {v8.4s, v9.4s}, [%[output0]]\n"                 \
                "b 24f\n"                                            \
                "23:\n"                                              \
                "st1 {v8.4s}, [%[output0]]\n"                        \
                "24:\n"

            #define STORE_C_QUANT                                    \
                "cmp %w[n_remain], #4\n"                             \
                "blt 21f\n"                                          \
                "str s8,  [%[output0], #0]\n"                        \
                "str s9,  [%[output0], #4]\n"                        \
                "str s10, [%[output0], #8]\n"                        \
                "str s11, [%[output0], #12]\n"                       \
                "b 24f\n"                                            \
                "21:\n"                                              \
                "cmp %w[n_remain], #3\n"                             \
                "blt 22f\n"                                          \
                "str s8,  [%[output0], #0]\n"                        \
                "str s9,  [%[output0], #4]\n"                        \
                "str s10, [%[output0], #8]\n"                        \
                "b 24f\n"                                            \
                "22:\n"                                              \
                "cmp %w[n_remain], #2\n"                             \
                "blt 23f\n"                                          \
                "str s8,  [%[output0], #0]\n"                        \
                "str s9,  [%[output0], #4]\n"                        \
                "b 24f\n"                                            \
                "23:\n"                                              \
                "str s8,  [%[output0], #0]\n"                        \
                "24:\n"
                //clang-format on

                asm volatile(     )");
    if (with_bias) {
        writer << R"(
                // load accumulator C
                "ld1 {v30.4s}, [%[bias_ptr]], #16\n"
                "mov v8.16b, v30.16b            \n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "mov v9.16b, v30.16b            \n"
                "ld1 {v0.16b}, [%[a_ptr]], #16\n"
                "mov v10.16b, v30.16b            \n"
                "mov v11.16b, v30.16b            \n")";
    } else {
        writer << R"(
                "eor  v8.16b, v8.16b, v8.16b     \n"
                "ld1 {v0.16b}, [%[a_ptr]], #16    \n"
                "eor  v9.16b, v9.16b, v9.16b     \n"
                "eor  v10.16b, v10.16b, v10.16b  \n"
                "ld1 {v2.4s}, [%[b_ptr]], #16    \n"
                "eor  v11.16b, v11.16b, v11.16b  \n")";
    }
    std::string body_temp = R"(
                ${gen_postprocess_reg_init}

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "sdot v8.4s,  v0.16b, v2.4b[0]\n"
                "ld1 {v1.16b}, [%[a_ptr]], 16\n"
                "sdot v9.4s,  v0.16b, v2.4b[1]\n"
                "sdot v10.4s, v0.16b, v2.4b[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "sdot v11.4s, v0.16b, v2.4b[3]\n"

                "sdot v8.4s,  v1.16b, v3.4b[0]\n"
                "sdot v9.4s,  v1.16b, v3.4b[1]\n"
                "ld1 {v0.16b}, [%[a_ptr]], 16\n"
                "sdot v10.4s, v1.16b, v3.4b[2]\n"
                "sdot v11.4s, v1.16b, v3.4b[3]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "sdot v8.4s,  v0.16b, v2.4b[0]\n"
                "ld1 {v1.16b}, [%[a_ptr]], 16\n"
                "sdot v9.4s,  v0.16b, v2.4b[1]\n"
                "sdot v10.4s, v0.16b, v2.4b[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "sdot v11.4s, v0.16b, v2.4b[3]\n"

                "sdot v8.4s,  v1.16b, v3.4b[0]\n"
                "sdot v9.4s,  v1.16b, v3.4b[1]\n"
                "sdot v10.4s, v1.16b, v3.4b[2]\n"
                "sdot v11.4s, v1.16b, v3.4b[3]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "sdot v8.4s,  v0.16b, v2.4b[0]\n"
                "sdot v9.4s,  v0.16b, v2.4b[1]\n"
                "sdot v10.4s, v0.16b, v2.4b[2]\n"
                "sdot v11.4s, v0.16b, v2.4b[3]\n"

                "6:\n"
                ${GenAsmGenAsmQuantStore(v8, v9)}
                ${GenAsmGenAsmQuantStore(v10, v11)}
                
                ${gen_store}

                : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
                  [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
                  [ output0 ] "+r"(output0), [ n_remain ] "+r"(n_remain), [scale_ptr] "+r" (scale_ptr), [inv6_ptr] "+r" (inv_6_ptr)
                :
                : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "cc", "memory");
            #undef STORE_C
            #undef STORE_C_QUANT
        })";
    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v20", "v21", "v22", "v23", "v27"}, nonline_mode,
            {"inv6_ptr", "scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("gen_store", store_str)
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 2);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0], args[1]}, "v27", "None", 0,
                                       dst_specifier,
                                       {"v20", "v21", "v22", "v23", "v24",
                                        "v25"},
                                       nonline_mode, false);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(body_temp);
    return writer.str();
}

std::string gen_pack_a(const std::string& sig) {
    //! FIXME: opt it
    std::stringstream ss;
    ss << sig;
    ss << R"({
    const int pack_mk = 4;
    const int pack_m = 8;
    const int m_stride = pack_m * pack_mk;
    const int min_m_stride = pack_mk * pack_mk;
    int y = 0;
    for (; y + 7 < ymax; y += pack_m) {
        const int8_t* inptr0 = inptr + y / pack_mk * ldin;
        const int8_t* inptr1 = inptr0 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        int k = (kmax);
        for (; k > 3; k -= pack_mk) {
            interleave_2x4_4_b(inptr0, inptr1, outptr);
            outptr += m_stride;
            inptr0 += min_m_stride;
            inptr1 += min_m_stride;
        }
    }
    for (; y < ymax; y += pack_mk) {
        const int8_t* inptr0 = inptr + y / pack_mk * ldin;
        prefetch_2x(inptr0);
        int K = (kmax);
        for (; K > 3; K -= pack_mk) {
            interleave_1x4_4_b(inptr0, outptr);
            outptr += min_m_stride;
            inptr0 += min_m_stride;
        }
    }
})";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        float tmpbuff[16] = {0.0f};

        int PACK_C_SIZE = 4;
        int ksize = kmax - k0;
        int ksize12 = ksize * 12;
        int ksize4 = (ksize << 2);
        int8_t* outptr_base = outptr;
        int8_t* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const int8_t* temp_inptr = inptr + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
            prefetch_3x(temp_inptr);

            int x = x0;
            int8_t* temp_outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                memcpy(temp_outptr, temp_inptr, 48);
                temp_outptr += ksize12;
                temp_inptr += 4 * 12;
            }
            temp_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                memcpy(temp_outptr, temp_inptr, 16);
                temp_outptr += ksize4;
                temp_inptr += 16;
            }
            if (x < xmax) {
                int i = 0;
                for (; i < xmax - x; i++) {
                    *temp_outptr++ = *temp_inptr++;
                    *temp_outptr++ = *temp_inptr++;
                    *temp_outptr++ = *temp_inptr++;
                    *temp_outptr++ = *temp_inptr++;
                }
                for (; i < 4; i++) {
                    *temp_outptr++ = 0;
                    *temp_outptr++ = 0;
                    *temp_outptr++ = 0;
                    *temp_outptr++ = 0;
                }
            }
            outptr_base += 12 * 4;
            outptr_base4 += 4 * 4;
        }
    })";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)(kmax - k0) * (ymax - y0) * sizeof(int8_t);
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
        size_t res = (size_t)(kmax - k0) * packed_hw * sizeof(int8_t);
        return res;
    })";
    return ss.str();
}

std::string gen_kernel(const std::string& dst_specifier, const std::string& sig,
                       TContext* ctx, const std::string& postprocess_call,
                       const std::string& preset_str = "",
                       bool with_temp_dst = false) {
    auto post_process_strs = gen_postprocess_inline(ctx);
    std::string gemm_output = "C";
    if (with_temp_dst) {
        gemm_output = "workspace";
    }
    std::string keren_body =
            R"(
    ${kernel_sig}{
        ${preset_str}
        const int m_block = 8;
        const int m_block_4 = 4;
        const int n_block = 12;
        const int pack_mk = 4;
        const int K12 = K * 12;
        const int K8 = K * 8;
        const int K4 = K * 4;
        size_t m = 0;
        ${dst_specifier}* gemm_output = (${dst_specifier}*)${gen_gemm_output};
        for (; m + m_block <= M; m += m_block) {
            ${dst_specifier}* output = gemm_output + (m / pack_mk * LDC);

            size_t n = 0;
            const int8_t* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_8x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr, scale);
                output += n_block * pack_mk;
                cur_pack_b += K12;
            }

            for (; n < N; n += 4) {                
                kern_8x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, N - n > 4 ? 4 : N - n, scale);
                output += 4 * pack_mk;
                cur_pack_b += K4;
            }
            pack_a += K8;
            bias_ptr += m_block;
        }
        for (; m < M; m += m_block_4) {
            ${dst_specifier}* output = gemm_output + (m / pack_mk * LDC);
            size_t n = 0;
            const int8_t* cur_pack_b = pack_b;
            for (; n + n_block - 1 < N; n += n_block) {                
                kern_4x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr, scale);
                output += n_block * pack_mk;
                cur_pack_b += K12;
            }
            for (; n < N; n += 4) {                
                kern_4x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, N - n > 4 ? 4 : N - n, scale);
                output += 4 * pack_mk;
                cur_pack_b += K4;
            }
            pack_a += K4;
            bias_ptr += m_block_4;
        }
        ${postprocess_call}
    }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("gen_gemm_output", gemm_output)
            .add("dst_specifier", dst_specifier)
            .add("postprocess_call", postprocess_call)
            .add("preset_str", preset_str)
            .add("kernel_sig", sig)
            .render(keren_body);
}

}  // namespace

std::string MatmulInt8DotM8N12MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Arm64_int8_dot_m8_n12_mk4_gemm";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") &&
        ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    auto dtype = ctx->getAttrStr("dtype");
    if (Utils::is_quant_dtype(dtype)) {
        ss << "_qsi8";
    } else {
        CC_ASSERT(dtype == "8832");
        ss << "_" << dtype;
    }
    if (ctx->haveAttr("last_dtype")) {
        auto last_dtype = ctx->getAttrStr("last_dtype");
        ss << "_"
           << "output_dtype_" << last_dtype;
    }
    return ss.str();
}

bool MatmulInt8DotM8N12MK4Kernel::need_post_process(TContext* ctx) const {
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    return nonline_mode == "SIGMOID";
}

std::vector<KernelObj> MatmulInt8DotM8N12MK4Kernel::GetDependInternalSymbol(
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

std::string MatmulInt8DotM8N12MK4Kernel::GetKernelBody(TContext* ctx) const {
    auto postprocess_pair = gen_postprocess_inline(ctx, need_post_process(ctx));
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <arm_neon.h>\n";
    writer << prefetch();
    writer << transpose_1x12();
    auto dtype = ctx->getAttrStr("dtype");
    std::string last_dtype = "si8";
    if (ctx->haveAttr("last_dtype")) {
        last_dtype = ctx->getAttrStr("last_dtype");
    }
    std::string dst_specifier = "int32_t";
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    if (Utils::is_quant_dtype(dtype) &&
        (nonline_mode == "RELU" || nonline_mode == "IDENTITY" ||
         nonline_mode == "H_SWISH")) {
        dst_specifier = Utils::cvt_dtype_specifier(last_dtype);
    }
    //! sigmoid use explicit postprocess
    bool need_temp_dst = need_post_process(ctx);
    auto gen_nonline_mode = need_temp_dst ? "IDENTITY" : nonline_mode;

    writer << kern_8x12(ctx, dst_specifier, gen_nonline_mode);
    writer << kern_8x4(ctx, dst_specifier, gen_nonline_mode);
    writer << kern_4x12(ctx, dst_specifier, gen_nonline_mode);
    writer << kern_4x4(ctx, dst_specifier, gen_nonline_mode);
    writer << interleave_2x4_4_b();
    writer << interleave_1x4_4_b();
    writer << gen_pack_a(GetPackASignature(ctx));
    writer << gen_pack_b(GetPackBSignature(ctx));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << postprocess_pair.first;
    writer << gen_kernel(dst_specifier, GetNakedKernelSignature(ctx), ctx,
                         postprocess_pair.second, "", need_temp_dst);

    std::string preset_temp = R"(
        size_t pack_a_size = ${packa_workspace_sym}(0, M, 0, K);
        int8_t* pack_a = workspace;
        int8_t* pack_b = workspace + pack_a_size;
        ${packa_sym}(pack_a, A, LDA, 0, M, 0, K);
        ${packb_sym}(pack_b, B, LDB, 0, N, 0, K);
    )";
    std::string preset_str =
            StringTemplate::StringTemplateArgs()
                    .add("packa_workspace_sym", GetPackAWorkspaceSymbol(ctx))
                    .add("packa_sym", GetPackASymbol(ctx))
                    .add("packb_sym", GetPackBSymbol(ctx))
                    .render(preset_temp);
    writer << gen_kernel(dst_specifier, GetKernelSignature(ctx), ctx,
                         postprocess_pair.second, preset_str, need_temp_dst);
    return writer.str();
}

std::string MatmulInt8DotM8N12MK4Kernel::GetPackAWorkspaceBody(
        TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulInt8DotM8N12MK4Kernel::GetPackBWorkspaceBody(
        TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
