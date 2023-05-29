#include "Arm/Arm64/Activation.h"
#include "Arm/ArmCommon/MatmulCommon.h"
#include "Arm/ArmCommon/common_asm_utils.h"
#include "InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;

namespace {
std::string interleave_2x4_4_s() {
    return std::string{
            R"(
    static inline void interleave_2x4_4_s(const float* inptr0, const float* inptr1,
                                      float* outptr) {
        asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[inptr0]], #64\n"
            "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[inptr1]], #64\n"
            "stp q0, q4, [%[outptr]]\n"
            "stp q1, q5, [%[outptr], #32]\n"
            "stp q2, q6, [%[outptr], #64]\n"
            "stp q3, q7, [%[outptr], #96]\n"
            : [ inptr0 ] "+r"(inptr0), [ inptr1 ] "+r"(inptr1),
              [ outptr ] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
    })"};
}

std::string interleave_1x4_4_s() {
    return std::string{
            R"(
    static inline void interleave_1x4_4_s(const float* inptr0, float* outptr) {
        asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[inptr0]], #64\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[outptr]]\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "memory");
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

static std::string kern_8x12(TContext* ctx) {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    //! kern_8x12
    writer << R"(
    // Overview of register layout:
    //
    // A 1x12 cell of Rhs is stored in 32bit in v2-v7
    // A 8x1 cell of Lhs is stored in 32bit in (v0-v1)
    // A 8x12 block of accumulators is stored in 32bit in v8-v31.
    //
    //                 +--------+--------+--------+
    //                 | v2[0-3]| v3[0-3]| v4[0-3]|
    //                 | v5[0-3]| v6[0-3]| v7[0-3]|
    //           Rhs   +--------+--------+--------+
    //
    //                 |        |        |        |
    //
    //    Lhs          |        |        |        |
    //
    //  +--+   ---  -  +--------+--------+--------+
    //  |v0|           | v8[0-3]| v9[0-3]|v10[0-3]|
    //  |v0|           |v11[0-3]|v12[0-3]|v13[0-3]|
    //  |v0|           |v14[0-3]|v15[0-3]|v16[0-3]|
    //  |v0|           |v17[0-3]|v18[0-3]|v19[0-3]|
    //  |v1|           |v20[0-3]|v21[0-3]|v22[0-3]|
    //  |v1|           |v23[0-3]|v24[0-3]|v25[0-3]|
    //  |v1|           |v26[0-3]|v27[0-3]|v28[0-3]|
    //  |v1|           |v29[0-3]|v30[0-3]|v31[0-3]|
    //  +--+   ---  -  +--------+--------+--------+
    //
    //                        Accumulator
    static inline void kern_8x12_bias_relu(const float* packA, const float* packB,
                          int K, float* output, int LDC,
                          const float* bias_ptr) {                              
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        float* output0 = output;
        float* output1 = output0 + LDC;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()";
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
    std::string body_temp = R"(
            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "fmla v8.4s,  v0.4s, v2.s[0]\n"
            "fmla v9.4s,  v0.4s, v2.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v10.4s, v0.4s, v2.s[2]\n"
            "fmla v11.4s, v0.4s, v2.s[3]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "fmla v12.4s, v0.4s, v3.s[0]\n"
            "fmla v13.4s, v0.4s, v3.s[1]\n"
            "ld1 {v6.4s}, [%[b_ptr]], #16\n"
            "fmla v14.4s, v0.4s, v3.s[2]\n"
            "fmla v15.4s, v0.4s, v3.s[3]\n"
            "ld1 {v7.4s}, [%[b_ptr]], #16\n"
            "fmla v16.4s, v0.4s, v4.s[0]\n"
            "fmla v17.4s, v0.4s, v4.s[1]\n"
            "fmla v18.4s, v0.4s, v4.s[2]\n"
            "fmla v19.4s, v0.4s, v4.s[3]\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"

            "fmla v20.4s, v1.4s, v2.s[0]\n"
            "fmla v21.4s, v1.4s, v2.s[1]\n"
            "fmla v22.4s, v1.4s, v2.s[2]\n"
            "fmla v23.4s, v1.4s, v2.s[3]\n"
            "fmla v24.4s, v1.4s, v3.s[0]\n"
            "fmla v25.4s, v1.4s, v3.s[1]\n"
            "fmla v26.4s, v1.4s, v3.s[2]\n"
            "fmla v27.4s, v1.4s, v3.s[3]\n"
            "fmla v28.4s, v1.4s, v4.s[0]\n"
            "fmla v29.4s, v1.4s, v4.s[1]\n"
            "fmla v30.4s, v1.4s, v4.s[2]\n"
            "fmla v31.4s, v1.4s, v4.s[3]\n"

            "fmla v8.4s,  v0.4s, v5.s[0]\n"
            "fmla v9.4s,  v0.4s, v5.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v10.4s, v0.4s, v5.s[2]\n"
            "fmla v11.4s, v0.4s, v5.s[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmla v12.4s, v0.4s, v6.s[0]\n"
            "fmla v13.4s, v0.4s, v6.s[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], 16\n"
            "fmla v14.4s, v0.4s, v6.s[2]\n"
            "fmla v15.4s, v0.4s, v6.s[3]\n"
            "ld1 {v4.4s}, [%[b_ptr]], 16\n"
            "fmla v16.4s, v0.4s, v7.s[0]\n"
            "fmla v17.4s, v0.4s, v7.s[1]\n"
            "fmla v18.4s, v0.4s, v7.s[2]\n"
            "fmla v19.4s, v0.4s, v7.s[3]\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"

            "fmla v20.4s, v1.4s, v5.s[0]\n"
            "fmla v21.4s, v1.4s, v5.s[1]\n"
            "fmla v22.4s, v1.4s, v5.s[2]\n"
            "fmla v23.4s, v1.4s, v5.s[3]\n"
            "fmla v24.4s, v1.4s, v6.s[0]\n"
            "subs %w[K], %w[K], #1\n"
            "fmla v25.4s, v1.4s, v6.s[1]\n"
            "fmla v26.4s, v1.4s, v6.s[2]\n"
            "fmla v27.4s, v1.4s, v6.s[3]\n"
            "fmla v28.4s, v1.4s, v7.s[0]\n"
            "fmla v29.4s, v1.4s, v7.s[1]\n"
            "fmla v30.4s, v1.4s, v7.s[2]\n"
            "fmla v31.4s, v1.4s, v7.s[3]\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "fmla v8.4s,  v0.4s, v2.s[0]\n"
            "fmla v9.4s,  v0.4s, v2.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v10.4s, v0.4s, v2.s[2]\n"
            "fmla v11.4s, v0.4s, v2.s[3]\n"
            "fmla v12.4s, v0.4s, v3.s[0]\n"
            "fmla v13.4s, v0.4s, v3.s[1]\n"
            "fmla v14.4s, v0.4s, v3.s[2]\n"
            "fmla v15.4s, v0.4s, v3.s[3]\n"
            "fmla v16.4s, v0.4s, v4.s[0]\n"
            "fmla v17.4s, v0.4s, v4.s[1]\n"
            "fmla v18.4s, v0.4s, v4.s[2]\n"
            "fmla v19.4s, v0.4s, v4.s[3]\n"

            "fmla v20.4s, v1.4s, v2.s[0]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "fmla v21.4s, v1.4s, v2.s[1]\n"
            "fmla v22.4s, v1.4s, v2.s[2]\n"
            "ld1 {v6.4s}, [%[b_ptr]], #16\n"
            "fmla v23.4s, v1.4s, v2.s[3]\n"
            "fmla v24.4s, v1.4s, v3.s[0]\n"
            "fmla v25.4s, v1.4s, v3.s[1]\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v26.4s, v1.4s, v3.s[2]\n"
            "fmla v27.4s, v1.4s, v3.s[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "fmla v28.4s, v1.4s, v4.s[0]\n"

            "eor v7.16b, v7.16b, v7.16b\n"

            "fmla v29.4s, v1.4s, v4.s[1]\n"
            "fmla v30.4s, v1.4s, v4.s[2]\n"
            "fmla v31.4s, v1.4s, v4.s[3]\n"

            "fmla v8.4s,  v0.4s, v5.s[0]\n"
            "fmla v9.4s,  v0.4s, v5.s[1]\n"
            "fmla v10.4s, v0.4s, v5.s[2]\n"
            "fmla v11.4s, v0.4s, v5.s[3]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v12.4s, v0.4s, v6.s[0]\n"
            ${GenAsmFloat(v8, v7)}
            "fmla v13.4s, v0.4s, v6.s[1]\n"

            "fmla v14.4s, v0.4s, v6.s[2]\n"
            ${GenAsmFloat(v9, v7)}
            "fmla v15.4s, v0.4s, v6.s[3]\n"
            ${GenAsmFloat(v10, v7)}
            "st1 {v8.4s}, [%[output0]], #16\n"

            "fmla v16.4s, v0.4s, v2.s[0]\n"
            ${GenAsmFloat(v11, v7)}
            "st1 {v9.4s}, [%[output0]], #16\n"
            ${GenAsmFloat(v12, v7)}
            "fmla v17.4s, v0.4s, v2.s[1]\n"
            "st1 {v10.4s}, [%[output0]], #16\n"
            "fmla v18.4s, v0.4s, v2.s[2]\n"
            ${GenAsmFloat(v13, v7)}
            "st1 {v11.4s}, [%[output0]], #16\n"
            "fmla v19.4s, v0.4s, v2.s[3]\n"
            ${GenAsmFloat(v14, v7)}
            "st1 {v12.4s}, [%[output0]], #16\n"

            "fmla v20.4s, v1.4s, v5.s[0]\n"
            ${GenAsmFloat(v15, v7)}
            "st1 {v13.4s}, [%[output0]], #16\n"
            "fmla v21.4s, v1.4s, v5.s[1]\n"
            ${GenAsmFloat(v16, v7)}
            "st1 {v14.4s}, [%[output0]], #16\n"
            "fmla v22.4s, v1.4s, v5.s[2]\n"
            ${GenAsmFloat(v17, v7)}
            "st1 {v15.4s}, [%[output0]], #16\n"
            "fmla v23.4s, v1.4s, v5.s[3]\n"
            ${GenAsmFloat(v18, v7)}
            "st1 {v16.4s}, [%[output0]], #16\n"
            "fmla v24.4s, v1.4s, v6.s[0]\n"
            ${GenAsmFloat(v19, v7)}
            "st1 {v17.4s}, [%[output0]], #16\n"
            "fmla v25.4s, v1.4s, v6.s[1]\n"
            ${GenAsmFloat(v20, v7)}
            "st1 {v18.4s}, [%[output0]], #16\n"
            "fmla v26.4s, v1.4s, v6.s[2]\n"
            ${GenAsmFloat(v21, v7)}
            "st1 {v19.4s}, [%[output0]], #16\n"
            "fmla v27.4s, v1.4s, v6.s[3]\n"
            ${GenAsmFloat(v22, v7)}
            "st1 {v20.4s}, [%[output1]], #16\n"
            "fmla v28.4s, v1.4s, v2.s[0]\n"
            ${GenAsmFloat(v23, v7)}
            "st1 {v21.4s}, [%[output1]], #16\n"
            "fmla v29.4s, v1.4s, v2.s[1]\n"
            ${GenAsmFloat(v24, v7)}
            "st1 {v22.4s}, [%[output1]], #16\n"
            "fmla v30.4s, v1.4s, v2.s[2]\n"
            ${GenAsmFloat(v25, v7)}
            "st1 {v23.4s}, [%[output1]], #16\n"
            "fmla v31.4s, v1.4s, v2.s[3]\n"
            ${GenAsmFloat(v26, v27, v28, v29, v30, v31, v7)}
            "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output1]], #64\n"
            "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output1]], #64\n"
            "b 6f\n"

            // odd tail
            "5:\n"
            "fmla v8.4s,  v0.4s, v2.s[0]\n"
            "eor  v7.16b, v7.16b, v7.16b \n"
            "fmla v9.4s,  v0.4s, v2.s[1]\n"
            "fmla v10.4s, v0.4s, v2.s[2]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v11.4s, v0.4s, v2.s[3]\n"
            ${GenAsmFloat(v8, v7)}
            "fmla v12.4s, v0.4s, v3.s[0]\n"
            ${GenAsmFloat(v9, v7)}
            "fmla v13.4s, v0.4s, v3.s[1]\n"
            ${GenAsmFloat(v10, v7)}
            "fmla v14.4s, v0.4s, v3.s[2]\n"
            ${GenAsmFloat(v11, v7)}
            "st1 {v8.4s}, [%[output0]], #16\n"
            "fmla v15.4s, v0.4s, v3.s[3]\n"
            ${GenAsmFloat(v12, v7)}
            "st1 {v9.4s}, [%[output0]], #16\n"
            "fmla v16.4s, v0.4s, v4.s[0]\n"
            ${GenAsmFloat(v13, v7)}
            "st1 {v10.4s}, [%[output0]], #16\n"
            "fmla v17.4s, v0.4s, v4.s[1]\n"
            ${GenAsmFloat(v14, v7)}
            "st1 {v11.4s}, [%[output0]], #16\n"
            "fmla v18.4s, v0.4s, v4.s[2]\n"
            ${GenAsmFloat(v15, v7)}
            "st1 {v12.4s}, [%[output0]], #16\n"
            "fmla v19.4s, v0.4s, v4.s[3]\n"
            ${GenAsmFloat(v16, v7)}
            "st1 {v13.4s}, [%[output0]], #16\n"

            "fmla v20.4s, v1.4s, v2.s[0]\n"
            ${GenAsmFloat(v17, v7)}
            "st1 {v14.4s}, [%[output0]], #16\n"
            "fmla v21.4s, v1.4s, v2.s[1]\n"
            ${GenAsmFloat(v18, v7)}
            "st1 {v15.4s}, [%[output0]], #16\n"
            "fmla v22.4s, v1.4s, v2.s[2]\n"
            ${GenAsmFloat(v19, v7)}
            "st1 {v16.4s}, [%[output0]], #16\n"
            "fmla v23.4s, v1.4s, v2.s[3]\n"
            ${GenAsmFloat(v20, v7)}
            "st1 {v17.4s}, [%[output0]], #16\n"
            "fmla v24.4s, v1.4s, v3.s[0]\n"
            ${GenAsmFloat(v21, v7)}
            "st1 {v18.4s}, [%[output0]], #16\n"
            "fmla v25.4s, v1.4s, v3.s[1]\n"
            ${GenAsmFloat(v22, v7)}
            "st1 {v19.4s}, [%[output0]], #16\n"
            "fmla v26.4s, v1.4s, v3.s[2]\n"
            ${GenAsmFloat(v23, v7)}
            "st1 {v20.4s}, [%[output1]], #16\n"
            "fmla v27.4s, v1.4s, v3.s[3]\n"
            ${GenAsmFloat(v24, v7)}
            "st1 {v21.4s}, [%[output1]], #16\n"
            "fmla v28.4s, v1.4s, v4.s[0]\n"
            ${GenAsmFloat(v25, v7)}
            "st1 {v22.4s}, [%[output1]], #16\n"
            "fmla v29.4s, v1.4s, v4.s[1]\n"
            ${GenAsmFloat(v26, v7)}
            "st1 {v23.4s}, [%[output1]], #16\n"
            "fmla v30.4s, v1.4s, v4.s[2]\n"
            ${GenAsmFloat(v27, v7)}
            "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output1]], #64\n"
            "fmla v31.4s, v1.4s, v4.s[3]\n"
            ${GenAsmFloat(v28, v29, v30, v31, v7)}
            "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output1]], #64\n"

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
        })";
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenAsmFloat",
                           [=](std::vector<std::string> args) {
                               return activation_gen->GenAsmFloat(args);
                           })
                      .render(body_temp);
    return writer.str();
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
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], #48\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "fmla v8.4s,  v1.4s, v5.s[0]\n"
                "fmla v9.4s,  v1.4s, v5.s[1]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "fmla v10.4s, v1.4s, v5.s[2]\n"
                "fmla v11.4s, v1.4s, v5.s[3]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v12.4s, v1.4s, v6.s[0]\n"
                "fmla v13.4s, v1.4s, v6.s[1]\n"
                "subs %w[K], %w[K], #1\n"
                "fmla v14.4s, v1.4s, v6.s[2]\n"
                "fmla v15.4s, v1.4s, v6.s[3]\n"
                "fmla v16.4s, v1.4s, v7.s[0]\n"
                "fmla v17.4s, v1.4s, v7.s[1]\n"
                "fmla v18.4s, v1.4s, v7.s[2]\n"
                "fmla v19.4s, v1.4s, v7.s[3]\n"
                "bne 3b\n"

                "4:\n"        
                "eor  v20.16b, v20.16b, v20.16b \n"        
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], #48\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "fmla v8.4s,  v1.4s, v5.s[0]\n"
                "fmla v9.4s,  v1.4s, v5.s[1]\n"
                "fmla v10.4s, v1.4s, v5.s[2]\n"
                "fmla v11.4s, v1.4s, v5.s[3]\n"
                ${GenAsmFloat(v8, v9, v10, v11, v20)}
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v12.4s, v1.4s, v6.s[0]\n"
                "fmla v13.4s, v1.4s, v6.s[1]\n"
                
                "fmla v14.4s, v1.4s, v6.s[2]\n"
                "fmla v15.4s, v1.4s, v6.s[3]\n"
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]], #64\n"
                ${GenAsmFloat(v12, v13, v14, v15, v20)}
                "fmla v16.4s, v1.4s, v7.s[0]\n"
                "fmla v17.4s, v1.4s, v7.s[1]\n"
                "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[output0]], #64\n"
                "fmla v18.4s, v1.4s, v7.s[2]\n"
                "fmla v19.4s, v1.4s, v7.s[3]\n"
                ${GenAsmFloat(v16, v17, v18, v19, v20)}
                "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output0]], #64\n"

                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                ${GenAsmFloat(v8, v9, v10, v11, v20)}
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]], #64\n"
                ${GenAsmFloat(v12, v13, v14, v15, v20)}
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[output0]], #64\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"
                ${GenAsmFloat(v16, v17, v18, v19, v20)}
                "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output0]], #64\n"

                "6:\n"
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [oddk] "+r"(oddk), [bias_ptr] "+r"(bias_ptr),
                  [output0] "+r"(output0)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "x1", "cc", "memory");
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

static std::string kern_8x4(TContext* ctx) {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    //! kern_8x4
    writer << R"(
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
        static inline void kern_8x4_bias_relu(const float* packA, const float* packB, int K,
                            float* output, int LDC, const float* bias_ptr, int n_remain) {
            const float* a_ptr = packA;
            const float* b_ptr = packB;
            float* output0 = output;
            float* output1 = output0 + LDC;

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
            //clang-format on

            asm volatile()";
    if (with_bias) {
        writer << R"(
            "ld1 {v30.4s}, [%[bias_ptr]], #16\n"
            "ld1 {v31.4s}, [%[bias_ptr]], #16\n"
            "mov v8.16b, v30.16b            \n"
            "mov v9.16b, v30.16b            \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
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
            "ld1 {v0.4s}, [%[a_ptr]], #16    \n"
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
            "eor  v30.16b, v30.16b, v30.16b \n"

            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "fmla v8.4s,  v0.4s, v2.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], #16\n"
            "fmla v9.4s,  v0.4s, v2.s[1]\n"
            "fmla v10.4s, v0.4s, v2.s[2]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "fmla v11.4s, v0.4s, v2.s[3]\n"
            "fmla v12.4s, v1.4s, v2.s[0]\n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
            "fmla v13.4s, v1.4s, v2.s[1]\n"
            "fmla v14.4s, v1.4s, v2.s[2]\n"
            "fmla v15.4s, v1.4s, v2.s[3]\n"

            "fmla v8.4s,  v0.4s, v3.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], #16\n"
            "fmla v9.4s,  v0.4s, v3.s[1]\n"
            "fmla v10.4s, v0.4s, v3.s[2]\n"
            "fmla v11.4s, v0.4s, v3.s[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "fmla v12.4s, v1.4s, v3.s[0]\n"
            "subs %w[K], %w[K], #1\n"
            "fmla v13.4s, v1.4s, v3.s[1]\n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
            "fmla v14.4s, v1.4s, v3.s[2]\n"
            "fmla v15.4s, v1.4s, v3.s[3]\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "fmla v8.4s,  v0.4s, v2.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], #16\n"
            "fmla v9.4s,  v0.4s, v2.s[1]\n"
            "fmla v10.4s, v0.4s, v2.s[2]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "fmla v11.4s, v0.4s, v2.s[3]\n"
            "fmla v12.4s, v1.4s, v2.s[0]\n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
            "fmla v13.4s, v1.4s, v2.s[1]\n"
            "fmla v14.4s, v1.4s, v2.s[2]\n"
            "fmla v15.4s, v1.4s, v2.s[3]\n"

            "fmla v8.4s,  v0.4s, v3.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], #16\n"
            "fmla v9.4s,  v0.4s, v3.s[1]\n"
            "fmla v10.4s, v0.4s, v3.s[2]\n"
            "fmla v11.4s, v0.4s, v3.s[3]\n"
            "fmla v12.4s, v1.4s, v3.s[0]\n"
            "fmla v13.4s, v1.4s, v3.s[1]\n"
            "fmla v14.4s, v1.4s, v3.s[2]\n"
            "fmla v15.4s, v1.4s, v3.s[3]\n"
            "b 6f\n"

            // odd tail
            "5:\n"
            "fmla v8.4s,  v0.4s, v2.s[0]\n"
            "ld1 {v1.4s}, [%[a_ptr]], #16\n"
            "fmla v9.4s,  v0.4s, v2.s[1]\n"
            "fmla v10.4s, v0.4s, v2.s[2]\n"
            "fmla v11.4s, v0.4s, v2.s[3]\n"
            "fmla v12.4s, v1.4s, v2.s[0]\n"
            "fmla v13.4s, v1.4s, v2.s[1]\n"
            "fmla v14.4s, v1.4s, v2.s[2]\n"
            "fmla v15.4s, v1.4s, v2.s[3]\n"

            "6:\n"
            ${GenAsmFloat(v8, v9, v10, v11, v12, v13, v14, v15, v30)}
            STORE_C

            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1),
              [ n_remain ] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13",
              "v14", "v15", "v30", "v31", "cc", "memory");
            #undef STORE_C
    })";
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
    //! kern_4x4
    writer << R"(
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
            static inline void kern_4x4_bias_relu(const float* packA, const float* packB, int K,
                                           float* output, int LDC, const float* bias_ptr,
                                           int n_remain) {
                const float* a_ptr = packA;
                const float* b_ptr = packB;
                float* output0 = output;

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
                //clang-format on

                asm volatile(     )";
    if (with_bias) {
        writer << R"(
                // load accumulator C
                "ld1 {v30.4s}, [%[bias_ptr]], #16\n"
                "mov v8.16b, v30.16b            \n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "mov v9.16b, v30.16b            \n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "mov v10.16b, v30.16b            \n"
                "mov v11.16b, v30.16b            \n")";
    } else {
        writer << R"(
                "eor  v8.16b, v8.16b, v8.16b     \n"
                "ld1 {v0.4s}, [%[a_ptr]], #16    \n"
                "eor  v9.16b, v9.16b, v9.16b     \n"
                "eor  v10.16b, v10.16b, v10.16b  \n"
                "ld1 {v2.4s}, [%[b_ptr]], #16    \n"
                "eor  v11.16b, v11.16b, v11.16b  \n")";
    }
    std::string body_temp = R"(
                "eor  v30.16b, v30.16b, v30.16b \n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "fmla v8.4s,  v1.4s, v3.s[0]\n"
                "fmla v9.4s,  v1.4s, v3.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v1.4s, v3.s[2]\n"
                "fmla v11.4s, v1.4s, v3.s[3]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "fmla v8.4s,  v1.4s, v3.s[0]\n"
                "fmla v9.4s,  v1.4s, v3.s[1]\n"
                "fmla v10.4s, v1.4s, v3.s[2]\n"
                "fmla v11.4s, v1.4s, v3.s[3]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "6:\n"
                ${GenAsmFloat(v8, v9, v10, v11, v30)}
                
                STORE_C

                : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
                  [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
                  [ output0 ] "+r"(output0), [ n_remain ] "+r"(n_remain)
                :
                : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v30", "v31",
                  "cc", "memory");
            #undef STORE_C
        })";
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
    const int pack_mk = 4;
    const int pack_m = 8;
    const int m_stride = pack_m * pack_mk;
    const int min_m_stride = pack_mk * pack_mk;
    int y = 0;
    for (; y + 7 < ymax; y += pack_m) {
        const float* inptr0 = inptr + y / pack_mk * ldin;
        const float* inptr1 = inptr0 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        int k = (kmax);
        for (; k > 3; k -= pack_mk) {
            interleave_2x4_4_s(inptr0, inptr1, outptr);
            outptr += m_stride;
            inptr0 += min_m_stride;
            inptr1 += min_m_stride;
        }
    }
    for (; y < ymax; y += pack_mk) {
        const float* inptr0 = inptr + y / pack_mk * ldin;
        prefetch_2x(inptr0);
        int K = (kmax);
        for (; K > 3; K -= pack_mk) {
            interleave_1x4_4_s(inptr0, outptr);
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
        float* outptr_base = outptr;
        float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* temp_inptr = inptr + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
            prefetch_3x(temp_inptr);

            int x = x0;
            float* temp_outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                float* outptr_interleave = temp_outptr;
                transpose_1x12_4_s(temp_inptr, outptr_interleave);
                temp_outptr += ksize12;
                temp_inptr += 4 * 12;
            }
            temp_outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                float* outptr_interleave = temp_outptr;
                asm volatile(
                        "ld4 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[inptr0]], #64\n"
                        "st1 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[outptr0]]\n"
                        : [ inptr0 ] "+r"(temp_inptr), [ outptr0 ] "+r"(outptr_interleave)
                        :
                        : "v0", "v1", "v2", "v3", "memory");
                temp_outptr += ksize4;
            }
            if (x < xmax) {
                memcpy(tmpbuff, temp_inptr, sizeof(float) * (xmax - x) * PACK_C_SIZE);
                float* outptr_interleave = temp_outptr;
                const float* tmp_ptr = &tmpbuff[0];
                asm volatile(
                        "ld4 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[inptr0]], #64\n"
                        "st1 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[outptr0]]\n"
                        :
                        [ inptr0 ] "+r"(tmp_ptr), [ outptr0 ] "+r"(outptr_interleave)
                        :
                        : "v0", "v1", "v2", "v3", "memory");
                temp_outptr += ksize4;
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
        const int m_block = 8;
        const int m_block_4 = 4;
        const int n_block = 12;
        const int pack_mk = 4;
        const int K12 = K * 12;
        const int K8 = K * 8;
        const int K4 = K * 4;
        size_t m = 0;        
        for (; m + m_block <= M; m += m_block) {
            float* output = C + (m / pack_mk * LDC);

            size_t n = 0;
            const float* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_8x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr);
                output += n_block * pack_mk;
                cur_pack_b += K12;
            }

            for (; n < N; n += 4) {                
                kern_8x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, N - n > 4 ? 4 : N - n);
                output += 4 * pack_mk;
                cur_pack_b += K4;
            }
            pack_a += K8;
            bias_ptr += m_block;
        }        
        for (; m < M; m += m_block_4) {
            float* output = C + (m / pack_mk * LDC);
            size_t n = 0;
            const float* cur_pack_b = pack_b;
            for (; n + n_block - 1 < N; n += n_block) {                
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
            bias_ptr += m_block_4;
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

std::string MatmulM8N12MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Arm64_fp32_m8_n12_mk4_matmul";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    return ss.str();
}

std::vector<KernelObj> MatmulM8N12MK4Kernel::GetDependInternalSymbol(
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

std::string MatmulM8N12MK4Kernel::GetKernelBody(TContext* ctx) const {
    auto postprocess_pair = gen_postprocess_inline(ctx);
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <arm_neon.h>\n";
    writer << "#include <math.h>\n";
    writer << prefetch();
    writer << transpose_1x12();
    writer << kern_8x12(ctx);
    writer << kern_8x4(ctx);
    writer << kern_4x12(ctx);
    writer << kern_4x4(ctx);
    writer << interleave_2x4_4_s();
    writer << interleave_1x4_4_s();
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

std::string MatmulM8N12MK4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulM8N12MK4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
