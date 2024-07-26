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
std::string interleave_8x1_4_s() {
    return std::string{
            R"(
    static inline void interleave_8x1_4_s(
        const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2,
        const int8_t* inptr3, const int8_t* inptr4, const int8_t* inptr5,
        const int8_t* inptr6, const int8_t* inptr7, int8_t* outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "ld1 {v2.4s}, [%[inptr2]], #16\n"  // d2 = C0C1C2C3
            "ld1 {v3.4s}, [%[inptr3]], #16\n"  // d3 = D0D1D2D3
            "zip1 v8.4s, v0.4s, v2.4s\n"       // d8 = A0C0A1C1
            "zip2 v9.4s, v0.4s, v2.4s\n"       // d9 = A2C2A3C3
            "zip1 v10.4s, v1.4s, v3.4s\n"      // d10 = B0D0B1D1
            "zip2 v11.4s, v1.4s, v3.4s\n"      // d11 = B2D2B3D3
            "zip1 v12.4s, v8.4s, v10.4s\n"     // d12 = A0B0C0D0
            "zip2 v13.4s, v8.4s, v10.4s\n"     // d13 = A1B1C1D1
            "zip1 v14.4s, v9.4s, v11.4s\n"     // d14 = A2B2C2D2
            "zip2 v15.4s, v9.4s, v11.4s\n"     // d15 = A3B3C3D3

            "ld1 {v4.4s}, [%[inptr4]], #16\n"  // d4 = E0E1E2E3
            "ld1 {v5.4s}, [%[inptr5]], #16\n"  // d5 = F0F1F2F3
            "ld1 {v6.4s}, [%[inptr6]], #16\n"  // d6 = G0G1G2G3
            "ld1 {v7.4s}, [%[inptr7]], #16\n"  // d7 = H0H1H2H3
            "zip1 v16.4s, v4.4s, v6.4s\n"      // d16 = E0G0E1G1
            "zip2 v17.4s, v4.4s, v6.4s\n"      // d17 = E2G2E3G3
            "zip1 v18.4s, v5.4s, v7.4s\n"      // d18 = F0H0F1H1
            "zip2 v19.4s, v5.4s, v7.4s\n"      // d19 = F2H2F3H3
            "zip1 v20.4s, v16.4s, v18.4s\n"    // d20 = E0F0G0H0
            "zip2 v21.4s, v16.4s, v18.4s\n"    // d21 = E1F1G1H1
            "zip1 v22.4s, v17.4s, v19.4s\n"    // d22 = E2F2G2H2
            "zip2 v23.4s, v17.4s, v19.4s\n"    // d23 = E3F3G3H3

            "st1 {v12.4s}, [%[outptr]], #16\n"
            "st1 {v20.4s}, [%[outptr]], #16\n"
            "st1 {v13.4s}, [%[outptr]], #16\n"
            "st1 {v21.4s}, [%[outptr]], #16\n"
            "st1 {v14.4s}, [%[outptr]], #16\n"
            "st1 {v22.4s}, [%[outptr]], #16\n"
            "st1 {v15.4s}, [%[outptr]], #16\n"
            "st1 {v23.4s}, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
              [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "cc", "memory");
})"};
}

std::string interleave_4x1_4_s() {
    return std::string{
            R"(
    static inline void interleave_4x1_4_s(
        const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2,
        const int8_t* inptr3, int8_t* outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "ld1 {v2.4s}, [%[inptr2]], #16\n"  // d2 = C0C1C2C3
            "ld1 {v3.4s}, [%[inptr3]], #16\n"  // d3 = D0D1D2D3
            "zip1 v8.4s, v0.4s, v2.4s\n"       // d8 = A0C0A1C1
            "zip2 v9.4s, v0.4s, v2.4s\n"       // d9 = A2C2A3C3
            "zip1 v10.4s, v1.4s, v3.4s\n"      // d10 = B0D0B1D1
            "zip2 v11.4s, v1.4s, v3.4s\n"      // d11 = B2D2B3D3
            "zip1 v12.4s, v8.4s, v10.4s\n"     // d12 = A0B0C0D0
            "zip2 v13.4s, v8.4s, v10.4s\n"     // d13 = A1B1C1D1
            "zip1 v14.4s, v9.4s, v11.4s\n"     // d14 = A2B2C2D2
            "zip2 v15.4s, v9.4s, v11.4s\n"     // d15 = A3B3C3D3

            "st1 {v12.4s}, [%[outptr]], #16\n"
            "st1 {v13.4s}, [%[outptr]], #16\n"
            "st1 {v14.4s}, [%[outptr]], #16\n"
            "st1 {v15.4s}, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
              [inptr3] "+r"(inptr3), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "cc", "memory");
})"};
}

std::string interleave_8() {
    return std::string{R"(
    static inline void interleave_helper(
            const int8_t* inptr, int8_t* outptr, int unroll_k, int ksize) {
        int k = 0;
        for (; k < ksize; k++) {
            *outptr++ = *inptr++;
        }
        for (; k < unroll_k; k++) {
            *outptr++ = 0;
        }
    }
    static inline void interleave_8(
            const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2, const int8_t* inptr3,
            const int8_t* inptr4, const int8_t* inptr5, const int8_t* inptr6, const int8_t* inptr7,
            int8_t* outptr, int unroll_k, int ksize) {
        for (int k = 0; k < ksize; k += unroll_k) {
            int size = unroll_k < ksize - k ? unroll_k : ksize - k;
            interleave_helper(inptr0, outptr, unroll_k, size);
            inptr0 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr1, outptr, unroll_k, size);
            inptr1 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr2, outptr, unroll_k, size);
            inptr2 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr3, outptr, unroll_k, size);
            inptr3 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr4, outptr, unroll_k, size);
            inptr4 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr5, outptr, unroll_k, size);
            inptr5 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr6, outptr, unroll_k, size);
            inptr6 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr7, outptr, unroll_k, size);
            inptr7 += unroll_k;
            outptr += unroll_k;
        }
    }
    )"};
}

std::string interleave_4() {
    return std::string{R"(
    static inline void interleave_4(
            const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2, const int8_t* inptr3,
            int8_t* outptr, int unroll_k, int ksize) {
        for (int k = 0; k < ksize; k += unroll_k) {
            int size = unroll_k < ksize - k ? unroll_k : ksize - k;
            interleave_helper(inptr0, outptr, unroll_k, size);
            inptr0 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr1, outptr, unroll_k, size);
            inptr1 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr2, outptr, unroll_k, size);
            inptr2 += unroll_k;
            outptr += unroll_k;
            interleave_helper(inptr3, outptr, unroll_k, size);
            inptr3 += unroll_k;
            outptr += unroll_k;
        }
    }
    )"};
}

std::string transpose_12x4_1_b() {
    return std::string{R"(
static inline void transpose_12x4_1_b(
        const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2, const int8_t* inptr3,
        int8_t* outptr) {
    asm volatile(
            "ldr q0,  [%[inptr0]], #12\n"  // A1A2A3A4A5A6A7A8A9A10A11A12A13A14A15A16
            "ldr q1,  [%[inptr1]], #12\n"  // B1B2B3B4B5B6B7B8B9B10B11B12B13B14B15B16
            "ldr q2,  [%[inptr2]], #12\n"  // C1C2C3C4C5C6C7C8C9C10C11C12C13C14C15C16
            //! \warning the last inptr3 may less than 16bytes, so we should
            //! split read it
            "ldr d3,  [%[inptr3]], #8\n"  // D1D2D3D4D5D6D7D8D9D10D11D12D13D14D15D16
            "ldr w1, [%[inptr3]], #4\n"
            "ins v3.s[2], w1\n"

            "trn1 v4.16b, v0.16b, v1.16b\n"  // v4: A1B1A3B3....
            "trn2 v5.16b, v0.16b, v1.16b\n"  // v5: A2B2A4B4....
            "trn1 v6.16b, v2.16b, v3.16b\n"  // v6: C1D1C3D3....
            "trn2 v7.16b, v2.16b, v3.16b\n"  // v7: C2D2C4D4....

            "trn1 v8.8h, v4.8h, v6.8h\n"   // v8: A1B1C1D1A5B5C5D5...
            "trn2 v9.8h, v4.8h, v6.8h\n"   // v9: A3B3C3D3A7B7C7D7...
            "trn1 v10.8h, v5.8h, v7.8h\n"  // v10: A2B2C2D2A6B6C6D6...
            "trn2 v11.8h, v5.8h, v7.8h\n"  // v11: A4B4C4D4A8B8C8D8...

            //! ABCD=E then
            //! v8: E1E5E9E13 v10: E2E6E10E14 v9: E3E7E11E15 v11:
            //! E4E8E12E16
            "zip1 v12.4s, v8.4s, v10.4s\n"   // v12: E1E2E5E6
            "zip2 v13.4s, v8.4s, v10.4s\n"   // v13: E9E10E13E14
            "zip1 v14.4s, v9.4s, v11.4s\n"   // v14: E3E4E7E8
            "zip2 v15.4s, v9.4s, v11.4s\n"   // v15: E11E12E15E16
            "zip1 v17.2d, v12.2d, v14.2d\n"  // v17: E1E2E3E4
            "zip2 v18.2d, v12.2d, v14.2d\n"  // v18: E5E6E7E8
            "zip1 v19.2d, v13.2d, v15.2d\n"  // v19: E8E10E11E12
            "zip2 v20.2d, v13.2d, v15.2d\n"  // v19: E13E14E15E16

            "stp q17, q18, [%[outptr]], #32\n"
            "str q19, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
              [inptr3] "+r"(inptr3), [outptr] "+r"(outptr)
            :
            : "w1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "memory");
}
    )"};
}

std::string transpose_4() {
    return std::string{R"(
static inline void transpose_4(
        const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2, const int8_t* inptr3,
        int8_t* outptr, int interleave, int size) {
    TINYNN_ASSERT(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
    }
    for (; i < interleave; i++) {
        *outptr++ = 0;
        *outptr++ = 0;
        *outptr++ = 0;
        *outptr++ = 0;
    }
}
    )"};
}

std::string prefetch() {
    return R"(
        #define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_2x_f32() +
           KernelGen::ArmCommon::gen_common_prefetch_3x_f32();
}

static std::string kern_8x12(
        TContext* ctx, const std::string& dst_specifier,
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
                          const int32_t* bias_ptr, float src_scale,  float dst_scale) {
        K /= 4;
        const int8_t* a_ptr = packA;
        const int8_t* b_ptr = packB;
        ${dst_specifier}* output0 = output;
        ${dst_specifier}* output1 = output0 + LDC;
        ${dst_specifier}* output2 = output1 + LDC;
        ${dst_specifier}* output3 = output2 + LDC;
        ${dst_specifier}* output4 = output3 + LDC;
        ${dst_specifier}* output5 = output4 + LDC;
        ${dst_specifier}* output6 = output5 + LDC;
        ${dst_specifier}* output7 = output6 + LDC;
        float* src_scale_ptr = &src_scale;
        float* dst_scale_ptr = &dst_scale; 
        const float inv_6 = 1.f / 6.f;
        const float* inv_6_ptr = &inv_6;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()");
    //! if convolution with bias
    if (with_bias) {
        writer << R"(
            "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
            "dup    v8.4s, v6.s[0]             \n"
            "dup    v9.4s, v6.s[0]             \n"
            "dup    v10.4s, v6.s[0]             \n"
            "prfm pstl1keep, [%[output0]]\n"
            "dup    v11.4s, v6.s[1]             \n"
            "dup    v12.4s, v6.s[1]             \n"
            "dup    v13.4s, v6.s[1]             \n"
            "prfm pstl1keep, [%[output1]]\n"
            "dup    v14.4s, v6.s[2]             \n"
            "dup    v15.4s, v6.s[2]             \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "dup    v16.4s, v6.s[2]             \n"
            "ld1 {v7.4s}, [%[bias_ptr]], #16\n"
            "dup    v17.4s, v6.s[3]             \n"
            "dup    v18.4s, v6.s[3]             \n"
            "dup    v19.4s, v6.s[3]             \n"
            "dup    v20.4s, v7.s[0]             \n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "dup    v21.4s, v7.s[0]             \n"
            "dup    v22.4s, v7.s[0]             \n"
            "dup    v23.4s, v7.s[1]             \n"
            "ld1 {v4.4s}, [%[b_ptr]], #16\n"
            "dup    v24.4s, v7.s[1]             \n"
            "dup    v25.4s, v7.s[1]             \n"
            "dup    v26.4s, v7.s[2]             \n"
            "dup    v27.4s, v7.s[2]             \n"
            "dup    v28.4s, v7.s[2]             \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
            "dup    v29.4s, v7.s[3]             \n"
            "dup    v30.4s, v7.s[3]             \n"
            "dup    v31.4s, v7.s[3]             \n")";

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
            "sdot v8.4s , v2.16b,  v0.4b[0]\n"
            "sdot v11.4s, v2.16b,  v0.4b[1]\n"
            "sdot v14.4s, v2.16b,  v0.4b[2]\n"
            "sdot v17.4s, v2.16b,  v0.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v9.4s , v3.16b,  v0.4b[0]\n"
            "sdot v12.4s, v3.16b,  v0.4b[1]\n"
            "sdot v15.4s, v3.16b,  v0.4b[2]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v18.4s, v3.16b,  v0.4b[3]\n"
            "sdot v10.4s, v4.16b,  v0.4b[0]\n"
            "sdot v13.4s, v4.16b,  v0.4b[1]\n"
            "ld1 {v6.4s}, [%[b_ptr]], #16\n"
            "sdot v16.4s, v4.16b,  v0.4b[2]\n"
            "sdot v19.4s, v4.16b,  v0.4b[3]\n"
            "ld1 {v7.4s}, [%[b_ptr]], #16\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v20.4s, v2.16b, v1.4b[0]\n"
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"
            "sdot v21.4s, v3.16b, v1.4b[0]\n"
            "sdot v24.4s, v3.16b, v1.4b[1]\n"
            "sdot v27.4s, v3.16b, v1.4b[2]\n"
            "sdot v30.4s, v3.16b, v1.4b[3]\n"
            "sdot v22.4s, v4.16b, v1.4b[0]\n"
            "sdot v25.4s, v4.16b, v1.4b[1]\n"
            "sdot v28.4s, v4.16b, v1.4b[2]\n"
            "sdot v31.4s, v4.16b, v1.4b[3]\n"

            "sdot v8.4s , v5.16b,  v0.4b[0]\n"
            "sdot v11.4s, v5.16b,  v0.4b[1]\n"
            "sdot v14.4s, v5.16b,  v0.4b[2]\n"
            "sdot v17.4s, v5.16b,  v0.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v9.4s , v6.16b,  v0.4b[0]\n"
            "sdot v12.4s, v6.16b,  v0.4b[1]\n"
            "sdot v15.4s, v6.16b,  v0.4b[2]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v18.4s, v6.16b,  v0.4b[3]\n"
            "sdot v10.4s, v7.16b,  v0.4b[0]\n"
            "sdot v13.4s, v7.16b,  v0.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v16.4s, v7.16b,  v0.4b[2]\n"
            "sdot v19.4s, v7.16b,  v0.4b[3]\n"
            "ld1 {v4.4s}, [%[b_ptr]], #16\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v20.4s, v5.16b, v1.4b[0]\n"
            "sdot v23.4s, v5.16b, v1.4b[1]\n"
            "sdot v26.4s, v5.16b, v1.4b[2]\n"
            "sdot v29.4s, v5.16b, v1.4b[3]\n"
            "sdot v21.4s, v6.16b, v1.4b[0]\n"
            "sdot v24.4s, v6.16b, v1.4b[1]\n"
            "subs %w[K], %w[K], #1\n"
            "sdot v27.4s, v6.16b, v1.4b[2]\n"
            "sdot v30.4s, v6.16b, v1.4b[3]\n"
            "sdot v22.4s, v7.16b, v1.4b[0]\n"
            "sdot v25.4s, v7.16b, v1.4b[1]\n"
            "sdot v28.4s, v7.16b, v1.4b[2]\n"
            "sdot v31.4s, v7.16b, v1.4b[3]\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
)";
    std::string tail_temp = R"(
            // Even tail
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"
            "sdot v9.4s, v3.16b, v0.4b[0]\n"
            "sdot v12.4s, v3.16b, v0.4b[1]\n"
            "sdot v15.4s, v3.16b, v0.4b[2]\n"
            "sdot v18.4s, v3.16b, v0.4b[3]\n"
            "sdot v10.4s, v4.16b, v0.4b[0]\n"
            "sdot v13.4s, v4.16b, v0.4b[1]\n"
            "sdot v16.4s, v4.16b, v0.4b[2]\n"
            "sdot v19.4s, v4.16b, v0.4b[3]\n"

            "sdot v20.4s, v2.16b, v1.4b[0]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"
            "sdot v21.4s, v3.16b, v1.4b[0]\n"
            "sdot v24.4s, v3.16b, v1.4b[1]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v27.4s, v3.16b, v1.4b[2]\n"
            "sdot v30.4s, v3.16b, v1.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v22.4s, v4.16b, v1.4b[0]\n"

            ${gen_postprocess_reg_init}

            "sdot v25.4s, v4.16b, v1.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v28.4s, v4.16b, v1.4b[2]\n"
            "sdot v31.4s, v4.16b, v1.4b[3]\n"

            "sdot v8.4s,  v5.16b, v0.4b[0]\n"
            "sdot v11.4s,  v5.16b, v0.4b[1]\n"
            "sdot v14.4s, v5.16b, v0.4b[2]\n"
            "sdot v17.4s, v5.16b, v0.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v9.4s, v2.16b, v0.4b[0]\n"
            
            "sdot v12.4s, v2.16b, v0.4b[1]\n"

            "sdot v15.4s, v2.16b, v0.4b[2]\n"
            
            "sdot v18.4s, v2.16b, v0.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}

            "sdot v10.4s, v3.16b, v0.4b[0]\n"
            
            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            
            
            "sdot v13.4s, v3.16b, v0.4b[1]\n"
            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            "sdot v16.4s, v3.16b, v0.4b[2]\n"
            
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            "sdot v19.4s, v3.16b, v0.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v12, output1, 4)}

            "sdot v20.4s, v5.16b, v1.4b[0]\n"
            
            ${GenAsmGenAsmQuantStore(v13, output1, 8)}
            "sdot v23.4s, v5.16b, v1.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            "sdot v26.4s, v5.16b, v1.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v15, output2, 4)}
            "sdot v29.4s, v5.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v16, output2, 8)}
            "sdot v21.4s, v2.16b, v1.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            "sdot v24.4s, v2.16b, v1.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v18, output3, 4)}
            "sdot v27.4s, v2.16b, v1.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v19, output3, 8)}
            "sdot v30.4s, v2.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v20, output4, 0)}
            "sdot v22.4s, v3.16b, v1.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v21, output4, 4)}
            "sdot v25.4s, v3.16b, v1.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v22, output4, 8)}
            "sdot v28.4s, v3.16b, v1.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v23, output5, 0)}
            "sdot v31.4s, v3.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v24, output5, 4)}
            ${GenAsmGenAsmQuantStore(v25, output5, 8)}
            ${GenAsmGenAsmQuantStore(v26, output6, 0)}
            ${GenAsmGenAsmQuantStore(v27, output6, 4)}
            ${GenAsmGenAsmQuantStore(v28, output6, 8)}
            ${GenAsmGenAsmQuantStore(v29, output7, 0)}
            ${GenAsmGenAsmQuantStore(v30, output7, 4)}
            ${GenAsmGenAsmQuantStore(v31, output7, 8)}
            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            ${gen_postprocess_reg_init}
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"

            "sdot v9.4s, v3.16b, v0.4b[0]\n"

            "sdot v12.4s, v3.16b, v0.4b[1]\n"

            "sdot v15.4s, v3.16b, v0.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            "sdot v18.4s, v3.16b, v0.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            "sdot v10.4s, v4.16b, v0.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            "sdot v13.4s, v4.16b, v0.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            "sdot v16.4s, v4.16b, v0.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v12, output1, 4)}
            "sdot v19.4s, v4.16b, v0.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v13, output1, 8)}

            "sdot v20.4s, v2.16b, v1.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            "sdot v23.4s, v2.16b, v1.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v15, output2, 4)}
            "sdot v26.4s, v2.16b, v1.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v16, output2, 8)}
            "sdot v29.4s, v2.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            "sdot v21.4s, v3.16b, v1.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v18, output3, 4)}
            "sdot v24.4s, v3.16b, v1.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v19, output3, 8)}
            "sdot v27.4s, v3.16b, v1.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v20, output4, 0)}
            "sdot v30.4s, v3.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v21, output4, 4)}
            "sdot v22.4s, v4.16b, v1.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v22, output4, 8)}
            "sdot v25.4s, v4.16b, v1.4b[1]\n"

            ${GenAsmGenAsmQuantStore(v23, output5, 0)}
            "sdot v28.4s, v4.16b, v1.4b[2]\n"

            ${GenAsmGenAsmQuantStore(v24, output5, 4)}
            ${GenAsmGenAsmQuantStore(v25, output5, 8)}
            ${GenAsmGenAsmQuantStore(v26, output6, 0)}
            ${GenAsmGenAsmQuantStore(v27, output6, 4)}
            "sdot v31.4s, v4.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v28, output6, 8)}
            ${GenAsmGenAsmQuantStore(v29, output7, 0)}
            ${GenAsmGenAsmQuantStore(v30, output7, 4)}
            ${GenAsmGenAsmQuantStore(v31, output7, 8)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1), [ output2 ] "+r"(output2), [ output3 ] "+r"(output3),
              [ output4 ] "+r"(output4), [ output5 ] "+r"(output5), [ output6 ] "+r"(output6), [ output7 ] "+r"(output7),
              [src_scale_ptr] "+r" (src_scale_ptr), [dst_scale_ptr] "+r" (dst_scale_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
        })";

    std::string tail_many_reg_temp = R"(
            // Even tail
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"
            "sdot v9.4s, v3.16b, v0.4b[0]\n"
            "sdot v12.4s, v3.16b, v0.4b[1]\n"
            "sdot v15.4s, v3.16b, v0.4b[2]\n"
            "sdot v18.4s, v3.16b, v0.4b[3]\n"
            "sdot v10.4s, v4.16b, v0.4b[0]\n"
            "sdot v13.4s, v4.16b, v0.4b[1]\n"
            "sdot v16.4s, v4.16b, v0.4b[2]\n"
            "sdot v19.4s, v4.16b, v0.4b[3]\n"

            "sdot v20.4s, v2.16b, v1.4b[0]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"
            "sdot v21.4s, v3.16b, v1.4b[0]\n"
            "sdot v24.4s, v3.16b, v1.4b[1]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "sdot v27.4s, v3.16b, v1.4b[2]\n"
            "sdot v30.4s, v3.16b, v1.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v22.4s, v4.16b, v1.4b[0]\n"


            "sdot v25.4s, v4.16b, v1.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v28.4s, v4.16b, v1.4b[2]\n"
            "sdot v31.4s, v4.16b, v1.4b[3]\n"

            "sdot v8.4s,  v5.16b, v0.4b[0]\n"
            "sdot v11.4s,  v5.16b, v0.4b[1]\n"
            "sdot v14.4s, v5.16b, v0.4b[2]\n"
            "sdot v17.4s, v5.16b, v0.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v9.4s, v2.16b, v0.4b[0]\n"
            "sdot v12.4s, v2.16b, v0.4b[1]\n"
            "sdot v15.4s, v2.16b, v0.4b[2]\n"
            "sdot v18.4s, v2.16b, v0.4b[3]\n"
            "sdot v10.4s, v3.16b, v0.4b[0]\n"
            "sdot v13.4s, v3.16b, v0.4b[1]\n"
            "sdot v16.4s, v3.16b, v0.4b[2]\n"
            "sdot v19.4s, v3.16b, v0.4b[3]\n"

            "sdot v20.4s, v5.16b, v1.4b[0]\n"
            "sdot v23.4s, v5.16b, v1.4b[1]\n"
            "sdot v26.4s, v5.16b, v1.4b[2]\n"
            "sdot v29.4s, v5.16b, v1.4b[3]\n"
            "sdot v21.4s, v2.16b, v1.4b[0]\n"
            "sdot v24.4s, v2.16b, v1.4b[1]\n"
            "sdot v27.4s, v2.16b, v1.4b[2]\n"
            "sdot v30.4s, v2.16b, v1.4b[3]\n"
            "sdot v22.4s, v3.16b, v1.4b[0]\n"
            "sdot v25.4s, v3.16b, v1.4b[1]\n"
            "sdot v28.4s, v3.16b, v1.4b[2]\n"
            "sdot v31.4s, v3.16b, v1.4b[3]\n"

            ${gen_postprocess_reg_init}
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            ${GenAsmGenAsmQuantStore(v12, output1, 4)}
            ${GenAsmGenAsmQuantStore(v13, output1, 8)}
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            ${GenAsmGenAsmQuantStore(v15, output2, 4)}
            ${GenAsmGenAsmQuantStore(v16, output2, 8)}
            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            ${GenAsmGenAsmQuantStore(v18, output3, 4)}
            ${GenAsmGenAsmQuantStore(v19, output3, 8)}
            ${GenAsmGenAsmQuantStore(v20, output4, 0)}
            ${GenAsmGenAsmQuantStore(v21, output4, 4)}
            ${GenAsmGenAsmQuantStore(v22, output4, 8)}
            ${GenAsmGenAsmQuantStore(v23, output5, 0)}
            ${GenAsmGenAsmQuantStore(v24, output5, 4)}
            ${GenAsmGenAsmQuantStore(v25, output5, 8)}
            ${GenAsmGenAsmQuantStore(v26, output6, 0)}
            ${GenAsmGenAsmQuantStore(v27, output6, 4)}
            ${GenAsmGenAsmQuantStore(v28, output6, 8)}
            ${GenAsmGenAsmQuantStore(v29, output7, 0)}
            ${GenAsmGenAsmQuantStore(v30, output7, 4)}
            ${GenAsmGenAsmQuantStore(v31, output7, 8)}
            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"
            "sdot v9.4s, v3.16b, v0.4b[0]\n"
            "sdot v12.4s, v3.16b, v0.4b[1]\n"
            "sdot v15.4s, v3.16b, v0.4b[2]\n"
            "sdot v18.4s, v3.16b, v0.4b[3]\n"
            "sdot v10.4s, v4.16b, v0.4b[0]\n"
            "sdot v13.4s, v4.16b, v0.4b[1]\n"
            "sdot v16.4s, v4.16b, v0.4b[2]\n"
            "sdot v19.4s, v4.16b, v0.4b[3]\n"

            "sdot v20.4s, v2.16b, v1.4b[0]\n"
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"
            "sdot v21.4s, v3.16b, v1.4b[0]\n"
            "sdot v24.4s, v3.16b, v1.4b[1]\n"
            "sdot v27.4s, v3.16b, v1.4b[2]\n"
            "sdot v30.4s, v3.16b, v1.4b[3]\n"
            "sdot v22.4s, v4.16b, v1.4b[0]\n"
            "sdot v25.4s, v4.16b, v1.4b[1]\n"
            "sdot v28.4s, v4.16b, v1.4b[2]\n"
            "sdot v31.4s, v4.16b, v1.4b[3]\n"

            ${gen_postprocess_reg_init}
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            ${GenAsmGenAsmQuantStore(v12, output1, 4)}
            ${GenAsmGenAsmQuantStore(v13, output1, 8)}
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            ${GenAsmGenAsmQuantStore(v15, output2, 4)}
            ${GenAsmGenAsmQuantStore(v16, output2, 8)}
            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            ${GenAsmGenAsmQuantStore(v18, output3, 4)}
            ${GenAsmGenAsmQuantStore(v19, output3, 8)}
            ${GenAsmGenAsmQuantStore(v20, output4, 0)}
            ${GenAsmGenAsmQuantStore(v21, output4, 4)}
            ${GenAsmGenAsmQuantStore(v22, output4, 8)}
            ${GenAsmGenAsmQuantStore(v23, output5, 0)}
            ${GenAsmGenAsmQuantStore(v24, output5, 4)}
            ${GenAsmGenAsmQuantStore(v25, output5, 8)}
            ${GenAsmGenAsmQuantStore(v26, output6, 0)}
            ${GenAsmGenAsmQuantStore(v27, output6, 4)}
            ${GenAsmGenAsmQuantStore(v28, output6, 8)}
            ${GenAsmGenAsmQuantStore(v29, output7, 0)}
            ${GenAsmGenAsmQuantStore(v30, output7, 4)}
            ${GenAsmGenAsmQuantStore(v31, output7, 8)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1), [ output2 ] "+r"(output2), [ output3 ] "+r"(output3),
              [ output4 ] "+r"(output4), [ output5 ] "+r"(output5), [ output6 ] "+r"(output6), [ output7 ] "+r"(output7),
              [src_scale_ptr] "+r" (src_scale_ptr), [inv6_ptr] "+r" (inv_6_ptr), [dst_scale_ptr] "+r" (dst_scale_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
        })";
    if (nonline_mode == "H_SWISH") {
        std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
                {"v0", "v1", "v2", "v3", "v7"}, nonline_mode,
                {"inv6_ptr", "src_scale_ptr"});
        writer << StringTemplate::StringTemplateArgs()
                          .add("dst_specifier", dst_specifier)
                          .add("GenAsmGenAsmQuantStore",
                               [=](std::vector<std::string> args) {
                                   CC_ASSERT(args.size() == 3);
                                   return activation_gen->GenAsmQuantStore(
                                           {args[0]}, "v7", "dst_scale_ptr",
                                           "src_scale_ptr", args[1], std::stoi(args[2]),
                                           dst_specifier,
                                           {"v0", "v1", "v2", "v3", "v4"},
                                           nonline_mode);
                               })
                          .add("gen_postprocess_reg_init", postprocess_reg_init)
                          .render(tail_many_reg_temp);
    } else {
        std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
                {"v7", "v6"}, nonline_mode, {"src_scale_ptr"});
        writer << StringTemplate::StringTemplateArgs()
                          .add("dst_specifier", dst_specifier)
                          .add("GenAsmGenAsmQuantStore",
                               [=](std::vector<std::string> args) {
                                   CC_ASSERT(args.size() == 3);
                                   return activation_gen->GenAsmQuantStore(
                                           {args[0]}, "v6", "dst_scale_ptr",
                                           "src_scale_ptr", args[1], std::stoi(args[2]),
                                           dst_specifier, {"v7"}, nonline_mode);
                               })
                          .add("gen_postprocess_reg_init", postprocess_reg_init)
                          .render(tail_temp);
    }
    return writer.str();
}

static std::string kern_4x12(
        TContext* ctx, const std::string& dst_specifier,
        const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
    __attribute__((target("dotprod")))
    static inline void kern_4x12_bias_relu(const int8_t* packA, const int8_t* packB,
                          int K, ${dst_specifier}* output, int LDC,
                          const int32_t* bias_ptr, float src_scale,  float dst_scale, int m_remain) {
        K /= 4;
        const int8_t* a_ptr = packA;
        const int8_t* b_ptr = packB;
        ${dst_specifier}* output0 = output;
        ${dst_specifier}* output1 = output0 + LDC;
        ${dst_specifier}* output2 = output1 + LDC;
        ${dst_specifier}* output3 = output2 + LDC;
        float* src_scale_ptr = &src_scale;
        float* dst_scale_ptr = &dst_scale; 
        const float inv_6 = 1.f / 6.f;
        const float* inv_6_ptr = &inv_6;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()");
    //! if convolution with bias
    if (with_bias) {
        writer << R"(
            "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
            "dup    v8.4s, v6.s[0]             \n"
            "dup    v9.4s, v6.s[0]             \n"
            "dup    v10.4s, v6.s[0]             \n"
            "prfm pstl1keep, [%[output0]]\n"
            "dup    v11.4s, v6.s[1]             \n"
            "dup    v12.4s, v6.s[1]             \n"
            "dup    v13.4s, v6.s[1]             \n"
            "prfm pstl1keep, [%[output1]]\n"
            "dup    v14.4s, v6.s[2]             \n"
            "dup    v15.4s, v6.s[2]             \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "dup    v16.4s, v6.s[2]             \n"
            "dup    v17.4s, v6.s[3]             \n"
            "dup    v18.4s, v6.s[3]             \n"
            "dup    v19.4s, v6.s[3]             \n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "ld1 {v4.4s}, [%[b_ptr]], #16\n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n")";

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
            "ld1 {v3.4s}, [%[b_ptr]], #16    \n"
            "ld1 {v4.4s}, [%[b_ptr]], #16    \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16    \n")";
    }
    writer << R"(
            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "sdot v8.4s , v2.16b,  v0.4b[0]\n"
            "sdot v11.4s, v2.16b,  v0.4b[1]\n"
            "sdot v14.4s, v2.16b,  v0.4b[2]\n"
            "sdot v17.4s, v2.16b,  v0.4b[3]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v9.4s , v3.16b,  v0.4b[0]\n"
            "sdot v12.4s, v3.16b,  v0.4b[1]\n"
            "sdot v15.4s, v3.16b,  v0.4b[2]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v18.4s, v3.16b,  v0.4b[3]\n"
            "sdot v10.4s, v4.16b,  v0.4b[0]\n"
            "sdot v13.4s, v4.16b,  v0.4b[1]\n"
            "ld1 {v6.4s}, [%[b_ptr]], #16\n"
            "sdot v16.4s, v4.16b,  v0.4b[2]\n"
            "sdot v19.4s, v4.16b,  v0.4b[3]\n"
            "ld1 {v7.4s}, [%[b_ptr]], #16\n"

            "sdot v8.4s , v5.16b,  v1.4b[0]\n"
            "sdot v11.4s, v5.16b,  v1.4b[1]\n"
            "sdot v14.4s, v5.16b,  v1.4b[2]\n"
            "sdot v17.4s, v5.16b,  v1.4b[3]\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"
            "sdot v9.4s , v6.16b,  v1.4b[0]\n"
            "sdot v12.4s, v6.16b,  v1.4b[1]\n"
            "sdot v15.4s, v6.16b,  v1.4b[2]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "sdot v18.4s, v6.16b,  v1.4b[3]\n"
            "sdot v10.4s, v7.16b,  v1.4b[0]\n"
            "sdot v13.4s, v7.16b,  v1.4b[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], #16\n"
            "sdot v16.4s, v7.16b,  v1.4b[2]\n"
            "sdot v19.4s, v7.16b,  v1.4b[3]\n"
            "ld1 {v4.4s}, [%[b_ptr]], #16\n"

            "subs %w[K], %w[K], #1\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
)";
    std::string tail_temp = R"(
            // Even tail
            "sdot v8.4s , v2.16b, v0.4b[0]\n"
            "sdot v11.4s, v2.16b, v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"
            "sdot v9.4s , v3.16b, v0.4b[0]\n"
            "sdot v12.4s, v3.16b, v0.4b[1]\n"
            "sdot v15.4s, v3.16b, v0.4b[2]\n"
            "sdot v18.4s, v3.16b, v0.4b[3]\n"
            "sdot v10.4s, v4.16b, v0.4b[0]\n"
            "sdot v13.4s, v4.16b, v0.4b[1]\n"
            "sdot v16.4s, v4.16b, v0.4b[2]\n"
            "sdot v19.4s, v4.16b, v0.4b[3]\n"

            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"

            ${gen_postprocess_reg_init}

            "ld1 {v3.4s}, [%[b_ptr]], #16\n"

            "sdot v8.4s,  v5.16b, v1.4b[0]\n"
            "sdot v11.4s,  v5.16b, v1.4b[1]\n"
            "sdot v14.4s, v5.16b, v1.4b[2]\n"
            "sdot v17.4s, v5.16b, v1.4b[3]\n"
            "sdot v9.4s, v2.16b, v1.4b[0]\n"
            "sdot v12.4s, v2.16b, v1.4b[1]\n"
            "sdot v15.4s, v2.16b, v1.4b[2]\n"
            "sdot v18.4s, v2.16b, v1.4b[3]\n"
            "sdot v10.4s, v3.16b, v1.4b[0]\n"
            "sdot v13.4s, v3.16b, v1.4b[1]\n"
            "sdot v16.4s, v3.16b, v1.4b[2]\n"
            "sdot v19.4s, v3.16b, v1.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            "cmp %w[m_remain], #2\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            ${GenAsmGenAsmQuantStore(v12, output1, 4)}
            ${GenAsmGenAsmQuantStore(v13, output1, 8)}
            "cmp %w[m_remain], #3\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            ${GenAsmGenAsmQuantStore(v15, output2, 4)}
            ${GenAsmGenAsmQuantStore(v16, output2, 8)}
            "cmp %w[m_remain], #4\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            ${GenAsmGenAsmQuantStore(v18, output3, 4)}
            ${GenAsmGenAsmQuantStore(v19, output3, 8)}

            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s , v2.16b, v0.4b[0]\n"
            ${gen_postprocess_reg_init}
            "sdot v11.4s, v2.16b, v0.4b[1]\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"
            "sdot v9.4s , v3.16b, v0.4b[0]\n"
            "sdot v12.4s, v3.16b, v0.4b[1]\n"
            "sdot v15.4s, v3.16b, v0.4b[2]\n"
            "sdot v18.4s, v3.16b, v0.4b[3]\n"
            "sdot v10.4s, v4.16b, v0.4b[0]\n"
            "sdot v13.4s, v4.16b, v0.4b[1]\n"
            "sdot v16.4s, v4.16b, v0.4b[2]\n"
            "sdot v19.4s, v4.16b, v0.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            ${GenAsmGenAsmQuantStore(v9, output0, 4)}
            ${GenAsmGenAsmQuantStore(v10, output0, 8)}
            "cmp %w[m_remain], #2\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            ${GenAsmGenAsmQuantStore(v12, output1, 4)}
            ${GenAsmGenAsmQuantStore(v13, output1, 8)}
            "cmp %w[m_remain], #3\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            ${GenAsmGenAsmQuantStore(v15, output2, 4)}
            ${GenAsmGenAsmQuantStore(v16, output2, 8)}
            "cmp %w[m_remain], #4\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            ${GenAsmGenAsmQuantStore(v18, output3, 4)}
            ${GenAsmGenAsmQuantStore(v19, output3, 8)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk), [ m_remain ] "+r"(m_remain),
              [ output0 ] "+r"(output0), [ output1 ] "+r"(output1), [ output2 ] "+r"(output2), [ output3 ] "+r"(output3),
              [src_scale_ptr] "+r" (src_scale_ptr), [inv6_ptr] "+r" (inv_6_ptr), [dst_scale_ptr] "+r" (dst_scale_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
        })";

    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v20", "v21", "v22", "v23", "v27"}, nonline_mode,
            {"inv6_ptr", "src_scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 3);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0]}, "v27", "dst_scale_ptr",
                                       "src_scale_ptr", args[1], std::stoi(args[2]),
                                       dst_specifier,
                                       {"v20", "v21", "v22", "v23", "v24", "v25"},
                                       nonline_mode);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(tail_temp);
    return writer.str();
}

static std::string kern_8x4(
        TContext* ctx, const std::string& dst_specifier,
        const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    //! kern_8x4
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
    __attribute__((target("dotprod")))
    static inline void kern_8x4_bias_relu(const int8_t* packA, const int8_t* packB,
                          int K, ${dst_specifier}* output, int LDC,
                          const int32_t* bias_ptr, float src_scale,  float dst_scale, int n_remain) {
        K /= 4;
        const int8_t* a_ptr = packA;
        const int8_t* b_ptr = packB;
        ${dst_specifier}* output0 = output;
        ${dst_specifier}* output1 = output0 + LDC;
        ${dst_specifier}* output2 = output1 + LDC;
        ${dst_specifier}* output3 = output2 + LDC;
        ${dst_specifier}* output4 = output3 + LDC;
        ${dst_specifier}* output5 = output4 + LDC;
        ${dst_specifier}* output6 = output5 + LDC;
        ${dst_specifier}* output7 = output6 + LDC;
        ${dst_specifier} tmp_output0[4];
        ${dst_specifier} tmp_output1[4];
        ${dst_specifier} tmp_output2[4];
        ${dst_specifier} tmp_output3[4];
        ${dst_specifier} tmp_output4[4];
        ${dst_specifier} tmp_output5[4];
        ${dst_specifier} tmp_output6[4];
        ${dst_specifier} tmp_output7[4];
        ${dst_specifier}* tmp_outputs[8] = {tmp_output0, tmp_output1, tmp_output2, tmp_output3, tmp_output4, tmp_output5, tmp_output6, tmp_output7};
        float* src_scale_ptr = &src_scale;
        float* dst_scale_ptr = &dst_scale; 
        const float inv_6 = 1.f / 6.f;
        const float* inv_6_ptr = &inv_6;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()");
    //! if convolution with bias
    if (with_bias) {
        writer << R"(
            "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
            "dup    v8.4s, v6.s[0]             \n"
            "dup    v11.4s, v6.s[1]             \n"
            "dup    v14.4s, v6.s[2]             \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "ld1 {v7.4s}, [%[bias_ptr]], #16\n"
            "dup    v17.4s, v6.s[3]             \n"
            "dup    v20.4s, v7.s[0]             \n"
            "dup    v23.4s, v7.s[1]             \n"
            "dup    v26.4s, v7.s[2]             \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n"
            "dup    v29.4s, v7.s[3]             \n")";

        //! if convolution without bias
    } else {
        writer << R"(
            "eor  v8.16b, v8.16b, v8.16b     \n"
            "prfm pstl1keep, [%[output0]]    \n"
            "eor  v11.16b, v11.16b, v11.16b  \n"
            "prfm pstl1keep, [%[output1]]    \n"
            "eor  v14.16b, v14.16b, v14.16b  \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16    \n"
            "eor  v17.16b, v17.16b, v17.16b  \n"
            "eor  v20.16b, v20.16b, v20.16b  \n"
            "eor  v23.16b, v23.16b, v23.16b  \n"
            "eor  v26.16b, v26.16b, v26.16b  \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16    \n"
            "eor  v29.16b, v29.16b, v29.16b  \n")";
    }
    writer << R"(
            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "sdot v8.4s , v2.16b,  v0.4b[0]\n"
            "sdot v11.4s, v2.16b,  v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b,  v0.4b[2]\n"
            "sdot v17.4s, v2.16b,  v0.4b[3]\n"

            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v20.4s, v2.16b, v1.4b[0]\n"
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"

            "sdot v8.4s , v5.16b,  v0.4b[0]\n"
            "sdot v11.4s, v5.16b,  v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v5.16b,  v0.4b[2]\n"
            "sdot v17.4s, v5.16b,  v0.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v20.4s, v5.16b, v1.4b[0]\n"
            "sdot v23.4s, v5.16b, v1.4b[1]\n"
            "sdot v26.4s, v5.16b, v1.4b[2]\n"
            "sdot v29.4s, v5.16b, v1.4b[3]\n"
            "subs %w[K], %w[K], #1\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
)";
    std::string tail_temp = R"(
            // Even tail
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"

            "sdot v20.4s, v2.16b, v1.4b[0]\n"
            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"

            ${gen_postprocess_reg_init}

            "sdot v8.4s,  v5.16b, v0.4b[0]\n"
            "sdot v11.4s,  v5.16b, v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v5.16b, v0.4b[2]\n"
            "sdot v17.4s, v5.16b, v0.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            "sdot v20.4s, v5.16b, v1.4b[0]\n"
            "sdot v23.4s, v5.16b, v1.4b[1]\n"
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            "sdot v26.4s, v5.16b, v1.4b[2]\n"
            "sdot v29.4s, v5.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            ${GenAsmGenAsmQuantStore(v20, output4, 0)}
            ${GenAsmGenAsmQuantStore(v23, output5, 0)}
            ${GenAsmGenAsmQuantStore(v26, output6, 0)}
            ${GenAsmGenAsmQuantStore(v29, output7, 0)}
            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            ${gen_postprocess_reg_init}
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v8, output0, 0)}

            ${GenAsmGenAsmQuantStore(v11, output1, 0)}

            "sdot v20.4s, v2.16b, v1.4b[0]\n"

            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            "sdot v23.4s, v2.16b, v1.4b[1]\n"
            "sdot v26.4s, v2.16b, v1.4b[2]\n"
            "sdot v29.4s, v2.16b, v1.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            ${GenAsmGenAsmQuantStore(v20, output4, 0)}
            ${GenAsmGenAsmQuantStore(v23, output5, 0)}
            ${GenAsmGenAsmQuantStore(v26, output6, 0)}
            ${GenAsmGenAsmQuantStore(v29, output7, 0)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(tmp_outputs[0]), [ output1 ] "+r"(tmp_outputs[1]), [ output2 ] "+r"(tmp_outputs[2]), [ output3 ] "+r"(tmp_outputs[3]),
              [ output4 ] "+r"(tmp_outputs[4]), [ output5 ] "+r"(tmp_outputs[5]), [ output6 ] "+r"(tmp_outputs[6]), [ output7 ] "+r"(tmp_outputs[7]),
              [src_scale_ptr] "+r" (src_scale_ptr), [inv6_ptr] "+r" (inv_6_ptr), [dst_scale_ptr] "+r" (dst_scale_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
            ${dst_specifier}* outputs[8] = {output0, output1, output2, output3, output4, output5, output6, output7};
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < n_remain; ++j) {
                    outputs[i][j] = tmp_outputs[i][j];
                }
            }
        })";

    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v21", "v24", "v27", "v30", "v31"}, nonline_mode,
            {"inv6_ptr", "src_scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 3);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0]}, "v31", "dst_scale_ptr",
                                       "src_scale_ptr", args[1], std::stoi(args[2]),
                                       dst_specifier,
                                       {"v21", "v24", "v27", "v30", "v22"},
                                       nonline_mode);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(tail_temp);
    return writer.str();
}

static std::string kern_4x4(
        TContext* ctx, const std::string& dst_specifier,
        const std::string& nonline_mode) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");

    std::stringstream writer;
    //! kern_4x4
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
    __attribute__((target("dotprod")))
    static inline void kern_4x4_bias_relu(const int8_t* packA, const int8_t* packB,
                          int K, ${dst_specifier}* output, int LDC,
                          const int32_t* bias_ptr, float src_scale,  float dst_scale, int m_remain, int n_remain) {
        K /= 4;
        const int8_t* a_ptr = packA;
        const int8_t* b_ptr = packB;
        ${dst_specifier}* output0 = output;
        ${dst_specifier}* output1 = output0 + LDC;
        ${dst_specifier}* output2 = output1 + LDC;
        ${dst_specifier}* output3 = output2 + LDC;
        ${dst_specifier} tmp_output0[4];
        ${dst_specifier} tmp_output1[4];
        ${dst_specifier} tmp_output2[4];
        ${dst_specifier} tmp_output3[4];
        ${dst_specifier}* tmp_outputs[4] = {tmp_output0, tmp_output1, tmp_output2, tmp_output3};
        float* src_scale_ptr = &src_scale;
        float* dst_scale_ptr = &dst_scale; 
        const float inv_6 = 1.f / 6.f;
        const float* inv_6_ptr = &inv_6;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile()");
    //! if convolution with bias
    if (with_bias) {
        writer << R"(
            "ld1 {v6.4s}, [%[bias_ptr]], #16\n"
            "dup    v8.4s, v6.s[0]             \n"
            "dup    v11.4s, v6.s[1]             \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "dup    v14.4s, v6.s[2]             \n"
            "dup    v17.4s, v6.s[3]             \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16\n")";

        //! if convolution without bias
    } else {
        writer << R"(
            "eor  v8.16b, v8.16b, v8.16b     \n"
            "prfm pstl1keep, [%[output0]]    \n"
            "eor  v11.16b, v11.16b, v11.16b  \n"
            "prfm pstl1keep, [%[output1]]    \n"
            "eor  v14.16b, v14.16b, v14.16b  \n"
            "ld1 {v2.4s}, [%[b_ptr]], #16    \n"
            "eor  v17.16b, v17.16b, v17.16b  \n"
            "ld1 {v0.4s}, [%[a_ptr]], #16    \n")";
    }
    writer << R"(
            "2: \n"
            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "sdot v8.4s , v2.16b,  v0.4b[0]\n"
            "sdot v11.4s, v2.16b,  v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b,  v0.4b[2]\n"
            "sdot v17.4s, v2.16b,  v0.4b[3]\n"

            "ld1 {v5.4s}, [%[b_ptr]], #16\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"

            "sdot v8.4s , v5.16b,  v1.4b[0]\n"
            "sdot v11.4s, v5.16b,  v1.4b[1]\n"
            "sdot v14.4s, v5.16b,  v1.4b[2]\n"
            "sdot v17.4s, v5.16b,  v1.4b[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], #16\n"
            "subs %w[K], %w[K], #1\n"

            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
)";
    std::string tail_temp = R"(
            // Even tail
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"

            "ld1 {v5.4s}, [%[b_ptr]], #16\n"

            ${gen_postprocess_reg_init}

            "sdot v8.4s,  v5.16b, v1.4b[0]\n"
            "sdot v11.4s,  v5.16b, v1.4b[1]\n"
            "sdot v14.4s, v5.16b, v1.4b[2]\n"
            "sdot v17.4s, v5.16b, v1.4b[3]\n"
            
            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            "cmp %w[m_remain], #2\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            "cmp %w[m_remain], #3\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            "cmp %w[m_remain], #4\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v17, output3, 0)}
            "b 6f\n"

            // odd tail
            "5:\n"
            "sdot v8.4s,  v2.16b, v0.4b[0]\n"
            ${gen_postprocess_reg_init}
            "sdot v11.4s,  v2.16b, v0.4b[1]\n"
            "sdot v14.4s, v2.16b, v0.4b[2]\n"
            "sdot v17.4s, v2.16b, v0.4b[3]\n"

            ${GenAsmGenAsmQuantStore(v8, output0, 0)}
            "cmp %w[m_remain], #2\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v11, output1, 0)}
            "cmp %w[m_remain], #3\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v14, output2, 0)}
            "cmp %w[m_remain], #4\n"
            "blt 6f\n"
            ${GenAsmGenAsmQuantStore(v17, output3, 0)}

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K), [ m_remain ] "+r"(m_remain),
              [ bias_ptr ] "+r"(bias_ptr), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(tmp_outputs[0]), [ output1 ] "+r"(tmp_outputs[1]), [ output2 ] "+r"(tmp_outputs[2]), [ output3 ] "+r"(tmp_outputs[3]),
              [src_scale_ptr] "+r" (src_scale_ptr), [inv6_ptr] "+r" (inv_6_ptr), [dst_scale_ptr] "+r" (dst_scale_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
            ${dst_specifier}* outputs[4] = {output0, output1, output2, output3};
            for (int i = 0; i < m_remain; ++i) {
                for (int j = 0; j < n_remain; ++j) {
                    outputs[i][j] = tmp_outputs[i][j];
                }
            }
        })";

    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v21", "v24", "v27", "v30", "v31"}, nonline_mode,
            {"inv6_ptr", "src_scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 3);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0]}, "v31", "dst_scale_ptr",
                                       "src_scale_ptr", args[1], std::stoi(args[2]),
                                       dst_specifier,
                                       {"v21", "v24", "v27", "v30", "v22"},
                                       nonline_mode);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(tail_temp);
    return writer.str();
}

std::string gen_pack_a(const std::string& sig) {
    std::stringstream ss;
    ss << interleave_8x1_4_s();
    ss << interleave_4x1_4_s();
    ss << interleave_8();
    ss << interleave_4();
    ss << sig;
    ss << R"({
    int8_t zerobuff[16];
    memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 7 < ymax; y += 8) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int K = kmax - k0;
        //! read 8 * 4 in each row
        for (; K > 15; K -= 16) {
            interleave_8x1_4_s(
                    inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6, inptr7,
                    outptr);
            inptr0 += 16;
            inptr1 += 16;
            inptr2 += 16;
            inptr3 += 16;
            inptr4 += 16;
            inptr5 += 16;
            inptr6 += 16;
            inptr7 += 16;
            outptr += 128;
        }

        if (K > 0) {
            interleave_8(
                    inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6, inptr7,
                    outptr, 4, K);
            inptr0 += K;
            inptr1 += K;
            inptr2 += K;
            inptr3 += K;
            inptr4 += K;
            inptr5 += K;
            inptr6 += K;
            inptr7 += K;
            outptr += ((K + 3) / 4 * 4 * 8);
        }
    }
    for (; y < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        //! read 4 * 4 in each row
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }

            interleave_4x1_4_s(inptr0, inptr1, inptr2, inptr3, outptr);
            inptr0 += 16;
            inptr1 += 16;
            inptr2 += 16;
            inptr3 += 16;
            outptr += 64;
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
                    default:
                        TINYNN_ASSERT(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, K);
            inptr0 += K;
            inptr1 += K;
            inptr2 += K;
            inptr3 += K;
            outptr += ((K + 3) / 4 * 4 * 4);
        }
    }
})";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << transpose_12x4_1_b();
    ss << transpose_4();
    ss << sig;
    ss << R"({
    int8_t zerobuff[16];
    memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize12 = (ksize + 3) / 4 * 4 * 12;
    const int ksize4 = (ksize + 3) / 4 * 4 * 4;
    int8_t* outptr0 = outptr;
    int8_t* outptr_base = outptr;
    //! 4x4 block output start pos
    int8_t* outptr_base4 = outptr + ((xmax - x0) / 12) * ksize12;

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = inptr + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = x0;
        outptr0 = outptr_base;
        for (; x + 11 < xmax; x += 12) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }

            transpose_12x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr0);
            inptr0 += 12;
            inptr1 += 12;
            inptr2 += 12;
            inptr3 += 12;
            outptr0 += ksize12;
        }

        outptr0 = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr0, 4, 4);
            inptr0 += 4;
            inptr1 += 4;
            inptr2 += 4;
            inptr3 += 4;
            outptr0 += ksize4;
        }

        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr0, 4, xmax - x);
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
        size_t res = (size_t)((kmax - k0 + 3) / 4 * 4) * ((ymax - y0 + 3) / 4 * 4) * sizeof(int8_t);
        return res;
    })";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)((kmax - k0 + 3) / 4 * 4) * ((xmax - x0 + 3) / 4 * 4) * sizeof(int8_t);
        return res;
    })";
    return ss.str();
}

std::string gen_kernel(
        const std::string& dst_specifier, const std::string& sig, TContext* ctx,
        const std::string& postprocess_call, const std::string& preset_str = "",
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
        K = (K + 3) / 4 * 4;
        const int K12 = K * 12;
        const int K8 = K * 8;
        const int K4 = K * 4;
        size_t m = 0;
        ${dst_specifier}* gemm_output = (${dst_specifier}*)${gen_gemm_output};
        for (; m + m_block <= M; m += m_block) {
            ${dst_specifier}* output = gemm_output + (m * LDC);

            size_t n = 0;
            const int8_t* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_8x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr, temp_scale, dst_scale_inv);
                output += n_block;
                cur_pack_b += K12;
            }

            for (; n < N; n += 4) {                
                kern_8x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, temp_scale, dst_scale_inv, N - n > 4 ? 4 : N - n);
                output += 4;
                cur_pack_b += K4;
            }
            pack_a += K8;
            bias_ptr += m_block;
        }
        for (; m < M; m += m_block_4) {
            ${dst_specifier}* output = gemm_output + (m * LDC);
            size_t n = 0;
            const int8_t* cur_pack_b = pack_b;
            for (; n + n_block - 1 < N; n += n_block) {                
                kern_4x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr, temp_scale, dst_scale_inv, M - m < 4 ? M - m : 4);
                output += n_block;
                cur_pack_b += K12;
            }
            for (; n < N; n += 4) {                
                kern_4x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, temp_scale, dst_scale_inv, M - m > 4 ? 4 : M - m, N - n > 4 ? 4 : N - n);
                output += 4;
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

std::string MatmulInt8M8N12K4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Arm64_int8_m8_n12_k4_gemm";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
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

bool MatmulInt8M8N12K4Kernel::need_post_process(TContext* ctx) const {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    return nonline_mode == "SIGMOID";
}

std::vector<KernelObj> MatmulInt8M8N12K4Kernel::GetDependInternalSymbol(
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

std::string MatmulInt8M8N12K4Kernel::GetKernelBody(TContext* ctx) const {
    auto postprocess_pair = gen_postprocess_inline(ctx, need_post_process(ctx));
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <arm_neon.h>\n";
    writer << "#include \"utils.h\"\n";
    writer << prefetch();
    auto dtype = ctx->getAttrStr("dtype");
    std::string last_dtype = "si8";
    if (ctx->haveAttr("last_dtype")) {
        last_dtype = ctx->getAttrStr("last_dtype");
    }
    std::string dst_specifier = "int32_t";
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
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
    writer << gen_pack_a(GetPackASignature(ctx));
    writer << gen_pack_b(GetPackBSignature(ctx));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << postprocess_pair.first;
    writer << gen_kernel(
            dst_specifier, GetNakedKernelSignature(ctx), ctx, postprocess_pair.second,
            "", need_temp_dst);

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
    writer << gen_kernel(
            dst_specifier, GetKernelSignature(ctx), ctx, postprocess_pair.second,
            preset_str, need_temp_dst);
    return writer.str();
}

std::string MatmulInt8M8N12K4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulInt8M8N12K4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
