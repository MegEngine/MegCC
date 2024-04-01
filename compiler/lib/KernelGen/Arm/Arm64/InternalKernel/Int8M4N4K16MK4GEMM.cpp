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
std::string transpose_interleave_4x4_4_b() {
    return std::string{
            R"(
    static inline void transpose_interleave_4x4_4_b(const int8_t* inptr0, const int8_t* inptr1, const int8_t* inptr2, const int8_t* inptr3,
                                      int8_t* outptr, int stride) {
        asm volatile(
            "ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [%[inptr0]]\n"
            "ld4 {v4.16b, v5.16b, v6.16b, v7.16b}, [%[inptr1]]\n"
            "ld4 {v8.16b, v9.16b, v10.16b, v11.16b}, [%[inptr2]]\n"
            "ld4 {v12.16b, v13.16b, v14.16b, v15.16b}, [%[inptr3]]\n"
            
            "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%[outptr]], %x[stride]\n"
            "st1 {v4.16b, v5.16b, v6.16b, v7.16b}, [%[outptr]], %x[stride]\n"
            "st1 {v8.16b, v9.16b, v10.16b, v11.16b}, [%[outptr]], %x[stride]\n"
            "st1 {v12.16b, v13.16b, v14.16b, v15.16b}, [%[outptr]], %x[stride]\n"
            : [ inptr0 ] "+r"(inptr0), [ inptr1 ] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [ outptr ] "+r"(outptr), [stride] "+r"(stride)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory");
    })"};
}

std::string transpose_interleave_1x4_4_b() {
    return std::string{
            R"(
    static inline void transpose_interleave_1x4_4_b(const int8_t* inptr0, int8_t* outptr) {
        asm volatile(
            "ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [%[inptr0]]\n"
            
            "st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [%[outptr]]\n"
            : [ inptr0 ] "+r"(inptr0), [ outptr ] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "memory");
    })"};
}

std::string transpose_4x4_1_s() {
    return std::string(R"(
    static inline void transpose_4x4_1_s(const int32_t* inptr0, const int32_t* inptr1, const int32_t* inptr2, const int32_t* inptr3, int32_t* outptr) {
        asm volatile(
            "ld1 {v0.4s}, [%[inptr0]]\n" // A0B0C0D0
            "ld1 {v1.4s}, [%[inptr1]]\n" // A1B1C1D1
            "ld1 {v2.4s}, [%[inptr2]]\n" // A2B2C2D2
            "ld1 {v3.4s}, [%[inptr3]]\n" // A3B3C3D3

            "zip1 v4.4s, v0.4s, v1.4s\n" // A0A1B0B1
            "zip1 v5.4s, v2.4s, v3.4s\n" // A2A3B2B3
            "zip2 v6.4s, v0.4s, v1.4s\n" // C0C1D0D1
            "zip2 v7.4s, v2.4s, v3.4s\n" // C2C3D2D3

            "zip1 v0.2d, v4.2d, v5.2d\n" // A0A1A2A3
            "zip2 v1.2d, v4.2d, v5.2d\n" // B0B1B2B3
            "zip1 v2.2d, v6.2d, v7.2d\n" // C0C1C2C3
            "zip2 v3.2d, v6.2d, v7.2d\n" // D0D1D2D3

            "st1 {v0.4s}, [%[outptr]], 16\n"
            "st1 {v1.4s}, [%[outptr]], 16\n"
            "st1 {v2.4s}, [%[outptr]], 16\n"
            "st1 {v3.4s}, [%[outptr]], 16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
        );
    }
    )");
}

std::string prefetch() {
    return R"(
        #define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_2x_f32() +
           KernelGen::ArmCommon::gen_common_prefetch_3x_f32();
}

static std::string kern_4x4(
        TContext* ctx, const std::string& dst_specifier,
        const std::string& nonline_mode, bool remain) {
    auto activation_gen = create_activation_gener(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string store_str = "STORE_C";
    if (dst_specifier == "int8_t") {
        store_str = "STORE_C_QUANT";
    }
    std::string sig =
            StringTemplate::StringTemplateArgs()
                    .add("dst_specifier", dst_specifier)
                    .render(R"(static inline void kern_4x4_bias_activation (const int8_t* packA, const int8_t* packB, int K,
                                           ${dst_specifier}* output, const int32_t* bias_ptr,
                                           float src_scale, float dst_scale))");
    if (remain) {
        sig = StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(static inline void kern_4x4_bias_activation_remain (const int8_t* packA, const int8_t* packB, int K,
                                           ${dst_specifier}* output, const int32_t* bias_ptr, int n_remain,
                                           float src_scale, float dst_scale))");
    }
    std::stringstream writer;
    //! kern_4x4
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .add("sig", sig)
                      .render(R"(
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
            ${sig} {
                K /= 16;
                TINYNN_ASSERT(K > 0);
                const int8_t* a_ptr = packA;
                const int8_t* b_ptr = packB;
                ${dst_specifier}* output0 = output;
                float* src_scale_ptr = &src_scale;
                float* dst_scale_ptr = &dst_scale;
                const float inv_6 = 1.f / 6.f;
                const float* inv_6_ptr = &inv_6;)");
    if (remain) {
        writer << R"(

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
    )";
    } else {
        writer << R"(

                //clang-format off
            #define STORE_C                                          \
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n"

            #define STORE_C_QUANT                                    \
                "str s8,  [%[output0], #0]\n"                        \
                "str s9,  [%[output0], #4]\n"                        \
                "str s10, [%[output0], #8]\n"                        \
                "str s11, [%[output0], #12]\n"                       \
                //clang-format on
                )";
    }
    writer << R"(
                asm volatile(
                "ld1 {v0.16b}, [%[a_ptr]], #16\n"
                "eor v16.16b,  v16.16b,  v16.16b\n"
                "eor v17.16b,  v17.16b,  v17.16b\n"
                "eor v18.16b,  v18.16b,  v18.16b\n"
                "ld1 {v1.16b}, [%[a_ptr]], #16\n"
                "eor v19.16b,  v19.16b,  v19.16b\n"
                "eor v20.16b,  v19.16b,  v19.16b\n"
                "eor v21.16b,  v19.16b,  v19.16b\n"
                "ld1 {v4.16b, v5.16b}, [%[b_ptr]], #32\n"
                "eor v22.16b,  v19.16b,  v19.16b\n"
                "PRFM PLDL1KEEP, [%[a_ptr], #32]\n"
                "eor v23.16b,  v19.16b,  v19.16b\n"
                "eor v24.16b,  v19.16b,  v19.16b\n"
                "PRFM PLDL1KEEP, [%[b_ptr], #32]\n"
                "eor v25.16b,  v19.16b,  v19.16b\n"
                "eor v26.16b,  v19.16b,  v19.16b\n"
                "PRFM PLDL1KEEP, [%[b_ptr], #64]\n"
                "eor v27.16b,  v19.16b,  v19.16b\n"
                "eor v28.16b,  v19.16b,  v19.16b\n"
                "PRFM PLDL1KEEP, [%[a_ptr], #64]\n"
                "eor v29.16b,  v19.16b,  v19.16b\n"
                "eor v30.16b,  v19.16b,  v19.16b\n"
                "PRFM PLDL1KEEP, [%[b_ptr], #128]\n"
                "eor v31.16b,  v19.16b,  v19.16b\n"

                "2: \n"
                "cmp %w[K], #2\n"
                "blt 4f\n"

                "3:\n"
                // [v0, v1] * [v4, v5] -> [v8, v9, v12, v13]
                "smull v8.8h, v0.8b, v4.8b\n"
                "smull v9.8h, v0.8b, v5.8b\n"
                "ld1 {v6.16b}, [%[b_ptr]], #16\n"
                "smull v12.8h, v1.8b, v4.8b\n"
                "smull v13.8h, v1.8b, v5.8b\n"
                "ld1 {v7.16b}, [%[b_ptr]], #16\n"
                "smlal2 v8.8h, v0.16b, v4.16b\n"
                "smlal2 v9.8h, v0.16b, v5.16b\n"
                "smlal2 v12.8h, v1.16b, v4.16b\n"
                "smlal2 v13.8h, v1.16b, v5.16b\n"

                // [v0, v1] * [v6, v7] -> [v10, v11, v14, v15]
                // v8 -> v16, v9 -> v17, v12 -> v20, v13 -> v21
                "smull v10.8h, v0.8b, v6.8b\n"
                "ld1 {v2.16b}, [%[a_ptr]], #16\n"
                "smull v11.8h, v0.8b, v7.8b\n"
                "smull v14.8h, v1.8b, v6.8b\n"
                "ld1 {v3.16b}, [%[a_ptr]], #16\n"
                "smull v15.8h, v1.8b, v7.8b\n"
                "sadalp v16.4s, v8.8h\n"
                "smlal2 v10.8h, v0.16b, v6.16b\n"
                "sadalp v17.4s, v9.8h\n"
                "smlal2 v11.8h, v0.16b, v7.16b\n"
                "sadalp v20.4s, v12.8h\n"
                "smlal2 v14.8h, v1.16b, v6.16b\n"
                "sadalp v21.4s, v13.8h\n"
                "smlal2 v15.8h, v1.16b, v7.16b\n"

                // [v2, v3] * [v4, v5] -> [v8, v9, v12, v13]
                // v10 -> v18, v11 -> v19, v14 -> v22, v15 -> v23
                "smull v8.8h, v2.8b, v4.8b\n"
                "smull v9.8h, v2.8b, v5.8b\n"
                "ld1 {v0.16b}, [%[a_ptr]], #16\n"
                "smull v12.8h, v3.8b, v4.8b\n"
                "smull v13.8h, v3.8b, v5.8b\n"
                "ld1 {v1.16b}, [%[a_ptr]], #16\n"
                "sadalp v18.4s, v10.8h\n"
                "smlal2 v8.8h, v2.16b, v4.16b\n"
                "sadalp v19.4s, v11.8h\n"
                "smlal2 v9.8h, v2.16b, v5.16b\n"
                "sadalp v22.4s, v14.8h\n"
                "smlal2 v12.8h, v3.16b, v4.16b\n"
                "sadalp v23.4s, v15.8h\n"
                "smlal2 v13.8h, v3.16b, v5.16b\n"

                // [v2, v3] * [v6, v7] -> [v10, v11, v14, v15]
                // v8 -> v24, v9 -> v25, v12 -> v28, v13 -> v29
                "smull v10.8h, v2.8b, v6.8b\n"
                "ld1 {v4.16b}, [%[b_ptr]], #16\n"
                "smull v11.8h, v2.8b, v7.8b\n"
                "smull v14.8h, v3.8b, v6.8b\n"
                "ld1 {v5.16b}, [%[b_ptr]], #16\n"
                "smull v15.8h, v3.8b, v7.8b\n"
                "sadalp v24.4s, v8.8h\n"
                "smlal2 v10.8h, v2.16b, v6.16b\n"
                "sadalp v25.4s, v9.8h\n"
                "smlal2 v11.8h, v2.16b, v7.16b\n"
                "sadalp v28.4s, v12.8h\n"
                "smlal2 v14.8h, v3.16b, v6.16b\n"
                "sadalp v29.4s, v13.8h\n"
                "smlal2 v15.8h, v3.16b, v7.16b\n"

                // [v0, v1] * [v4, v5] -> [v8, v9, v12, v13]
                // v10 -> v26, v11 -> v27, v14 -> v30, v15 -> v31
                "smull v8.8h, v0.8b, v4.8b\n"
                "smull v9.8h, v0.8b, v5.8b\n"
                "ld1 {v6.16b}, [%[b_ptr]], #16\n"
                "smull v12.8h, v1.8b, v4.8b\n"
                "smull v13.8h, v1.8b, v5.8b\n"
                "ld1 {v7.16b}, [%[b_ptr]], #16\n"
                "sadalp v26.4s, v10.8h\n"
                "smlal2 v8.8h, v0.16b, v4.16b\n"
                "sadalp v27.4s, v11.8h\n"
                "smlal2 v9.8h, v0.16b, v5.16b\n"
                "sadalp v30.4s, v14.8h\n"
                "smlal2 v12.8h, v1.16b, v4.16b\n"
                "sadalp v31.4s, v15.8h\n"
                "smlal2 v13.8h, v1.16b, v5.16b\n"

                // [v0, v1] * [v6, v7] -> [v10, v11, v14, v15]
                // v8 -> v16, v9 -> v17, v12 -> v20, v13 -> v21
                "smull v10.8h, v0.8b, v6.8b\n"
                "ld1 {v2.16b}, [%[a_ptr]], #16\n"
                "smull v11.8h, v0.8b, v7.8b\n"
                "smull v14.8h, v1.8b, v6.8b\n"
                "ld1 {v3.16b}, [%[a_ptr]], #16\n"
                "smull v15.8h, v1.8b, v7.8b\n"
                "sadalp v16.4s, v8.8h\n"
                "smlal2 v10.8h, v0.16b, v6.16b\n"
                "sadalp v17.4s, v9.8h\n"
                "smlal2 v11.8h, v0.16b, v7.16b\n"
                "sadalp v20.4s, v12.8h\n"
                "smlal2 v14.8h, v1.16b, v6.16b\n"
                "sadalp v21.4s, v13.8h\n"
                "smlal2 v15.8h, v1.16b, v7.16b\n"

                // [v2, v3] * [v4, v5] -> [v8, v9, v12, v13]
                // v10 -> v18, v11 -> v19, v14 -> v22, v15 -> v23
                "smull v8.8h, v2.8b, v4.8b\n"
                "smull v9.8h, v2.8b, v5.8b\n"
                "ld1 {v0.16b}, [%[a_ptr]], #16\n"
                "smull v12.8h, v3.8b, v4.8b\n"
                "smull v13.8h, v3.8b, v5.8b\n"
                "ld1 {v1.16b}, [%[a_ptr]], #16\n"
                "sadalp v18.4s, v10.8h\n"
                "smlal2 v8.8h, v2.16b, v4.16b\n"
                "sadalp v19.4s, v11.8h\n"
                "smlal2 v9.8h, v2.16b, v5.16b\n"
                "sadalp v22.4s, v14.8h\n"
                "smlal2 v12.8h, v3.16b, v4.16b\n"
                "sadalp v23.4s, v15.8h\n"
                "smlal2 v13.8h, v3.16b, v5.16b\n"

                // [v2, v3] * [v6, v7] -> [v10, v11, v14, v15]
                // v8 -> v24, v9 -> v25, v12 -> v28, v13 -> v29
                "smull v10.8h, v2.8b, v6.8b\n"
                "ld1 {v4.16b}, [%[b_ptr]], #16\n"
                "smull v11.8h, v2.8b, v7.8b\n"
                "smull v14.8h, v3.8b, v6.8b\n"
                "ld1 {v5.16b}, [%[b_ptr]], #16\n"
                "smull v15.8h, v3.8b, v7.8b\n"
                "sadalp v24.4s, v8.8h\n"
                "smlal2 v10.8h, v2.16b, v6.16b\n"
                "sadalp v25.4s, v9.8h\n"
                "smlal2 v11.8h, v2.16b, v7.16b\n"
                "sadalp v28.4s, v12.8h\n"
                "smlal2 v14.8h, v3.16b, v6.16b\n"
                "sadalp v29.4s, v13.8h\n"
                "smlal2 v15.8h, v3.16b, v7.16b\n"

                "sadalp v26.4s, v10.8h\n"
                "sadalp v27.4s, v11.8h\n"
                "sadalp v30.4s, v14.8h\n"
                "sadalp v31.4s, v15.8h\n"

                "subs %w[K], %w[K], #2\n"
                "cmp %w[K], #2\n"
                "bge 3b\n"

                "4:\n"
                "cmp %w[K], #0\n"
                "beq 5f\n"

                // [v0, v1] * [v4, v5] -> [v8, v9, v12, v13]
                "smull v8.8h, v0.8b, v4.8b\n"
                "smull v9.8h, v0.8b, v5.8b\n"
                "ld1 {v6.16b}, [%[b_ptr]], #16\n"
                "smull v12.8h, v1.8b, v4.8b\n"
                "smull v13.8h, v1.8b, v5.8b\n"
                "ld1 {v7.16b}, [%[b_ptr]], #16\n"
                "smlal2 v8.8h, v0.16b, v4.16b\n"
                "smlal2 v9.8h, v0.16b, v5.16b\n"
                "smlal2 v12.8h, v1.16b, v4.16b\n"
                "smlal2 v13.8h, v1.16b, v5.16b\n"

                // [v0, v1] * [v6, v7] -> [v10, v11, v14, v15]
                // v8 -> v16, v9 -> v17, v12 -> v20, v13 -> v21
                "smull v10.8h, v0.8b, v6.8b\n"
                "ld1 {v2.16b}, [%[a_ptr]], #16\n"
                "smull v11.8h, v0.8b, v7.8b\n"
                "smull v14.8h, v1.8b, v6.8b\n"
                "ld1 {v3.16b}, [%[a_ptr]], #16\n"
                "smull v15.8h, v1.8b, v7.8b\n"
                "sadalp v16.4s, v8.8h\n"
                "smlal2 v10.8h, v0.16b, v6.16b\n"
                "sadalp v17.4s, v9.8h\n"
                "smlal2 v11.8h, v0.16b, v7.16b\n"
                "sadalp v20.4s, v12.8h\n"
                "smlal2 v14.8h, v1.16b, v6.16b\n"
                "sadalp v21.4s, v13.8h\n"
                "smlal2 v15.8h, v1.16b, v7.16b\n"

                // [v2, v3] * [v4, v5] -> [v8, v9, v12, v13]
                // v10 -> v18, v11 -> v19, v14 -> v22, v15 -> v23
                "smull v8.8h, v2.8b, v4.8b\n"
                "smull v9.8h, v2.8b, v5.8b\n"
                "smull v12.8h, v3.8b, v4.8b\n"
                "smull v13.8h, v3.8b, v5.8b\n"
                "sadalp v18.4s, v10.8h\n"
                "smlal2 v8.8h, v2.16b, v4.16b\n"
                "sadalp v19.4s, v11.8h\n"
                "smlal2 v9.8h, v2.16b, v5.16b\n"
                "sadalp v22.4s, v14.8h\n"
                "smlal2 v12.8h, v3.16b, v4.16b\n"
                "sadalp v23.4s, v15.8h\n"
                "smlal2 v13.8h, v3.16b, v5.16b\n"

                // [v2, v3] * [v6, v7] -> [v10, v11, v14, v15]
                // v8 -> v24, v9 -> v25, v12 -> v28, v13 -> v29
                "smull v10.8h, v2.8b, v6.8b\n"
                "smull v11.8h, v2.8b, v7.8b\n"
                "smull v14.8h, v3.8b, v6.8b\n"
                "smull v15.8h, v3.8b, v7.8b\n"
                "sadalp v24.4s, v8.8h\n"
                "smlal2 v10.8h, v2.16b, v6.16b\n"
                "sadalp v25.4s, v9.8h\n"
                "smlal2 v11.8h, v2.16b, v7.16b\n"
                "sadalp v28.4s, v12.8h\n"
                "smlal2 v14.8h, v3.16b, v6.16b\n"
                "sadalp v29.4s, v13.8h\n"
                "smlal2 v15.8h, v3.16b, v7.16b\n"

                "sadalp v26.4s, v10.8h\n"
                "sadalp v27.4s, v11.8h\n"
                "sadalp v30.4s, v14.8h\n"
                "sadalp v31.4s, v15.8h\n"

                "5:\n"
                "addp v0.4s, v16.4s, v20.4s\n"
                "addp v1.4s, v24.4s, v28.4s\n"
                "addp v2.4s, v17.4s, v21.4s\n"
                "addp v3.4s, v25.4s, v29.4s\n"
                "addp v4.4s, v18.4s, v22.4s\n"
                "addp v5.4s, v26.4s, v30.4s\n"
                "addp v6.4s, v19.4s, v23.4s\n"
                "addp v7.4s, v27.4s, v31.4s\n")";

    std::string temp = R"(
                ${gen_postprocess_reg_init}

                "addp v8.4s, v0.4s, v1.4s\n"
                "addp v9.4s, v2.4s, v3.4s\n"
                "addp v10.4s, v4.4s, v5.4s\n"
                "addp v11.4s, v6.4s, v7.4s\n"

                ${add_bias}

                ${GenAsmGenAsmQuantStore(v8, v9)}
                ${GenAsmGenAsmQuantStore(v10, v11)}
                
                ${gen_store}

                : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
                  [ bias_ptr ] "+r"(bias_ptr), [ output0 ] "+r"(output0), ${remain_input_list}
                  [src_scale_ptr] "+r" (src_scale_ptr), [inv6_ptr] "+r" (inv_6_ptr), [dst_scale_ptr] "+r" (dst_scale_ptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                  "cc", "memory");
            #undef STORE_C
            #undef STORE_C_QUANT
        })";
    std::string add_bias = "";
    if (with_bias) {
        add_bias = R"(
                "ld1 {v0.4s}, [%[bias_ptr]], #16\n"
                "add v8.4s, v8.4s, v0.4s\n"
                "add v9.4s, v9.4s, v0.4s\n"
                "add v10.4s, v10.4s, v0.4s\n"
                "add v11.4s, v11.4s, v0.4s\n")";
    }
    std::string remain_input_list = "";
    if (remain) {
        remain_input_list = R"([ n_remain ] "+r"(n_remain), )";
    }
    std::string postprocess_reg_init = activation_gen->GenAsmQuantInit(
            {"v20", "v21", "v22", "v23", "v27"}, nonline_mode,
            {"inv6_ptr", "src_scale_ptr"});
    writer << StringTemplate::StringTemplateArgs()
                      .add("gen_store", store_str)
                      .add("add_bias", add_bias)
                      .add("remain_input_list", remain_input_list)
                      .add("GenAsmGenAsmQuantStore",
                           [=](std::vector<std::string> args) {
                               CC_ASSERT(args.size() == 2);
                               return activation_gen->GenAsmQuantStore(
                                       {args[0], args[1]}, "v27", "dst_scale_ptr",
                                       "src_scale_ptr", "None", 0, dst_specifier,
                                       {"v20", "v21", "v22", "v23", "v24", "v25"},
                                       nonline_mode, false);
                           })
                      .add("gen_postprocess_reg_init", postprocess_reg_init)
                      .render(temp);
    return writer.str();
}

std::string gen_pack_a(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    int8_t zero_buf[4][64];
    memset(zero_buf, 0, 4 * 64 * sizeof(int8_t));
    TINYNN_ASSERT(ymax % 4 == 0 && y0 % 4 == 0);
    TINYNN_ASSERT(kmax % 4 == 0 && k0 % 4 == 0);
    const int pack_k = 16;
    const int pack_m = 4;
    const int mk_stride = pack_m * pack_k;
    int round_k = (kmax - k0 + pack_k - 1) / pack_k * pack_k;
    int out_stride = round_k * pack_m;

    int y = y0;
    //! unroll 4
    for (; y + 15 < ymax; y += 16) {
        const int8_t* inptr0 = inptr + y / pack_m * ldin + k0 * 4; // ldin is stride[0]
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        int8_t* outptr0 = outptr + (y - y0) / pack_m * out_stride;

        int k = k0;
        for (; k + 15 < kmax; k += pack_k) {
            transpose_interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, outptr0, out_stride);
            outptr0 += mk_stride;
            inptr0 += mk_stride;
            inptr1 += mk_stride;
            inptr2 += mk_stride;
            inptr3 += mk_stride;
        }
        if (k < kmax) {
            int remain_bytes = (kmax - k) * pack_m * sizeof(int8_t);
            memcpy(zero_buf[0], inptr0, remain_bytes);
            memcpy(zero_buf[1], inptr1, remain_bytes);
            memcpy(zero_buf[2], inptr2, remain_bytes);
            memcpy(zero_buf[3], inptr3, remain_bytes);
            transpose_interleave_4x4_4_b(zero_buf[0], zero_buf[1], zero_buf[2], zero_buf[3], outptr0, out_stride);
        }
    }
    for (; y + 3 < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y / pack_m * ldin + k0 * 4; // ldin is stride[0]
        prefetch_2x(inptr0);
        int8_t* outptr0 = outptr + (y - y0) / pack_m * out_stride;

        int k = k0;
        for (; k + 15 < kmax; k += pack_k) {
            transpose_interleave_1x4_4_b(inptr0, outptr0);
            outptr0 += mk_stride;
            inptr0 += mk_stride;
        }
        if (k < kmax) {
            int remain_bytes = (kmax - k) * pack_m * sizeof(int8_t);
            memcpy(zero_buf[0], inptr0, remain_bytes);
            transpose_interleave_1x4_4_b(zero_buf[0], outptr0);
        }
    }
})";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        TINYNN_ASSERT(k0 % 4 == 0 && kmax % 4 == 0);
        const int pack_k = 16;
        const int pack_n = 4;
        const int round_k = (kmax - k0 + pack_k - 1) / pack_k * pack_k;
        const int out_stride = round_k * pack_n;

        int k = k0;
        for (; k + 15 < kmax; k += pack_k) {
            int n = x0;
            int32_t* outptr0 = (int32_t*)(outptr + (k - k0) * pack_n);
            const int32_t* inptr0 = (int32_t*)(inptr + k / 4 * ldin + n * 4);
            const int32_t* inptr1 = inptr0 + ldin / 4;
            const int32_t* inptr2 = inptr1 + ldin / 4;
            const int32_t* inptr3 = inptr2 + ldin / 4;
            for (; n + 3 < xmax; n += pack_n) {
                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr0);
                inptr0 += pack_n;
                inptr1 += pack_n;
                inptr2 += pack_n;
                inptr3 += pack_n;
                outptr0 += round_k;
            }
            for (; n < xmax; ++n) {
                *outptr0++ = *inptr0++;
                *outptr0++ = *inptr1++;
                *outptr0++ = *inptr2++;
                *outptr0++ = *inptr3++;
            }
        }
        if (k < kmax) {
            int32_t zero[4] = {0};
            int n = x0;
            int32_t* outptr0 = (int32_t*)(outptr + (k - k0) * pack_n);
            const int32_t* inptr0 = (int32_t*)(inptr + k / 4 * ldin + n * 4);
            const int32_t* inptr1 = inptr0 + ldin / 4;
            const int32_t* inptr2 = inptr1 + ldin / 4;
            const int32_t* inptr3 = inptr2 + ldin / 4;
            for (; n + 3 < xmax; n += pack_n) {
                switch (kmax - k) {
                    case 4:
                        inptr1 = zero;
                    case 8:
                        inptr2 = zero;
                    case 12:
                        inptr3 = zero;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr0);
                inptr0 += pack_n;
                inptr1 += pack_n;
                inptr2 += pack_n;
                inptr3 += pack_n;
                outptr0 += round_k;
            }
            if (n < xmax) {
                switch (kmax - k) {
                    case 4:
                        inptr1 = zero;
                    case 8:
                        inptr2 = zero;
                    case 12:
                        inptr3 = zero;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
                for (; n < xmax; ++n) {
                    *outptr0++ = *inptr0++;
                    *outptr0++ = *inptr1++;
                    *outptr0++ = *inptr2++;
                    *outptr0++ = *inptr3++;
                }
            }
        }
    })";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_k = 16;
        size_t res = (size_t)((kmax - k0 + packed_k - 1) / packed_k * packed_k) * (ymax - y0) * sizeof(int8_t);
        return res;
    })";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_k = 16;
        const int packed_n = 4;
        const int round_k = (kmax - k0 + packed_k - 1) / packed_k * packed_k;
        const int round_n = (xmax - x0 + packed_n - 1) / packed_n * packed_n;
        size_t res = (size_t)(round_k) * round_n * sizeof(int8_t);
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
        const int m_block = 4;
        const int n_block = 4;
        const int round_k = (K + 15) / 16 * 16;
        const int K4 = round_k * 4;
        size_t m = 0;
        ${dst_specifier}* gemm_output = (${dst_specifier}*)${gen_gemm_output};
        for (; m + m_block <= M; m += m_block) {
            ${dst_specifier}* output = gemm_output + (m / m_block * LDC);

            size_t n = 0;
            const int8_t* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_4x4_bias_activation(pack_a, cur_pack_b, round_k, output,
                                    bias_ptr, temp_scale, dst_scale_inv);
                output += n_block * m_block;
                cur_pack_b += K4;
            }

            if (n < N) {
                kern_4x4_bias_activation_remain(pack_a, cur_pack_b, round_k, output,
                                   bias_ptr, N - n, temp_scale, dst_scale_inv);
            }
            pack_a += K4;
            bias_ptr += m_block;
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

std::string MatmulInt8M4N4K16MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Arm64_int8_m4_n4_k16_mk4_gemm";
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

bool MatmulInt8M4N4K16MK4Kernel::need_post_process(TContext* ctx) const {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    return nonline_mode == "SIGMOID";
}

std::vector<KernelObj> MatmulInt8M4N4K16MK4Kernel::GetDependInternalSymbol(
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

std::string MatmulInt8M4N4K16MK4Kernel::GetKernelBody(TContext* ctx) const {
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

    writer << kern_4x4(ctx, dst_specifier, gen_nonline_mode, false);
    writer << kern_4x4(ctx, dst_specifier, gen_nonline_mode, true);
    writer << transpose_interleave_4x4_4_b();
    writer << transpose_interleave_1x4_4_b();
    writer << transpose_4x4_1_s();
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

std::string MatmulInt8M4N4K16MK4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulInt8M4N4K16MK4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
