/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/ElemwiseHelper/UnaryHelper.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "ElemwiseHelper.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
std::string ElemwiseGenUnary::GenCodeBody(std::vector<std::string> strs) const {
    std::stringstream body_ss;
    if (m_inline_mode) {
        body_ss << R"(static inline void ${inline_func_name}(const ${src_specifier}* src, ${dst_specifier}* dst, size_t nr_elem)";
        body_ss << "){";
    } else {
        body_ss << R"(
            Layout in_layout = inputs[0]->layout;
            size_t nr_elem = 1;
            for (int i = 0; i < in_layout.nr_dim; ++i) {
                nr_elem *= in_layout.dims[i];
            }
            const ${src_specifier} * src = ${source};
            ${dst_specifier}* dst = ${dst};
        )";
    }
    body_ss << R"(
       
        ${kernel_init()}

        size_t index = offset;
        for(; index + 7 < nr_elem; index += 8) {
            ${src_simd_specifier} vsrc0 = ${src_ld1q}(src);
            ${src_simd_specifier} vsrc1 = ${src_ld1q}(src + 4);
            ${kernel_simd_unroll(2, vsrc0, vdst0, vsrc1, vdst1)}
            ${dst_store(dst, vdst0)};
            ${dst_store(dst + 4, vdst1)};
            src += 8;
            dst += 8;
        }
        for(; index + 3 < nr_elem; index += 4) {
            ${src_simd_specifier} vsrc0 = ${src_ld1q}(src);
            ${kernel_simd_unroll(1, vsrc0, vdst0)}
            ${dst_store(dst, vdst0)};
            src += 4;
            dst += 4;
        }
        for(; index < nr_elem; index++) {
            ${kernel_naive_unroll(1, src, dst)}
            src += 1;
            dst += 1;
        })";
    if (m_inline_mode) {
        body_ss << "}";
    }
    auto kernel_init = [this](std::vector<std::string> strs) {
        return GenKernelAsmInit(strs);
    };
    auto kernel_simd_unroll = [this](std::vector<std::string> strs) {
        return GenKernelSimdUnroll(strs);
    };
    auto kernel_naive_unroll = [this](std::vector<std::string> strs) {
        return GenKernelNaiveUnroll(strs);
    };
    std::stringstream ss;
    auto body_render = StringTemplate::StringTemplateArgs()
                               .add("kernel_init", kernel_init)
                               .add("kernel_simd_unroll", kernel_simd_unroll)
                               .add("kernel_naive_unroll", kernel_naive_unroll)
                               .add("src_specifier",
                                    Utils::cvt_dtype_specifier(m_src_dtype))
                               .add("dst_specifier",
                                    Utils::cvt_dtype_specifier(m_dst_dtype))
                               .add("src_ld1q", m_src_simd->get_ld1q_symbol())
                               .add("dst_store",
                                    [=](std::string ptr, std::string dst_reg) {
                                        return m_dst_simd->get_st1q_symbol() +
                                               "(" + ptr + "," + dst_reg +
                                               ")\n";
                                    })
                               .add("dst_st1q", m_dst_simd->get_st1q_symbol())
                               .add("src_simd_specifier",
                                    m_src_simd->get_specifier_q_symbol());

    if (m_inline_mode) {
        body_render.add("inline_func_name", GenInlineName());
    } else {
        auto input = strs[0];
        auto output = strs[1];
        body_render.add("source", input).add("dst", output);
    }
    ss << body_render.render(body_ss.str());

    return ss.str();
}

//! Sigmoid
std::string ElemwiseGenUnarySigmoid::GenInlineName() const {
    return "ElemwiseGenUnarySigmoid";
}
std::string ElemwiseGenUnarySigmoid::GenKernelAsmInit(
        std::vector<std::string>) const {
    std::stringstream writer;
    writer << R"(
        size_t x6_iter = nr_elem / (4 * 6);
        size_t offset = x6_iter * 4 * 6;
        float32x4_t lower_range;
        float32x4_t upper_range;
        float32x4_t alpha_9;
        float32x4_t alpha_7;
        float32x4_t alpha_5;
        float32x4_t alpha_3;
        float32x4_t alpha_1;
        float32x4_t beta_10;
        float32x4_t beta_8;
        float32x4_t beta_6;
        float32x4_t beta_4;
        float32x4_t beta_2;
        float32x4_t beta_0;
        float32x4_t one_half;

        const float* const_ptr = &(sigmoid_constants.lower_range);
        if (x6_iter > 0) {
            /**
             * q0 - q5   : squared
             * q6 - q11  : p
             * q12- q17  : val(temp), q
             * q18- q31  : const
             */
            asm volatile(
                    "ld1r {%[lower_range].4s}, [%[const_ptr]], #4\n"
                    "ld1r {%[upper_range].4s}, [%[const_ptr]], #4\n"
                    "ld1r {%[alpha_9].4s},     [%[const_ptr]], #4\n"
                    "ld1r {%[alpha_7].4s},     [%[const_ptr]], #4\n"
                    "ld1r {%[alpha_5].4s},     [%[const_ptr]], #4\n"
                    "ld1r {%[alpha_3].4s},     [%[const_ptr]], #4\n"
                    "ld1r {%[alpha_1].4s},     [%[const_ptr]], #4\n"
                    "ld1r {%[beta_10].4s},     [%[const_ptr]], #4\n"
                    "ld1r {%[beta_8].4s},      [%[const_ptr]], #4\n"
                    "ld1r {%[beta_6].4s},      [%[const_ptr]], #4\n"
                    "ld1r {%[beta_4].4s},      [%[const_ptr]], #4\n"
                    "ld1r {%[beta_2].4s},      [%[const_ptr]], #4\n"
                    "ld1r {%[beta_0].4s},      [%[const_ptr]], #4\n"
                    "ld1r {%[one_half].4s},    [%[const_ptr]], #4\n"

                    "1:\n"
                    "ldr  q12, [%[a_ptr]]     \n"
                    "ldr  q13, [%[a_ptr], #16]\n"
                    "ldr  q14, [%[a_ptr], #32]\n"
                    "ldr  q15, [%[a_ptr], #48]\n"
                    "ldr  q16, [%[a_ptr], #64]\n"
                    "ldr  q17, [%[a_ptr], #80]\n"
                    // auto val = vmaxq_f32(vdupq_n_f32(sigmoid_constants.lower_range),
                    // src);
                    "fmax v12.4s, v12.4s, %[lower_range].4s\n"
                    "fmax v13.4s, v13.4s, %[lower_range].4s\n"
                    "fmax v14.4s, v14.4s, %[lower_range].4s\n"
                    "fmax v15.4s, v15.4s, %[lower_range].4s\n"
                    "fmax v16.4s, v16.4s, %[lower_range].4s\n"
                    "fmax v17.4s, v17.4s, %[lower_range].4s\n"
                    "add %[a_ptr], %[a_ptr], #96\n"

                    //  val = vminq_f32(vdupq_n_f32(sigmoid_constants.upper_range), val);
                    "fmin v12.4s, v12.4s, %[upper_range].4s\n"
                    "fmin v13.4s, v13.4s, %[upper_range].4s\n"
                    "fmin v14.4s, v14.4s, %[upper_range].4s\n"
                    "fmin v15.4s, v15.4s, %[upper_range].4s\n"
                    "fmin v16.4s, v16.4s, %[upper_range].4s\n"
                    "fmin v17.4s, v17.4s, %[upper_range].4s\n"

                    //! auto squared = vmulq_f32(val, val);
                    "fmul v0.4s, v12.4s, v12.4s\n"
                    "fmul v1.4s, v13.4s, v13.4s\n"
                    "fmul v2.4s, v14.4s, v14.4s\n"
                    "fmul v3.4s, v15.4s, v15.4s\n"
                    "fmul v4.4s, v16.4s, v16.4s\n"
                    "fmul v5.4s, v17.4s, v17.4s\n"
                    //    auto p = fma_ps_f32(
                    //             vdupq_n_f32(sigmoid_constants.alpha_7), squared,
                    //             vdupq_n_f32(sigmoid_constants.alpha_9));
                    "fmul v6.4s,  v0.4s,  %[alpha_9].4s\n"
                    "fmul v7.4s,  v1.4s,  %[alpha_9].4s\n"
                    "fmul v8.4s,  v2.4s,  %[alpha_9].4s\n"
                    "fmul v9.4s,  v3.4s,  %[alpha_9].4s\n"
                    "fmul v10.4s, v4.4s,  %[alpha_9].4s\n"
                    "fmul v11.4s, v5.4s,  %[alpha_9].4s\n"
                    "fadd v6.4s,  v6.4s,  %[alpha_7].4s\n"
                    "fadd v7.4s,  v7.4s,  %[alpha_7].4s\n"
                    "fadd v8.4s,  v8.4s,  %[alpha_7].4s\n"
                    "fadd v9.4s,  v9.4s,  %[alpha_7].4s\n"
                    "fadd v10.4s, v10.4s, %[alpha_7].4s\n"
                    "fadd v11.4s, v11.4s, %[alpha_7].4s\n"

                    // p = fma_ps_f32(vdupq_n_f32(sigmoid_constants.alpha_5), p, squared);
                    "fmul v6.4s,   v6.4s,  v0.4s\n"
                    "fmul v7.4s,   v7.4s,  v1.4s\n"
                    "fmul v8.4s,   v8.4s,  v2.4s\n"
                    "fmul v9.4s,   v9.4s,  v3.4s\n"
                    "fmul v10.4s,  v10.4s, v4.4s\n"
                    "fmul v11.4s,  v11.4s, v5.4s\n"
                    "fadd v6.4s,  v6.4s,  %[alpha_5].4s\n"
                    "fadd v7.4s,  v7.4s,  %[alpha_5].4s\n"
                    "fadd v8.4s,  v8.4s,  %[alpha_5].4s\n"
                    "fadd v9.4s,  v9.4s,  %[alpha_5].4s\n"
                    "fadd v10.4s, v10.4s, %[alpha_5].4s\n"
                    "fadd v11.4s, v11.4s, %[alpha_5].4s\n"

                    // p = fma_ps_f32(vdupq_n_f32(sigmoid_constants.alpha_3), p, squared);
                    "fmul v6.4s,   v6.4s,  v0.4s\n"
                    "fmul v7.4s,   v7.4s,  v1.4s\n"
                    "fmul v8.4s,   v8.4s,  v2.4s\n"
                    "fmul v9.4s,   v9.4s,  v3.4s\n"
                    "fmul v10.4s,  v10.4s, v4.4s\n"
                    "fmul v11.4s,  v11.4s, v5.4s\n"
                    "fadd v6.4s,  v6.4s,  %[alpha_3].4s\n"
                    "fadd v7.4s,  v7.4s,  %[alpha_3].4s\n"
                    "fadd v8.4s,  v8.4s,  %[alpha_3].4s\n"
                    "fadd v9.4s,  v9.4s,  %[alpha_3].4s\n"
                    "fadd v10.4s, v10.4s, %[alpha_3].4s\n"
                    "fadd v11.4s, v11.4s, %[alpha_3].4s\n"

                    // p = fma_ps_f32(vdupq_n_f32(sigmoid_constants.alpha_1), p, squared);
                    "fmul v6.4s,   v6.4s,  v0.4s\n"
                    "fmul v7.4s,   v7.4s,  v1.4s\n"
                    "fmul v8.4s,   v8.4s,  v2.4s\n"
                    "fmul v9.4s,   v9.4s,  v3.4s\n"
                    "fmul v10.4s,  v10.4s, v4.4s\n"
                    "fmul v11.4s,  v11.4s, v5.4s\n"
                    "fadd v6.4s,  v6.4s,  %[alpha_1].4s\n"
                    "fadd v7.4s,  v7.4s,  %[alpha_1].4s\n"
                    "fadd v8.4s,  v8.4s,  %[alpha_1].4s\n"
                    "fadd v9.4s,  v9.4s,  %[alpha_1].4s\n"
                    "fadd v10.4s, v10.4s, %[alpha_1].4s\n"
                    "fadd v11.4s, v11.4s, %[alpha_1].4s\n"

                    //     p = vmulq_f32(p, val);
                    "fmul v6.4s,   v6.4s,  v12.4s\n"
                    "fmul v7.4s,   v7.4s,  v13.4s\n"
                    "fmul v8.4s,   v8.4s,  v14.4s\n"
                    "fmul v9.4s,   v9.4s,  v15.4s\n"
                    "fmul v10.4s,  v10.4s, v16.4s\n"
                    "fmul v11.4s,  v11.4s, v17.4s\n"

                    //     auto q = fma_ps_f32(
                    //             vdupq_n_f32(sigmoid_constants.beta_8), squared,
                    //             vdupq_n_f32(sigmoid_constants.beta_10));
                    "fmul v12.4s, v0.4s,  %[beta_10].4s\n"
                    "fmul v13.4s, v1.4s,  %[beta_10].4s\n"
                    "fmul v14.4s, v2.4s,  %[beta_10].4s\n"
                    "fmul v15.4s, v3.4s,  %[beta_10].4s\n"
                    "fmul v16.4s, v4.4s,  %[beta_10].4s\n"
                    "fmul v17.4s, v5.4s,  %[beta_10].4s\n"
                    "fadd v12.4s, v12.4s, %[beta_8].4s\n"
                    "fadd v13.4s, v13.4s, %[beta_8].4s\n"
                    "fadd v14.4s, v14.4s, %[beta_8].4s\n"
                    "fadd v15.4s, v15.4s, %[beta_8].4s\n"
                    "fadd v16.4s, v16.4s, %[beta_8].4s\n"
                    "fadd v17.4s, v17.4s, %[beta_8].4s\n"

                    //     q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_6), q,
                    //     squared);
                    "fmul v12.4s, v12.4s,  v0.4s\n"
                    "fmul v13.4s, v13.4s,  v1.4s\n"
                    "fmul v14.4s, v14.4s,  v2.4s\n"
                    "fmul v15.4s, v15.4s,  v3.4s\n"
                    "fmul v16.4s, v16.4s,  v4.4s\n"
                    "fmul v17.4s, v17.4s,  v5.4s\n"
                    "fadd v12.4s, v12.4s, %[beta_6].4s\n"
                    "fadd v13.4s, v13.4s, %[beta_6].4s\n"
                    "fadd v14.4s, v14.4s, %[beta_6].4s\n"
                    "fadd v15.4s, v15.4s, %[beta_6].4s\n"
                    "fadd v16.4s, v16.4s, %[beta_6].4s\n"
                    "fadd v17.4s, v17.4s, %[beta_6].4s\n"

                    //     q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_4), q,
                    //     squared);
                    "fmul v12.4s, v12.4s,  v0.4s\n"
                    "fmul v13.4s, v13.4s,  v1.4s\n"
                    "fmul v14.4s, v14.4s,  v2.4s\n"
                    "fmul v15.4s, v15.4s,  v3.4s\n"
                    "fmul v16.4s, v16.4s,  v4.4s\n"
                    "fmul v17.4s, v17.4s,  v5.4s\n"
                    "fadd v12.4s, v12.4s, %[beta_4].4s\n"
                    "fadd v13.4s, v13.4s, %[beta_4].4s\n"
                    "fadd v14.4s, v14.4s, %[beta_4].4s\n"
                    "fadd v15.4s, v15.4s, %[beta_4].4s\n"
                    "fadd v16.4s, v16.4s, %[beta_4].4s\n"
                    "fadd v17.4s, v17.4s, %[beta_4].4s\n"

                    //     q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_2), q,
                    //     squared);
                    "fmul v12.4s, v12.4s,  v0.4s\n"
                    "fmul v13.4s, v13.4s,  v1.4s\n"
                    "fmul v14.4s, v14.4s,  v2.4s\n"
                    "fmul v15.4s, v15.4s,  v3.4s\n"
                    "fmul v16.4s, v16.4s,  v4.4s\n"
                    "fmul v17.4s, v17.4s,  v5.4s\n"
                    "fadd v12.4s, v12.4s, %[beta_2].4s\n"
                    "fadd v13.4s, v13.4s, %[beta_2].4s\n"
                    "fadd v14.4s, v14.4s, %[beta_2].4s\n"
                    "fadd v15.4s, v15.4s, %[beta_2].4s\n"
                    "fadd v16.4s, v16.4s, %[beta_2].4s\n"
                    "fadd v17.4s, v17.4s, %[beta_2].4s\n"

                    // q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_0), q, squared);
                    "fmul v12.4s, v12.4s,  v0.4s\n"
                    "fmul v13.4s, v13.4s,  v1.4s\n"
                    "fmul v14.4s, v14.4s,  v2.4s\n"
                    "fmul v15.4s, v15.4s,  v3.4s\n"
                    "fmul v16.4s, v16.4s,  v4.4s\n"
                    "fmul v17.4s, v17.4s,  v5.4s\n"
                    "fadd v12.4s, v12.4s, %[beta_0].4s\n"
                    "fadd v13.4s, v13.4s, %[beta_0].4s\n"
                    "fadd v14.4s, v14.4s, %[beta_0].4s\n"
                    "fadd v15.4s, v15.4s, %[beta_0].4s\n"
                    "fadd v16.4s, v16.4s, %[beta_0].4s\n"
                    "fadd v17.4s, v17.4s, %[beta_0].4s\n"

                    // vaddq_f32(div_ps_f32(p, q),
                    // vdupq_n_f32(sigmoid_constants.one_half));
                    "fdiv v12.4s, v6.4s,  v12.4s\n"
                    "fdiv v13.4s, v7.4s,  v13.4s\n"
                    "fdiv v14.4s, v8.4s,  v14.4s\n"
                    "fdiv v15.4s, v9.4s,  v15.4s\n"
                    "fdiv v16.4s, v10.4s, v16.4s\n"
                    "fdiv v17.4s, v11.4s, v17.4s\n"
                    "subs %w[x6_iter], %w[x6_iter], #1\n"
                    "fadd v12.4s, v12.4s, %[one_half].4s\n"
                    "fadd v13.4s, v13.4s, %[one_half].4s\n"
                    "fadd v14.4s, v14.4s, %[one_half].4s\n"
                    "fadd v15.4s, v15.4s, %[one_half].4s\n"
                    "fadd v16.4s, v16.4s, %[one_half].4s\n"
                    "fadd v17.4s, v17.4s, %[one_half].4s\n"

                    // save it
                    "str  q12, [%[d_ptr]]     \n"
                    "str  q13, [%[d_ptr], #16]\n"
                    "str  q14, [%[d_ptr], #32]\n"
                    "str  q15, [%[d_ptr], #48]\n"
                    "str  q16, [%[d_ptr], #64]\n"
                    "str  q17, [%[d_ptr], #80]\n"
                    "add %[d_ptr], %[d_ptr], #96\n"

                    "bne 1b\n"

                    "2:\n"
                    : [a_ptr] "+r"(src), [d_ptr] "+r"(dst), [const_ptr] "+r"(const_ptr),
                      [x6_iter] "+r"(x6_iter), [lower_range] "=w"(lower_range),
                      [alpha_9] "=w"(alpha_9), [upper_range] "=w"(upper_range),
                      [alpha_7] "=w"(alpha_7), [alpha_5] "=w"(alpha_5),
                      [alpha_3] "=w"(alpha_3), [alpha_1] "=w"(alpha_1),
                      [beta_10] "=w"(beta_10), [beta_8] "=w"(beta_8),
                      [beta_6] "=w"(beta_6), [beta_4] "=w"(beta_4),
                      [beta_2] "=w"(beta_2), [beta_0] "=w"(beta_0),
                      [one_half] "=w"(one_half)
                    :
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                      "v11", "v12", "v13", "v14", "v15", "v16", "v17", "x1", "x2", "x8",
                      "x9", "cc", "memory");
        }

    )";
    writer << "\nfloat32x4_t ones = vdupq_n_f32(1.f);";
    return writer.str();
}

std::string ElemwiseGenUnarySigmoid::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    return m_common_sigmoid_gen->GenKernelSimdUnroll(strs);
}

std::string ElemwiseGenUnarySigmoid::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    return m_common_sigmoid_gen->GenKernelNaiveUnroll(strs);
}

// vim: syntax=cpp.doxygen
