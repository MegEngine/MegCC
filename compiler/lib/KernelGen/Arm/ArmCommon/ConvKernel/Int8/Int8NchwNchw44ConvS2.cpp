#include <sstream>
#include <string>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/ArmCommon/ConvKernel.h"
#include "Common.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

namespace {
std::string pack_src() {
    std::string pack_nchw_src_for_nchw44_conv = R"(
    enum PAD_TYPE {
        TOP_PAD_E = 0,
        NO_PAD_E,
        BOTTOM_PAD_E,
    };
    static inline void pack_one_line(const int8_t *src, int8_t *dst, const int iw, 
            const int pw, enum PAD_TYPE pad_type) {
        const int8_t *src_row0 = src;
        const int8_t *src_row1 = src + iw;
        const int src_expand = 4;
        const int combine_rows = 2;
        const int iw_unroll = 16;
        const int iw_unroll_end = iw / iw_unroll * iw_unroll;

        memset(dst, 0, sizeof(int8_t) * pw * combine_rows * src_expand);
        dst += pw * combine_rows * src_expand;
        int iw_idx = 0;
        for(; iw_idx < iw_unroll_end; iw_idx += iw_unroll) {
            int8x16_t src0 = vld1q_s8(src_row0 + iw_idx);
            int8x16_t src1 = vdupq_n_s8(0);
            if(pad_type == NO_PAD_E) {
                src1 = vld1q_s8(src_row1 + iw_idx);
            } else if (pad_type == TOP_PAD_E) {
                src1 = src0;
                src0 = vdupq_n_s8(0);
            }
            int8x16x2_t src_comb = vzipq_s8(src0, src1);
#define cb(i) \
vst1_s8(dst + i * 8, vreinterpret_s8_s16(vdup_laneq_s16(vreinterpretq_s16_s8(src_comb.val[0]), i)));
            UNROLL_CALL_NOWRAPPER(8, cb)
#undef cb
#define cb(i) \
vst1_s8(dst + (i + 8) * 8, vreinterpret_s8_s16(vdup_laneq_s16(vreinterpretq_s16_s8(src_comb.val[1]), i)));
            UNROLL_CALL_NOWRAPPER(8, cb)
#undef cb
            dst += iw_unroll * combine_rows * src_expand;
        }
        for(; iw_idx < iw; ++iw_idx) {
            int8x8_t src0 = vld1_dup_s8(src_row0 + iw_idx);
            int8x8_t src1 = vdup_n_s8(0);
            if(pad_type == NO_PAD_E) {
                src1 = vld1_dup_s8(src_row1 + iw_idx);
            } else if (pad_type == TOP_PAD_E) {
                src1 = src0;
                src0 = vdup_n_s8(0);
            }
            int8x8x2_t src_comb = vzip_s8(src0, src1);
            vst1_s8(dst, src_comb.val[0]);
            dst += combine_rows * src_expand;
        }
        memset(dst, 0, sizeof(int8_t) * pw * combine_rows * src_expand);
    }

    static inline void pack_nchw_src_for_nchw44_conv(const int8_t *src, int8_t *dst, const int ic,
            const int pad_top, const int pad_bottom, const int ih, const int iw,
            const int iw_pad, const int pw) {
        const int src_expand = 4;
        const int ih_step = 2;
        const int src_ic_stride = iw * ih;
        const int dst_ih_stride = iw_pad * ih_step * src_expand;
        const int dst_ih_step_end = (pad_top + ih) / ih_step * ih_step;
        const int dst_ih = ih + pad_top + pad_bottom;

        for(int ic_idx = 0; ic_idx < ic; ++ic_idx) {
            const int8_t *src_ic = src + ic_idx * src_ic_stride;
            int ih_idx = 0;
            for(; ih_idx < pad_top; ih_idx += ih_step) {
                if(ih_idx + ih_step - 1 < pad_top) {
                    memset(dst, 0, dst_ih_stride * sizeof(int8_t));
                } else {
                    pack_one_line(src_ic, dst, iw, pw, TOP_PAD_E);
                    src_ic += iw;
                }
                dst += dst_ih_stride;
            }
            for(; ih_idx < dst_ih_step_end; ih_idx += ih_step) {
                pack_one_line(src_ic, dst, iw, pw, NO_PAD_E);
                dst += dst_ih_stride;
                src_ic += ih_step * iw;
            }
            for(; ih_idx < dst_ih; ih_idx += ih_step) {
                if(ih_idx >= pad_top + ih) {
                    memset(dst, 0, dst_ih_stride * sizeof(int8_t));
                } else {
                    pack_one_line(src_ic, dst, iw, pw, BOTTOM_PAD_E);
                    src_ic += iw;
                }
                dst += dst_ih_stride;
            }
        }
    }
)";
    return pack_nchw_src_for_nchw44_conv;
}

std::string core_calculate_s2(const int oc_block, const int filter_size) {
    std::stringstream core_calculate;
    core_calculate << R"(
                            const int8_t *weight_kh = weight_oc + ic * weight->layout.stride[1];
                            const int8_t *src_packed_ih = src_packed + 
                                    (ic * IH_pad * IW_pad + ih * stride * IW_pad) * src_expand;
                        )";
    for (int k = 0; k < filter_size / 2; ++k) {
        core_calculate << "{";
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .add("filter_size_2", filter_size / 2)
                                  .render(R"(
                            int8x16_t weight_v[${oc_block}][${filter_size_2}];
                        )");
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"(
                            weight_v[${i}][${j}] = vld1q_s8(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 16);
                        )");
            }
        }
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("nr_src", 4 + filter_size / 2 - 1)
                                  .render(R"(
                            int8x16_t src_v[${nr_src}];
#define cb(iw_inner) \
    src_v[iw_inner] = vld1q_s8(src_packed_ih + (iw + iw_inner) * stride * src_expand * ih_step);
                            UNROLL_CALL_NOWRAPPER(${nr_src}, cb)
#undef cb
    )");
        core_calculate << R"(
#define cb(ow_inner))";
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"( \
    dst_v[${i}][ow_inner] = vdotq_s32_h(weight_v[${i}][${j}], src_v[ow_inner + ${j}], dst_v[${i}][ow_inner], tmp);)");
            }
        }
        core_calculate << R"(
                            UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
        )";

        if (filter_size % 2) {
            core_calculate << StringTemplate::StringTemplateArgs()
                                      .add("oc_block", oc_block)
                                      .render(R"(
                            int8x8_t weight_hv[${oc_block}];
                        )");
            for (int i = 0; i < oc_block; ++i) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", filter_size / 2)
                                          .render(R"(
                            weight_hv[${i}] = vld1_s8(weight_kh + 
                                                    ${i} * weight->layout.stride[0] +
                                                    ${j} * 16);
                        )");
            }
            core_calculate << StringTemplate::StringTemplateArgs()
                                      .add("j", filter_size / 2)
                                      .render(R"(
                            int8x8_t src_hv[4];
#define cb(iw_inner) \
    src_hv[iw_inner] = vld1_s8(src_packed_ih + (iw + iw_inner + ${j}) * stride * src_expand * ih_step);
                            UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
    )");
            core_calculate << R"(
#define cb(ow_inner))";
            for (int i = 0; i < oc_block; ++i) {
                core_calculate
                        << StringTemplate::StringTemplateArgs().add("i", i).render(
                                   R"( \
    dst_v[${i}][ow_inner] = vdot2_s32_h(weight_hv[${i}], src_hv[ow_inner], dst_v[${i}][ow_inner], tmp);)");
            }
            core_calculate << R"(
                            UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
        )";
        }

        core_calculate << R"(
                            weight_kh += 2 * ksize * oc_step;
                            src_packed_ih += 2 * IW_pad * src_expand;
                        )";
        core_calculate << "}";
    }

    if (filter_size % 2) {
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .add("filter_size_2", filter_size / 2)
                                  .render(R"(
                            //! last line
                            int8x8_t weight_hv[${oc_block}][${filter_size_2}];
                        )");
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"(
                            weight_hv[${i}][${j}] = vld1_s8(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 8);
                )");
            }
        }
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("nr_src", 4 + filter_size / 2 - 1)
                                  .render(R"(
                            int8x8_t src_hv[${nr_src}];
#define cb(iw_inner) \
    src_hv[iw_inner] = vldq_tbl_low_s8(src_packed_ih + (iw + iw_inner) * stride * src_expand * ih_step, tbl);
                            UNROLL_CALL_NOWRAPPER(${nr_src}, cb)
#undef cb
    )");
        core_calculate << R"(
#define cb(ow_inner))";
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"( \
    dst_v[${i}][ow_inner] = vdot2_s32_h(weight_hv[${i}][${j}], src_hv[ow_inner + ${j}], dst_v[${i}][ow_inner], tmp);)");
            }
        }
        core_calculate << R"(
                            UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
        )";

        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .render(R"(
                            int16x8_t weight_v16[${oc_block}];
                            int16x8_t src_v16[4];
                            )");
        for (int i = 0; i < oc_block; ++i) {
            core_calculate << StringTemplate::StringTemplateArgs()
                                      .add("i", i)
                                      .add("j", filter_size / 2)
                                      .render(
                                              R"(
                            weight_v16[${i}] = vldq_dup_4s8_8s16(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 8);
            )");
        }
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("j", filter_size / 2)
                                  .render(R"(
#define cb(ow_inner) \
    src_v16[ow_inner] = vld1_dup_s8_s16(src_packed_ih + (iw + ow_inner) * stride * src_expand * ih_step + 16 * ${j});
                            UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
        )");
        for (int i = 0; i < oc_block; ++i) {
            core_calculate << StringTemplate::StringTemplateArgs().add("i", i).render(
                    R"(
#define cb(ow_inner) \
    dst_v[${i}][ow_inner] = vmlal_s16(dst_v[${i}][ow_inner], vget_low_s16(weight_v16[${i}]), vget_low_s16(src_v16[ow_inner]));
                            UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
            )");
        }
    }
    return core_calculate.str();
}

std::string core_calculate_s2_assembly(const int filter_size) {
    std::string core_calculate;
    switch (filter_size) {
        case 3:
            core_calculate = R"(
                const int8_t *weight_oc0_ic = weight_oc + ic * weight->layout.stride[1];
                const int8_t *weight_oc1_ic = weight_oc0_ic + weight->layout.stride[0];
                const int8_t *src_packed_iw = src_packed +
                                                (ic * IH_pad * IW_pad +
                                                ih * stride * IW_pad +
                                                iw * stride * ih_step) * src_expand;
                const int8_t *src_packed_iw_last_line = src_packed_iw + 2 * IW_pad * src_expand;
                const size_t weight_step = weight->layout.stride[1];
                /**
                 * r0-r7 c
                 * r24-r31 temp
                 * r8-r15 src
                 * r16-r19 weight
                 * r20-vtbl
                 */
                asm volatile(
                        //! load src 0,1
                        "ldp q8,q9,  [%[nchw_src_ptr]]\n"
                        "ldr q16, [%[weight_ptr]]\n"
                        "ldp q10,q11,  [%[nchw_src_ptr], #32]\n"
                        "add	x5, %[weight_ptr], #32\n"
                        "smull v24.8h, v8.8b, v16.8b\n"
                        "ldr q17, [%[weight_ptr_oc]]\n"
                        "smull v25.8h, v9.8b, v16.8b\n"
                        "add x6, %[weight_ptr_oc], #32\n"
                        "smull v26.8h, v10.8b, v16.8b\n"
                        "smull v27.8h, v11.8b, v16.8b\n"
                        "smlal2 v24.8h, v8.16b, v16.16b\n"
                        "add	x7, %[nchw_src_ptr_last_line], #64\n"
                        "smlal2 v25.8h, v9.16b, v16.16b\n"
                        "smlal2 v26.8h, v10.16b, v16.16b\n"
                        "smlal2 v27.8h, v11.16b, v16.16b\n"
                        "sadalp %[c00].4s, v24.8h\n"
                        "smull v28.8h, v8.8b, v17.8b\n"
                        "ldr d12,  [%[nchw_src_ptr],#16]\n"
                        "sadalp %[c01].4s, v25.8h\n"
                        "smull v29.8h, v9.8b, v17.8b\n"
                        "ldr d13,  [%[nchw_src_ptr],#32]\n"
                        "sadalp %[c02].4s, v26.8h\n"
                        "smull v30.8h, v10.8b, v17.8b\n"
                        "ldr d14,  [%[nchw_src_ptr],#48]\n"
                        "sadalp %[c03].4s, v27.8h\n"
                        "smull v31.8h, v11.8b, v17.8b\n"
                        "ldr d18,  [%[weight_ptr],#16]\n"
                        "smlal2 v28.8h, v8.16b, v17.16b\n"
                        "ldr d19,  [%[weight_ptr_oc],#16]\n"
                        "smlal2 v29.8h, v9.16b, v17.16b\n"
                        "ldr d15,  [%[nchw_src_ptr],#64]\n"
                        "smlal2 v30.8h, v10.16b, v17.16b\n"
                        "ldp q8,q9,  [%[nchw_src_ptr_last_line]]\n"
                        "smull v24.8h, v12.8b, v18.8b\n"
                        "sadalp %[c10].4s, v28.8h\n"
                        "smlal2 v31.8h, v11.16b, v17.16b\n"
                        "ldp q10,q11,  [%[nchw_src_ptr_last_line], #32]\n"
                        "sadalp %[c11].4s, v29.8h\n"
                        "smull v25.8h, v13.8b, v18.8b\n"
                        "tbl v8.16b, {v8.16b}, %[vtbl].16b\n"
                        "sadalp %[c12].4s, v30.8h\n"
                        "smull v26.8h, v14.8b, v18.8b\n"
                        "ldr d16,  [%[weight_ptr],#24]\n"
                        "sadalp %[c13].4s, v31.8h\n"
                        "ldr d17,  [%[weight_ptr_oc],#24]\n"
                        "smull v27.8h, v15.8b, v18.8b\n"
                        "tbl v9.16b, {v9.16b}, %[vtbl].16b\n"
                        "sadalp %[c00].4s, v24.8h\n"
                        "smull v28.8h, v12.8b, v19.8b\n"
                        "tbl v10.16b, {v10.16b}, %[vtbl].16b\n"
                        "sadalp %[c01].4s, v25.8h\n"
                        "smull v29.8h, v13.8b, v19.8b\n"
                        "tbl v11.16b, {v11.16b}, %[vtbl].16b\n"
                        "sadalp %[c02].4s, v26.8h\n"
                        "smull v30.8h, v14.8b, v19.8b\n"
                        "ld1r {v18.2s}, [x5]\n"
                        "sadalp %[c03].4s, v27.8h\n"
                        "smull v31.8h, v15.8b, v19.8b\n"
                        "ld1r {v19.2s}, [x6]\n"
                        "sadalp %[c10].4s, v28.8h\n"
                        "smull v24.8h, v8.8b, v16.8b\n"
                        "sadalp %[c11].4s, v29.8h\n"
                        "smull v25.8h, v9.8b, v16.8b\n"
                        "dup v12.8b, v9.b[0]\n"
                        "sadalp %[c12].4s, v30.8h\n"
                        "smull v26.8h, v10.8b, v16.8b\n"
                        "dup v12.8b, v9.b[0]\n"
                        "sadalp %[c13].4s, v31.8h\n"
                        "smull v27.8h, v11.8b, v16.8b\n"
                        "dup v13.8b, v10.b[0]\n"
                        "smull v28.8h, v8.8b, v17.8b\n"
                        "dup v14.8b, v11.b[0]\n"
                        "sadalp %[c00].4s, v24.8h\n"
                        "smull v29.8h, v9.8b, v17.8b\n"
                        "ld1r {v15.8b}, [x7]\n"
                        "sadalp %[c01].4s, v25.8h\n"
                        "smull v30.8h, v10.8b, v17.8b\n"
                        "sxtl v12.8h, v12.8b\n"
                        "sxtl v18.8h, v18.8b\n"
                        "sadalp %[c02].4s, v26.8h\n"
                        "smull v31.8h, v11.8b, v17.8b\n"
                        "sxtl v13.8h, v13.8b\n"
                        "sadalp %[c03].4s, v27.8h\n"
                        "smlal %[c00].4s, v12.4h, v18.4h\n"
                        "sxtl v14.8h, v14.8b\n"
                        "sadalp %[c10].4s, v28.8h\n"
                        "smlal %[c01].4s, v13.4h, v18.4h\n"
                        "sxtl v15.8h, v15.8b\n"
                        "sadalp %[c11].4s, v29.8h\n"
                        "smlal %[c02].4s, v14.4h, v18.4h\n"
                        "sxtl v19.8h, v19.8b\n"
                        "sadalp %[c12].4s, v30.8h\n"
                        "add %[weight_ptr], %[weight_ptr], %[weight_step]\n"
                        "smlal %[c03].4s, v15.4h, v18.4h\n"
                        "sadalp %[c13].4s, v31.8h\n"
                        "smlal %[c10].4s, v12.4h, v19.4h\n"
                        "smlal %[c11].4s, v13.4h, v19.4h\n"
                        "smlal %[c12].4s, v14.4h, v19.4h\n"
                        "smlal %[c13].4s, v15.4h, v19.4h\n"
                        :

                        [c00] "+w"(dst_v[0][0]), [c10] "+w"(dst_v[1][0]), [c01] "+w"(dst_v[0][1]),
                        [c11] "+w"(dst_v[1][1]), [c02] "+w"(dst_v[0][2]), [c12] "+w"(dst_v[1][2]),
                        [c03] "+w"(dst_v[0][3]), [c13] "+w"(dst_v[1][3]),

                        [weight_ptr] "+r"(weight_oc0_ic), [weight_ptr_oc] "+r"(weight_oc1_ic)
                        : [vtbl] "w"(tbl), [nchw_src_ptr] "r"(src_packed_iw),
                        [nchw_src_ptr_last_line] "r"(src_packed_iw_last_line),
                        [weight_step] "r"(weight_step)
                        : "x5", "x6", "x7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                        "v15", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27",
                        "v28", "v29", "v30", "v31", "cc", "memory");
            )";
            break;
        case 7:
            core_calculate = R"(
            const int8_t *weight_oc0_ic = weight_oc + ic * weight->layout.stride[1];
            const int8_t *weight_oc1_ic = weight_oc0_ic + weight->layout.stride[0];
            const int8_t *src_packed_iw = src_packed +
                                            (ic * IH_pad * IW_pad +
                                            ih * stride * IW_pad +
                                            iw * stride * ih_step) * src_expand;
            const int8_t *src_packed_iw_last_line = src_packed_iw + 6 * IW_pad * src_expand;
            const size_t src_ih_step = ih_step * src_expand * IW_pad;
            const size_t weight_step = ksize * oc_step * 2;
            const size_t weight_step_small = ksize * oc_step;
            /**
             * r0-r7 c
             * r24-r31 temp
             * r8-r15 src
             * r16-r22 weight
             * r23 vtbl
             */
            asm volatile(

                    "ldp q8, q9, [%[nchw_src_ptr]]\n"
                    "ldp q16, q17, [%[weight_ptr]]\n"
                    "ldp q10, q11, [%[nchw_src_ptr], #32]\n"
                    "smull v24.8h, v8.8b, v16.8b\n"
                    "ldp q19, q20, [%[weight_ptr_oc]]\n"
                    "smull v25.8h, v9.8b, v16.8b\n"
                    "ldp q12, q13, [%[nchw_src_ptr], #64]\n"
                    "smull v26.8h, v10.8b, v16.8b\n"
                    "ldr q18, [%[weight_ptr],#32]\n"
                    "smull v27.8h, v11.8b, v16.8b\n"
                    "ldr q21, [%[weight_ptr_oc],#32]\n"
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "smlal2 v24.8h, v8.16b, v16.16b\n"
                    "smlal2 v25.8h, v9.16b, v16.16b\n"
                    "smlal2 v26.8h, v10.16b, v16.16b\n"
                    "smlal2 v27.8h, v11.16b, v16.16b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v10.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v11.8b, v19.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v8.16b, v19.16b\n"
                    "ldr d8,  [%[nchw_src_ptr],#48]\n"
                    "smlal2 v29.8h, v9.16b, v19.16b\n"
                    "smlal2 v30.8h, v10.16b, v19.16b\n"
                    "smlal2 v31.8h, v11.16b, v19.16b\n"
                    "smull v24.8h, v9.8b, v17.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v10.8b, v17.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v11.8b, v17.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v12.8b, v17.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smlal2 v24.8h, v9.16b, v17.16b\n"
                    "smlal2 v25.8h, v10.16b, v17.16b\n"
                    "smlal2 v26.8h, v11.16b, v17.16b\n"
                    "smlal2 v27.8h, v12.16b, v17.16b\n"
                    "smull v28.8h, v9.8b, v20.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v10.8b, v20.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v11.8b, v20.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v12.8b, v20.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v9.16b, v20.16b\n"
                    "ldr d9,  [%[nchw_src_ptr],#64]\n"
                    "smlal2 v29.8h, v10.16b, v20.16b\n"
                    "ldr d14,  [%[nchw_src_ptr],#80]\n"
                    "smlal2 v30.8h, v11.16b, v20.16b\n"
                    "smlal2 v31.8h, v12.16b, v20.16b\n"
                    "smull v24.8h, v10.8b, v18.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v11.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v12.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v13.8b, v18.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smlal2 v24.8h, v10.16b, v18.16b\n"
                    "ldr d19,  [%[weight_ptr_oc],#48]\n"
                    "smlal2 v25.8h, v11.16b, v18.16b\n"
                    "ldr d15,  [%[nchw_src_ptr],#96]\n"
                    "smlal2 v26.8h, v12.16b, v18.16b\n"
                    "smlal2 v27.8h, v13.16b, v18.16b\n"
                    "ldr d18,  [%[weight_ptr],#48]\n"
                    "smull v28.8h, v10.8b, v21.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v11.8b, v21.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v12.8b, v21.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v13.8b, v21.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v10.16b, v21.16b\n"
                    "add %[nchw_src_ptr], %[nchw_src_ptr], %[src_step]\n"
                    "smlal2 v29.8h, v11.16b, v21.16b\n"
                    "ldp q10, q11, [%[nchw_src_ptr], #32]\n"
                    "add %[weight_ptr], %[weight_ptr], %[weight_step]\n"
                    "smlal2 v30.8h, v12.16b, v21.16b\n"
                    "add %[weight_ptr_oc], %[weight_ptr_oc], "
                    "%[weight_step]\n"
                    "smlal2 v31.8h, v13.16b, v21.16b\n"
                    "ldp q16, q17, [%[weight_ptr]]\n"
                    "smull v24.8h, v8.8b, v18.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v9.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v14.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v15.8b, v18.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "ldp q8, q9, [%[nchw_src_ptr]]\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v14.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v15.8b, v19.8b\n"
                    "ldp q19, q20, [%[weight_ptr_oc]]\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smull v24.8h, v8.8b, v16.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v9.8b, v16.8b\n"
                    "ldp q12, q13, [%[nchw_src_ptr], #64]\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v10.8b, v16.8b\n"
                    "ldr q18, [%[weight_ptr],#32]\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v11.8b, v16.8b\n"
                    "ldr q21, [%[weight_ptr_oc],#32]\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    //! fh = 2
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "smlal2 v24.8h, v8.16b, v16.16b\n"
                    "smlal2 v25.8h, v9.16b, v16.16b\n"
                    "smlal2 v26.8h, v10.16b, v16.16b\n"
                    "smlal2 v27.8h, v11.16b, v16.16b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v10.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v11.8b, v19.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v8.16b, v19.16b\n"
                    "ldr d8,  [%[nchw_src_ptr],#48]\n"
                    "smlal2 v29.8h, v9.16b, v19.16b\n"
                    "smlal2 v30.8h, v10.16b, v19.16b\n"
                    "smlal2 v31.8h, v11.16b, v19.16b\n"
                    "smull v24.8h, v9.8b, v17.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v10.8b, v17.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v11.8b, v17.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v12.8b, v17.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smlal2 v24.8h, v9.16b, v17.16b\n"
                    "smlal2 v25.8h, v10.16b, v17.16b\n"
                    "smlal2 v26.8h, v11.16b, v17.16b\n"
                    "smlal2 v27.8h, v12.16b, v17.16b\n"
                    "smull v28.8h, v9.8b, v20.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v10.8b, v20.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v11.8b, v20.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v12.8b, v20.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v9.16b, v20.16b\n"
                    "ldr d9,  [%[nchw_src_ptr],#64]\n"
                    "smlal2 v29.8h, v10.16b, v20.16b\n"
                    "ldr d14,  [%[nchw_src_ptr],#80]\n"
                    "smlal2 v30.8h, v11.16b, v20.16b\n"
                    "smlal2 v31.8h, v12.16b, v20.16b\n"
                    "smull v24.8h, v10.8b, v18.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v11.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v12.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v13.8b, v18.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smlal2 v24.8h, v10.16b, v18.16b\n"
                    "ldr d19,  [%[weight_ptr_oc],#48]\n"
                    "smlal2 v25.8h, v11.16b, v18.16b\n"
                    "ldr d15,  [%[nchw_src_ptr],#96]\n"
                    "smlal2 v26.8h, v12.16b, v18.16b\n"
                    "smlal2 v27.8h, v13.16b, v18.16b\n"
                    "ldr d18,  [%[weight_ptr],#48]\n"
                    "smull v28.8h, v10.8b, v21.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v11.8b, v21.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v12.8b, v21.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v13.8b, v21.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v10.16b, v21.16b\n"
                    "add %[nchw_src_ptr], %[nchw_src_ptr], %[src_step]\n"
                    "smlal2 v29.8h, v11.16b, v21.16b\n"
                    "add %[weight_ptr], %[weight_ptr], %[weight_step]\n"
                    "smlal2 v30.8h, v12.16b, v21.16b\n"
                    "add %[weight_ptr_oc], %[weight_ptr_oc], "
                    "%[weight_step]\n"
                    "smlal2 v31.8h, v13.16b, v21.16b\n"
                    "ldp q16, q17, [%[weight_ptr]]\n"
                    "smull v24.8h, v8.8b, v18.8b\n"
                    "ldp q10, q11, [%[nchw_src_ptr], #32]\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v9.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v14.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v15.8b, v18.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "ldp q8, q9, [%[nchw_src_ptr]]\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v14.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v15.8b, v19.8b\n"
                    "ldp q19, q20, [%[weight_ptr_oc]]\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smull v24.8h, v8.8b, v16.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v9.8b, v16.8b\n"
                    "ldp q12, q13, [%[nchw_src_ptr], #64]\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v10.8b, v16.8b\n"
                    "ldr q18, [%[weight_ptr],#32]\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v11.8b, v16.8b\n"
                    "ldr q21, [%[weight_ptr_oc],#32]\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    //! fh = 4
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "smlal2 v24.8h, v8.16b, v16.16b\n"
                    "smlal2 v25.8h, v9.16b, v16.16b\n"
                    "smlal2 v26.8h, v10.16b, v16.16b\n"
                    "smlal2 v27.8h, v11.16b, v16.16b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v10.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v11.8b, v19.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v8.16b, v19.16b\n"
                    "ldr d8,  [%[nchw_src_ptr],#48]\n"
                    "smlal2 v29.8h, v9.16b, v19.16b\n"
                    "smlal2 v30.8h, v10.16b, v19.16b\n"
                    "smlal2 v31.8h, v11.16b, v19.16b\n"
                    "smull v24.8h, v9.8b, v17.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v10.8b, v17.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v11.8b, v17.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v12.8b, v17.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smlal2 v24.8h, v9.16b, v17.16b\n"
                    "smlal2 v25.8h, v10.16b, v17.16b\n"
                    "smlal2 v26.8h, v11.16b, v17.16b\n"
                    "smlal2 v27.8h, v12.16b, v17.16b\n"
                    "smull v28.8h, v9.8b, v20.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v10.8b, v20.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v11.8b, v20.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v12.8b, v20.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v9.16b, v20.16b\n"
                    "ldr d9,  [%[nchw_src_ptr],#64]\n"
                    "smlal2 v29.8h, v10.16b, v20.16b\n"
                    "ldr d14,  [%[nchw_src_ptr],#80]\n"
                    "smlal2 v30.8h, v11.16b, v20.16b\n"
                    "smlal2 v31.8h, v12.16b, v20.16b\n"
                    "smull v24.8h, v10.8b, v18.8b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v11.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v12.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v13.8b, v18.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smlal2 v24.8h, v10.16b, v18.16b\n"
                    "ldr d19,  [%[weight_ptr_oc],#48]\n"
                    "smlal2 v25.8h, v11.16b, v18.16b\n"
                    "ldr d15,  [%[nchw_src_ptr],#96]\n"
                    "smlal2 v26.8h, v12.16b, v18.16b\n"
                    "smlal2 v27.8h, v13.16b, v18.16b\n"
                    "ldr d18,  [%[weight_ptr],#48]\n"
                    "smull v28.8h, v10.8b, v21.8b\n"
                    "add %[weight_ptr], %[weight_ptr], %[weight_step]\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v11.8b, v21.8b\n"
                    "add %[weight_ptr_oc], %[weight_ptr_oc], %[weight_step]\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v12.8b, v21.8b\n"
                    "ldr q16, [%[weight_ptr]]\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v13.8b, v21.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smlal2 v28.8h, v10.16b, v21.16b\n"
                    "smlal2 v29.8h, v11.16b, v21.16b\n"
                    "ldp q10, q11, [%[nchw_src_ptr_last_line], #32]\n"
                    "smlal2 v30.8h, v12.16b, v21.16b\n"
                    "smlal2 v31.8h, v13.16b, v21.16b\n"
                    "ldp q12, q13, [%[nchw_src_ptr_last_line], #64]\n"
                    "smull v24.8h, v8.8b, v18.8b\n"
                    "ldr d21, [%[weight_ptr_oc],#16]\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v25.8h, v9.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v26.8h, v14.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v27.8h, v15.8b, v18.8b\n"
                    "ldr d18, [%[weight_ptr],#16]\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "ldp q8, q9, [%[nchw_src_ptr_last_line]]\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v30.8h, v14.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v31.8h, v15.8b, v19.8b\n"
                    "ldr q19, [%[weight_ptr_oc]]\n"
                    "tbl v8.16b, {v8.16b}, %[vtbl].16b\n"
                    "tbl v9.16b, {v9.16b}, %[vtbl].16b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "tbl v10.16b, {v10.16b}, %[vtbl].16b\n"
                    "tbl v11.16b, {v11.16b}, %[vtbl].16b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "tbl v12.16b, {v12.16b}, %[vtbl].16b\n"
                    "tbl v13.16b, {v13.16b}, %[vtbl].16b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    /// last line////
                    "smull v24.8h, v8.8b, v16.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v25.8h, v9.8b, v16.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smull v26.8h, v10.8b, v16.8b\n"
                    "smull v27.8h, v11.8b, v16.8b\n"
                    "smlal2 v24.8h, v9.16b, v16.16b\n"
                    "smlal2 v25.8h, v10.16b, v16.16b\n"
                    "smlal2 v26.8h, v11.16b, v16.16b\n"
                    "smlal2 v27.8h, v12.16b, v16.16b\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v28.8h, v8.8b, v19.8b\n"
                    "sadalp %[c01].4s, v25.8h\n"
                    "smull v29.8h, v9.8b, v19.8b\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v30.8h, v10.8b, v19.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smull v31.8h, v11.8b, v19.8b\n"
                    "smlal2 v28.8h, v9.16b, v19.16b\n"
                    "dup v9.8b, v11.b[0]\n"
                    "smlal2 v29.8h, v10.16b, v19.16b\n"
                    "smlal2 v30.8h, v11.16b, v19.16b\n"
                    "smlal2 v31.8h, v12.16b, v19.16b\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "smull v24.8h, v10.8b, v18.8b\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "smull v25.8h, v11.8b, v18.8b\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "smull v26.8h, v12.8b, v18.8b\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "smull v27.8h, v13.8b, v18.8b\n"
                    "add x10, %[nchw_src_ptr_last_line], #96\n"
                    "sadalp %[c00].4s, v24.8h\n"
                    "smull v28.8h, v10.8b, v21.8b\n"

                    "sadalp %[c01].4s, v25.8h\n"
                    "add x5, %[weight_ptr], #24\n"
                    "smull v29.8h, v11.8b, v21.8b\n"
                    "add x6, %[weight_ptr_oc], #24\n"
                    "sadalp %[c02].4s, v26.8h\n"
                    "smull v30.8h, v12.8b, v21.8b\n"
                    "sadalp %[c03].4s, v27.8h\n"
                    "smull v31.8h, v13.8b, v21.8b\n"
                    "dup v10.8b, v12.b[0]\n"
                    "sadalp %[c10].4s, v28.8h\n"
                    "ld1r {v12.8b}, [x10]\n"
                    "sadalp %[c11].4s, v29.8h\n"
                    "dup v11.8b, v13.b[0]\n"
                    "sadalp %[c12].4s, v30.8h\n"
                    "ld1r {v16.2s}, [x5]\n"
                    "sadalp %[c13].4s, v31.8h\n"
                    "sxtl v16.8h, v16.8b\n"
                    ///////////////last element/////////
                    "add %[weight_ptr], %[weight_ptr], %[weight_step_small]\n"
                    "sxtl v9.8h, v9.8b\n"
                    "ld1r {v19.2s}, [x6]\n"
                    "sxtl v10.8h, v10.8b\n"
                    "sxtl v11.8h, v11.8b\n"
                    "smlal %[c00].4s, v9.4h, v16.4h\n"
                    "sxtl v12.8h, v12.8b\n"
                    "smlal %[c01].4s, v10.4h, v16.4h\n"
                    "sxtl v19.8h, v19.8b\n"
                    "smlal %[c02].4s, v11.4h, v16.4h\n"
                    "smlal %[c03].4s, v12.4h, v16.4h\n"
                    "smlal %[c10].4s, v9.4h, v19.4h\n"
                    "smlal %[c11].4s, v10.4h, v19.4h\n"
                    "smlal %[c12].4s, v11.4h, v19.4h\n"
                    "smlal %[c13].4s, v12.4h, v19.4h\n"
                    :

                    [c00] "+w"(dst_v[0][0]), [c10] "+w"(dst_v[1][0]), [c01] "+w"(dst_v[0][1]),
                    [c11] "+w"(dst_v[1][1]), [c02] "+w"(dst_v[0][2]), [c12] "+w"(dst_v[1][2]),
                    [c03] "+w"(dst_v[0][3]), [c13] "+w"(dst_v[1][3]),
                    [nchw_src_ptr] "+r"(src_packed_iw), [weight_ptr] "+r"(weight_oc0_ic),
                    [weight_ptr_oc] "+r"(weight_oc1_ic)

                    : [vtbl] "w"(tbl),
                      [nchw_src_ptr_last_line] "r"(src_packed_iw_last_line),
                      [src_step] "r"(src_ih_step), [weight_step] "r"(weight_step),
                      [weight_step_small] "r"(weight_step_small)
                    : "x5", "x6", "x7", "x8", "x9", "x10", "v8", "v9", "v10", "v11",
                      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                      "v21", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
                      "cc", "memory");
        )";
            break;
        default:
            CC_ABORT;
    }
    return core_calculate;
}

std::string core_calculate_ow_remain_s2(const int oc_block, const int filter_size) {
    std::stringstream core_calculate;
    core_calculate << StringTemplate::StringTemplateArgs()
                              .add("oc_block", oc_block)
                              .add("filter_size_2", (filter_size + 1) / 2)
                              .render(R"(
                            int8x16_t weight_v[${oc_block}][${filter_size_2}];
                            const int8_t *weight_kh = weight_oc + ic * weight->layout.stride[1];
                            const int8_t *src_packed_ih = src_packed + 
                                    (ic * IH_pad * IW_pad + ih * stride * IW_pad) * src_expand;
                        )");
    for (int k = 0; k < filter_size / 2; ++k) {
        core_calculate << "{";
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"(
                            weight_v[${i}][${j}] = vld1q_s8(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 16);
                        )");
            }
            if (filter_size % 2) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", filter_size / 2)
                                          .render(R"(
                            weight_v[${i}][${j}] = vcombine_s8(
                                                        vld1_s8(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 16),
                                                        vdup_n_s8(0));
                        )");
            }
        }
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("nr_src", 4 + (filter_size + 1) / 2 - 1)
                                  .render(R"(
                            int8x16_t src_v[${nr_src}];
                            for(int iw_inner = 0; iw_inner < ${nr_src} - 4 + ow_remain; ++iw_inner)
                                src_v[iw_inner] = vld1q_s8(src_packed_ih + (iw + iw_inner) * stride * src_expand * ih_step);
    )");
        core_calculate << R"(
                            for(int ow_inner = 0; ow_inner < ow_remain; ++ow_inner){
    )";
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < (filter_size + 1) / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"(
                                dst_v[${i}][ow_inner] = vdotq_s32_h(weight_v[${i}][${j}], src_v[ow_inner + ${j}], dst_v[${i}][ow_inner], tmp);)");
            }
        }
        core_calculate << R"(
                            }

                            weight_kh += 2 * ksize * oc_step;
                            src_packed_ih += 2 * IW_pad * src_expand;
                        )";
        core_calculate << "}";
    }

    if (filter_size % 2) {
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .add("filter_size_2", filter_size / 2)
                                  .render(R"(
                            //! last line
                            int8x8_t weight_hv[${oc_block}][${filter_size_2}];
                        )");
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"(
                            weight_hv[${i}][${j}] = vld1_s8(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 8);
                )");
            }
        }
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("nr_src", 4 + filter_size / 2 - 1)
                                  .render(R"(
                            int8x8_t src_hv[${nr_src}];
                            for(int iw_inner = 0; iw_inner < ${nr_src} - 4 + ow_remain; ++iw_inner)
                                src_hv[iw_inner] = vldq_tbl_low_s8(src_packed_ih + (iw + iw_inner) * stride * src_expand * ih_step, tbl);
    )");
        core_calculate << R"(
                            for(int ow_inner = 0; ow_inner < ow_remain; ++ow_inner){
    )";
        for (int i = 0; i < oc_block; ++i) {
            for (int j = 0; j < filter_size / 2; ++j) {
                core_calculate << StringTemplate::StringTemplateArgs()
                                          .add("i", i)
                                          .add("j", j)
                                          .render(R"(
                                dst_v[${i}][ow_inner] = vdot2_s32_h(weight_hv[${i}][${j}], src_hv[ow_inner + ${j}], dst_v[${i}][ow_inner], tmp);)");
            }
        }
        core_calculate << R"(
                            }
        )";

        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .render(R"(
                            int16x8_t weight_v16[${oc_block}];
                            int16x8_t src_v16[4];
                            )");
        for (int i = 0; i < oc_block; ++i) {
            core_calculate << StringTemplate::StringTemplateArgs()
                                      .add("i", i)
                                      .add("j", filter_size / 2)
                                      .render(
                                              R"(
                            weight_v16[${i}] = vldq_dup_4s8_8s16(weight_kh + 
                                                            ${i} * weight->layout.stride[0] +
                                                            ${j} * 8);
            )");
        }
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("j", filter_size / 2)
                                  .render(R"(
                            for(int ow_inner = 0; ow_inner < ow_remain; ++ow_inner)
                                src_v16[ow_inner] = vld1_dup_s8_s16(src_packed_ih + (iw + ow_inner) * stride * src_expand * ih_step + 16 * ${j});
        )");
        for (int i = 0; i < oc_block; ++i) {
            core_calculate << StringTemplate::StringTemplateArgs().add("i", i).render(
                    R"(
                            for(int ow_inner = 0; ow_inner < ow_remain; ++ow_inner)
                                dst_v[${i}][ow_inner] = vmlal_s16(dst_v[${i}][ow_inner], vget_low_s16(weight_v16[${i}]), vget_low_s16(src_v16[ow_inner]));
            )");
        }
    }
    return core_calculate.str();
}
}  // namespace

std::string Int8NchwNchw44ConvS2::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    writer << "#include \"marm_neon.h\"\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    Tensor* in_weights = inputs[1];
                      )";
    std::string fill_weight_attr = R"(
        out_weights->layout.nr_dim = 4;
        out_weights->layout.dims[0] = in_weights->layout.dims[0];
        out_weights->layout.dims[1] = in_weights->layout.dims[3];
        out_weights->layout.dims[2] = in_weights->layout.dims[1] * in_weights->layout.dims[2];
        out_weights->layout.dims[3] = in_weights->layout.dims[4];

        out_weights->layout.stride[out_weights->layout.nr_dim - 1] = 1;
        for(int i = out_weights->layout.nr_dim - 2; i >= 0; --i){
            out_weights->layout.stride[i] = 
                out_weights->layout.stride[i + 1] * out_weights->layout.dims[i + 1];
        }
        out_weights->dtype.type_enum = TinyNN_QINT8;
        out_weights->name = in_weights->name;
        out_weights->dtype.param.scale = in_weights->dtype.param.scale;
    )";
    std::string fill_weight_transform_template = R"(
        const int oc_div_4 = out_weights->layout.dims[0];
        const int ic = out_weights->layout.dims[1];
        const int kh = ${kh};
        const int kw = ${kw};
        const int oc_step = 4;

        const int ic_step = 2;
        const int kh_step = 2;
        const int kw_step = 2;
        const int src_oc_stride = kh * kw * ic * oc_step;
        const int src_kh_stride = kw * ic * oc_step;
        const int src_kw_stride = ic * oc_step;
        const int dst_oc_stride = src_oc_stride;
        const int dst_ic_stride = kh * kw * oc_step;
        const int dst_kh_stride = kw * oc_step;

        int8_t* outptr = out_weights->ptr;
        const int8_t* inptr = in_weights->ptr;

        static const uint8_t reorder_idx[16] = {0, 8, 1, 9, 2, 10, 3, 11,
                                                    4, 12, 5, 13, 6, 14, 7, 15};
        uint8x16_t tbl_reorder = vld1q_u8(reorder_idx);

        for (int oc_idx = 0; oc_idx < oc_div_4; ++oc_idx) {
            const int8_t *src_ptr_oc = inptr + oc_idx * src_oc_stride;
            int8_t *dst_ptr_oc = outptr + oc_idx * dst_oc_stride;
            int ic_idx = 0;
            for (; ic_idx + ic_step <= ic; ic_idx += ic_step) {
                const int8_t *src_ptr_ic = src_ptr_oc + ic_idx * oc_step;
                int8_t *dst_ptr_ic0 = dst_ptr_oc + ic_idx * dst_ic_stride;
                int8_t *dst_ptr_ic1 = dst_ptr_ic0 + dst_ic_stride;
                int kh_idx = 0;
                //! pack to (oc/4, ic, floor(fh/2), fw, 4(of oc), 2(of fh))
                for (; kh_idx + kh_step <= kh; kh_idx += kh_step) {
                    const int8_t *src_ptr_kh = src_ptr_ic + kh_idx * src_kh_stride;
                    for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                        const int8_t *src_ptr_kw0 = src_ptr_kh + kw_idx * src_kw_stride;
                        const int8_t *src_ptr_kw1 = src_ptr_kw0 + src_kh_stride;
                        int8x8_t src_kh_row0 = vld1_s8(src_ptr_kw0);
                        int8x8_t src_kh_row1 = vld1_s8(src_ptr_kw1);
                        int8x16_t src_combine = vcombine_s8(src_kh_row0, src_kh_row1);
                        int8x16_t src_reorder = vqtbl1q_s8(src_combine, tbl_reorder);
                        vst1_s8(dst_ptr_ic0, vget_low_s8(src_reorder));
                        vst1_s8(dst_ptr_ic1, vget_high_s8(src_reorder));
                        dst_ptr_ic0 += 8;
                        dst_ptr_ic1 += 8;
                    }
                }
                if (kh_idx < kh) {
                    const int8_t *src_ptr_kh = src_ptr_ic + kh_idx * src_kh_stride;
                    int kw_idx = 0;
                    //! pack to (oc/4, ic, 1(last fh), fw/2, 4(of oc), 2(of fw))
                    for(; kw_idx + kw_step <= kw; kw_idx += kw_step) {
                        const int8_t *src_ptr_kw0 = src_ptr_kh + kw_idx * src_kw_stride;
                        const int8_t *src_ptr_kw1 = src_ptr_kw0 + src_kw_stride;
                        int8x8_t src_kw0 = vld1_s8(src_ptr_kw0);
                        int8x8_t src_kw1 = vld1_s8(src_ptr_kw1);
                        int8x16_t src_combine = vcombine_s8(src_kw0, src_kw1);
                        int8x16_t src_reorder = vqtbl1q_s8(src_combine, tbl_reorder);
                        vst1_s8(dst_ptr_ic0, vget_low_s8(src_reorder));
                        vst1_s8(dst_ptr_ic1, vget_high_s8(src_reorder));
                        dst_ptr_ic0 += 8;
                        dst_ptr_ic1 += 8;
                    }
                    //! pack to (oc/4, ic, 1(last fh), 1(last fw), 4(of oc))
                    if (kw_idx < kw) {
                        const int8_t *src_ptr_kw = src_ptr_kh + kw_idx * src_kw_stride;
                        *(int32_t*)dst_ptr_ic0 = *(int32_t*)src_ptr_kw;
                        *(int32_t*)dst_ptr_ic1 = *((int32_t*)src_ptr_kw + 1);
                        dst_ptr_ic0 += 4;
                        dst_ptr_ic1 += 4;
                    }
                }
            }
            if (ic_idx < ic) {
                const int8_t *src_ptr_ic = src_ptr_oc + ic_idx * oc_step;
                int8_t *dst_ptr_ic0 = dst_ptr_oc + ic_idx * dst_ic_stride;
                int kh_idx = 0;
                for (; kh_idx + kh_step <= kh; kh_idx += kh_step) {
                    const int8_t *src_ptr_kh = src_ptr_ic + kh_idx * src_kh_stride;
                    for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                        const int8_t *src_ptr_kw0 = src_ptr_kh + kw_idx * src_kw_stride;
                        const int8_t *src_ptr_kw1 = src_ptr_kw0 + src_kh_stride;
                        int8x8_t src_kh_row0 = vreinterpret_s8_s32(vld1_dup_s32((int32_t*)src_ptr_kw0));
                        int8x8_t src_kh_row1 = vreinterpret_s8_s32(vld1_dup_s32((int32_t*)src_ptr_kw1));
                        int8x16_t src_combine = vcombine_s8(src_kh_row0, src_kh_row1);
                        int8x16_t src_reorder = vqtbl1q_s8(src_combine, tbl_reorder);
                        vst1_s8(dst_ptr_ic0, vget_low_s8(src_reorder));
                        dst_ptr_ic0 += 8;
                    }
                }
                if (kh_idx < kh) {
                    const int8_t *src_ptr_kh = src_ptr_ic + kh_idx * src_kh_stride;
                    int kw_idx = 0;
                    for(; kw_idx + kw_step <= kw; kw_idx += kw_step) {
                        const int8_t *src_ptr_kw0 = src_ptr_kh + kw_idx * src_kw_stride;
                        const int8_t *src_ptr_kw1 = src_ptr_kw0 + src_kw_stride;
                        int8x8_t src_kw0 = vreinterpret_s8_s32(vld1_dup_s32((int32_t*)src_ptr_kw0));
                        int8x8_t src_kw1 = vreinterpret_s8_s32(vld1_dup_s32((int32_t*)src_ptr_kw1));
                        int8x16_t src_combine = vcombine_s8(src_kw0, src_kw1);
                        int8x16_t src_reorder = vqtbl1q_s8(src_combine, tbl_reorder);
                        vst1_s8(dst_ptr_ic0, vget_low_s8(src_reorder));
                        dst_ptr_ic0 += 8;
                    }
                    if (kw_idx < kw) {
                        const int8_t *src_ptr_kw = src_ptr_kh + kw_idx * src_kw_stride;
                        *(int32_t*)dst_ptr_ic0 = *(int32_t*)src_ptr_kw;
                        dst_ptr_ic0 += 4;
                    }
                }
            }
        }
    )";
    std::string fill_weight_transform = StringTemplate::StringTemplateArgs()
                                                .add("kh", filter_size)
                                                .add("kw", filter_size)
                                                .render(fill_weight_transform_template);
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string Int8NchwNchw44ConvS2::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1];
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const uint32_t ih_pad = (ih + 2 * ${pad_h} + 1) / 2 * 2;
        const uint32_t iw_pad = iw + 2 * ${pad_w};
        const uint32_t src_expand = 4;
        size_t res = ic * ih_pad * iw_pad * src_expand * sizeof(int8_t);

        *workspace = res;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .render(workspace_temp);
    return ss.str();
}

bool Int8NchwNchw44ConvS2::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            (ctx->getAttrUInt("kernel_h") == 2 || ctx->getAttrUInt("kernel_h") == 3 ||
             ctx->getAttrUInt("kernel_h") == 5 || ctx->getAttrUInt("kernel_h") == 7) &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 2 && ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU";

    bool type_ok = is_qint8_conv_dtype(ctx);
    bool layout_ok =
            ctx->getAttrOprand("operand:0").shape.size() == 4 &&
            ctx->getAttrOprand(
                       "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1))
                            .shape.size() == 5;
    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           bias_ok;
}

std::string Int8NchwNchw44ConvS2::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include \"marm_neon.h\"\n";
    writer << "#include \"unroll_macro.h\"\n";
    writer << pack_src();
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{";
    std::string init_bias_big_oc = ConvImpl::is_bias(ctx) ? R"(
                        int32x4_t bias0 = vld1q_s32((int32_t*)(inputs[2]->ptr) + oc * oc_step);
                        int32x4_t bias1 = vld1q_s32((int32_t*)(inputs[2]->ptr) + (oc + 1) * oc_step);
                    )"
                                                          : R"(
                        int32x4_t bias0 = vdupq_n_s32(0);
                        int32x4_t bias1 = vdupq_n_s32(0);
                    )";
    std::string init_bias_small_oc = ConvImpl::is_bias(ctx) ? R"(
                        int32x4_t bias0 = vld1q_s32((int32_t*)(inputs[2]->ptr) + oc * oc_step);
                    )"
                                                            : R"(
                        int32x4_t bias0 = vdupq_n_s32(0);
                    )";
    std::string template_kernel = R"(
        Tensor *src = inputs[0];
        Tensor *weight = inputs[1];
        TINYNN_ASSERT(nr_output == 1);
        Tensor *dst = outputs[0];

        const int OC_div_4 = weight->layout.dims[0];
        const int IC = weight->layout.dims[1];
        const int ksize = ${fh};
        const int oc_step = 4;
        TINYNN_ASSERT(src->layout.nr_dim == 4);
        const int N = src->layout.dims[0];
        TINYNN_ASSERT(src->layout.dims[1] == IC);
        const int IH = src->layout.dims[2];
        const int IW = src->layout.dims[3];
        TINYNN_ASSERT(dst->layout.nr_dim == 5);
        TINYNN_ASSERT(N == dst->layout.dims[0]);
        TINYNN_ASSERT(dst->layout.dims[1] == OC_div_4);
        const int OH = dst->layout.dims[2];
        const int OW = dst->layout.dims[3];
        TINYNN_ASSERT(dst->layout.dims[4] == 4);

        const int stride = 2;
        const int src_expand = 4;
        const int ih_step = 2;

        const int ph = ${pad_h};
        const int pw = ${pad_w};
        const int IW_pad = IW + 2 * pw;
        const int IH_pad = ((IH + 2 * ph) + stride - 1) / stride * stride;
        const int ow_unroll = 4;
        const int ow_unroll_end = OW / ow_unroll * ow_unroll;

        const int8_t *src_ptr = src->ptr;
        const int8_t *weight_ptr = weight->ptr;
        int8_t *dst_ptr = dst->ptr;
        int8_t *src_packed = (int8_t*)(workspace->ptr);

        const float src_scale = src->dtype.param.scale;
        const float flt_scale = weight->dtype.param.scale;
        const float dst_scale = 1.f / dst->dtype.param.scale;
        const float bias_scale = src_scale * flt_scale;

        uint8_t tbl_idx[] = {0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8};
        uint8x16_t tbl = vld1q_u8(tbl_idx);

        int16x8_t tmp;
        ${init_store}
        for(int n = 0; n < N; ++n) {
            const int8_t *src_n = src_ptr + n * src->layout.stride[0];
            pack_nchw_src_for_nchw44_conv(src_n, src_packed, IC, ph, ph, IH, IW, IW_pad, pw);
            int oc = 0;
            for(; oc + 1 < OC_div_4; oc += 2) {
                const int8_t *weight_oc = weight_ptr + oc * weight->layout.stride[0];
                ${init_bias_big_oc}
                for(int oh = 0; oh < OH; ++oh) {
                    int ow = 0;
                    for(; ow < ow_unroll_end; ow += ow_unroll) {
                        const int ih = oh;
                        const int iw = ow;
                        int8_t *dst_w = dst_ptr + dst->layout.stride[0] * n +
                                        dst->layout.stride[1] * oc + dst->layout.stride[2] * oh +
                                        dst->layout.stride[3] * ow;
                        int32x4_t dst_v[2][4] = {{bias0, bias0, bias0, bias0}, 
                                                    {bias1, bias1, bias1, bias1}};
                        for(int ic = 0; ic < IC; ++ic) {
#ifdef __aarch64__
                            ${core_calculate_big_oc_step_aarch64}
#else
                            ${core_calculate_big_oc_step}
#endif
                        }
                        ${store_oc8_ow4(dst_w, dst_v, bias_scale, dst_scale)};
                    }
                    if(ow < OW) {
                        const int ow_remain = OW - ow;
                        const int ih = oh;
                        const int iw = ow;
                        int8_t *dst_w = dst_ptr + dst->layout.stride[0] * n +
                                        dst->layout.stride[1] * oc + dst->layout.stride[2] * oh +
                                        dst->layout.stride[3] * ow;
                        int32x4_t dst_v[2][4] = {{bias0, bias0, bias0, bias0},
                                                    {bias1, bias1, bias1, bias1}};
                        for(int ic = 0; ic < IC; ++ic) {
                            ${core_calculate_ow_remain_big_oc_step}
                        }
                        ${store_oc8_ow_remain(dst_w, dst_v, bias_scale, dst_scale, ow_remain)};
                    }
                }
            }
            for(; oc < OC_div_4; ++oc) {
                const int8_t *weight_oc = weight_ptr + oc * weight->layout.stride[0];
                ${init_bias_small_oc}
                for(int oh = 0; oh < OH; ++oh) {
                    int ow = 0;
                    for(; ow < ow_unroll_end; ow += ow_unroll) {
                        const int ih = oh;
                        const int iw = ow;
                        int8_t *dst_w = dst_ptr + dst->layout.stride[0] * n +
                                        dst->layout.stride[1] * oc + dst->layout.stride[2] * oh +
                                        dst->layout.stride[3] * ow;
                        int32x4_t dst_v[1][4] = {{bias0, bias0, bias0, bias0}};
                        for(int ic = 0; ic < IC; ++ic) {
                            ${core_calculate_small_oc_step}
                        }
                        ${store_oc4_ow4(dst_w, dst_v, bias_scale, dst_scale)};
                    }
                    if(ow < OW) {
                        const int ow_remain = OW - ow;
                        const int ih = oh;
                        const int iw = ow;
                        int8_t *dst_w = dst_ptr + dst->layout.stride[0] * n +
                                        dst->layout.stride[1] * oc + dst->layout.stride[2] * oh +
                                        dst->layout.stride[3] * ow;
                        int32x4_t dst_v[1][4] = {{bias0, bias0, bias0, bias0}};
                        for(int ic = 0; ic < IC; ++ic) {
                            ${core_calculate_ow_remain_small_oc_step}
                        }
                        ${store_oc4_ow_remain(dst_w, dst_v, bias_scale, dst_scale, ow_remain)};
                    }
                }
            }
        }
        return TinyNN_SUCCESS;
    )";

    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    std::string core_calculate_small_oc_step = core_calculate_s2(1, filter_size),
                core_calculate_big_oc_step = core_calculate_s2(2, filter_size);
    std::string core_calculate_big_oc_step_aarch64;
    if (filter_size == 3 || filter_size == 7) {
        core_calculate_big_oc_step_aarch64 = core_calculate_s2_assembly(filter_size);
    } else {
        core_calculate_big_oc_step_aarch64 = core_calculate_big_oc_step;
    }
    std::string core_calculate_ow_remain_small_oc_step =
                        core_calculate_ow_remain_s2(1, filter_size),
                core_calculate_ow_remain_big_oc_step =
                        core_calculate_ow_remain_s2(2, filter_size);

    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add("fh", filter_size)
                      .add("init_store", activate_gen->GenIntrinsicInitFloat())
                      .add("store_oc4_ow4",
                           [=](const std::string& dst, const std::string& sum,
                               const std::string& bias_scale,
                               const std::string& dst_scale) -> std::string {
                               return store_ocx_owx(
                                       dst, sum, bias_scale, dst_scale,
                                       *activate_gen.get(), 1, 4);
                           })
                      .add("store_oc8_ow4",
                           [=](const std::string& dst, const std::string& sum,
                               const std::string& bias_scale,
                               const std::string& dst_scale) -> std::string {
                               return store_ocx_owx(
                                       dst, sum, bias_scale, dst_scale,
                                       *activate_gen.get(), 2, 4);
                           })
                      .add("store_oc4_ow_remain",
                           [=](const std::string& dst, const std::string& sum,
                               const std::string& bias_scale,
                               const std::string& dst_scale,
                               const std::string& ow_remain) -> std::string {
                               return store_ocx_ow_remain(
                                       dst, sum, bias_scale, dst_scale, ow_remain,
                                       *activate_gen.get(), 1);
                           })
                      .add("store_oc8_ow_remain",
                           [=](const std::string& dst, const std::string& sum,
                               const std::string& bias_scale,
                               const std::string& dst_scale,
                               const std::string& ow_remain) -> std::string {
                               return store_ocx_ow_remain(
                                       dst, sum, bias_scale, dst_scale, ow_remain,
                                       *activate_gen.get(), 2);
                           })
                      .add("init_bias_big_oc", init_bias_big_oc)
                      .add("init_bias_small_oc", init_bias_small_oc)
                      .add("core_calculate_small_oc_step", core_calculate_small_oc_step)
                      .add("core_calculate_big_oc_step", core_calculate_big_oc_step)
                      .add("core_calculate_big_oc_step_aarch64",
                           core_calculate_big_oc_step_aarch64)
                      .add("core_calculate_ow_remain_small_oc_step",
                           core_calculate_ow_remain_small_oc_step)
                      .add("core_calculate_ow_remain_big_oc_step",
                           core_calculate_ow_remain_big_oc_step)
                      .render(template_kernel);
    writer << "}";
    return writer.str();
}