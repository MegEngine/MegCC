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
    static inline void pack_nchw_src_for_nchw44_conv(const int8_t *src, int8_t *dst, const int ic,
            const int pad_top, const int pad_bottom, const int ih, const int iw,
            const int iw_pad, const int pw, int8_t *buff) {
        static const uint8_t reorder_idx[16] = {0, 1, 0, 1, 0, 1, 0, 1,
                                                2, 3, 2, 3, 2, 3, 2, 3};
        uint8x16_t reorder_tbl = vld1q_u8(reorder_idx);

        const int iw_unroll = 4;
        const int pack_iw_len = 16;
        const int iw_pad_unroll = iw_pad / iw_unroll * iw_unroll;
        const int src_ic_stride = iw * ih;

        memset(buff, 0, sizeof(int8_t) * (iw + 2 * pw));
        for(int ic_idx = 0; ic_idx < ic; ++ic_idx) {
            const int8_t *src_ic = src + ic_idx * src_ic_stride;
            memset(dst, 0, sizeof(int8_t) * iw_pad * (ih + pad_top + pad_bottom) * pack_iw_len);
            dst += iw_pad * pad_top * pack_iw_len;
            for(int ih_idx = 0; ih_idx < ih; ++ih_idx) {
                memcpy(buff + pw, src_ic + iw * ih_idx, sizeof(int8_t) * iw);
                for(int iw_idx = 0; iw_idx < iw_pad_unroll; iw_idx += iw_unroll) {
                    int8x16_t src0 = vld1q_s8(buff + iw_idx);
                    int8x16_t src1 = vld1q_s8(buff + iw_idx + 1);
                    int8x16_t src2 = vld1q_s8(buff + iw_idx + 2);
                    int8x16_t src3 = vld1q_s8(buff + iw_idx + 3);

                    int8x16_t dst0 = vqtbl1q_s8(src0, reorder_tbl);
                    int8x16_t dst1 = vqtbl1q_s8(src1, reorder_tbl);
                    int8x16_t dst2 = vqtbl1q_s8(src2, reorder_tbl);
                    int8x16_t dst3 = vqtbl1q_s8(src3, reorder_tbl);

                    vst1q_s8(dst + iw_idx * pack_iw_len, dst0);
                    vst1q_s8(dst + (iw_idx + 1) * pack_iw_len, dst1);
                    vst1q_s8(dst + (iw_idx + 2) * pack_iw_len, dst2);
                    vst1q_s8(dst + (iw_idx + 3) * pack_iw_len, dst3);
                }
                for(int iw_idx = iw_pad_unroll; iw_idx < iw_pad; ++iw_idx) {
                    int8x16_t src0 = vld1q_s8(buff + iw_idx);
                    int8x16_t dst0 = vqtbl1q_s8(src0, reorder_tbl);
                    vst1q_s8(dst + iw_idx * pack_iw_len, dst0);
                }
                dst += iw_pad * pack_iw_len;
            }
            dst += iw_pad * pad_bottom * pack_iw_len;
        }
    }
)";
    return pack_nchw_src_for_nchw44_conv;
}

std::string core_calculate_s1(const int oc_block, const int filter_size) {
    std::stringstream core_calculate;
    if (filter_size <= 4) {
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .render(R"(
                            for(int fh = 0; fh < ksize; ++fh) {
                                int8x16_t weight_v[${oc_block}];
                            )");
        for (int i = 0; i < oc_block; ++i)
            core_calculate << StringTemplate::StringTemplateArgs().add("i", i).render(
                    R"(
                                weight_v[${i}] = vld1q_s8(weight_oc + 
                                                        ${i} * weight->layout.stride[0] +
                                                        ic * weight->layout.stride[1] + 
                                                        fh * weight->layout.stride[2]);
                    )");
        core_calculate << R"(
                                int8x16_t src_v[8];
#define cb(ow_inner) \
    src_v[ow_inner] = vld1q_s8(src_packed + (ic * IH_pad * IW_pad + (ih + fh) * IW_pad + (iw + ow_inner)) * pack_iw_len); \
    dst_v[0][ow_inner] = vdotq_s32_h(weight_v[0], src_v[ow_inner], dst_v[0][ow_inner], tmp);)";
        if (oc_block == 2)
            core_calculate << R"( \
    dst_v[1][ow_inner] = vdotq_s32_h(weight_v[1], src_v[ow_inner], dst_v[1][ow_inner], tmp);
    )";
        core_calculate << R"(
                                UNROLL_CALL_NOWRAPPER(8, cb)
#undef cb
                        )";
        core_calculate << R"(
                            }
    )";
    } else {
        CC_ASSERT(filter_size <= 8);
        core_calculate << StringTemplate::StringTemplateArgs()
                                  .add("oc_block", oc_block)
                                  .render(R"(
                            for(int fh = 0; fh < ksize; ++fh) {
                                int8x16_t weight_v[${oc_block}][2];
                            )");
        for (int i = 0; i < oc_block; ++i)
            core_calculate << StringTemplate::StringTemplateArgs().add("i", i).render(
                    R"(
                                weight_v[${i}][0] = vld1q_s8(weight_oc + 
                                                        ${i} * weight->layout.stride[0] +
                                                        ic * weight->layout.stride[1] + 
                                                        fh * weight->layout.stride[2]);
                                weight_v[${i}][1] = vld1q_s8(weight_oc + 
                                                        ${i} * weight->layout.stride[0] +
                                                        ic * weight->layout.stride[1] + 
                                                        fh * weight->layout.stride[2] + 16);
                    )");
        core_calculate << R"(
                                int8x16_t src_v[12];
#define cb(iw_inner) \
    src_v[iw_inner] = vld1q_s8(src_packed + (ic * IH_pad * IW_pad + (ih + fh) * IW_pad + (iw + iw_inner)) * pack_iw_len);
                                UNROLL_CALL_NOWRAPPER(12, cb)
#undef cb
                        )";
        core_calculate << R"(
#define cb(ow_inner) \
    dst_v[0][ow_inner] = vdotq_s32_h(weight_v[0][0], src_v[ow_inner], dst_v[0][ow_inner], tmp); \
    dst_v[0][ow_inner] = vdotq_s32_h(weight_v[0][1], src_v[ow_inner + 4], dst_v[0][ow_inner], tmp);)";
        if (oc_block == 2)
            core_calculate << R"( \
    dst_v[1][ow_inner] = vdotq_s32_h(weight_v[1][0], src_v[ow_inner], dst_v[1][ow_inner], tmp); \
    dst_v[1][ow_inner] = vdotq_s32_h(weight_v[1][1], src_v[ow_inner + 4], dst_v[1][ow_inner], tmp);
    )";
        core_calculate << R"(
                                UNROLL_CALL_NOWRAPPER(8, cb)
#undef cb
                        )";
        core_calculate << R"(
                            }
    )";
    }
    return core_calculate.str();
}

std::string core_calculate_ow_remain_s1(const int oc_block, const int filter_size) {
    std::stringstream core_calculate_ow_remain;
    if (filter_size <= 4) {
        core_calculate_ow_remain << StringTemplate::StringTemplateArgs()
                                            .add("oc_block", oc_block)
                                            .render(R"(
                            for(int fh = 0; fh < ksize; ++fh) {
                                int8x16_t weight_v[${oc_block}];
                            )");
        for (int i = 0; i < oc_block; ++i)
            core_calculate_ow_remain
                    << StringTemplate::StringTemplateArgs().add("i", i).render(
                               R"(
                                weight_v[${i}] = vld1q_s8(weight_oc + 
                                                        ${i} * weight->layout.stride[0] +
                                                        ic * weight->layout.stride[1] + 
                                                        fh * weight->layout.stride[2]);
                    )");
        core_calculate_ow_remain << R"(
                                int8x16_t src_v[8];
                                for(int ow_inner = 0; ow_inner < ow_remain; ++ow_inner) {
                                    src_v[ow_inner] = vld1q_s8(src_packed + (ic * IH_pad * IW_pad + (ih + fh) * IW_pad + (iw + ow_inner)) * pack_iw_len);
                                    dst_v[0][ow_inner] = vdotq_s32_h(weight_v[0], src_v[ow_inner], dst_v[0][ow_inner], tmp);)";
        if (oc_block == 2)
            core_calculate_ow_remain << R"(
                                    dst_v[1][ow_inner] = vdotq_s32_h(weight_v[1], src_v[ow_inner], dst_v[1][ow_inner], tmp);
    )";
        core_calculate_ow_remain << R"(
                                }
                            }
    )";
    } else {
        CC_ASSERT(filter_size <= 8);
        core_calculate_ow_remain << StringTemplate::StringTemplateArgs()
                                            .add("oc_block", oc_block)
                                            .render(R"(
                            for(int fh = 0; fh < ksize; ++fh) {
                                int8x16_t weight_v[${oc_block}][2];
                            )");
        for (int i = 0; i < oc_block; ++i)
            core_calculate_ow_remain
                    << StringTemplate::StringTemplateArgs().add("i", i).render(
                               R"(
                                weight_v[${i}][0] = vld1q_s8(weight_oc + 
                                                        ${i} * weight->layout.stride[0] +
                                                        ic * weight->layout.stride[1] + 
                                                        fh * weight->layout.stride[2]);
                                weight_v[${i}][1] = vld1q_s8(weight_oc + 
                                                        ${i} * weight->layout.stride[0] +
                                                        ic * weight->layout.stride[1] + 
                                                        fh * weight->layout.stride[2] + 16);
                    )");
        core_calculate_ow_remain << R"(
                                int8x16_t src_v[12];
                                for(int iw_inner = 0; iw_inner < ow_remain + 4; ++iw_inner)
                                    src_v[iw_inner] = vld1q_s8(src_packed + (ic * IH_pad * IW_pad + (ih + fh) * IW_pad + (iw + iw_inner)) * pack_iw_len);
                        )";
        core_calculate_ow_remain << R"(
                                for(int ow_inner = 0; ow_inner < ow_remain; ++ow_inner) {
                                    dst_v[0][ow_inner] = vdotq_s32_h(weight_v[0][0], src_v[ow_inner], dst_v[0][ow_inner], tmp);
                                    dst_v[0][ow_inner] = vdotq_s32_h(weight_v[0][1], src_v[ow_inner + 4], dst_v[0][ow_inner], tmp);)";
        if (oc_block == 2)
            core_calculate_ow_remain << R"(
                                    dst_v[1][ow_inner] = vdotq_s32_h(weight_v[1][0], src_v[ow_inner], dst_v[1][ow_inner], tmp);
                                    dst_v[1][ow_inner] = vdotq_s32_h(weight_v[1][1], src_v[ow_inner + 4], dst_v[1][ow_inner], tmp);
                                )";
        core_calculate_ow_remain << R"(
                                }
                            }
                            )";
    }
    return core_calculate_ow_remain.str();
}
}  // namespace

std::string Int8NchwNchw44ConvS1::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    writer << "#include \"marm_neon.h\"\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    Tensor* in_weights = inputs[1];
                      )";
    std::string fill_weight_attr = R"(
            out_weights->layout.nr_dim = 5;
            out_weights->layout.dims[0] = in_weights->layout.dims[0];
            out_weights->layout.dims[1] = in_weights->layout.dims[3];
            out_weights->layout.dims[2] = in_weights->layout.dims[1];
            out_weights->layout.dims[3] = (in_weights->layout.dims[2] + 3) & (~3);
            out_weights->layout.dims[4] = in_weights->layout.dims[4];

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
            const int kw_pad = out_weights->layout.dims[3];
            const int kw_remain = kw_pad - ${kw};
            const int oc_step = 4;
            const int oc_div_4 = out_weights->layout.dims[0];
            const int ic = out_weights->layout.dims[1];
            const int src_oc_stride = ${kh} * ${kw} * ic;
            const int dst_oc_stride = ${kh} * kw_pad * ic;
            const int dst_ic_stride = ${kh} * kw_pad;
            int8_t* outptr = out_weights->ptr;
            const int8_t* inptr = in_weights->ptr;
            static const uint8_t transpose_4x4_idx[16] = {0, 4, 1, 5, 2, 6, 3, 7,
                                                        8, 12, 9, 13, 10, 14, 11, 15};
            uint8x16_t tbl_transpose_4x4 = vld1q_u8(transpose_4x4_idx);
            for (int oc_idx = 0; oc_idx < oc_div_4; ++oc_idx) {
                const int32_t* src_ptr_oc = (const int32_t*)inptr + oc_idx * src_oc_stride;
                int32_t* dst_ptr_oc = (int32_t*)outptr + oc_idx * dst_oc_stride;
                //! weight: (oc/4, fh, fw, ic, 4) -> (oc/4, ic, fh, fw_pad, 4)
                for (int kh_idx = 0; kh_idx < ${kh}; ++kh_idx) {
                    for (int kw_idx = 0; kw_idx < ${kw}; ++kw_idx) {
                        for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                            *(dst_ptr_oc + ic_idx * dst_ic_stride) = *src_ptr_oc;
                            ++src_ptr_oc;
                        }
                        ++dst_ptr_oc;
                    }
                    if (kw_remain) {
                        for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                            memset(dst_ptr_oc + ic_idx * dst_ic_stride, 0, sizeof(int32_t) * kw_remain);
                        }
                        dst_ptr_oc += kw_remain;
                    }
                }
                //! pack for dot-like: <fw_pad, 4(of oc)>
                //! <0,0> <0,1> <0,2> <0,3>
                //! <1,0> <1,1> <1,2> <1,3>
                //! <2,0> <2,1> <2,2> <2,3>
                //! <3,0> <3,1> <3,2> <3,3>
                //! --->
                //! <0,0> <1,0> <0,1> <1,1>
                //! <0,2> <1,2> <0,3> <1,3>
                //! <2,0> <3,0> <2,1> <3,1>
                //! <2,2> <3,2> <2,3> <3,3>
                int8_t *trans_dst_ptr = outptr + dst_oc_stride * oc_step * oc_idx;
                for (int i = 0; i < dst_oc_stride * oc_step; i += 16) {
                    int8x16_t tmp = vld1q_s8(trans_dst_ptr + i);
                    vst1q_s8(trans_dst_ptr + i, vqtbl1q_s8(tmp, tbl_transpose_4x4));
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

std::string Int8NchwNchw44ConvS1::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1];
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const uint32_t ih_pad = ih + 2 * ${pad_h};
        const uint32_t iw_pad = iw + 2 * ${pad_w};
        const uint32_t src_expand = 16;
        size_t pack_src_size = ic * ih_pad * iw_pad * src_expand * sizeof(int8_t);

        size_t buff_size = iw_pad * sizeof(int8_t);

        *workspace = pack_src_size + buff_size;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .render(workspace_temp);
    return ss.str();
}

bool Int8NchwNchw44ConvS1::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            (ctx->getAttrUInt("kernel_h") == 1 || ctx->getAttrUInt("kernel_h") == 2 ||
             ctx->getAttrUInt("kernel_h") == 3 || ctx->getAttrUInt("kernel_h") == 5 ||
             ctx->getAttrUInt("kernel_h") == 7) &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 1 && ctx->getAttrUInt("dilate_h") == 1 &&
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
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}

std::string Int8NchwNchw44ConvS1::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include \"marm_neon.h\"\n";
    writer << "#include \"unroll_macro.h\"\n";
    writer << pack_src();
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{";
    std::string init_bias = ConvImpl::is_bias(ctx) ? R"(
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

        const int ph = ${pad_h};
        const int pw = ${pad_w};
        const int IW_pad = IW + 2 * pw;
        const int IH_pad = IH + 2 * ph;
        const int pack_iw_len = 16;
        const int ow_unroll = 8;
        const int ow_unroll_end = OW / ow_unroll * ow_unroll;

        const int8_t *src_ptr = src->ptr;
        const int8_t *weight_ptr = weight->ptr;
        int8_t *dst_ptr = dst->ptr;
        int8_t *buff = (int8_t*)workspace->ptr;
        int8_t *src_packed = (int8_t*)workspace->ptr + IW_pad;

        const float src_scale = src->dtype.param.scale;
        const float flt_scale = weight->dtype.param.scale;
        const float dst_scale = 1.f / dst->dtype.param.scale;
        const float bias_scale = src_scale * flt_scale;

        int16x8_t tmp;
        ${init_store}
        for(int n = 0; n < N; ++n) {
            const int8_t *src_n = src_ptr + n * src->layout.stride[0];
            pack_nchw_src_for_nchw44_conv(src_n, src_packed, IC, ph, ph, IH, IW, IW_pad, pw, buff);
            int oc = 0;
            for(; oc < OC_div_4; ++oc) {
                const int8_t *weight_oc = weight_ptr + oc * weight->layout.stride[0];
                ${init_bias}
                for(int oh = 0; oh < OH; ++oh) {
                    int ow = 0;
                    for(; ow < ow_unroll_end; ow += ow_unroll) {
                        const int ih = oh;
                        const int iw = ow;
                        int8_t *dst_w = dst_ptr + dst->layout.stride[0] * n +
                                        dst->layout.stride[1] * oc + dst->layout.stride[2] * oh +
                                        dst->layout.stride[3] * ow;
                        int32x4_t dst_v[1][8] = {{bias0, bias0, bias0, bias0, bias0, bias0, bias0, bias0}};
                        for(int ic = 0; ic < IC; ++ic) {
                            ${core_calculate}
                        }
                        ${store_oc4_ow8(dst_w, dst_v, bias_scale, dst_scale)};
                    }
                    if(ow < OW) {
                        const int ow_remain = OW - ow;
                        const int ih = oh;
                        const int iw = ow;
                        int8_t *dst_w = dst_ptr + dst->layout.stride[0] * n +
                                        dst->layout.stride[1] * oc + dst->layout.stride[2] * oh +
                                        dst->layout.stride[3] * ow;
                        int32x4_t dst_v[1][8] = {{bias0, bias0, bias0, bias0, bias0, bias0, bias0, bias0}};
                        for(int ic = 0; ic < IC; ++ic) {
                            ${core_calculate_ow_remain}
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
    std::string core_calculate = core_calculate_s1(1, filter_size);
    std::string core_calculate_ow_remain = core_calculate_ow_remain_s1(1, filter_size);
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add("fh", filter_size)
                      .add("init_store", activate_gen->GenIntrinsicInitFloat())
                      .add("store_oc4_ow8",
                           [=](const std::string& dst, const std::string& sum,
                               const std::string& bias_scale,
                               const std::string& dst_scale) -> std::string {
                               return store_ocx_owx(
                                       dst, sum, bias_scale, dst_scale,
                                       *activate_gen.get(), 1, 8);
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
                      .add("init_bias", init_bias)
                      .add("core_calculate", core_calculate)
                      .add("core_calculate_ow_remain", core_calculate_ow_remain)
                      .render(template_kernel);
    writer << "}";
    return writer.str();
}