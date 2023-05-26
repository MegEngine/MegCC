#include <sstream>
#include <string>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/ArmCommon/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

namespace {

std::string gen_common_code() {
    std::stringstream ss;
    ss << R"(
#define rep(i, n)            for (int i = 0; i < (n); ++i)
#define rep_step(i, n, step) for (int i = 0; i < (n); i += (step))

static inline int div_ceil(int x, int r) {
    return (x + r - 1) / r;
}
)";
    return ss.str();
}

std::string gen_copy_padding_code(TContext* ctx) {
    std::stringstream ss;
    std::string body_temp = R"(
static inline void nchw44_pack_src(const int8_t* src, int8_t* dst, int length) {
    static const uint8_t src_idx_buffer[16] = {0, 1, 2, 3, 0, 1, 2, 3,
                                               3, 2, 1, 0, 3, 2, 1, 0};
    const int pack_ic = 4;
    const int simd_len = 16;
    uint8x16_t src_idx = vld1q_u8(src_idx_buffer);
    for (int i = 0; i < length; i++) {
        int8x16_t result = vld_dup_tbl_s32(src + i * pack_ic, src_idx);
        vst1q_s8(dst + i * simd_len, result);
    }
}
 
static void copy_padding_kern(const int8_t* sptr, int8_t* ws_sptr_base, int IH, int IW, int nr_ic, int IH2, int IW2) {
    //! Used for get the workspace offset
    const int pack_ic = 4;
    const int expend_element = 4;

    int nr_pad_w = ${pad_w} * pack_ic * expend_element;
    int nr_pad_h = ${pad_h} * IW2 * pack_ic * expend_element;
    int row_last_pad = (IW2 - IW - ${pad_w}) * pack_ic * expend_element;
    int col_last_pad = (IH2 - IH - ${pad_h}) * IW2 * pack_ic * expend_element;
    
    //! copy to ws_sptr_base to eliminate padding effect
    rep_step(ic_idx, nr_ic, pack_ic) {
        memset(ws_sptr_base, 0, nr_pad_h * sizeof(int8_t));
        ws_sptr_base += nr_pad_h;
        rep(ih_idx, IH) {
            memset(ws_sptr_base, 0, nr_pad_w * sizeof(int8_t));
            ws_sptr_base += nr_pad_w;
            nchw44_pack_src(sptr, ws_sptr_base, IW);
            ws_sptr_base += IW * pack_ic * expend_element;
            sptr += IW * pack_ic;
            memset(ws_sptr_base, 0, row_last_pad * sizeof(int8_t));
            ws_sptr_base += row_last_pad;
        }
        memset(ws_sptr_base, 0, col_last_pad * sizeof(int8_t));
        ws_sptr_base += col_last_pad;
    }
}
)";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .render(body_temp);
    return ss.str();
}

std::string gen_do_conv_code(TContext* ctx) {
    std::stringstream ss;
    std::string code_body = R"(
static void do_conv_kern(int8_t* sptr_base, int8_t* packed_weight_base, int32_t* bptr_base, int8_t* dptr_base, size_t OH, size_t OW, size_t ICPG, size_t OCPG, size_t GROUP, size_t IH2, size_t IW2, 
    size_t workspace_batch_id, size_t workspace_group_id, size_t batch_id, size_t group_id, size_t oc_id, size_t oc_block_num, float scale, float dst_scale) {
    size_t padding_group_size = IH2 * IW2 * ICPG;
    const size_t pack_c = 4;
    const size_t src_expand_size = 4;
    size_t nr_pack_per_step = div_ceil(div_ceil(OCPG, pack_c), oc_block_num);
    size_t oc_block = nr_pack_per_step * pack_c;
    const size_t oc_idx = oc_id * oc_block;
    if (oc_id == (oc_block_num - 1)) {
        oc_block = OCPG - oc_id * nr_pack_per_step * pack_c;
    }
    TINYNN_ASSERT_MSG(
            oc_block % pack_c == 0, "oc must be devisible by 4, but oc = %zu",
            oc_block);
    const int8_t* sptr = sptr_base +
            workspace_batch_id * GROUP * padding_group_size * src_expand_size +
            workspace_group_id * padding_group_size * src_expand_size;

    int8_t* dst_ptr = dptr_base + (batch_id * GROUP * OCPG * OH * OW + group_id * OCPG * OH * OW + oc_idx * OH * OW) * sizeof(int8_t);
    int32_t* bptr = NULL;
    if (bptr_base) {
        bptr = bptr_base + (group_id * OCPG + oc_idx);
    }
    int8_t* packed_weight = packed_weight_base + (group_id * OCPG * ICPG * ${kernel_h} * ${kernel_w} + oc_idx * ICPG * ${kernel_h} * ${kernel_w}) * sizeof(int8_t);
    nchw44_conv_direct_${kernel_h}x${kernel_w}_int8(sptr, packed_weight, bptr, dst_ptr, oc_block, ICPG, IH2, IW2, OH, OW, scale, dst_scale);
}
)";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .render(code_body);
    return ss.str();
}

std::string gen_oc4_kernel_common_code(TContext* ctx) {
    std::stringstream ss;
    std::string body_temp = R"(
static inline void nchw44_conv_direct_${kernel_h}x${kernel_w}_int8(const int8_t* src, const int8_t* filter, const int32_t* bias, int8_t* dst,
        const size_t oc, const size_t ic, const size_t ih, const size_t iw, const size_t oh, const size_t ow, float scale, float dst_scale){
    const size_t fh = ${kernel_h};
    const size_t fw = ${kernel_w};
    const size_t ic_step = 4;
    const size_t oc_step = 4;
    const size_t oh_step = 1;
    const size_t ow_step = 8;
    const size_t stride_h = ${stride_h};
    const size_t stride_w = ${stride_w};
    const int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const int ld_dst_oc = oh * ow * oc_step;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;

    for (size_t oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;  
                nchw44_conv_direct_${kernel_h}x${kernel_w}_int8_impl(src + src_offset, filter + weight_offset, bias + oc_idx,
                             dst + dst_offset, ic, ih, iw, scale,dst_scale);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                nchw44_conv_direct_${kernel_h}x${kernel_w}_int8_impl_remain(src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, scale,dst_scale, ow_remain);
            }
        }
    }
}
)";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .render(body_temp);
    return ss.str();
}

std::string gen_2x2_kernel_common_code(TContext* ctx) {
    std::stringstream ss;
    std::string body_temp = R"(
static inline void nchw44_conv_direct_${kernel_h}x${kernel_w}_int8(const int8_t* src, const int8_t* filter, const int32_t* bias, int8_t* dst,
        const size_t oc, const size_t ic, const size_t ih, const size_t iw, const size_t oh, const size_t ow, float scale, float dst_scale){
    const int fh = ${kernel_h};
    const int fw = ${kernel_w};
    const int oc_step = 4;
    const int ic_step = 4;
    const int big_oc_step = 8;
    const int oh_step = 1;
    const int ow_step = 8;
    const size_t stride_h = ${stride_h};
    const size_t stride_w = ${stride_w};
    const int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const size_t oc_end = oc / big_oc_step * big_oc_step;
    const size_t oc_remain = oc - oc_end;
    const int ld_oc = oh * ow * oc_step;

    for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s${stride_h}_oc8_ow8(src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc, scale, dst_scale);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_2x2s${stride_h}_oc8_ow8_remain(src + src_offset, filter + weight_offset, bias + oc_idx,
                    dst + dst_offset, ic, ih, iw, ld_oc, scale,dst_scale, ow_remain);
            }
        }
    }
    if (oc_remain > 0) {
        const size_t oc_idx = oc_end;
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s${stride_h}_oc4_ow8(src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc, scale,dst_scale);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_2x2s${stride_h}_oc4_ow8_remain(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc, scale, dst_scale,ow_remain);
            }
        }
    }
}
)";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .render(body_temp);
    return ss.str();
}

std::string gen_bias_init_code(bool with_bias, int c_dim) {
    std::stringstream ss;
    std::string bias_init_func, bias_init_code;
    if (c_dim == 1) {
        bias_init_func = with_bias ? "c[0][step] = vld1q_s32(bias_ptr)"
                                   : "c[0][step] = vdupq_n_s32(0)";
        bias_init_code = R"(
#define BAIS_INIT_C1(step) ${bias_init_func};
    UNROLL_CALL_RAW(8, BAIS_INIT_C1);
#undef BAIS_INIT_C1
)";
    } else if (c_dim == 2) {
        bias_init_func = with_bias ? R"(c[0][step] = vld1q_s32(bias_ptr); \
    c[1][step] = vld1q_s32(bias_ptr + oc_step);
)"
                                   : R"(c[0][step] = vdupq_n_s32(0); \
    c[1][step] = vdupq_n_s32(0);
)";
        bias_init_code = R"(
#define BAIS_INIT_C2(step) \
    ${bias_init_func}
    UNROLL_CALL_RAW(8, BAIS_INIT_C2);
#undef BAIS_INIT_C2
)";
    }
    ss << StringTemplate::StringTemplateArgs()
                    .add("bias_init_func", bias_init_func)
                    .render(bias_init_code);
    return ss.str();
}

std::string gen_bias_init_code_remain(bool with_bias, std::string remain, int c_dim) {
    std::stringstream ss;
    std::string bias_init_func, bias_init_remain;
    bias_init_remain = R"(
        for(int i=0; i < ${remain}; i++){
            ${bias_init_func};
        }
)";
    if (c_dim == 1) {
        bias_init_func = with_bias ? "c[0][i] = vld1q_s32(bias_ptr)"
                                   : "c[0][i] = vdupq_n_s32(0)";
    } else if (c_dim == 2) {
        bias_init_func = with_bias ? R"(
            c[0][i] = vld1q_s32(bias_ptr);
            c[1][i] = vld1q_s32(bias_ptr + oc_step);
)"
                                   : R"(
            c[0][i] = vdupq_n_s32(0);
            c[1][i] = vdupq_n_s32(0);
)";
    }
    ss << StringTemplate::StringTemplateArgs()
                    .add("remain", remain)
                    .add("bias_init_func", bias_init_func)
                    .render(bias_init_remain);
    return ss.str();
}

std::string gen_res_store_code(
        std::string reg_name, std::string dst_name,
        const ActivationGenIntrinsicBase& act) {
    std::stringstream ss;
    for (int i = 0; i < 8; ++i) {
        ss << act.GenIntrinsicQuantStore(
                reg_name + "[" + std::to_string(i) + "]",
                dst_name + " + " + std::to_string(i) + " * 4", "scale", "dst_scale");
    }
    return ss.str();
}

std::string gen_res_store_code_remain(
        std::string reg_name, std::string dst_name,
        const ActivationGenIntrinsicBase& act, std::string remain) {
    std::stringstream ss;
    ss << "rep(i, " << remain << ")\n";
    ss << act.GenIntrinsicQuantStore(
            reg_name + "[i]", dst_name + " + i * 4", "scale", "dst_scale");
    return ss.str();
}

std::string gen_2x2_s1_oc8_ow8_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, int ld_dst_oc, float scale, float dst_scale ${remain_param}){
    const int filter_size = 2;
    const int fh = filter_size;
    const int fw = filter_size;
    const int ic_step = 4;
    const int oc_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    int32x4_t c[2][8];
    int8x16_t weight[2][2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[4];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));

            const int8_t* read_weight_ptr = weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);

            c[0][0] = vdotq_s32_h(weight[0][0], src[0], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][0], src[0], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][0], src[1], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][0], src[1], c[1][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[0][1], src[1], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][1], src[1], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][1], src[2], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][1], src[2], c[1][1], temp_c[3]);

            c[0][2] = vdotq_s32_h(weight[0][0], src[2], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][0], src[2], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][0], src[3], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][0], src[3], c[1][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[0][1], src[3], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][1], src[3], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][1], src[4], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][1], src[4], c[1][3], temp_c[3]);

            c[0][4] = vdotq_s32_h(weight[0][0], src[4], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][0], src[4], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][0], src[5], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][0], src[5], c[1][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[0][1], src[5], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][1], src[5], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][1], src[6], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][1], src[6], c[1][5], temp_c[3]);

            c[0][6] = vdotq_s32_h(weight[0][0], src[6], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][0], src[6], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][0], src[7], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][0], src[7], c[1][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[0][1], src[7], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][1], src[7], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][1], src[8], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][1], src[8], c[1][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s1_oc8_ow8")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 2))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()) +
                                 gen_res_store_code(
                                         "c[1]", "dst_ptr + ld_dst_oc",
                                         *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s1_oc8_ow8_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 2))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w") +
                                 gen_res_store_code_remain(
                                         "c[1]", "dst_ptr + ld_dst_oc",
                                         *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_2x2_s1_oc4_ow8_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, int ld_dst_oc, float scale, float dst_scale ${remain_param}){
    const int filter_size = 2;
    const int fh = filter_size;
    const int fw = filter_size;
    const int ic_step = 4;
    const int oc_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[1][2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[2];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));

            const int8_t* read_weight_ptr = weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);

            c[0][0] = vdotq_s32_h(weight[0][0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0][0], src[1], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[0][1], src[1], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0][1], src[2], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0][0], src[2], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0][0], src[3], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[0][1], src[3], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0][1], src[4], c[0][3], temp_c[1]);

            c[0][4] = vdotq_s32_h(weight[0][0], src[4], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0][0], src[5], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[0][1], src[5], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0][1], src[6], c[0][5], temp_c[1]);

            c[0][6] = vdotq_s32_h(weight[0][0], src[6], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0][0], src[7], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[0][1], src[7], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0][1], src[8], c[0][7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s1_oc4_ow8")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s1_oc4_ow8_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_2x2_s2_oc8_ow8_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, int ld_dst_oc, float scale, float dst_scale ${remain_param}){
    const int filter_size = 2;
    const int fh = filter_size;
    const int fw = filter_size;
    const int ic_step = 4;
    const int oc_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    int32x4_t c[2][8];
    int8x16_t weight[2][2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[4];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));

            const int8_t* read_weight_ptr = weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);

            c[0][0] = vdotq_s32_h(weight[0][0], src[0], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][0], src[0], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][0], src[2], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][0], src[2], c[1][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[0][1], src[1], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][1], src[1], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][1], src[3], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][1], src[3], c[1][1], temp_c[3]);

            c[0][2] = vdotq_s32_h(weight[0][0], src[4], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][0], src[4], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][0], src[6], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][0], src[6], c[1][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[0][1], src[5], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][1], src[5], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][1], src[7], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][1], src[7], c[1][3], temp_c[3]);

            src[0] = vld1q_s8(src_ic_0_3 + 9 * 16);
            src[1] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[2] = vld1q_s8(src_ic_0_3 + 11 * 16);
            c[0][4] = vdotq_s32_h(weight[0][0], src[8], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][0], src[8], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][0], src[1], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][0], src[1], c[1][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[0][1], src[0], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][1], src[0], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][1], src[2], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][1], src[2], c[1][5], temp_c[3]);

            src[3] = vld1q_s8(src_ic_0_3 + 12 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 13 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 14 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 15 * 16);
            c[0][6] = vdotq_s32_h(weight[0][0], src[3], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][0], src[3], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][0], src[5], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][0], src[5], c[1][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[0][1], src[4], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][1], src[4], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][1], src[6], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][1], src[6], c[1][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s2_oc8_ow8")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 2))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()) +
                                 gen_res_store_code(
                                         "c[1]", "dst_ptr + ld_dst_oc",
                                         *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s2_oc8_ow8_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 2))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w") +
                                 gen_res_store_code_remain(
                                         "c[1]", "dst_ptr + ld_dst_oc",
                                         *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_2x2_s2_oc4_ow8_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, int ld_dst_oc, float scale,float dst_scale ${remain_param}){
    const int filter_size = 2;
    const int fh = filter_size;
    const int fw = filter_size;
    const int ic_step = 4;
    const int oc_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[2];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));

            const int8_t* read_weight_ptr = weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 9 * 16);
            src[1] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[2] = vld1q_s8(src_ic_0_3 + 11 * 16);
            c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[1], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[0], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[1], src[2], c[0][5], temp_c[1]);

            src[3] = vld1q_s8(src_ic_0_3 + 12 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 13 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 14 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 15 * 16);
            c[0][6] = vdotq_s32_h(weight[0], src[3], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0], src[5], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[1], src[4], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[6], c[0][7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s2_oc4_ow8")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "ker_neon_dirctconv_2x2s2_oc4_ow8_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_3x3_s1_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, float scale, float dst_scale ${remain_param}){
    const int filter_size = 3;
    const int fh = filter_size;
    const int fw = filter_size;
    const int oc_step = 4;
    const int ic_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[3];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[2];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                        fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[1], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[1], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[2], src[3], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0], src[2], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0], src[3], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[1], src[3], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[4], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[2], src[4], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[2], src[5], c[0][3], temp_c[1]);

            c[0][4] = vdotq_s32_h(weight[0], src[4], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[5], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[5], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[1], src[6], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[2], src[6], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[2], src[7], c[0][5], temp_c[1]);

            c[0][6] = vdotq_s32_h(weight[0], src[6], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0], src[7], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[1], src[7], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[8], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[2], src[8], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[2], src[9], c[0][7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_3x3_int8_impl")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_3x3_int8_impl_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_3x3_s2_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, float scale,float dst_scale ${remain_param}){
    const int filter_size = 3;
    const int fh = filter_size;
    const int fw = filter_size;
    const int oc_step = 4;
    const int ic_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[3];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[4];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                        fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[2]);
            c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[2], src[4], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[2], src[6], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[2], src[8], c[0][3], temp_c[3]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));
            src[2] = vld1q_s8((src_ic_0_3 + 12 * 16));
            c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[0], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[9], c[0][4], temp_c[2]);
            c[0][5] = vdotq_s32_h(weight[1], src[1], c[0][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[2], src[0], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[2], src[2], c[0][5], temp_c[1]);

            src[3] = vld1q_s8((src_ic_0_3 + 13 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 14 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 15 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 16 * 16));
            c[0][6] = vdotq_s32_h(weight[0], src[2], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[0], src[4], c[0][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[1], src[3], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[5], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[2], src[4], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[2], src[6], c[0][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_3x3_int8_impl")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_3x3_int8_impl_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_5x5_s1_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, float scale,float dst_scale ${remain_param}){
    const int filter_size = 5;
    const int fh = filter_size;
    const int fw = filter_size;
    const int oc_step = 4;
    const int ic_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[5];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[2];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                        fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
            weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
            weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[1], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[1], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[2], src[3], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[3], src[3], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[3], src[4], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[4], src[4], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[4], src[5], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0], src[2], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0], src[3], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[1], src[3], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[4], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[2], src[4], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[2], src[5], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[3], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[3], src[6], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[4], src[6], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[4], src[7], c[0][3], temp_c[1]);

            c[0][4] = vdotq_s32_h(weight[0], src[4], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[5], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[5], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[1], src[6], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[2], src[6], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[2], src[7], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[3], src[7], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[3], src[8], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[4], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[4], src[9], c[0][5], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));

            c[0][6] = vdotq_s32_h(weight[0], src[6], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0], src[7], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[1], src[7], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[8], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[2], src[8], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[2], src[9], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[3], src[9], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[3], src[0], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[4], src[0], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[4], src[1], c[0][7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_5x5_int8_impl")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_5x5_int8_impl_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_5x5_s2_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, float scale,float dst_scale ${remain_param}){
    const int filter_size = 5;
    const int fh = filter_size;
    const int fw = filter_size;
    const int oc_step = 4;
    const int ic_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[5];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[4];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                        fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
            weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
            weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[2]);
            c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[2], src[4], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[3], src[3], c[0][0], temp_c[2]);
            c[0][1] = vdotq_s32_h(weight[3], src[5], c[0][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[4], src[4], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[4], src[6], c[0][1], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[2], src[6], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[2], src[8], c[0][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[3], src[7], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[3], src[9], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[4], src[8], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[4], src[0], c[0][3], temp_c[3]);

            src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));
            src[2] = vld1q_s8((src_ic_0_3 + 12 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 13 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 14 * 16));
            c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[0], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[9], c[0][4], temp_c[2]);
            c[0][5] = vdotq_s32_h(weight[1], src[1], c[0][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[2], src[0], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[2], src[2], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[3], src[1], c[0][4], temp_c[2]);
            c[0][5] = vdotq_s32_h(weight[3], src[3], c[0][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[4], src[2], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[4], src[4], c[0][5], temp_c[1]);

            src[5] = vld1q_s8((src_ic_0_3 + 15 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 16 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 17 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 18 * 16));
            c[0][6] = vdotq_s32_h(weight[0], src[2], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[0], src[4], c[0][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[1], src[3], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[5], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[2], src[4], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[2], src[6], c[0][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[3], src[5], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[3], src[7], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[4], src[6], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[4], src[8], c[0][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_5x5_int8_impl")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_5x5_int8_impl_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_7x7_s1_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, float scale,float dst_scale ${remain_param}){
    const int filter_size = 7;
    const int fh = filter_size;
    const int fw = filter_size;
    const int oc_step = 4;
    const int ic_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[7];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[2];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                        fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
            weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
            weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);
            weight[5] = vld1q_s8(read_weight_ptr + 5 * 16);
            weight[6] = vld1q_s8(read_weight_ptr + 6 * 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[1], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[1], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[2], src[3], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[3], src[3], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[3], src[4], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[4], src[4], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[4], src[5], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[5], src[5], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[5], src[6], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[6], src[6], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[6], src[7], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0], src[2], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0], src[3], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[1], src[3], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[4], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[2], src[4], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[2], src[5], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[3], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[3], src[6], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[4], src[6], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[4], src[7], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[5], src[7], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[5], src[8], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[6], src[8], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[6], src[9], c[0][3], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));

            c[0][4] = vdotq_s32_h(weight[0], src[4], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[5], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[5], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[1], src[6], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[2], src[6], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[2], src[7], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[3], src[7], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[3], src[8], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[4], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[4], src[9], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[5], src[9], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[5], src[0], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[6], src[0], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[6], src[1], c[0][5], temp_c[1]);

            src[2] = vld1q_s8(src_ic_0_3 + 12 * 16);
            src[3] = vld1q_s8((src_ic_0_3 + 13 * 16));

            c[0][6] = vdotq_s32_h(weight[0], src[6], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0], src[7], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[1], src[7], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[8], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[2], src[8], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[2], src[9], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[3], src[9], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[3], src[0], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[4], src[0], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[4], src[1], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[5], src[1], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[5], src[2], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[6], src[2], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[6], src[3], c[0][7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_7x7_int8_impl")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_7x7_int8_impl_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}

std::string gen_7x7_s2_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline void ${func_name}(const int8_t* src_ptr, const int8_t* weight_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
        int ic, int ih, int iw, float scale,float dst_scale ${remain_param}){
    const int filter_size = 7;
    const int fh = filter_size;
    const int fw = filter_size;
    const int oc_step = 4;
    const int ic_step = 4;
    const int loop_ic_step = 4;
    const int ld_weight_ic4 = 16;
    const int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[1][8];
    int8x16_t weight[7];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[4];

    ${bias_init_func}

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                        fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8(src_ic_0_3 + 1 * 16);
            src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);
            src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 4 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 5 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 6 * 16);
            src[7] = vld1q_s8(src_ic_0_3 + 7 * 16);
            src[8] = vld1q_s8(src_ic_0_3 + 8 * 16);
            src[9] = vld1q_s8(src_ic_0_3 + 9 * 16);

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
            weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
            weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);
            weight[5] = vld1q_s8(read_weight_ptr + 5 * 16);
            weight[6] = vld1q_s8(read_weight_ptr + 6 * 16);

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[2]);
            c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[2], src[4], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[3], src[3], c[0][0], temp_c[2]);
            c[0][1] = vdotq_s32_h(weight[3], src[5], c[0][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[4], src[4], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[4], src[6], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[5], src[5], c[0][0], temp_c[2]);
            c[0][1] = vdotq_s32_h(weight[5], src[7], c[0][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[6], src[6], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[6], src[8], c[0][1], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[1] = vld1q_s8(src_ic_0_3 + 11 * 16);
            src[2] = vld1q_s8(src_ic_0_3 + 12 * 16);
            c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[2], src[6], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[2], src[8], c[0][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[3], src[7], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[3], src[9], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[4], src[8], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[4], src[0], c[0][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[5], src[9], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[5], src[1], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[6], src[0], c[0][2], temp_c[2]);
            c[0][3] = vdotq_s32_h(weight[6], src[2], c[0][3], temp_c[3]);

            src[3] = vld1q_s8(src_ic_0_3 + 13 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 14 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 15 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 16 * 16);
            c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[0], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[9], c[0][4], temp_c[2]);
            c[0][5] = vdotq_s32_h(weight[1], src[1], c[0][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[2], src[0], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[2], src[2], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[3], src[1], c[0][4], temp_c[2]);
            c[0][5] = vdotq_s32_h(weight[3], src[3], c[0][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[4], src[2], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[4], src[4], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[5], src[3], c[0][4], temp_c[2]);
            c[0][5] = vdotq_s32_h(weight[5], src[5], c[0][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[6], src[4], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[6], src[6], c[0][5], temp_c[1]);

            src[7] = vld1q_s8(src_ic_0_3 + 17 * 16);
            src[8] = vld1q_s8(src_ic_0_3 + 18 * 16);
            src[9] = vld1q_s8(src_ic_0_3 + 19 * 16);
            src[0] = vld1q_s8(src_ic_0_3 + 20 * 16);
            c[0][6] = vdotq_s32_h(weight[0], src[2], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[0], src[4], c[0][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[1], src[3], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[5], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[2], src[4], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[2], src[6], c[0][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[3], src[5], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[3], src[7], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[4], src[6], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[4], src[8], c[0][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[5], src[7], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[5], src[9], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[6], src[8], c[0][6], temp_c[2]);
            c[0][7] = vdotq_s32_h(weight[6], src[0], c[0][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    ${init_store}
    ${store_func}
}
)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_7x7_int8_impl")
                    .add("remain_param", "")
                    .add("bias_init_func", gen_bias_init_code(with_bias, 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code("c[0]", "dst_ptr", *activate_gen.get()))
                    .render(kernel_impl);
    ss << StringTemplate::StringTemplateArgs()
                    .add("func_name", "nchw44_conv_direct_7x7_int8_impl_remain")
                    .add("remain_param", ", size_t remain_w")
                    .add("bias_init_func",
                         gen_bias_init_code_remain(with_bias, "remain_w", 1))
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         gen_res_store_code_remain(
                                 "c[0]", "dst_ptr", *activate_gen.get(), "remain_w"))
                    .render(kernel_impl);
    return ss.str();
}
}  // namespace

bool DirectInt8NCHW44::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            (ctx->getAttrUInt("kernel_h") == 2 || ctx->getAttrUInt("kernel_h") == 3 ||
             ctx->getAttrUInt("kernel_h") == 5 || ctx->getAttrUInt("kernel_h") == 7) &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            (ctx->getAttrUInt("stride_w") == 2 || ctx->getAttrUInt("stride_w") == 1) &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (ctx->getAttrStr("sparse") == "DENSE" ||
                          ctx->getAttrStr("sparse") == "GROUP") &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU";
    bool type_ok = is_qint8_conv_dtype(ctx);
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;

    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}

std::string DirectInt8NCHW44::GetKernelSymbol(TContext* ctx) const {
    return "ArmCommon_direct_" + ConvImpl::GetKernelSymbol(ctx);
}

std::string DirectInt8NCHW44::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const size_t src_expand = 4;
        const int pack_c_size = ${pack_c_size};
        const Layout in_layout = inputs[0]->layout;
        const uint32_t batch = in_layout.dims[0];
        const uint32_t icb = in_layout.dims[1];
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        uint32_t ph = ${pad_h};
        uint32_t pw = ${pad_w};

        const uint32_t ih2 = ih + ph * 2;
        const uint32_t iw2 = iw + pw * 2;
        size_t res = ${calc_workspace_func};

        //! The extra 9*16B is used to avoid invalid read in kernel compute
        *workspace = res + 9 * 16;
        return TinyNN_SUCCESS;
    })";
    bool is_group = context->getAttrStr("sparse") == "GROUP";
    std::string calc_workspace_func =
            is_group
                    ? R"(icb * pack_c_size * ih2 * iw2 * sizeof(int8_t) * src_expand)"
                    : R"(batch * icb * pack_c_size * ih2 * iw2 * sizeof(int8_t) * src_expand)";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add("pack_c_size", 4)
                    .add("calc_workspace_func", calc_workspace_func)
                    .render(workspace_temp);
    return ss.str();
}

std::string DirectInt8NCHW44::GetInitBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include <marm_neon.h>\n";
    writer << R"(
static inline void nchw44_pack_filter(const int8_t* src, int8_t* dst) {
    static const uint8_t weight_idx_buffer[16] = {0,  4, 9, 13, 2,  6,  11, 15,
                                                  12, 8, 5, 1,  14, 10, 7,  3};
    uint8x16_t weight_idx = vld1q_u8(weight_idx_buffer);
    int8x16_t result = vldq_tbl_s8(src, weight_idx);
    vst1q_s8(dst, result);
}
)";
    writer << GenCommonRet() << " " << GetInitSignature(context) << "\n";
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    int PACK_C_SIZE = 4;
    Tensor* in_weights = inputs[1];
    Layout in_weight_layout = inputs[1]->layout;

    uint32_t OCB, ICB, Group, FH, FW;
    if(in_weight_layout.nr_dim == 7){
        Group = in_weight_layout.dims[0];
        OCB = in_weight_layout.dims[1];
        ICB = in_weight_layout.dims[2];
        FH = in_weight_layout.dims[3];
        FW = in_weight_layout.dims[4];
    } else {
        Group = 1;
        OCB = in_weight_layout.dims[0];
        ICB = in_weight_layout.dims[1];
        FH = in_weight_layout.dims[2];
        FW = in_weight_layout.dims[3];
    }
)";
    std::string fill_weight_attr = R"(
        out_weights->layout.nr_dim = 6;
        out_weights->layout.dims[0] = Group;
        out_weights->layout.dims[1] = OCB;
        out_weights->layout.dims[2] = ICB;
        out_weights->layout.dims[3] = FH;
        out_weights->layout.dims[4] = FW;
        out_weights->layout.dims[5] = PACK_C_SIZE * PACK_C_SIZE;
        out_weights->layout.stride[5] = 1;
        out_weights->layout.stride[4] = out_weights->layout.dims[5] * out_weights->layout.stride[5];
        out_weights->layout.stride[3] = out_weights->layout.dims[4] * out_weights->layout.stride[4];
        out_weights->layout.stride[2] = out_weights->layout.dims[3] * out_weights->layout.stride[3];
        out_weights->layout.stride[1] = out_weights->layout.dims[2] * out_weights->layout.stride[2];
        out_weights->layout.stride[0] = out_weights->layout.dims[1] * out_weights->layout.stride[1];
        out_weights->dtype.type_enum = TinyNN_QINT8;
        out_weights->name = in_weights->name;
        out_weights->dtype.param.scale = in_weights->dtype.param.scale;
)";
    std::string fill_weight_transform = R"(
        int8_t* outptr = out_weights->ptr;
        int8_t* inptr = in_weights->ptr;
    
        for (size_t group = 0; group < Group; group++) {
            for (size_t ocb = 0; ocb < OCB; ocb++) {
                for (size_t icb = 0; icb < ICB; icb++) {
                    for (size_t fh = 0; fh < FH; fh++) {
                        for (size_t fw = 0; fw < FW; fw++) {
                            nchw44_pack_filter(inptr, outptr);
                            inptr += 16;
                            outptr += 16;
                        }
                    }
                }
            }
        }
)";

    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string DirectInt8NCHW44::GetKernelBody(TContext* context) const {
    int stride = context->getAttrUInt("stride_h");
    int kernel = context->getAttrUInt("kernel_h");
    bool with_bias = ConvImpl::is_bias(context);
    std::string bias_str = with_bias ? "inputs[2]->ptr" : "0";
    std::string nonline_mode = context->haveAttr("nonlineMode")
                                     ? context->getAttrStr("nonlineMode")
                                     : "IDENTITY";

    std::stringstream writer;
    writer << R"(
        #include <marm_neon.h>
        #include <string.h>
        #include "unroll_macro.h"
    )";
    writer << gen_common_code();
    writer << "\n\n";
    if (1 == stride) {
        if (kernel == 2) {
            writer << gen_2x2_s1_oc8_ow8_kern(context, with_bias, nonline_mode);
            writer << gen_2x2_s1_oc4_ow8_kern(context, with_bias, nonline_mode);
        } else if (kernel == 3) {
            writer << gen_3x3_s1_kern(context, with_bias, nonline_mode);
        } else if (kernel == 5) {
            writer << gen_5x5_s1_kern(context, with_bias, nonline_mode);
        } else if (kernel == 7) {
            writer << gen_7x7_s1_kern(context, with_bias, nonline_mode);
        } else {
            CC_ABORT
                    << "unsupported kernel size for stride1 in int8 nchw44 direct conv "
                       "kernel.\n";
        }
    } else if (2 == stride) {
        if (kernel == 2) {
            writer << gen_2x2_s2_oc8_ow8_kern(context, with_bias, nonline_mode);
            writer << gen_2x2_s2_oc4_ow8_kern(context, with_bias, nonline_mode);
        } else if (kernel == 3) {
            writer << gen_3x3_s2_kern(context, with_bias, nonline_mode);
        } else if (kernel == 5) {
            writer << gen_5x5_s2_kern(context, with_bias, nonline_mode);
        } else if (kernel == 7) {
            writer << gen_7x7_s2_kern(context, with_bias, nonline_mode);
        } else {
            CC_ABORT
                    << "unsupported kernel size for stride2 in int8 nchw44 direct conv "
                       "kernel.\n";
        }
    } else {
        CC_ABORT << "unsupported stride in int8 nchw44 direct conv kernel.\n";
    }
    if (kernel == 2) {
        writer << gen_2x2_kernel_common_code(context);
    } else {
        writer << gen_oc4_kernel_common_code(context);
    }

    writer << gen_copy_padding_code(context);
    writer << gen_do_conv_code(context);

    writer << GenCommonRet() << " " << GetKernelSignature(context) << "{\n";
    std::string body_temp = R"(
    const int pack_ic_size = 4;
    const int N = inputs[0]->layout.dims[0];
    const int ICB = inputs[0]->layout.dims[1];
    const int IH = inputs[0]->layout.dims[2];
    const int IW = inputs[0]->layout.dims[3];
    const int IC = ICB * pack_ic_size;
    const int ICB_stride = IH * IW * pack_ic_size;
    const int in_batch_stride = IC * IH * IW;

    const size_t pack_oc_size = 4;
    const int Group = inputs[1]->layout.dims[0];
    const int OCB = inputs[1]->layout.dims[1];
    const int OH = outputs[0]->layout.dims[2];
    const int OW = outputs[0]->layout.dims[3];
    const int OC = Group * OCB * pack_oc_size;
    const int OCB_stride = OH * OW * pack_oc_size;

    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    // this must be 1.f/dst_scale for quant data
    const float dst_scale_inv = 1.f / dst_scale;
    const float scale = src_scale * flt_scale;

    int8_t* input_data = inputs[0]->ptr;
    int8_t* output_data = outputs[0]->ptr;
    int8_t* weight_data = inputs[1]->ptr;
    int8_t* padding_src = workspace->ptr;
    int32_t* bias_data = ${bias_str};
    
    int IH2, IW2;
    IH2 = IH + ${pad_h} * 2;
    IW2 = IW + ${pad_w} * 2;
    const int icpg = div_ceil(IC, Group);
    const int ocpg = div_ceil(OC, Group);
    int padding_group_size = icpg * IH2 * IW2;
    const int expend_element = 4;

    size_t oc_step = pack_oc_size;
    if (${kernel_h} == 2 && ${kernel_w} == 2 && ocpg >= 8) {
        oc_step = 8;
    }
    const uint32_t oc_block_num = div_ceil(ocpg, oc_step);

    ${padding_do_conv_body}
    return TinyNN_SUCCESS;
    })";

    bool is_group = context->getAttrStr("sparse") == "GROUP";
    std::string padding_do_conv_body = is_group ? R"(
    rep(batch_id, N){
        rep(group_id, Group){
            size_t batch_offset = batch_id * in_batch_stride * sizeof(int8_t);
            size_t group_offset = group_id * icpg * IH * IW * sizeof(int8_t);
            const int8_t* src_ptr = input_data + batch_offset + group_offset;
            copy_padding_kern(src_ptr, padding_src, IH, IW, icpg, IH2, IW2);
            do_conv_kern(padding_src, weight_data, bias_data, output_data, OH, OW, icpg, ocpg, Group, IH2, IW2, 
                0, 0, batch_id, group_id, 0, 1, scale, dst_scale_inv);
        }
    }
)"
                                                : R"(
    rep(batch_id, N){
        rep(icb, ICB){
            size_t src_batch_offset = batch_id * in_batch_stride * sizeof(int8_t);
            size_t src_channel_offset = icb * ICB_stride * sizeof(int8_t);
            const int8_t* src_ptr = input_data + src_batch_offset + src_channel_offset;
            int8_t* ws_sptr_base = padding_src + (batch_id * padding_group_size + 
                                                        icb * 4 * IH2 * IW2) * sizeof(int8_t) * expend_element;
            copy_padding_kern(src_ptr, ws_sptr_base, IH, IW, pack_ic_size, IH2, IW2);
        }
    }
    rep(batch_id, N){
        rep(oc_id, oc_block_num){
            do_conv_kern(padding_src, weight_data, bias_data, output_data, OH, OW, icpg, ocpg, Group, IH2, IW2, 
                batch_id, 0, batch_id, 0, oc_id, oc_block_num, scale, dst_scale_inv);
        }
    }
)";

    writer << StringTemplate::StringTemplateArgs(context)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add("padding_do_conv_body", padding_do_conv_body)
                      .add("bias_str", bias_str)
                      .render(body_temp);
    return writer.str();
}
