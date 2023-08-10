#include "Int8DirectNchwBase.h"

namespace megcc {
namespace KernelGen {
namespace Armv7 {
std::string Int8DirectNchwHelperBase::gen_common_code() {
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

std::string Int8DirectNchwHelperBase::gen_copy_padding_code(TContext* ctx) {
    std::stringstream ss;
    std::string body_temp = R"(
static void copy_padding_kern(const int8_t* sptr, int8_t* ws_sptr_base, int IH, int IW, int nr_ic, int IH2, int IW2) {
    size_t PH = ${pad_h};
    size_t PW = ${pad_w};
    size_t padding_group_size = IH2 * IW2 * nr_ic;

    memset(ws_sptr_base, 0, padding_group_size);
    rep(ic, nr_ic) {
        //! copy to sptr_base to eliminate padding effect
        const int8_t* src_base = sptr + ic * IH * IW;
        int8_t* dst_base = ws_sptr_base + ic * IH2 * IW2;
        rep(ih, IH) {
            memcpy(
                    dst_base + (ih + PH) * IW2 + PW, src_base + ih * IW,
                    sizeof(int8_t) * IW);
        }
    }
}
)";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .render(body_temp);
    return ss.str();
}

std::string Int8DirectNchwHelperBase::gen_do_conv_code(
        TContext* ctx, bool with_bias, std::string nonline_mode) {
    std::string func_name = gen_kern_name(ctx);
    std::string code_body =
            StringTemplate::StringTemplateArgs()
                    .add("func_name", func_name)
                    .add("kern", gen_kern(ctx, with_bias, nonline_mode, func_name))
                    .render(R"(
${kern}

//! calculate one output channel
static void do_conv_kern(const int8_t* src, const int8_t* filter, const int32_t* bias, int32_t* temp, int8_t* dst, size_t OH2, size_t OW2, size_t ICPG, size_t IH2, size_t IW2, 
    size_t FH, size_t FW, const float bias_scale, const float inv_dst_scale) {
    const size_t LDA = IH2 * IW2;
    const size_t LDB = FH * FW;
    ${func_name}(src, filter, bias, temp, dst, IH2, IW2, OH2, OW2, 1, ICPG == 1, bias_scale, inv_dst_scale);
    src += LDA;
    filter += LDB;
    for(int i = 1; i < ICPG - 1; ++i){
        ${func_name}(src, filter, bias, temp, dst, IH2, IW2, OH2, OW2, 0, 0, bias_scale, inv_dst_scale);
        src += LDA;
        filter += LDB;
    }
    if(ICPG > 1){
        ${func_name}(src, filter, bias, temp, dst, IH2, IW2, OH2, OW2, 0, 1, bias_scale, inv_dst_scale);
    }
}
)");
    return code_body;
}

std::string Int8DirectNchwHelperBase::gen_res_store_code(
        const std::string& reg_name, const std::string& dst_name,
        const ArmCommon::ActivationGenIntrinsicBase& act) {
    std::stringstream ss;
    ss << act.GenIntrinsicQuantStore(reg_name, dst_name, "bias_scale", "inv_dst_scale");
    return ss.str();
}

std::string Int8DirectNchwHelperBase::gen_kern_name(TContext* ctx) {
    return StringTemplate::StringTemplateArgs(ctx)
            .add_ctx_int("stride_h")
            .add_ctx_int("kernel_h")
            .add_ctx_int("kernel_w")
            .render("int8_direct_s${stride_h}_conv${kernel_h}x${kernel_w}");
}

std::string Int8DirectNchwBase::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << helper->gen_need_copy_padding();
    ss << helper->gen_get_rectified_size(context);
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t batch = in_layout.dims[0];
        const uint32_t ic = in_layout.dims[1];
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        uint32_t ph = ${pad_h};
        uint32_t pw = ${pad_w};
        uint32_t fh = ${kernel_h};
        uint32_t fw = ${kernel_w};
        uint32_t sh = ${stride_h};
        uint32_t sw = ${stride_w};
        const uint32_t oh = (ih - fh + 2 * ph) / sh + 1;
        const uint32_t ow = (iw - fw + 2 * pw) / sw + 1;
        size_t ih2, iw2, oh2, ow2;
        get_rectified_size(ih, iw, oh, ow, &ih2, &iw2, &oh2, &ow2);
        size_t res = oh2 * ow2 * sizeof(int32_t);
        if (need_src_copy_padding(ph, pw, ow)) {
            const Layout filter_layout = inputs[1]->layout;
            uint32_t icpg = filter_layout.dims[1];
            if (filter_layout.nr_dim == 5) { //! group NCHW
                icpg = filter_layout.dims[2];
            }
            res += icpg * ih2 * iw2 * sizeof(int8_t);
        }
        if (need_dst_copy_padding(ow)){
            res += oh2 * ow2 * sizeof(int8_t);
        }

        *workspace = res;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .render(workspace_temp);
    return ss.str();
}

std::string Int8DirectNchwBase::GetKernelBody(TContext* context) const {
    bool with_bias = ConvImpl::is_bias(context);
    std::string bias_str = with_bias ? "inputs[2]->ptr" : "0";
    std::string nonline_mode = context->haveAttr("nonlineMode")
                                     ? context->getAttrStr("nonlineMode")
                                     : "IDENTITY";

    std::stringstream writer;
    writer << R"(
        #include <marm_neon.h>
        #include <string.h>
    )";
    writer << Int8DirectNchwHelperBase::gen_common_code();
    writer << "\n\n";

    writer << helper->gen_get_rectified_size(context);
    writer << helper->gen_need_copy_padding();

    writer << Int8DirectNchwHelperBase::gen_copy_padding_code(context);
    writer << helper->gen_do_conv_code(context, with_bias, nonline_mode);

    writer << GenCommonRet() << " " << GetKernelSignature(context) << "{\n";
    std::string body_temp = R"(
    const size_t N = inputs[0]->layout.dims[0];
    const size_t IC = inputs[0]->layout.dims[1];
    const size_t IH = inputs[0]->layout.dims[2];
    const size_t IW = inputs[0]->layout.dims[3];
    const size_t IC_stride = IH * IW;
    const size_t in_batch_stride = IC * IH * IW;

    const size_t Group = ${group_str};
    const size_t OC = outputs[0]->layout.dims[1];
    const size_t OH = outputs[0]->layout.dims[2];
    const size_t OW = outputs[0]->layout.dims[3];
    const size_t OC_stride = OH * OW;
    const size_t out_batch_stride = OC * OH * OW;

    const size_t FH = ${kernel_h};
    const size_t FW = ${kernel_w};

    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    // this must be 1.f/dst_scale for quant data
    const float dst_scale_inv = 1.f / dst_scale;
    const float scale = src_scale * flt_scale;

    int8_t* input_data = inputs[0]->ptr;
    int8_t* output_data = outputs[0]->ptr;
    int8_t* weight_data = inputs[1]->ptr;
    int32_t* bias_data = ${bias_str};
    
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(IH, IW, OH, OW, &IH2, &IW2, &OH2, &OW2);
    const int icpg = div_ceil(IC, Group);
    const int ocpg = div_ceil(OC, Group);

    int need_src_copy_padding_flag = need_src_copy_padding(${pad_h}, ${pad_w}, OW);
    int need_dst_copy_padding_flag = need_dst_copy_padding(OW);
    int32_t* temp_buffer = workspace->ptr;
    void* ws_usable = temp_buffer + OH2 * OW2;
    int8_t* padding_src = NULL;
    if (need_src_copy_padding_flag) {
        padding_src = ws_usable;
        ws_usable = padding_src + icpg * IH2 * IW2;
    }
    int8_t* dst_buffer = NULL;
    if (need_dst_copy_padding_flag) {
        dst_buffer = ws_usable;
    }

    ${padding_do_conv_body}
    return TinyNN_SUCCESS;
    })";

    std::string padding_do_conv_body = R"(
    rep(batch_id, N){
        rep(group_id, Group){
            size_t batch_offset = batch_id * in_batch_stride;
            size_t group_offset = group_id * icpg * IH * IW;
            const int8_t* src_ptr = input_data + batch_offset + group_offset;
            if (need_src_copy_padding_flag){
                copy_padding_kern(src_ptr, padding_src, IH, IW, icpg, IH2, IW2);
                src_ptr = padding_src;
            }

            const int8_t* filter_oc_ptr = weight_data + group_id * ocpg * icpg * FH * FW;
            int8_t* output_oc_ptr = output_data + batch_id * out_batch_stride + group_id * ocpg * OC_stride;
            rep(oc_idx, ocpg){
                if (need_dst_copy_padding_flag) {
                    do_conv_kern(src_ptr, filter_oc_ptr, bias_data + group_id * ocpg + oc_idx, temp_buffer, dst_buffer, OH2, OW2, icpg, IH2, IW2, 
                        FH, FW, scale, dst_scale_inv);
                    rep(oh, OH){
                        memcpy(output_oc_ptr + oh * OW, dst_buffer + oh * OW2, OW * sizeof(int8_t));
                    }
                } else {
                    do_conv_kern(src_ptr, filter_oc_ptr, bias_data + group_id * ocpg + oc_idx, temp_buffer, output_oc_ptr, OH2, OW2, icpg, IH2, IW2, 
                        FH, FW, scale, dst_scale_inv);
                }
                filter_oc_ptr += icpg * FH * FW;
                output_oc_ptr += OC_stride;
            }
        }
    }
)";

    bool is_group = context->getAttrStr("sparse") == "GROUP";
    std::string group_str = is_group ? "inputs[1]->layout.dims[0]" : "1";

    writer << StringTemplate::StringTemplateArgs(context)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add("padding_do_conv_body", padding_do_conv_body)
                      .add("bias_str", bias_str)
                      .add("group_str", group_str)
                      .render(body_temp);
    return writer.str();
}
}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc