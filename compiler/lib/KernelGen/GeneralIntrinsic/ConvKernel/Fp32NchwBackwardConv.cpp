#include <sstream>
#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ConvKernel/ConvKernel.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool Float32NchwBackward::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (ctx->getAttrStr("sparse") == "DENSE" ||
                          ctx->getAttrStr("sparse") == "GROUP") &&
                         ctx->getAttrStr("format") == "NCHW" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode");
    bool no_bias_ok = !ConvImpl::is_bias(ctx);

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f32" &&
                   ctx->getAttrOprand("operand:1").dtype == "f32" &&
                   ctx->getAttrOprand("operand:2").dtype == "f32";
    bool bias_ok = !is_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && no_bias_ok &&
           bias_ok;
}

std::string Float32NchwBackward::GetKernelSymbol(TContext* ctx) const {
    CC_ASSERT(ctx);
    std::string name_temp =
            "GI_kernel_back_data_conv2d_${format}_${kernel_h}x${kernel_w}_${"
            "sparse}_p${pad_h}x${pad_w}_s${stride_h}x${stride_w}_d${"
            "dilate_h}x${dilate_w}_f32f32f32";
    return StringTemplate::StringTemplateArgs(ctx)
            .add_ctx_int("kernel_h")
            .add_ctx_int("kernel_w")
            .add_ctx_str("format")
            .add_ctx_str("sparse")
            .add_ctx_int("pad_h")
            .add_ctx_int("pad_w")
            .add_ctx_int("stride_h")
            .add_ctx_int("stride_w")
            .add_ctx_int("dilate_h")
            .add_ctx_int("dilate_w")
            .render(name_temp);
}

std::string Float32NchwBackward::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_gemm_ctx = GetInnerGemmCtx(ctx);
    writer << m_inner_gemm.GetPackASignature(inner_gemm_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackAWorkspaceSignature(inner_gemm_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    Tensor* in_weights = inputs[0]; //! The weight is the first input in BackwardConv.
    const uint32_t nr_dim = in_weights->layout.nr_dim;
    const uint32_t group = nr_dim == 5 ? in_weights->layout.dims[0] : 1;
    const uint32_t ic_idx = nr_dim == 5 ? 1 : 0;
    int ymax = in_weights->layout.dims[ic_idx + 1] * in_weights->layout.dims[ic_idx + 2] * in_weights->layout.dims[ic_idx + 3];
    int kmax = in_weights->layout.dims[ic_idx + 0];
    int ldin = ymax;
                      )";
    std::string fill_weight_attr =
            R"(
    out_weights->layout.nr_dim = 2;
    out_weights->layout.dims[0] = group;
    out_weights->layout.dims[1] = )" +
            m_inner_gemm.GetPackAWorkspaceSymbol(inner_gemm_ctx.get()) +
            R"((0, ymax, 0, kmax)/sizeof(float);
    out_weights->layout.stride[0] = out_weights->layout.dims[1];
    out_weights->layout.stride[1] = 1;
    out_weights->dtype.type_enum=TinyNN_FLOAT;
    out_weights->name = in_weights->name;
                      )";
    std::string fill_weight_transform =
            R"(
    for (int g = 0; g < group; ++g) {
        float* outptr = (float*)(out_weights->ptr) + g * out_weights->layout.stride[0];
        float* inptr = (float*)(in_weights->ptr) + g * kmax * ymax;
    )" + m_inner_gemm.GetPackASymbol(inner_gemm_ctx.get()) +
            "(outptr, inptr, ldin, 0, ymax, 0, kmax);}";
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string Float32NchwBackward::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerGemmCtx(ctx);
    if (jit) {
        ss << m_inner_gemm.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout weight_layout = inputs[0]->layout;
        const uint32_t group = weight_layout.nr_dim == 5 ? weight_layout.dims[0] : 1;
        const Layout in_layout = inputs[1]->layout;
        const uint32_t ic = in_layout.dims[1] / group;
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const uint32_t hw = ih * iw;
        *workspace = )" +
            m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()) +
            StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .render(R"((0, hw, 0, ic);
        const uint32_t kh = ${kernel_h};
        const uint32_t kw = ${kernel_w};
        if (kh != 1 || kw != 1 || ${stride_h} != 1 || ${stride_w} != 1 || ${pad_h} != 0 || ${pad_w} != 0) {
            const uint32_t oc = weight_layout.nr_dim == 5 ? weight_layout.dims[2] : weight_layout.dims[1];
            *workspace = *workspace + oc * kh * kw * hw * sizeof(float);
        }
        return TinyNN_SUCCESS;
    })");
    ss << workspace_temp;
    return ss.str();
}

std::vector<KernelObj> Float32NchwBackward::GetDependInternalSymbol(
        TContext* ctx) const {
    auto inner_gemm_ctx = GetInnerGemmCtx(ctx);

    return {
            {m_inner_gemm.GetKernelSymbol(inner_gemm_ctx.get()),
             m_inner_gemm.GetKernelBody(inner_gemm_ctx.get()),
             m_inner_gemm.GetBodyGuardBegin(inner_gemm_ctx.get()),
             m_inner_gemm.GetBodyGuardEnd(inner_gemm_ctx.get())}};
}

std::shared_ptr<TContext> Float32NchwBackward::GetInnerGemmCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("transposeA", true);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "NCHW");
    inner_ctx->setAttr("dtype", "f32");
    return inner_ctx;
}

std::string Float32NchwBackward::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerGemmCtx(ctx);
    writer << m_inner_gemm.GetNakedKernelSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackBSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);

    std::string col2img = R"(
            for (int oc = 0; oc < out_c; ++oc) {
                for(int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                    for(int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                        for(int ih = 0; ih < in_h; ++ih) {
                            const int oh = ih * sh + kh_idx - ph;
                            for(int iw = 0; iw < in_w; ++iw){
                                const int ow = iw * sw + kw_idx - pw;
                                if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                    output_data[oc * out_h * out_w + oh * out_w + ow] += *out_ptr;
                                }
                                ++out_ptr;
                            }
                        }
                    }
                }
            }
    )";
    if (ctx->getAttrInt("pad_h") == 0 && ctx->getAttrInt("pad_w") == 0 &&
        ctx->getAttrInt("stride_h") == 1 && ctx->getAttrInt("stride_w") == 1) {
        col2img = R"(
            for (int oc = 0; oc < out_c; ++oc) {
                for(int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                    for(int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                        for(int ih = 0; ih < in_h; ++ih) {
                            const int oh = ih + kh_idx;
                            for(int iw = 0; iw < in_w; ++iw){
                                const int ow = iw + kw_idx;
                                output_data[oc * out_h * out_w + oh * out_w + ow] += *out_ptr++;
                            }
                        }
                    }
                }
            }
        )";
    }

    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .render(R"({
    float* input_data = inputs[1]->ptr;
    float* output_data = outputs[0]->ptr;

    Layout weight_layout = inputs[0]->layout;
    Layout in_layout = inputs[1]->layout;
    Layout out_layout = outputs[0]->layout;

    const int group = weight_layout.dims[0];

    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1] / group;
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];
    const size_t N = in_h * in_w;

    const int out_c = out_layout.dims[1] / group;
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];

    const int kh = ${kernel_h};
    const int kw = ${kernel_w};
    const int ph = ${pad_h};
    const int pw = ${pad_w};
    const int sh = ${stride_h};
    const int sw = ${stride_w};
    const size_t M = out_c * kh * kw;

    const int LDB = N;
    const int LDC = LDB;

    void* workspace_ptr = workspace->ptr;
    float* out_buffer = NULL;
    const int need_out_buffer = (kh != 1 || kw != 1 || sh != 1 || sw != 1 || ph != 0 || pw != 0);
    if (need_out_buffer) {
        out_buffer = workspace->ptr;
        workspace_ptr = out_buffer + M * N;
    }
    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        float* weight_data = inputs[0]->ptr;
        for (int g = 0; g < group; ++g) {
        )") << m_inner_gemm.GetPackBSymbol(inner_ctx.get())
           << R"(
            (workspace_ptr, input_data, LDB, 0, in_h * in_w, 0, in_c);
            float* out_ptr = need_out_buffer ? out_buffer : output_data;
        )" << m_inner_gemm.GetNakedKernelSymbol(inner_ctx.get())
           << StringTemplate::StringTemplateArgs()
                      .add("col2img", col2img)
                      .render(R"( (weight_data, workspace_ptr, out_ptr, LDC, M, N, in_c, NULL);
            if (need_out_buffer) {
                memset(output_data, 0, sizeof(float) * out_c * out_h * out_w);
                ${col2img}
            }
            weight_data += weight_layout.stride[0];
            input_data += in_c * in_h * in_w;
            output_data += out_c * out_h * out_w;
        }
    }
    return TinyNN_SUCCESS;
})");

    return writer.str();
}

// vim: syntax=cpp.doxygen
