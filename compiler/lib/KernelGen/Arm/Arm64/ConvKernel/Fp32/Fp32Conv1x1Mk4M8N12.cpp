#include <sstream>
#include <string>
#include "Arm/Arm64/Activation.h"
#include "Arm/Arm64/ConvKernel.h"
#include "Arm/Arm64/InternalKernel/InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;

namespace megcc {
namespace KernelGen {
namespace Arm64 {

bool Conv1x1FloatMk4::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == 1 && ctx->getAttrUInt("kernel_w") == 1 &&
            ctx->getAttrUInt("stride_h") == 1 && ctx->getAttrUInt("stride_w") == 1 &&
            ctx->getAttrUInt("pad_h") == 0 && ctx->getAttrUInt("pad_w") == 0 &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (ctx->getAttrStr("sparse") == "DENSE" ||
                          ctx->getAttrStr("sparse") == "GROUP") &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "SIGMOID" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f32" &&
                   ctx->getAttrOprand("operand:1").dtype == "f32" &&
                   ctx->getAttrOprand("operand:2").dtype == "f32";
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;
    bool bias_ok =
            !is_bias(ctx) || is_channel_broadcast_bias(ctx) || is_elemwise_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           bias_ok;
}

std::string Conv1x1FloatMk4::GetKernelSymbol(TContext* ctx) const {
    std::stringstream extra_ss;
    if (is_bias(ctx)) {
        if (is_channel_broadcast_bias(ctx))
            extra_ss << "_channel_broadcast_bias";
        else if (is_elemwise_bias(ctx))
            extra_ss << "_elemwise_bias";
        else
            CC_ABORT << "only support channel broadcast and elemwise bias mode.";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        extra_ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    std::string name_temp =
            "Arm64_kernel_conv2d_conv1x1_${format}_${kernel_h}x${kernel_w}_${"
            "sparse}_p${pad_h}x${pad_w}_s${stride_h}x${stride_w}_d${"
            "dilate_h}x${dilate_w}${extra}";
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
            .add("extra", extra_ss.str())
            .render(name_temp);
}

std::string Conv1x1FloatMk4::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_gemm.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    const bool is_group = ctx->getAttrStr("sparse") == "GROUP";
    const std::string group_str = is_group ? "in_weights->layout.dims[0]" : "1";
    const int ocpg_offset = is_group ? 1 : 0;
    uint32_t nr_out_weight = 1;
    std::string common_def = StringTemplate::StringTemplateArgs()
                                     .add("group_str", group_str)
                                     .add("ocpg_offset", ocpg_offset)
                                     .render(R"(
    int PACK_SIZE_32 = 4 * 8;
    int PACK_SIZE_16 = 4 * 4;
    int PACK_C_SIZE = 4;
    Tensor* in_weights = inputs[1];
    const int group = ${group_str};
    int ymax = in_weights->layout.dims[${ocpg_offset} + 0] * PACK_C_SIZE;
    int kmax = in_weights->layout.dims[${ocpg_offset} + 1] * PACK_C_SIZE;
    int ldin = kmax * PACK_C_SIZE;
                      )");
    std::string fill_weight_attr =
            R"(
    out_weights->layout.nr_dim = 2;
    out_weights->layout.dims[0] = group;
    out_weights->layout.dims[1] = )" +
            m_inner_gemm.GetPackAWorkspaceSymbol(inner_ctx.get()) +
            R"((0, ymax, 0, kmax)/sizeof(float);
    out_weights->layout.stride[0] = out_weights->layout.dims[1];
    out_weights->layout.stride[1] = 1;
    out_weights->dtype.type_enum=TinyNN_FLOAT;
    out_weights->name = in_weights->name;
                      )";
    std::string fill_weight_transform =
            StringTemplate::StringTemplateArgs()
                    .add("packa", m_inner_gemm.GetPackASymbol(inner_ctx.get()))
                    .render(R"(    
    float* outptr = out_weights->ptr;
    float* inptr = in_weights->ptr;
    for(int group_idx = 0; group_idx < group; ++group_idx) {
        ${packa}(outptr, inptr, ldin, 0, ymax, 0, kmax);
        outptr += out_weights->layout.dims[1];
        inptr += ymax * kmax;
    }
    )");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string Conv1x1FloatMk4::GetWorkspaceBodyCondition(TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
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
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1] * 4;
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const uint32_t hw = ih * iw;
        *workspace = )" +
            m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()) +
            R"((0, hw, 0, ic);
        return TinyNN_SUCCESS;
    })";
    ss << workspace_temp;
    return ss.str();
}

std::vector<KernelObj> Conv1x1FloatMk4::GetDependInternalSymbol(TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);

    return {
            {m_inner_gemm.GetKernelSymbol(inner_ctx.get()),
             m_inner_gemm.GetKernelBody(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardEnd(inner_ctx.get()),
             m_inner_gemm.GetDependInternalSymbol(inner_ctx.get())}};
}

std::shared_ptr<TContext> Conv1x1FloatMk4::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode", CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    std::string bias_mode = "NO_BIAS";
    if (is_channel_broadcast_bias(ctx)) {
        bias_mode = "CHANNEL_BROADCAST_BIAS";
    } else if (is_elemwise_bias(ctx)) {
        bias_mode = "ELEMWISE_BIAS";
    }
    inner_ctx->setAttr("bias_mode", bias_mode);
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("dtype", "f32");
    return inner_ctx;
}

std::string Conv1x1FloatMk4::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_gemm.GetNakedKernelSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackBSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    std::string elemwise_bias_init, channel_broadcast_bias_init;
    if (is_channel_broadcast_bias(ctx)) {
        channel_broadcast_bias_init = "bias_data = inputs[2]->ptr;";
    } else if (is_elemwise_bias(ctx)) {
        elemwise_bias_init = "bias_data = inputs[2]->ptr;";
    }
    std::string bias_offset = "0";
    if (is_channel_broadcast_bias(ctx)) {
        bias_offset = "ocpg";
    } else if (is_elemwise_bias(ctx)) {
        bias_offset = "ocpg * out_h * out_w";
    }
    writer << StringTemplate::StringTemplateArgs()
                      .add("elemwise_bias_init", elemwise_bias_init)
                      .add("channel_broadcast_bias_init", channel_broadcast_bias_init)
                      .add("bias_offset", bias_offset)
                      .add("packB", m_inner_gemm.GetPackBSymbol(inner_ctx.get()))
                      .add("kern", m_inner_gemm.GetNakedKernelSymbol(inner_ctx.get()))
                      .render(R"({
    float* input_data = inputs[0]->ptr;
    float* output_data = outputs[0]->ptr;

    Layout in_layout = inputs[0]->layout;
    Layout out_layout = outputs[0]->layout;
    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1] * in_layout.dims[4];
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];
    const int PACK_C_SIZE = 4;

    const int out_c = out_layout.dims[1] * out_layout.dims[4];
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];
    const size_t N = out_h * out_w;

    Layout weight_layout = inputs[1]->layout;
    const int group = weight_layout.dims[0];
    const int icpg = in_c / group;
    const int ocpg = out_c / group;

    const int K12 = in_c * 12;
    const int K8 = in_c * 8;
    const int K4 = in_c * 4;

    const int A_INTERLEAVE = 8;
    const int A_INTERLEAVE4 = 4;
    const int B_INTERLEAVE = 12;
    const int LDC = out_h * out_w * PACK_C_SIZE;
    const int LDB = in_h * in_w * PACK_C_SIZE;

    void* workspace_ptr = workspace->ptr;
    const float* bias_data = NULL;
    ${elemwise_bias_init}
    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        float* weight_data = inputs[1]->ptr;
        ${channel_broadcast_bias_init}
        for(int group_idx = 0; group_idx < group; ++group_idx){
            ${packB}(workspace_ptr, input_data, LDB, 0, in_h * in_w, 0, icpg);
            ${kern}(weight_data, workspace_ptr, output_data, LDC, ocpg, N, icpg, bias_data);
            input_data += icpg * in_h * in_w;
            output_data += ocpg * out_h * out_w;
            weight_data += weight_layout.stride[0];
            bias_data += ${bias_offset};
        }
    }
    return TinyNN_SUCCESS;
})");

    return writer.str();
}

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
