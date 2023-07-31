#include <sstream>
#include <string>
#include "Arm/Armv7/Activation.h"
#include "Arm/Armv7/ConvKernel/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace Armv7;

std::shared_ptr<TContext> DotInt8Conv1x1NCHWM6N8K4::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode", CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    inner_ctx->setAttr("with_bias", ConvImpl::is_bias(ctx));
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "MK");
    inner_ctx->setAttr("dtype", ctx->getAttrOprand("operand:0").dtype);
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    inner_ctx->setAttr("last_dtype", last_dtype_str);
    inner_ctx->setAttr("nr_operands", ctx->getAttrInt("nr_operands"));
    inner_ctx->setAttr("operand:2", ctx->getAttrOprand("operand:2"));
    return inner_ctx;
}

bool DotInt8Conv1x1NCHWM6N8K4::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("kernel_h") == 1 &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 1 && ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1 && ctx->getAttrUInt("pad_h") == 0 &&
            ctx->getAttrUInt("pad_w") == 0;
    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";
    bool type_ok = is_qint8_conv_dtype(ctx);
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 4;

    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}

std::string DotInt8Conv1x1NCHWM6N8K4::GetKernelSymbol(TContext* ctx) const {
    return "Armv7_int8_dot_Conv1x1_M6N8K4_" + ConvImpl::GetKernelSymbol(ctx);
}

std::vector<KernelObj> DotInt8Conv1x1NCHWM6N8K4::GetDependInternalSymbol(
        TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);

    return {
            {inner_gemm.GetKernelSymbol(inner_ctx.get()),
             inner_gemm.GetKernelBody(inner_ctx.get()),
             inner_gemm.GetBodyGuardBegin(inner_ctx.get()),
             inner_gemm.GetBodyGuardEnd(inner_ctx.get()),
             inner_gemm.GetDependInternalSymbol(inner_ctx.get())}};
}

std::string DotInt8Conv1x1NCHWM6N8K4::GetWorkspaceBodyCondition(
        TContext* context, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(context);
    if (jit) {
        ss << inner_gemm.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string temp_dst_workspace;
    //! NOTE: conv1x1 src hw shape is equal to dst
    temp_dst_workspace = R"(
        const Layout flt_layout = inputs[1]->layout;
        uint32_t oc = flt_layout.dims[0];
        res += 128 + oc * hw * sizeof(int32_t);
    )";
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1];
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const uint32_t hw = ih * iw;
        size_t res = ${packb_size_sym}(0, hw, 0, ic);

        ${temp_dst_workspace}
        *workspace = res;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("packb_size_sym",
                         inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .add("temp_dst_workspace", temp_dst_workspace)
                    .render(workspace_temp);
    return ss.str();
}

std::string DotInt8Conv1x1NCHWM6N8K4::GetInitBody(TContext* context) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(context);
    writer << inner_gemm.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << inner_gemm.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(context) << "\n";
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    Tensor* in_weights = inputs[1];
    int ymax = in_weights->layout.dims[0];
    int kmax = in_weights->layout.dims[1];
    int ldin = kmax;
)";
    std::string fill_weight_attr = R"(
        out_weights->layout.nr_dim = 1;
        out_weights->layout.dims[0] = )" +
                                   inner_gemm.GetPackAWorkspaceSymbol(inner_ctx.get()) +
                                   R"((0, ymax, 0, kmax);
        out_weights->layout.stride[0] = 1;
        out_weights->dtype.type_enum = TinyNN_QINT8;
        out_weights->name = in_weights->name;
        out_weights->dtype.param.scale = in_weights->dtype.param.scale;
)";
    std::string fill_weight_transform = R"(
    int8_t* outptr = out_weights->ptr;
    int8_t* inptr = in_weights->ptr;
    )" + inner_gemm.GetPackASymbol(inner_ctx.get()) +
                                        "(outptr, inptr, ldin, 0, ymax, 0, kmax);";
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);
    return writer.str();
}

std::string DotInt8Conv1x1NCHWM6N8K4::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << R"(
#include <marm_neon.h>
#include <string.h>
    )";
    auto inner_ctx = GetInnerCtx(context);
    writer << inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << inner_gemm.GetKernelSignature(inner_ctx.get()) << ";\n";
    writer << inner_gemm.GetPackBSignature(inner_ctx.get()) << ";\n";
    writer << inner_gemm.GetNakedKernelSignature(inner_ctx.get()) << ";\n";

    bool with_bias = ConvImpl::is_bias(context);
    std::string bias_ptr_str = with_bias ? "inputs[2]->ptr" : "0";
    std::string nonline_mode = context->haveAttr("nonlineMode")
                                     ? context->getAttrStr("nonlineMode")
                                     : "IDENTITY";

    writer << GenCommonRet() << " " << GetKernelSignature(context) << "{\n";
    std::string body_temp = R"(
    Layout in_layout = inputs[0]->layout;
    Layout out_layout = outputs[0]->layout;
    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1];
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];

    const int out_c = out_layout.dims[1];
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];
    const size_t N = out_h * out_w;

    const int LDB = in_h * in_w;
    const int LDC = out_h * out_w;

    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    const float temp_scale = src_scale * flt_scale;
    // this must be 1.f/dst_scale for quant data
    const float dst_scale_inv = 1.f / dst_scale;
    const float scale = src_scale * flt_scale * dst_scale_inv;

    int8_t* input_data = inputs[0]->ptr;
    int8_t* output_data = outputs[0]->ptr;
    int8_t* weight_data = inputs[1]->ptr;
    int32_t* bias_data = ${bias_ptr_str};

    const size_t pack_b_size = ${packb_size_sym}(0, in_h * in_w, 0, in_c);
    const size_t pack_b_align = (pack_b_size + 63) / 64 * 64;
    void* workspace_ptr = workspace->ptr;
    void* temp_dst = (int8_t*) workspace_ptr + pack_b_align;
    
    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        int8_t* weight_data = inputs[1]->ptr;
        int32_t* bias_data = ${bias_ptr_str};

        ${packb_sym}(workspace_ptr, input_data, LDB, 0, in_h * in_w, 0, in_c);
        ${naked_kern_sym}(weight_data, workspace_ptr, output_data, LDC, out_c, N, in_c, bias_data, temp_dst, scale, temp_scale, dst_scale_inv);
        input_data += in_c * in_h * in_w;
        output_data += out_c * out_h * out_w;
    }

    return TinyNN_SUCCESS;
}
)";

    writer << StringTemplate::StringTemplateArgs(context)
                      .add("bias_ptr_str", bias_ptr_str)
                      .add("packb_size_sym",
                           inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                      .add("packb_sym", inner_gemm.GetPackBSymbol(inner_ctx.get()))
                      .add("naked_kern_sym",
                           inner_gemm.GetNakedKernelSymbol(inner_ctx.get()))
                      .render(body_temp);
    return writer.str();
}
