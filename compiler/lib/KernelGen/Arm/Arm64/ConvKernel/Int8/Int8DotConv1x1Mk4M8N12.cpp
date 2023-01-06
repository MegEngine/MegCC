/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/ConvKernel/Fp32/Fp32Conv1x1Mk4M8N12.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <sstream>
#include <string>
#include "Arm/Arm64/Activation.h"
#include "Arm/Arm64/ConvKernel.h"
#include "Arm/Arm64/InternalKernel/InternalKernel.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;
namespace megcc {
namespace KernelGen {
namespace Arm64 {

bool Conv1x1DotMk4::IsAvailable(TContext* ctx) const {
    bool param_value_ok = ctx->getAttrUInt("kernel_h") == 1 &&
                          ctx->getAttrUInt("kernel_w") == 1 &&
                          ctx->getAttrUInt("stride_h") == 1 &&
                          ctx->getAttrUInt("stride_w") == 1 &&
                          ctx->getAttrUInt("pad_h") == 0 &&
                          ctx->getAttrUInt("pad_w") == 0 &&
                          ctx->getAttrUInt("dilate_h") == 1 &&
                          ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW44_DOT" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = is_qint8_conv_dtype(ctx, true);

    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}

std::string Conv1x1DotMk4::GetKernelSymbol(TContext* ctx) const {
    std::stringstream extra_ss;
    if (is_bias(ctx)) {
        extra_ss << "_bias";
    }
    extra_ss << "_" << SymbolHelper::gen_io_str(ctx);
    if (ctx->haveAttr("nonlineMode") &&
        ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        extra_ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    std::string name_temp =
            "Arm64_kernel_dot_conv2d_conv1x1_${format}_${kernel_h}x${kernel_w}_"
            "${"
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

std::string Conv1x1DotMk4::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_gemm.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    int PACK_SIZE_32 = 4 * 8;
    int PACK_SIZE_16 = 4 * 4;
    int PACK_C_SIZE = 4;
    Tensor* in_weights = inputs[1];
    int ymax = in_weights->layout.dims[0] * PACK_C_SIZE;
    int kmax = in_weights->layout.dims[1] * PACK_C_SIZE;
    int ldin = kmax * PACK_C_SIZE;
                      )";
    std::string fill_weight_attr =
            R"(
    out_weights->layout.nr_dim = 1;
    out_weights->layout.dims[0] = )" +
            m_inner_gemm.GetPackAWorkspaceSymbol(inner_ctx.get()) +
            R"((0, ymax, 0, kmax);
    out_weights->layout.stride[0] = 1;
    out_weights->dtype.type_enum=TinyNN_QINT8;
    out_weights->name = in_weights->name;
    out_weights->dtype.param.scale = in_weights->dtype.param.scale;
                      )";
    std::string fill_weight_transform =
            R"(    
    int8_t* outptr = out_weights->ptr;
    int8_t* inptr = in_weights->ptr;
    )" + m_inner_gemm.GetPackASymbol(inner_ctx.get()) +
            "(outptr, inptr, ldin, 0, ymax, 0, kmax);";
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string Conv1x1DotMk4::GetWorkspaceBodyCondition(TContext* ctx,
                                                     bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
    if (jit) {
        ss << m_inner_gemm.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern "
           << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get()) << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string temp_dst_workspace;
    if (need_temp_dst(ctx)) {
        //! NOTE: conv1x1 src hw shape is equal to dst
        temp_dst_workspace = R"(
            const Layout flt_layout = inputs[1]->layout;
            uint32_t oc = flt_layout.dims[0] * 4;
            res += 128 + oc * hw * sizeof(int32_t);
        )";
    }
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1] * 4;
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
                         m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .add("temp_dst_workspace", temp_dst_workspace)
                    .render(workspace_temp);

    return ss.str();
}

std::vector<KernelObj> Conv1x1DotMk4::GetDependInternalSymbol(
        TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);

    return {{m_inner_gemm.GetKernelSymbol(inner_ctx.get()),
             m_inner_gemm.GetKernelBody(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardEnd(inner_ctx.get()),
             m_inner_gemm.GetDependInternalSymbol(inner_ctx.get())}};
}

bool Conv1x1DotMk4::need_temp_dst(TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);
    return m_inner_gemm.need_post_process(inner_ctx.get());
}

std::shared_ptr<TContext> Conv1x1DotMk4::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode",
                           CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    inner_ctx->setAttr("with_bias", ConvImpl::is_bias(ctx));
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "MK4_DOT");
    inner_ctx->setAttr("dtype", ctx->getAttrOprand("operand:0").dtype);
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    inner_ctx->setAttr("last_dtype", last_dtype_str);
    return inner_ctx;
}

std::string Conv1x1DotMk4::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetNakedKernelSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackBSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    std::string bias_ptr_str = is_bias(ctx) ? "inputs[2]->ptr;" : "0;";
    std::string gen_temp_dst = "void* temp_dst = NULL;";
    if (need_temp_dst(ctx)) {
        gen_temp_dst =
                "void* temp_dst = (int8_t*) workspace_ptr + pack_b_align;";
    }
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    std::string dst_specifier = Utils::cvt_dtype_specifier(last_dtype_str);
    writer << StringTemplate::StringTemplateArgs()
                      .add("bias_ptr_str", bias_ptr_str)
                      .add("packb_size_sym",
                           m_inner_gemm.GetPackBWorkspaceSymbol(
                                   inner_ctx.get()))
                      .add("packb_sym",
                           m_inner_gemm.GetPackBSymbol(inner_ctx.get()))
                      .add("naked_kern_sym",
                           m_inner_gemm.GetNakedKernelSymbol(inner_ctx.get()))
                      .add("gen_temp_dst", gen_temp_dst)
                      .add("dst_specifier", dst_specifier)
                      .render(R"({
    int8_t* input_data = inputs[0]->ptr;
    ${dst_specifier}* output_data = outputs[0]->ptr;

    Layout in_layout = inputs[0]->layout;
    Layout out_layout = outputs[0]->layout;
    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1] * in_layout.dims[4];
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];
    const int PACK_C_SIZE = 4;
    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    const float temp_scale = src_scale * flt_scale;
    const float dst_scale_inv = 1.f / dst_scale;
    const float scale = src_scale * flt_scale / dst_scale;

    const int out_c = out_layout.dims[1] * out_layout.dims[4];
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];
    const size_t N = out_h * out_w;

    const int K12 = in_c * 12;
    const int K8 = in_c * 8;
    const int K4 = in_c * 4;

    const int A_INTERLEAVE = 8;
    const int A_INTERLEAVE4 = 4;
    const int B_INTERLEAVE = 12;
    const int LDC = out_h * out_w * PACK_C_SIZE;
    const int LDB = in_h * in_w * PACK_C_SIZE;

    const size_t pack_b_size = ${packb_size_sym}(0, in_h * in_w, 0, in_c);
    const size_t pack_b_align = (pack_b_size + 63) / 64 * 64;
    void* workspace_ptr = workspace->ptr;
    ${gen_temp_dst}
    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        int8_t* weight_data = inputs[1]->ptr;
        int32_t* bias_data = ${bias_ptr_str};

        ${packb_sym}(workspace_ptr, input_data, LDB, 0, in_h * in_w, 0, in_c);
        ${naked_kern_sym}(weight_data, workspace_ptr, output_data, LDC, out_c, N, in_c, bias_data, temp_dst, scale, temp_scale, dst_scale_inv);
        input_data += in_c * in_h * in_w;
        output_data += out_c * out_h * out_w;
    }
    return TinyNN_SUCCESS;
})");

    return writer.str();
}

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
