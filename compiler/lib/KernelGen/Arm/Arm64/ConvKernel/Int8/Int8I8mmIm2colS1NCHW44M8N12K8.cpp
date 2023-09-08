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

namespace {
std::string pad_src() {
    return R"(
static inline void pad_src(const int8_t *src, int8_t *dst, const int IC, const int IH,
             const int IW, const int PH, const int PW) {
  const int paded_H = IH + 2 * PH;
  const int paded_W = IW + 2 * PW;
  const int paded_HW = paded_H * paded_W;
  memset(dst, 0, IC * 4 * paded_HW * sizeof(int8_t));
  for (int ic = 0; ic < IC; ic++) {
    dst += PH * paded_W * 4;
    for (int ih = 0; ih < IH; ++ih) {
      memcpy(dst + ih * paded_W * 4 + PW * 4, src + ih * IW * 4,
             IW * 4 * sizeof(int8_t));
    }
    dst += (IH + PH) * paded_W * 4;
    src += IH * IW * 4;
  }
}
    )";
}

std::string im2col_s1() {
    return R"(
static inline void im2col(const int8_t *src, int8_t *dst, const int IC, const int OH,
            const int OW, const int FH, const int FW, const int paded_H,
            const int paded_W) {
  const int src_stride_ic = paded_H * paded_W * 4;
  const int src_stride_h = paded_W * 4;
  const int dst_stride_ic = OH * OW * 4;
  const int dst_stride_h = OW * 4;
  for (int ic = 0; ic < IC; ++ic) {
    for (int fh = 0; fh < FH; ++fh) {
      for (int fw = 0; fw < FW; ++fw) {
        const int8_t *src_base =
            src + ic * src_stride_ic + fh * src_stride_h + fw * 4;
        for (int oh = 0; oh < OH; ++oh) {
          memcpy(dst + oh * dst_stride_h, src_base + oh * src_stride_h,
                 dst_stride_h * sizeof(int8_t));
        }
        dst += dst_stride_ic;
      }
    }
  }
}
    )";
}

std::string im2col_s2() {
    return R"(
static inline void im2col(const int8_t *src, int8_t *dst, const int IC, const int OH,
            const int OW, const int FH, const int FW, const int paded_H,
            const int paded_W) {
  const int src_stride_ic = paded_H * paded_W * 4;
  const int src_stride_h = paded_W * 4;
  const int dst_stride_ic = OH * OW * 4;
  const int dst_stride_h = OW * 4;
  for (int ic = 0; ic < IC; ++ic) {
    for (int fh = 0; fh < FH; ++fh) {
      for (int fw = 0; fw < FW; ++fw) {
        const int8_t *src_base =
            src + ic * src_stride_ic + fh * src_stride_h + fw * 4;
        for (int oh = 0; oh < OH; ++oh) {
          const int32_t *src_ptr = (const int32_t*)(src_base + oh * 2 * src_stride_h);
          int32_t *dst_ptr = (int32_t*)(dst + oh * dst_stride_h);
          int ow = 0;
          for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t d = vld2q_s32(src_ptr + ow * 2);
            vst1q_s32(dst_ptr + ow, d.val[0]);
          }
          for (; ow < OW; ++ow) {
            dst_ptr[ow] = src_ptr[ow * 2];
          }
        }
        dst += dst_stride_ic;
      }
    }
  }
}
    )";
}
}  // namespace

bool ConvBiasIm2colI8mmNCHW44::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            (ctx->getAttrUInt("stride_w") == 1 || ctx->getAttrUInt("stride_w") == 2) &&
            ctx->getAttrUInt("pad_h") == ctx->getAttrUInt("pad_w") &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (ctx->getAttrStr("sparse") == "DENSE" ||
                          (ctx->getAttrStr("sparse") == "GROUP" &&
                           ctx->getAttrOprand("operand:1").shape.size() ==
                                   7 /*reject channel wise whose dimension is 6*/)) &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = is_qint8_conv_dtype(ctx, true);

    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;
    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           bias_ok;
}

std::string ConvBiasIm2colI8mmNCHW44::GetKernelSymbol(TContext* ctx) const {
    return "Arm64_kernel_im2col_i8mm_m8n12k8_" + ConvImpl::GetKernelSymbol(ctx);
}

std::string ConvBiasIm2colI8mmNCHW44::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    const bool is_group = ctx->getAttrStr("sparse") == "GROUP";
    const std::string group_str =
            is_group ? "const int group = in_weights->layout.dims[0];"
                     : "const int group = 1;";
    const int oc_idx = is_group ? 1 : 0;
    writer << m_inner_gemm.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    const uint32_t nr_out_weight = 1;
    const std::string common_def = StringTemplate::StringTemplateArgs()
                                           .add("group", group_str)
                                           .add("oc_idx", oc_idx)
                                           .render(R"(
        Tensor* in_weights = inputs[1];
        ${group}
        const int ymax = in_weights->layout.dims[${oc_idx}] * 4;
        const int kmax = in_weights->layout.dims[${oc_idx} + 1] * in_weights->layout.dims[${oc_idx} + 2] * in_weights->layout.dims[${oc_idx} + 3] * 4;
        const int ldin = kmax * 4;
    )");
    const std::string fill_weight_attr =
            R"(
        out_weights->layout.nr_dim = 2;
        out_weights->layout.dims[0] = group;
        out_weights->layout.dims[1] = )" +
            m_inner_gemm.GetPackAWorkspaceSymbol(inner_ctx.get()) +
            R"((0, ymax, 0, kmax);
        out_weights->layout.stride[0] = out_weights->layout.dims[1];
        out_weights->layout.stride[1] = 1;
        out_weights->dtype.type_enum = TinyNN_QINT8;
        out_weights->name = in_weights->name;
        out_weights->dtype.param.scale = in_weights->dtype.param.scale;
    )";
    const std::string fill_weight_transform =
            StringTemplate::StringTemplateArgs()
                    .add("packa_sym", m_inner_gemm.GetPackASymbol(inner_ctx.get()))
                    .render(
                            R"(    
        int8_t* outptr = out_weights->ptr;
        int8_t* inptr = in_weights->ptr;
        ${packa_sym}(outptr, inptr, ldin, 0, ymax, 0, kmax);
        for (int i = 1; i < group; ++i) {
            inptr += in_weights->layout.stride[0];
            outptr += out_weights->layout.stride[0];
            ${packa_sym}(outptr, inptr, ldin, 0, ymax, 0, kmax);
        }
    )");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string ConvBiasIm2colI8mmNCHW44::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
    const bool is_group = ctx->getAttrStr("sparse") == "GROUP";
    const std::string group_str = is_group
                                        ? "const int group = inputs[1]->layout.dims[0];"
                                        : "const int group = 1;";
    if (jit) {
        ss << m_inner_gemm.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        ${group}
        const Layout src_layout = inputs[0]->layout;
        const size_t IC = src_layout.dims[1] / group * 4;
        const size_t IH = src_layout.dims[2], IW = src_layout.dims[3];

        const size_t padded_IH = IH + 2 * ${pad_h};
        const size_t padded_IW = IW + 2 * ${pad_w};
        size_t pad_size = 0;
        if ((${pad_h} != 0) || (${pad_w} != 0)){
            pad_size = IC * padded_IH * padded_IW * sizeof(int8_t);
        }

        const size_t OH = (padded_IH - ${kernel_h}) / ${stride_h} + 1;
        const size_t OW = (padded_IW - ${kernel_w}) / ${stride_w} + 1;
        const size_t K = IC * ${kernel_h} * ${kernel_w}, N = OH * OW;
        size_t im2col_size = 0;
        if (${kernel_h} != 1 || ${kernel_w} != 1 || ${stride_h} != 1 || ${stride_w} != 1){
            im2col_size = K * N * sizeof(int8_t);
        }

        *workspace = pad_size + im2col_size + ${packb_workspace_sym}(0, N, 0, K);
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add("packb_workspace_sym",
                         m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .add("group", group_str)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .render(workspace_temp);
    return ss.str();
}

std::vector<KernelObj> ConvBiasIm2colI8mmNCHW44::GetDependInternalSymbol(
        TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);

    return {
            {m_inner_gemm.GetKernelSymbol(inner_ctx.get()),
             m_inner_gemm.GetKernelBody(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardEnd(inner_ctx.get()),
             m_inner_gemm.GetDependInternalSymbol(inner_ctx.get())}};
}

std::shared_ptr<TContext> ConvBiasIm2colI8mmNCHW44::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode", CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    inner_ctx->setAttr("with_bias", ConvImpl::is_bias(ctx));
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("dtype", ctx->getAttrOprand("operand:0").dtype);
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    inner_ctx->setAttr("last_dtype", last_dtype_str);
    return inner_ctx;
}

std::string ConvBiasIm2colI8mmNCHW44::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetNakedKernelSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackBSignature(inner_ctx.get()) << ";\n";
    const bool need_pad = (ctx->getAttrInt("pad_h") || ctx->getAttrInt("pad_w")),
               need_im2col =
                       (ctx->getAttrInt("kernel_h") != 1 ||
                        ctx->getAttrInt("kernel_w") != 1 ||
                        ctx->getAttrInt("stride_h") != 1 ||
                        ctx->getAttrInt("stride_w") != 1);
    if (need_pad) {
        writer << pad_src();
    }
    if (need_im2col) {
        if (ctx->getAttrInt("stride_h") == 1 && ctx->getAttrInt("stride_w") == 1) {
            writer << im2col_s1();
        } else {
            CC_ASSERT(
                    ctx->getAttrInt("stride_h") == 2 &&
                    ctx->getAttrInt("stride_w") == 2);
            writer << "#include <arm_neon.h>\n";
            writer << im2col_s2();
        }
    }
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    std::string bias_ptr_str = is_bias(ctx) ? "inputs[2]->ptr;" : "0;";
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    std::string dst_specifier = Utils::cvt_dtype_specifier(last_dtype_str);
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add("bias_ptr_str", bias_ptr_str)
                      .add("packb_size_sym",
                           m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                      .add("packb_sym", m_inner_gemm.GetPackBSymbol(inner_ctx.get()))
                      .add("naked_kern_sym",
                           m_inner_gemm.GetNakedKernelSymbol(inner_ctx.get()))
                      .add("dst_specifier", dst_specifier)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .render(R"({
    int8_t* input_data = inputs[0]->ptr;
    ${dst_specifier}* output_data = outputs[0]->ptr;

    Layout in_layout = inputs[0]->layout;
    Layout weight_layout = inputs[1]->layout;
    const int group = weight_layout.dims[0];
    Layout out_layout = outputs[0]->layout;
    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1] / group * in_layout.dims[4];
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];
    const int PACK_C_SIZE = 4;
    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    const float temp_scale = src_scale * flt_scale;
    const float dst_scale_inv = 1.f / dst_scale;
    const float scale = src_scale * flt_scale * dst_scale_inv;

    const int out_c = out_layout.dims[1] / group * out_layout.dims[4];
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];
    const size_t N = out_h * out_w, M = out_c, K = in_c * ${kernel_h} * ${kernel_w};

    const int LDC = out_h * out_w * PACK_C_SIZE;
    const int LDB = out_h * out_w * PACK_C_SIZE;

    const size_t padded_ih = in_h + 2 * ${pad_h}, padded_iw = in_w + 2 * ${pad_w};
    size_t pad_size = 0, im2col_size = 0;
    if ((${pad_h} != 0) || (${pad_w} != 0)) {
        pad_size = in_c * padded_ih * padded_iw * sizeof(int8_t);
    }
    if (${kernel_h} != 1 || ${kernel_w} != 1 || ${stride_h} != 1 || ${stride_w} != 1){
        im2col_size = K * N * sizeof(int8_t);
    }
    void *pad_ws = workspace->ptr;
    void *im2col_ws = pad_ws + pad_size;
    void *packb_ws = im2col_ws + im2col_size;
    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        int32_t* bias_data = ${bias_ptr_str};
        int8_t* weight_data = inputs[1]->ptr;
        for (int g = 0; g < group; ++g) {)" +
                              (need_pad ? std::string(R"(
            pad_src(input_data, pad_ws, in_c / 4, in_h, in_w, ${pad_h}, ${pad_w});)")
                                        : std::string(R"(
            pad_ws = input_data;)")) +
                              (need_im2col ? std::string(R"(
            im2col(pad_ws, im2col_ws, in_c / 4, out_h, out_w, ${kernel_h}, ${kernel_w}, padded_ih, padded_iw);)")
                                           : std::string(R"(
            im2col_ws = pad_ws;)")) +
                              std::string(R"(
            ${packb_sym}(packb_ws, im2col_ws, LDB, 0, N, 0, K);
            ${naked_kern_sym}(weight_data, packb_ws, output_data, LDC, M, N, K, bias_data, NULL, scale, temp_scale, dst_scale_inv);
            weight_data += weight_layout.stride[0];
            bias_data += out_c;
            input_data += in_c * in_h * in_w;
            output_data += out_c * out_h * out_w;
        }
    }
    return TinyNN_SUCCESS;
})"));

    return writer.str();
}

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
