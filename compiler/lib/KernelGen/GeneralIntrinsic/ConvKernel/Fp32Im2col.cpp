#include <sstream>
#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ConvKernel/ConvKernel.h"
#include "GeneralIntrinsic/ConvKernel/Im2colCommon.h"
#include "GeneralIntrinsic/ElemwiseHelper/ElemwiseHelper.h"
#include "GeneralIntrinsic/GIMathHelper.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
static std::pair<std::string, std::string> gen_postprocess_inline(
        TContext* ctx, bool need_postprocess = true) {
    std::string call_str;
    std::stringstream declare_ss;
    auto nonline_mode = ctx && ctx->haveAttr("nonlineMode")
                              ? ctx->getAttrStr("nonlineMode")
                              : "IDENTITY";
    if ((nonline_mode == "SIGMOID") && need_postprocess) {
        std::vector<CCOperand> operands;
        operands.resize(2);
        auto dtype = ctx->getAttrStr("dtype");
        auto create_elem =
                [=](std::string src_dtype,
                    std::string dst_dtype) -> std::shared_ptr<ElemwiseGenUnary> {
            return std::make_shared<ElemwiseGenUnarySigmoid>(
                    src_dtype, dst_dtype, true);
        };

        std::shared_ptr<ElemwiseGenUnary> ElemwiseImpl = create_elem("f32", "f32");

        if (Utils::is_quant_dtype(dtype)) {
            ElemwiseImpl = create_elem("si32", "si8");
        }

        std::string post_process_temp = R"(
            if (LDC == N){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                for(int m_idx = 0; m_idx < M; ++m_idx){
                    ${ElemwiseImplName}(C + m_idx * LDC, C + m_idx * LDC, N);
                }
            }
        )";
        if (ctx->getAttrStr("format") == "MK4") {
            post_process_temp = R"(
            if (LDC == (4 * N)){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < M; m_idx += 4){
                    ${ElemwiseImplName}(C + cnt * LDC, C + cnt * LDC, 4 * N);
                    ++cnt;
                }
            }
        )";
        } else {
            CC_ASSERT(ctx->getAttrStr("format") == "MK4_DOT");
            post_process_temp = R"(
            if (LDC == (4 * N)){
                ${ElemwiseImplName}(gemm_output, C, M * N, temp_scale, dst_scale_inv);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < M; m_idx += 4){
                    ${ElemwiseImplName}(gemm_output + cnt * LDC, C + cnt * LDC, 4 * N, temp_scale, dst_scale_inv);
                    ++cnt;
                }
            }
        )";
        }
        call_str = StringTemplate::StringTemplateArgs()
                           .add("ElemwiseImplName", ElemwiseImpl->GenInlineName())
                           .render(post_process_temp);
        GIMathHelper gi_math;
        declare_ss << R"(
#include "gi_int.h"
            )";
        declare_ss << gi_math.GiExpPsFloat32() << "\n";
        declare_ss << gi_math.GiSigmoidPsFloat32() << "\n";
        declare_ss << ElemwiseImpl->GenCodeBody({});
    }
    return {declare_ss.str(), call_str};
}

}  // namespace
std::string ConvIm2colFloat::GetKernelSymbol(TContext* ctx) const {
    std::stringstream extra_ss;
    if (ctx) {
        if (is_bias(ctx)) {
            extra_ss << "_bias";
        }
        if (ctx->haveAttr("nonlineMode") &&
            ctx->getAttrStr("nonlineMode") != "IDENTITY") {
            extra_ss << "_" << ctx->getAttrStr("nonlineMode");
        }
        extra_ss << ctx->getAttrOprand("operand:0").dtype;
        std::string name_temp =
                "GI_kernel_conv2d_im2col_${kernel_h}x${kernel_w}_${"
                "format}_${sparse}_p${pad_h}x${pad_w}_s${stride_h}x${stride_w}_"
                "d${"
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
    } else {
        return "GI_kernel_conv2d_im2col";
    }
}

bool ConvIm2colFloat::IsAvailable(TContext* ctx) const {
    auto fmt = ctx->getAttrStr("format");
    int nr_operands = ctx->getAttrInt("nr_operands");
    std::string dst_oprands = std::string("operand:") + std::to_string(nr_operands - 1);
    bool param_value_ok =
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (fmt == "NCHW44" || fmt == "NCHW") &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "SIGMOID" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";
    bool type_ok = nr_operands >= 3 && ctx->getAttrOprand("operand:0").dtype == "f32" &&
                   ctx->getAttrOprand("operand:1").dtype == "f32" &&
                   ctx->getAttrOprand("operand:2").dtype == "f32";
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 4 &&
                     ctx->getAttrOprand(dst_oprands).shape.size() == 4;
    bool weight_ok = (ctx->getAttrStr("sparse") == "GROUP" &&
                      ctx->getAttrOprand("operand:1").shape.size() == 5) ||
                     (ctx->getAttrStr("sparse") == "DENSE" &&
                      ctx->getAttrOprand("operand:1").shape.size() == 4);
    if (fmt == "NCHW44") {
        layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                    ctx->getAttrOprand(dst_oprands).shape.size() == 5;
        weight_ok = (ctx->getAttrStr("sparse") == "GROUP" &&
                     ctx->getAttrOprand("operand:1").shape.size() == 7) ||
                    (ctx->getAttrStr("sparse") == "DENSE" &&
                     ctx->getAttrOprand("operand:1").shape.size() == 6);
    }
    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    bool available = param_value_ok && param_mode_ok && type_ok && noline_ok &&
                     layout_ok && weight_ok && bias_ok;
    return available;
}

std::string ConvIm2colFloat::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);

    writer << m_strategy.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_strategy.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    writer << m_framework.GenInitCode(ctx, &m_strategy);

    return writer.str();
}

std::string ConvIm2colFloat::GetWorkspaceBodyCondition(TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    if (jit) {
        ss << static_cast<GeneralIntrinsic::GIMatmulInternal*>(
                      m_strategy.GetInnerCtxMatmul(inner_ctx.get()))
                        ->GetPackBWorkspaceBody(inner_ctx.get())
           << ";\n";
    } else {
        ss << m_strategy.GetPackBWorkspaceSignature(inner_ctx.get()) << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    ss << m_framework.GenGetWorkSpaceCode(ctx, &m_strategy);
    return ss.str();
}

std::vector<KernelObj> ConvIm2colFloat::GetDependInternalSymbol(TContext* ctx) const {
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    auto inner_gemm = m_strategy.GetInnerCtxMatmul(inner_ctx.get());
    return {
            {inner_gemm->GetKernelSymbol(inner_ctx.get()),
             inner_gemm->GetKernelBody(inner_ctx.get()),
             inner_gemm->GetBodyGuardBegin(inner_ctx.get()),
             inner_gemm->GetBodyGuardEnd(inner_ctx.get()),
             inner_gemm->GetDependInternalSymbol(inner_ctx.get())}};
}

std::string ConvIm2colFloat::GetKernelBody(TContext* ctx) const {
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    std::stringstream writer;
    writer << R"(
#include <stdbool.h>
#include <string.h>
#include "gi_float.h"
)";

    writer << "extern "
           << static_cast<GeneralIntrinsic::GIMatmulInternal*>(
                      m_strategy.GetInnerCtxMatmul(inner_ctx.get()))
                      ->GetNakedKernelSignature(inner_ctx.get())
           << ";\n";
    writer << m_strategy.GetPackBSignature(inner_ctx.get()) << ";\n";

    writer << m_strategy.PaddingSrc(ctx);
    writer << m_strategy.Im2col(ctx);
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    writer << m_framework.GenKernelBodyCode(ctx, &m_strategy);
    return writer.str();
}

std::string ConvIm2colFloatM4N8::GetKernelSymbol(TContext* ctx) const {
    std::stringstream extra_ss;
    if (ctx) {
        if (is_bias(ctx)) {
            extra_ss << "_bias";
        }
        if (ctx->haveAttr("nonlineMode") &&
            ctx->getAttrStr("nonlineMode") != "IDENTITY") {
            extra_ss << "_" << ctx->getAttrStr("nonlineMode");
        }
        extra_ss << ctx->getAttrOprand("operand:0").dtype;
        std::string name_temp =
                "GI_kernel_conv2d_im2colm4n8_${kernel_h}x${kernel_w}_${"
                "format}_${sparse}_p${pad_h}x${pad_w}_s${stride_h}x${stride_w}_"
                "d${"
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
    } else {
        return "GI_kernel_conv2d_im2colm4n8";
    }
}

bool ConvIm2colFloatM4N8::IsAvailable(TContext* ctx) const {
    auto fmt = ctx->getAttrStr("format");
    int nr_operands = ctx->getAttrInt("nr_operands");
    std::string dst_oprands = std::string("operand:") + std::to_string(nr_operands - 1);
    bool param_value_ok =
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok =
            (fmt == "NCHW44") && ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "SIGMOID" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";
    bool type_ok = nr_operands >= 3 && ctx->getAttrOprand("operand:0").dtype == "f32" &&
                   ctx->getAttrOprand("operand:1").dtype == "f32" &&
                   ctx->getAttrOprand("operand:2").dtype == "f32";
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand(dst_oprands).shape.size() == 5;
    bool weight_ok = (ctx->getAttrStr("sparse") == "GROUP" &&
                      ctx->getAttrOprand("operand:1").shape.size() == 7) ||
                     (ctx->getAttrStr("sparse") == "DENSE" &&
                      ctx->getAttrOprand("operand:1").shape.size() == 6);
    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    bool available = param_value_ok && param_mode_ok && type_ok && noline_ok &&
                     layout_ok && weight_ok && bias_ok;
    return available;
}

std::string ConvIm2colFloatM4N8::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    writer << m_strategy.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_strategy.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    writer << m_framework.GenInitCode(ctx, &m_strategy);

    return writer.str();
}

std::string ConvIm2colFloatM4N8::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    ss << m_strategy.GetPackBWorkspaceSignature(inner_ctx.get());
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    ss << m_framework.GenGetWorkSpaceCode(ctx, &m_strategy);
    return ss.str();
}

std::vector<KernelObj> ConvIm2colFloatM4N8::GetDependInternalSymbol(
        TContext* ctx) const {
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    auto inner_gemm = m_strategy.GetInnerCtxMatmul(inner_ctx.get());
    return {
            {inner_gemm->GetKernelSymbol(inner_ctx.get()),
             inner_gemm->GetKernelBody(inner_ctx.get()),
             inner_gemm->GetBodyGuardBegin(inner_ctx.get()),
             inner_gemm->GetBodyGuardEnd(inner_ctx.get()),
             inner_gemm->GetDependInternalSymbol(inner_ctx.get())}};
}

std::string ConvIm2colFloatM4N8::GetKernelBody(TContext* ctx) const {
    auto inner_ctx = m_strategy.cvt2matmul_ctx(ctx);
    std::stringstream writer;
    writer << R"(
#include <stdbool.h>
#include <string.h>
#include "gi_float.h"
)";
    writer << "extern "
           << m_strategy.GetInnerCtxMatmul(inner_ctx.get())
                      ->GetKernelSignature(inner_ctx.get())
           << ";\n";
    auto nonline_mode = ctx && ctx->haveAttr("nonlineMode")
                              ? ctx->getAttrStr("nonlineMode")
                              : "IDENTITY";
    auto postprocess_pair = gen_postprocess_inline(inner_ctx.get());
    auto activation_gen = create_activation_gener_instrinsic(nonline_mode);
    writer << postprocess_pair.first;
    writer << StringTemplate::StringTemplateArgs()
                      .add("inner_matmul_sym",
                           m_strategy.GetInnerCtxMatmul(inner_ctx.get())
                                   ->GetKernelSymbol(inner_ctx.get()))
                      .add("matmul_sym",
                           m_strategy.GetInnerCtxMatmulSym(inner_ctx.get()))
                      .add("bias_expression",
                           is_bias(ctx)
                                   ? "GI_FLOAT32_t vbias = GiLoadFloat32(bias_ptr_);"
                                   : "GI_FLOAT32_t vbias = GiBroadcastFloat32(0.0);")
                      .add("bias_inc", is_bias(ctx) ? "bias_ptr_+=4;" : "")
                      .add("act_init", activation_gen->GenIntrinsicInitFloat())
                      .add("act_func",
                           [=](std::vector<std::string> args) {
                               return activation_gen->GenIntrinsicFloat(
                                       args[0], args[1]);
                           })
                      .add("postprocess_call", postprocess_pair.second)
                      .render(R"(
        static inline void ${matmul_sym}(const float* pack_a, const float* pack_b, float* C, size_t LDC, 
        size_t M, size_t N, size_t K, const float* bias_ptr){
                size_t strideA = K*4;
                size_t strideB = N*4;
                ${act_init}
                ${inner_matmul_sym}(pack_a, strideA, pack_b, strideB, C, LDC, M, N, K);
                const float* bias_ptr_=bias_ptr; 
                for (size_t m = 0; m < M; m += 4) {
                    float* output = C + (m / 4) * LDC;
                    ${bias_expression}
                    for (size_t n = 0; n < N; n++) {
                        GI_FLOAT32_t out_vec= GiLoadFloat32(output+n*4);
                        out_vec= GiAddFloat32(vbias, out_vec);
                        ${act_func(out_vec, out_vec)}; 
                        GiStoreFloat32(output+n*4, out_vec);
                    }
                    ${bias_inc} 
                }
                ${postprocess_call}
        })");
    writer << m_strategy.GetPackBSignature(inner_ctx.get()) << ";\n";

    writer << m_strategy.PaddingSrc(ctx);
    writer << m_strategy.Im2col(ctx);
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    writer << m_framework.GenKernelBodyCode(ctx, &m_strategy);
    return writer.str();
}
// vim: syntax=cpp.doxygen
