#include <sstream>
#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ConvKernel/ConvKernel.h"
#include "GeneralIntrinsic/ElemwiseHelper/ElemwiseHelper.h"
#include "GeneralIntrinsic/GIMathHelper.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
static inline std::pair<std::string, std::string> gen_postprocess_inline(
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

        std::shared_ptr<ElemwiseGenUnary> ElemwiseImpl = create_elem("f16", "f16");

        auto ImpleGen = [=](std::vector<std::string> strs) {
            return ElemwiseImpl->GenCodeBody(strs);
        };
        std::string post_process_temp;
        if (ctx->getAttrStr("format") == "MK8") {
            post_process_temp = R"(
            if (LDC == (8 * N)){
                ${ElemwiseImplName}(output_data, output_data, out_c * N);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < out_c; m_idx += 8){
                    ${ElemwiseImplName}(output_data + cnt * LDC, output_data + cnt * LDC, 8 * N);
                    ++cnt;
                }
            }
        )";
        } else {
            CC_ABORT << "unsupported format for fp16 im2col\n";
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
        declare_ss << gi_math.GiSigmoidPsFloat16() << "\n";
        declare_ss << gi_math.FastFp16toFp32() << "\n";
        declare_ss << gi_math.FastFp32toFp16() << "\n";
        declare_ss << ElemwiseImpl->GenCodeBody({});
    }
    return {declare_ss.str(), call_str};
}

}  // namespace
bool Conv1x1Float16MK8::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == 1 && ctx->getAttrUInt("kernel_w") == 1 &&
            ctx->getAttrUInt("stride_h") == 1 && ctx->getAttrUInt("stride_w") == 1 &&
            ctx->getAttrUInt("pad_h") == 0 && ctx->getAttrUInt("pad_w") == 0 &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW88" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "SIGMOID" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f16" &&
                   ctx->getAttrOprand("operand:1").dtype == "f16" &&
                   ctx->getAttrOprand("operand:2").dtype == "f16";
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 8;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}

std::string Conv1x1Float16MK8::GetKernelSymbol(TContext* ctx) const {
    std::stringstream extra_ss;
    if (ctx) {
        if (is_bias(ctx)) {
            extra_ss << "_bias";
        }
        if (ctx->haveAttr("nonlineMode") &&
            ctx->getAttrStr("nonlineMode") != "IDENTITY") {
            extra_ss << "_" << ctx->getAttrStr("nonlineMode");
        }
        std::string name_temp =
                "GI_kernel_conv2d_conv1x1_fp16_${format}_${kernel_h}x${kernel_w}_${"
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
    } else {
        return "GI_kernel_conv2d_conv1x1_fp16";
    }
}

std::string Conv1x1Float16MK8::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << R"(
#include "gi_float16.h"
)";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    int PACK_C_SIZE = 8;
    Tensor* in_weights = inputs[1];
    int ymax = in_weights->layout.dims[0] * PACK_C_SIZE;
    int kmax = in_weights->layout.dims[1] * PACK_C_SIZE;
    int ldin = kmax * PACK_C_SIZE;
                      )";
    std::string fill_weight_attr =
            R"(
    out_weights->layout.nr_dim = 1;
    out_weights->layout.dims[0] = (ymax * kmax);
    out_weights->layout.stride[0] = 1;
    out_weights->dtype.type_enum=TinyNN_FLOAT16;
    out_weights->name = in_weights->name;
                      )";
    std::string fill_weight_transform =
            R"(    
    gi_float16_t* outptr = out_weights->ptr;
    gi_float16_t* inptr = in_weights->ptr;
    memcpy(outptr, inptr, ymax * kmax * 2);
    )";
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string Conv1x1Float16MK8::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        return TinyNN_SUCCESS;
    })";
    ss << workspace_temp;
    return ss.str();
}

std::vector<KernelObj> Conv1x1Float16MK8::GetDependInternalSymbol(TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);

    return {
            {m_inner_gemm.GetKernelSymbol(inner_ctx.get()),
             m_inner_gemm.GetKernelBody(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardEnd(inner_ctx.get())}};
}

std::shared_ptr<TContext> Conv1x1Float16MK8::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode", CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    inner_ctx->setAttr("with_bias", ConvImpl::is_bias(ctx));
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "MK8");
    inner_ctx->setAttr("dtype", "f16");
    return inner_ctx;
}

std::string Conv1x1Float16MK8::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);

    auto nonline_mode = ctx && ctx->haveAttr("nonlineMode")
                              ? ctx->getAttrStr("nonlineMode")
                              : "IDENTITY";
    auto postprocess_pair = gen_postprocess_inline(inner_ctx.get());
    auto activation_gen = create_activation_gener_instrinsic(nonline_mode, "f16");
    writer << R"(
#include "gi_float.h"
#include "gi_float16.h"
)";
    writer << postprocess_pair.first;
    writer << m_inner_gemm.GetKernelSignature(inner_ctx.get()) << ";\n";
    auto matmul_body =
            StringTemplate::StringTemplateArgs()
                    .add("inner_matmul_sym",
                         m_inner_gemm.GetKernelSymbol(inner_ctx.get()))
                    .add("bias_expression",
                         is_bias(ctx) ? "GI_FLOAT16_t vbias = GiLoadFloat16(bias_ptr_);"
                                      : "GI_FLOAT16_t vbias = GiBroadcastFloat16(0.0);")
                    .add("bias_inc", is_bias(ctx) ? "bias_ptr_+=8;" : "")
                    .add("act_init", activation_gen->GenIntrinsicInitFloat())
                    .add("act_func",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenIntrinsicFloat(args[0], args[1]);
                         })
                    .add("postprocess_call", postprocess_pair.second)
                    .render(R"(
                size_t strideA = in_c*8;
                size_t strideB = N*8;
                ${act_init}
                ${inner_matmul_sym}(weight_data, strideA, input_data, strideB, output_data, LDC, out_c, N, in_c);
                const gi_float16_t* bias_ptr_=bias_data; 
                for (size_t m = 0; m < out_c; m += 8) {
                    gi_float16_t* output = output_data + (m / 8) * LDC;
                    ${bias_expression}
                    for (size_t n = 0; n < N; n++) {
                        GI_FLOAT16_t out_vec= GiLoadFloat16(output+n*8);
                        out_vec= GiAddFloat16(vbias, out_vec);
                        ${act_func(out_vec, out_vec)}; 
                        GiStoreFloat16(output+n*8, out_vec);
                    }
                    ${bias_inc} 
                }
                ${postprocess_call}
        )");

    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    std::string bias_ptr_str = is_bias(ctx) ? "inputs[2]->ptr;" : "NULL;";
    writer << StringTemplate::StringTemplateArgs()
                      .add("bias_ptr", bias_ptr_str)
                      .add("matmul", matmul_body)
                      .render(
                              R"({
    gi_float16_t* input_data = inputs[0]->ptr;
    gi_float16_t* output_data = outputs[0]->ptr;


    Layout in_layout = inputs[0]->layout;
    Layout out_layout = outputs[0]->layout;
    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1] * in_layout.dims[4];
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];
    const int PACK_C_SIZE = 8;

    const int out_c = out_layout.dims[1] * out_layout.dims[4];
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];
    const size_t N = out_h * out_w;

    const int K8 = in_c * 8;

    const int LDC = out_h * out_w * PACK_C_SIZE;
    const int LDB = in_h * in_w * PACK_C_SIZE;

    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        gi_float16_t* weight_data = inputs[1]->ptr;
        gi_float16_t* bias_data = ${bias_ptr}
        ${matmul}
        input_data += in_c * in_h * in_w;
        output_data += out_c * out_h * out_w;
    }
    return TinyNN_SUCCESS;
})");

    return writer.str();
}

// vim: syntax=cpp.doxygen
