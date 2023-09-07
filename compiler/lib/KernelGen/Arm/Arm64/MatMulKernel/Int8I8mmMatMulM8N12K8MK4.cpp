#include "../InternalKernel/InternalKernel.h"
#include "MatMul.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
std::shared_ptr<TContext> Int8I8mmMatMulM8N12K8MK4::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("dtype", "8832");
    return inner_ctx;
}

bool Int8I8mmMatMulM8N12K8MK4::IsAvailable(TContext* context) const {
    bool ok_dtype = Utils::is_int_dtype(context->getAttrOprand("operand:0").dtype, 8) &&
                    Utils::is_int_dtype(context->getAttrOprand("operand:1").dtype, 8) &&
                    Utils::is_int_dtype(context->getAttrOprand("operand:2").dtype, 32);
    bool ok_mode = context->getAttrStr("format") == "MK4" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:0").shape.size() == 4 &&
                    context->getAttrOprand("operand:1").shape.size() == 3;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;
    return ok_dtype && ok_mode && ok_shape && ok_tran;
}
//! kernel gen
std::string Int8I8mmMatMulM8N12K8MK4::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Arm64_kernel_int8_i8mm_matmul_8x8x12mk4_";
    if (context->getAttrBool("transposeA")) {
        ss << "t";
    } else {
        ss << "n";
    }
    if (context->getAttrBool("transposeB")) {
        ss << "t";
    } else {
        ss << "n";
    }
    return ss.str();
}

std::string Int8I8mmMatMulM8N12K8MK4::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_matmul.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_inner_matmul.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    const uint32_t nr_out_weight = 1;
    const std::string common_def = R"(
        Tensor* in_weights = inputs[0];
        const int ymax = in_weights->layout.dims[0] * 4;
        const int kmax = in_weights->layout.dims[1] * 4;
        const int ldin = kmax * 4;
    )";
    const std::string fill_weight_attr =
            R"(
        out_weights->layout.nr_dim = 1;
        out_weights->layout.dims[0] = )" +
            m_inner_matmul.GetPackAWorkspaceSymbol(inner_ctx.get()) +
            R"((0, ymax, 0, kmax);
        out_weights->layout.stride[0] = 1;
        out_weights->dtype.type_enum = TinyNN_INT8;
        out_weights->name = in_weights->name;
    )";
    const std::string fill_weight_transform =
            StringTemplate::StringTemplateArgs()
                    .add("packa_sym", m_inner_matmul.GetPackASymbol(inner_ctx.get()))
                    .render(
                            R"(    
        int8_t* outptr = out_weights->ptr;
        int8_t* inptr = in_weights->ptr;
        ${packa_sym}(outptr, inptr, ldin, 0, ymax, 0, kmax);
    )");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::vector<KernelObj> Int8I8mmMatMulM8N12K8MK4::GetDependInternalSymbol(
        TContext* context) const {
    auto inner_ctx = GetInnerCtx(context);
    return {
            {m_inner_matmul.GetKernelSymbol(inner_ctx.get()),
             m_inner_matmul.GetKernelBody(inner_ctx.get()),
             m_inner_matmul.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_matmul.GetBodyGuardEnd(inner_ctx.get()),
             m_inner_matmul.GetDependInternalSymbol(inner_ctx.get())}};
}

std::string Int8I8mmMatMulM8N12K8MK4::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
    if (jit) {
        ss << m_inner_matmul.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << m_inner_matmul.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout b_layout = inputs[1]->layout;
        const size_t K = b_layout.dims[0] * 4;
        const size_t N = b_layout.dims[1];
        *workspace = ${packb_workspace_sym}(0, N, 0, K);
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("packb_workspace_sym",
                         m_inner_matmul.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .render(workspace_temp);
    return ss.str();
}

std::string Int8I8mmMatMulM8N12K8MK4::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(context);
    writer << "#include <arm_neon.h>\n";
    writer << "extern " << m_inner_matmul.GetKernelSignature(inner_ctx.get()) << ";\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);
    std::string body_temp = R"({
    int8_t* A = (int8_t*)inputs[0]->ptr;
    int8_t* B = (int8_t*)inputs[1]->ptr;
    int32_t* C = (int32_t*)outputs[0]->ptr;
    TINYNN_ASSERT(A);
    TINYNN_ASSERT(B);
    TINYNN_ASSERT(C);
    const Layout b_layout = inputs[1]->layout;
    const Layout c_layout = outputs[0]->layout;
    const size_t LDB = b_layout.stride[0];
    const size_t LDC = c_layout.stride[0];
    const size_t M = c_layout.dims[0] * 4;
    const size_t K = b_layout.dims[0] * 4;
    const size_t N = c_layout.dims[1];
    const size_t LDA = K * 4;

    TINYNN_ASSERT(4 == b_layout.dims[2]);
    TINYNN_ASSERT(4 == c_layout.dims[2]);

    TINYNN_ASSERT(b_layout.dims[1] == c_layout.dims[1]);

    void* workspace_ptr = workspace->ptr;
    TINYNN_ASSERT(workspace_ptr);

    ${matmul_symbol}(A, LDA, B, LDB, C, LDC, M, N, K, 0, workspace_ptr, 1.f, 1.f, 1.f);
    return TinyNN_SUCCESS;
    })";

    writer << StringTemplate::StringTemplateArgs()
                      .add("matmul_symbol",
                           m_inner_matmul.GetKernelSymbol(inner_ctx.get()))
                      .render(body_temp);
    return writer.str();
}

// vim: syntax=cpp.doxygen
