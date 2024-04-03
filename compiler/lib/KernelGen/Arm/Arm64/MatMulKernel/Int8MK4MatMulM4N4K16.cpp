#include "../InternalKernel/InternalKernel.h"
#include "MatMul.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
std::shared_ptr<TContext> Int8MK4MatMulM4N4K16::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("dtype", "8832");
    return inner_ctx;
}

bool Int8MK4MatMulM4N4K16::IsAvailable(TContext* context) const {
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
std::string Int8MK4MatMulM4N4K16::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Arm64_kernel_int8_matmul_m4n4k16_mk4_";
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

std::vector<KernelObj> Int8MK4MatMulM4N4K16::GetDependInternalSymbol(
        TContext* context) const {
    auto inner_ctx = GetInnerCtx(context);
    return {
            {m_inner_matmul.GetKernelSymbol(inner_ctx.get()),
             m_inner_matmul.GetKernelBody(inner_ctx.get()),
             m_inner_matmul.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_matmul.GetBodyGuardEnd(inner_ctx.get()),
             m_inner_matmul.GetDependInternalSymbol(inner_ctx.get())}};
}

std::string Int8MK4MatMulM4N4K16::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
    if (jit) {
        ss << m_inner_matmul.GetPackAWorkspaceBody(inner_ctx.get()) << ";\n";
        ss << m_inner_matmul.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << m_inner_matmul.GetPackAWorkspaceSignature(inner_ctx.get())
           << ";\n";
        ss << "extern " << m_inner_matmul.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout a_layout = inputs[0]->layout;
        const Layout b_layout = inputs[1]->layout;
        const size_t M = a_layout.dims[0] * 4;
        const size_t K = a_layout.dims[1] * 4;
        const size_t N = b_layout.dims[1];
        *workspace = ${packa_workspace_sym}(0, M, 0, K) + ${packb_workspace_sym}(0, N, 0, K);
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("packa_workspace_sym",
                         m_inner_matmul.GetPackAWorkspaceSymbol(inner_ctx.get()))
                    .add("packb_workspace_sym",
                         m_inner_matmul.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .render(workspace_temp);
    return ss.str();
}

std::string Int8MK4MatMulM4N4K16::GetKernelBody(TContext* context) const {
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
    const Layout a_layout = inputs[0]->layout;
    const Layout b_layout = inputs[1]->layout;
    const Layout c_layout = outputs[0]->layout;
    const size_t LDA = a_layout.stride[0];
    const size_t LDB = b_layout.stride[0];
    const size_t LDC = c_layout.stride[0];
    const size_t M = a_layout.dims[0] * 4;
    const size_t K = a_layout.dims[1] * 4;
    const size_t N = c_layout.dims[1];

    TINYNN_ASSERT(4 == a_layout.dims[3]);
    TINYNN_ASSERT(4 == a_layout.dims[2]);
    TINYNN_ASSERT(4 == b_layout.dims[2]);
    TINYNN_ASSERT(4 == c_layout.dims[2]);

    TINYNN_ASSERT(a_layout.dims[0] == c_layout.dims[0]);
    TINYNN_ASSERT(a_layout.dims[1] == b_layout.dims[0]);
    TINYNN_ASSERT(b_layout.dims[1] == b_layout.dims[1]);

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
