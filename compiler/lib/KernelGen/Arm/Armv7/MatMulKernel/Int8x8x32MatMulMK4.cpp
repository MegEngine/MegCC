#include "Int8MatMul.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
namespace {
std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("transposeA", ctx->getAttrBool("transposeA"));
    inner_ctx->setAttr("transposeB", ctx->getAttrBool("transposeB"));
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    inner_ctx->setAttr("last_dtype", last_dtype_str);
    inner_ctx->setAttr("dtype", "8832");
    return inner_ctx;
}
}  // namespace

bool Int8x8x32MatMulMK4::IsAvailable(TContext* context) const {
    bool ok_dtype = (context->getAttrOprand("operand:0").dtype == "i8" ||
                     context->getAttrOprand("operand:0").dtype == "si8") &&
                    (context->getAttrOprand("operand:1").dtype == "i8" ||
                     context->getAttrOprand("operand:1").dtype == "si8") &&
                    (context->getAttrOprand("operand:2").dtype == "i32" ||
                     context->getAttrOprand("operand:2").dtype == "si32");
    bool ok_mode = context->getAttrStr("format") == "MK4" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:0").shape.size() == 4 &&
                    context->getAttrOprand("operand:1").shape.size() == 3;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;

    return ok_dtype && ok_mode && ok_shape && ok_tran;
}

std::string Int8x8x32MatMulMK4::GetWorkspaceBodyCondition(
        TContext* context, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(context);
    if (jit) {
        ss << m_internal_kernel.GetPackAWorkspaceBody(inner_ctx.get()) << ";\n";
        ss << m_internal_kernel.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << m_internal_kernel.GetPackAWorkspaceSignature(inner_ctx.get())
           << ";\n";
        ss << "extern " << m_internal_kernel.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
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
                         m_internal_kernel.GetPackAWorkspaceSymbol(inner_ctx.get()))
                    .add("packb_workspace_sym",
                         m_internal_kernel.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .render(workspace_temp);
    return ss.str();
}

std::string Int8x8x32MatMulMK4::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Armv7_kernel_int8x8x32_matmul_mk4_";
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

std::vector<KernelObj> Int8x8x32MatMulMK4::GetDependInternalSymbol(
        TContext* context) const {
    auto ctx = GetInnerCtx(context);
    return {
            {m_internal_kernel.GetKernelSymbol(ctx.get()),
             m_internal_kernel.GetKernelBody(ctx.get()),
             m_internal_kernel.GetBodyGuardBegin(ctx.get()),
             m_internal_kernel.GetBodyGuardEnd(ctx.get()),
             m_internal_kernel.GetDependInternalSymbol(ctx.get())}};
}

std::string Int8x8x32MatMulMK4::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include <marm_neon.h>\n";
    writer << "extern "
           << m_internal_kernel.GetKernelSignature(GetInnerCtx(context).get()) << ";\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << R"(
    const int8_t* A = (int8_t*)inputs[0]->ptr;
    const int8_t* B = (int8_t*)inputs[1]->ptr;
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

    TINYNN_ASSERT(4 == a_layout.dims[2]);
    TINYNN_ASSERT(4 == a_layout.dims[3]);
    TINYNN_ASSERT(4 == b_layout.dims[2]);
    TINYNN_ASSERT(4 == c_layout.dims[2]);

    TINYNN_ASSERT(a_layout.dims[0] == c_layout.dims[0]);
    TINYNN_ASSERT(a_layout.dims[1] == b_layout.dims[0]);
    TINYNN_ASSERT(b_layout.dims[1] == b_layout.dims[1]);

    ${matmul_symbol}(A, LDA, B, LDB, C, LDC, M, N, K, 0, workspace->ptr, 1.f, 1.f, 1.f);
    )";

    writer << R"(
    return TinyNN_SUCCESS;
})";

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("matmul_symbol",
                         m_internal_kernel.GetKernelSymbol(GetInnerCtx(context).get()))
                    .render(writer.str());
    return ss.str();
}