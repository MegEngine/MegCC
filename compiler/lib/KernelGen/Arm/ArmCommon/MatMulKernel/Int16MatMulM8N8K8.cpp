#include "Int16MatMulM8N8K8.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;
namespace {
std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("format", "MK8");
    inner_ctx->setAttr("transposeA", ctx->getAttrBool("transposeA"));
    inner_ctx->setAttr("transposeB", ctx->getAttrBool("transposeB"));
    inner_ctx->setAttr("dtype", "i16");
    return inner_ctx;
}
}  // namespace

bool Int16MatMulM8N8K8::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "i16" &&
                    context->getAttrOprand("operand:1").dtype == "i16" &&
                    context->getAttrOprand("operand:2").dtype == "i32";
    bool ok_mode = context->getAttrStr("format") == "MK8" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:0").shape.size() == 4 &&
                    context->getAttrOprand("operand:1").shape.size() == 3;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;

    return ok_dtype && ok_mode && ok_shape && ok_tran;
}

std::string Int16MatMulM8N8K8::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "ArmCommon_kernel_int16_matmul_m8_n8_k8_";
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

std::vector<KernelObj> Int16MatMulM8N8K8::GetDependInternalSymbol(
        TContext* context) const {
    auto ctx = GetInnerCtx(context);
    return {
            {m_internal_kernel.GetKernelSymbol(ctx.get()),
             m_internal_kernel.GetKernelBody(ctx.get()),
             m_internal_kernel.GetBodyGuardBegin(ctx.get()),
             m_internal_kernel.GetBodyGuardEnd(ctx.get()),
             m_internal_kernel.GetDependInternalSymbol(ctx.get())}};
}

std::string Int16MatMulM8N8K8::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include <arm_neon.h>\n";
    writer << "#include <stddef.h>\n";
    writer << "extern " << m_internal_kernel.GetKernelSignature(context) << ";\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << R"(
    const int16_t* A = (int16_t*)inputs[0]->ptr;
    const int16_t* B = (int16_t*)inputs[1]->ptr;
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
    const size_t M = a_layout.dims[0] * 8;
    const size_t K = a_layout.dims[1] * 8;
    const size_t N = c_layout.dims[1];

    TINYNN_ASSERT(8 == a_layout.dims[3]);
    TINYNN_ASSERT(8 == a_layout.dims[2]);
    TINYNN_ASSERT(8 == b_layout.dims[2]);
    TINYNN_ASSERT(8 == c_layout.dims[2]);

    TINYNN_ASSERT(a_layout.dims[0] == c_layout.dims[0]);
    TINYNN_ASSERT(a_layout.dims[1] == b_layout.dims[0]);
    TINYNN_ASSERT(b_layout.dims[1] == b_layout.dims[1]);

    ${matmul_symbol}(A, LDA, B, LDB, C, LDC, M, N, K);
    )";

    writer << R"(
        return TinyNN_SUCCESS;
    })";

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("matmul_symbol", m_internal_kernel.GetKernelSymbol(context))
                    .render(writer.str());
    return ss.str();
}