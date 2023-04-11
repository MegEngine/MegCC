#include "../InternalKernel/InternalKernel.h"
#include "Fp32MatMul.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;

bool Fp32MatMulM4N16K4::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "MK4" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:0").shape.size() == 4 &&
                    context->getAttrOprand("operand:1").shape.size() == 3;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;

    return ok_dtype && ok_mode && ok_shape && ok_tran;
}
//! kernel gen
std::string Fp32MatMulM4N16K4::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Arm64_kernel_fp32_matmul_4x16mk4_";
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

namespace {
std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("transposeA", ctx->getAttrBool("transposeA"));
    inner_ctx->setAttr("transposeB", ctx->getAttrBool("transposeB"));
    inner_ctx->setAttr("dtype", "f32");
    return inner_ctx;
}
}  // namespace

std::vector<KernelObj> Fp32MatMulM4N16K4::GetDependInternalSymbol(
        TContext* context) const {
    auto matmul_kernel = MatmulM4N16MK4Kernel();
    auto ctx = GetInnerCtx(context);
    return {
            {matmul_kernel.GetKernelSymbol(ctx.get()),
             matmul_kernel.GetKernelBody(ctx.get()),
             matmul_kernel.GetBodyGuardBegin(ctx.get()),
             matmul_kernel.GetBodyGuardEnd(ctx.get()),
             matmul_kernel.GetDependInternalSymbol(ctx.get())}};
}

std::string Fp32MatMulM4N16K4::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    auto matmul_kernel = MatmulM4N16MK4Kernel();
    writer << "#include<arm_neon.h>\n";
    writer << "extern " << matmul_kernel.GetKernelSignature(context) << ";\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << R"(
    float* A = (float*)inputs[0]->ptr;
    float* B = (float*)inputs[1]->ptr;
    float* C = (float*)outputs[0]->ptr;
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

    ${matmul_symbol}(A, LDA, B, LDB, C, LDC, M, N, K);
    )";

    writer << R"(
        return TinyNN_SUCCESS;
    })";

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("matmul_symbol", matmul_kernel.GetKernelSymbol(context))
                    .render(writer.str());
    return ss.str();
}

// vim: syntax=cpp.doxygen
