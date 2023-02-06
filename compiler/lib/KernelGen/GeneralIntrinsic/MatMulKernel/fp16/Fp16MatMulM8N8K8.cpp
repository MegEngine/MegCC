/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/MatMulKernel/Fp16MatMulM8N8K8.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Fp16MatMul.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool Fp16MatMulM8N8K8::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f16" &&
                    context->getAttrOprand("operand:1").dtype == "f16" &&
                    context->getAttrOprand("operand:2").dtype == "f16";
    bool ok_mode = context->getAttrStr("format") == "MK8" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:0").shape.size() == 4 &&
                    context->getAttrOprand("operand:1").shape.size() == 3;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;

    return ok_dtype && ok_mode && ok_shape && ok_tran;
}
//! kernel gen
std::string Fp16MatMulM8N8K8::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "GI_kernel_fp16_matmul_8x8mk8_";
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
    inner_ctx->setAttr("format", "MK8");
    inner_ctx->setAttr("transposeA", ctx->getAttrBool("transposeA"));
    inner_ctx->setAttr("transposeB", ctx->getAttrBool("transposeB"));
    inner_ctx->setAttr("dtype", "f16");
    return inner_ctx;
}
}  // namespace

std::vector<KernelObj> Fp16MatMulM8N8K8::GetDependInternalSymbol(
        TContext* context) const {
    auto matmul_kernel = MatmulM8N8MK8Kernel();
    auto ctx = GetInnerCtx(context);
    return {{matmul_kernel.GetKernelSymbol(ctx.get()),
             matmul_kernel.GetKernelBody(ctx.get()),
             matmul_kernel.GetBodyGuardBegin(ctx.get()),
             matmul_kernel.GetBodyGuardEnd(ctx.get()),
             matmul_kernel.GetDependInternalSymbol(ctx.get())}};
}

std::string Fp16MatMulM8N8K8::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include \"gi_float16.h\"\n";
    auto matmul_kernel = MatmulM8N8MK8Kernel();
    writer << "extern " << matmul_kernel.GetKernelSignature(context) << ";\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << R"(
    const gi_float16_t* A = (gi_float16_t*)inputs[0]->ptr;
    const gi_float16_t* B = (gi_float16_t*)inputs[1]->ptr;
    gi_float16_t* C = (gi_float16_t*)outputs[0]->ptr;
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
                    .add("matmul_symbol",
                         matmul_kernel.GetKernelSymbol(context))
                    .render(writer.str());
    return ss.str();
}

// vim: syntax=cpp.doxygen
