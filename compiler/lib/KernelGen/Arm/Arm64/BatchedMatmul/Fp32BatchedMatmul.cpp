#include <string>
#include "../../../Utils/StringTemplate.h"
#include "BatchedMatmul.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;

namespace {
std::shared_ptr<TContext> GetInnerCtx(TContext* ctx) {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    inner_ctx->setAttr("with_bias", false);
    inner_ctx->setAttr("format", "NCHW");
    inner_ctx->setAttr("transposeA", ctx->getAttrBool("transposeA"));
    inner_ctx->setAttr("transposeB", ctx->getAttrBool("transposeB"));
    inner_ctx->setAttr("dtype", "f32");
    return inner_ctx;
}
}  // namespace

bool Fp32BatchedMatmul::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "DEFAULT" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    return ok_dtype && ok_mode;
}

std::string Fp32BatchedMatmul::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    bool trans_a = context->getAttrBool("transposeA");
    bool trans_b = context->getAttrBool("transposeB");
    std::string delare_K, delare_M, delare_N;

    if (trans_a == false) {
        delare_K = "const int K = a_layout.dims[2];";
        delare_M = "const int M = a_layout.dims[1];";
    } else {
        delare_K = "const int K = a_layout.dims[1];";
        delare_M = "const int M = a_layout.dims[2];";
    }
    if (trans_b == false) {
        delare_N = "const int N = b_layout.dims[2];";
    } else {
        delare_N = "const int N = b_layout.dims[1];";
    }

    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        const Layout a_layout = inputs[0]->layout;
        const Layout b_layout = inputs[1]->layout;
        )" + delare_K + delare_M +
                                 delare_N + R"(
        const int ev_M = ((M + 7) / 8 ) * 8;
        const int ev_N = ((N + 11) / 12 ) * 12;
        const int ev_K = K;
        *workspace = sizeof(float) * (ev_M * ev_K + ev_K * ev_N) + 64;
        return TinyNN_SUCCESS;
    })";
    ss << workspace_temp;
    return ss.str();
}

std::string Fp32BatchedMatmul::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Arm64_kernel_fp32_batched_matmul_";
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

std::string Fp32BatchedMatmul::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    bool trans_a = context->getAttrBool("transposeA");
    bool trans_b = context->getAttrBool("transposeB");
    std::string delare_K, check_tensor;
    if (trans_a == false) {
        delare_K = "const int K = a_layout.dims[2];";
        check_tensor += "TINYNN_ASSERT(a_layout.dims[1]==M);\n";
        check_tensor += "TINYNN_ASSERT(a_layout.dims[2]==K);\n";
    } else {
        delare_K = "const int K = a_layout.dims[1];";
        check_tensor += "TINYNN_ASSERT(a_layout.dims[2]==M);\n";
        check_tensor += "TINYNN_ASSERT(a_layout.dims[1]==K);\n";
    }
    if (trans_b == false) {
        check_tensor += "TINYNN_ASSERT(b_layout.dims[2]==N);\n";
        check_tensor += "TINYNN_ASSERT(b_layout.dims[1]==K);\n";
    } else {
        check_tensor += "TINYNN_ASSERT(b_layout.dims[2]==K);\n";
        check_tensor += "TINYNN_ASSERT(b_layout.dims[1]==N);\n";
    }
    writer << "#include<arm_neon.h>\n";
    auto inner_ctx = GetInnerCtx(context);
    writer << "extern " << gemm_kernel.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << "extern " << gemm_kernel.GetPackBSignature(inner_ctx.get()) << ";\n";
    writer << "extern " << gemm_kernel.GetNakedKernelSignature(inner_ctx.get())
           << ";\n";
    writer << "extern " << gemm_kernel.GetPackAWorkspaceSignature(inner_ctx.get())
           << ";\n";
    auto pack_a_func = gemm_kernel.GetPackASymbol(inner_ctx.get());
    auto pack_b_func = gemm_kernel.GetPackBSymbol(inner_ctx.get());
    auto kern_func = gemm_kernel.GetNakedKernelSymbol(inner_ctx.get());
    writer << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    float* a_data = (float*)inputs[0]->ptr;
    float* b_data = (float*)inputs[1]->ptr;
    float* c_data = (float*)outputs[0]->ptr;
    TINYNN_ASSERT(a_data);
    TINYNN_ASSERT(b_data);
    TINYNN_ASSERT(c_data);
    const Tensor* a_tensor = inputs[0];
    const Tensor* b_tensor = inputs[1];
    const Tensor* c_tensor = outputs[0];
    const Layout a_layout = a_tensor->layout;
    const Layout b_layout = b_tensor->layout;
    const Layout c_layout = c_tensor->layout;
    const int bc_stride_a = a_layout.stride[0];
    const int bc_stride_b = b_layout.stride[0];
    const int bc_stride_c = c_layout.stride[0];
    const int Astride = a_layout.stride[1];
    const int Bstride = b_layout.stride[1];
    const int Cstride = c_layout.stride[1];
    const int M = c_layout.dims[1];
    const int N = c_layout.dims[2]; 
    ${delare_K};
    ${check_tensor};

    void* total_workspace = workspace->ptr;
    const size_t a_workspace_byte = ${a_workspace_func}(0, M, 0, K);

    float* a_workspace = (float*)total_workspace;
    float* b_workspace = (float*)(total_workspace + a_workspace_byte);

    int batch_size = a_layout.dims[0];
    float *A, *B, *C;
    for(int i = 0; i < batch_size; ++i) {
        A = a_data + i * bc_stride_a;
        B = b_data + i * bc_stride_b;
        C = c_data + i * bc_stride_c;

        ${pack_a_func}(a_workspace, A, Astride, 0, M, 0, K);
        ${pack_b_func}(b_workspace, B, Bstride, 0, N, 0, K);
        ${kern_func}(a_workspace, b_workspace, C, Cstride, M, N, K, 0);
    }
    return TinyNN_SUCCESS;
    })";

    writer << StringTemplate::StringTemplateArgs()
                      .add("delare_K", delare_K)
                      .add("check_tensor", check_tensor)
                      .add("pack_a_func", pack_a_func)
                      .add("pack_b_func", pack_b_func)
                      .add("kern_func", kern_func)
                      .add("a_workspace_func",
                           gemm_kernel.GetPackAWorkspaceSymbol(inner_ctx.get()))
                      .render(body_temp);
    return writer.str();
}

std::vector<KernelObj> Fp32BatchedMatmul::GetDependInternalSymbol(
        TContext* context) const {
    auto inner_ctx = GetInnerCtx(context);
    return {
            {gemm_kernel.GetKernelSymbol(inner_ctx.get()),
             gemm_kernel.GetKernelBody(inner_ctx.get()),
             gemm_kernel.GetBodyGuardBegin(inner_ctx.get()),
             gemm_kernel.GetBodyGuardEnd(inner_ctx.get())}};
}
