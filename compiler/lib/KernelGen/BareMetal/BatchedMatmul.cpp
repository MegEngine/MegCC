#include <float.h>
#include <sstream>

#include "BatchedMatmul.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool BatchedMatrixMulKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "DEFAULT" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    return ok_dtype && ok_mode;
}

//! kernel gen
std::string BatchedMatrixMulKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_batched_matmul_";
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
std::string emit_k(bool trans_a) {
    if (trans_a) {
        return "const int k = a_layout.dims[1];";
    } else {
        return "const int k = a_layout.dims[2];";
    }
}

std::string emit_a_val(bool trans_a) {
    if (trans_a) {
        return "float a_val = a_data[k_idx * lda + m_idx];";
    } else {
        return "float a_val = a_data[m_idx * lda + k_idx];";
    }
}

std::string emit_b_val(bool trans_b) {
    if (trans_b) {
        return "float b_val = b_data[n_idx * ldb + k_idx];";
    } else {
        return "float b_val = b_data[k_idx * ldb + n_idx];";
    }
}

}  // namespace

std::string BatchedMatrixMulKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    bool trans_a = context->getAttrBool("transposeA");
    bool trans_b = context->getAttrBool("transposeB");

    ss << GenCommonRet() << " " << GetKernelSignature(context);
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
    const int stride_a = a_layout.stride[0];
    const int stride_b = b_layout.stride[0];
    const int stride_c = c_layout.stride[0];
    const int lda = a_layout.stride[1];
    const int ldb = b_layout.stride[1];
    const int ldc = c_layout.stride[1];
    const int b = c_layout.dims[0];
    const int m = c_layout.dims[1];
    const int n = c_layout.dims[2];
    ${k_init}
    for(int b_idx = 0; b_idx < b; ++b_idx){
        for (int m_idx = 0; m_idx < m; ++m_idx) {
            for (int n_idx = 0; n_idx < n; ++n_idx) {
                float sum = 0.f;
                for (int k_idx = 0; k_idx < k; ++k_idx) {
                    ${a_init}
                    ${b_init}
                    sum += a_val * b_val;
                }
                c_data[m_idx * ldc + n_idx] = sum;
            }
        }
        a_data += stride_a;
        b_data += stride_b;
        c_data += stride_c;
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("k_init", emit_k(trans_a))
                    .add("a_init", emit_a_val(trans_a))
                    .add("b_init", emit_b_val(trans_b))
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen
