#include "Fp32Gevm.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

namespace {
std::string vec_mul_matrix_n_n(void) {
    return R"(
    memset(C, 0, sizeof(float) * M * N);
    for (size_t m = 0; m < M; m++) {
      for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            C[m * Cstride + n] += A[m * Astride + k] * B[k * Bstride + n];
        }
      }
    }        
  )";
}

std::string vec_mul_matrix_n_t(void) {
    return R"(
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float acc = 0.f;
            for(size_t k = 0;k < K;k++){
                acc += A[m*Astride+k] * B[n*Bstride+k];
            }
            C[m*Cstride+n] = acc;
        }
    }
    )";
}

}  // namespace

bool Fp32GevmKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "DEFAULT" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:0").shape.size() == 2 &&
                    context->getAttrOprand("operand:0").shape[0] <= 4;
    bool ok_tran = context->getAttrBool("transposeA") == false;
    return ok_dtype && ok_mode && ok_shape && ok_tran;
}
//! kernel gen
std::string Fp32GevmKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_gevm";
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

std::string Fp32GevmKernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include <string.h>\n";
    bool trans_b = context->getAttrBool("transposeB");

    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    // clang-format off
    writer << R"(
    float* A = (float*)inputs[0]->ptr;
    float* B = (float*)inputs[1]->ptr;
    float* C = (float*)outputs[0]->ptr;
    TINYNN_ASSERT(A);
    TINYNN_ASSERT(B);
    TINYNN_ASSERT(C);
    const Tensor* a_tensor = inputs[0];
    const Tensor* b_tensor = inputs[1];
    const Tensor* c_tensor = outputs[0];
    const Layout a_layout = a_tensor->layout;
    const Layout b_layout = b_tensor->layout;
    const Layout c_layout = c_tensor->layout;
    const int Astride = a_layout.stride[0];
    const int Bstride = b_layout.stride[0];
    const int Cstride = c_layout.stride[0];
    const int M = c_layout.dims[0];
    const int K = a_layout.dims[1];
    const int N = c_layout.dims[1];
    )";
    if(trans_b)
        writer << vec_mul_matrix_n_t();
    else 
        writer << vec_mul_matrix_n_n();
    writer << R"(
        return TinyNN_SUCCESS;
    })";
    // clang-format on
    return writer.str();
}

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen