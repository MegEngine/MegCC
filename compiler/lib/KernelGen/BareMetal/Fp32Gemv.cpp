#include "Fp32Gemv.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {
namespace {
std::string matrix_mul_vec_nn(void) {
    return R"(
      for(size_t n = 0;n < N;n++){
        for(size_t m = 0;m < M;m++){
          float acc = 0.f;
          for(size_t k = 0;k < K;k++){
            acc += A[m*Astride + k] * D[n*Bstride + k];
          }
          C[m*Cstride + n] = acc;
        }
      }
    )";
}
}  // namespace

bool Fp32GemvKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "DEFAULT" &&
                   context->getAttrStr("compute_mode") == "DEFAULT";
    bool ok_shape = context->getAttrOprand("operand:1").shape.size() == 2 &&
                    context->getAttrOprand("operand:1").shape[1] <= 4;
    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;
    return ok_dtype && ok_mode && ok_shape && ok_tran;
}
//! kernel gen
std::string Fp32GemvKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_gemv";
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

std::string Fp32GemvKernel::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[1]->layout;
        const uint32_t ic = in_layout.dims[1];
        if(ic == 1)*workspace = 0;
        else *workspace = in_layout.dims[0] * ic * sizeof(float);
        return TinyNN_SUCCESS;
    })";
    ss << workspace_temp;
    return ss.str();
}

std::string Fp32GemvKernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include <string.h>\n";
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
    const int Cstride = c_layout.stride[0];
    const int M = c_layout.dims[0];
    const int K = a_layout.dims[1];
    const int N = c_layout.dims[1];
    const int Bstride = K;


  float* D;
  if(N == 1)D = B;
  else{
    D = (float*)workspace->ptr;
    for(size_t n = 0;n < N;n++){
      for(size_t k = 0;k < K;k++){
        D[n * K + k] = B[k * N + n];
      }
    }
  }
  )";
  writer << matrix_mul_vec_nn();
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
