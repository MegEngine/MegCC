#include "Fp32MatMul.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

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
    ss << "GI_kernel_gemv_";
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
    writer << "#include \"gi_float.h\"\n";
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
#define UNROLL_FUNC1(cb, a...) cb(0, ##a)
#define UNROLL_FUNC2(cb, a...) cb(0, ##a) cb(1, ##a)
#define UNROLL_FUNC3(cb, a...) UNROLL_FUNC2(cb, ##a) cb(2, ##a)
#define UNROLL_FUNC4(cb, a...) UNROLL_FUNC3(cb, ##a) cb(3, ##a)
#define UNROLL_FUNC5(cb, a...) UNROLL_FUNC4(cb, ##a) cb(4, ##a)
#define UNROLL_FUNC6(cb, a...) UNROLL_FUNC5(cb, ##a) cb(5, ##a)
#define UNROLL_FUNC7(cb, a...) UNROLL_FUNC6(cb, ##a) cb(6, ##a)
#define UNROLL_FUNC8(cb, a...) UNROLL_FUNC7(cb, ##a) cb(7, ##a)
#define UNROLL_FUNC(cb, i, a...) UNROLL_FUNC##i(cb, ##a)

#define INIT(i) GI_FLOAT32_t sum##i = GiBroadcastFloat32(0);
#define LOADA(i) GI_FLOAT32_t a##i = GiLoadFloat32(A + (m + i) * Astride + k);
#define CALADD(i) sum##i = GiMlaqFloat32(sum##i, a##i, b0);
#define CALOTHER(i) acc[i] += A[(m + i) * Astride + k] * D[n * Bstride + k];
#define CUMULATE(i) \
acc[i] += GiReduceAddFloat32(sum##i);

#define STORE(i) C[(m + i) * Cstride + n] = acc[i];

#define EXE_BLOCK(i)                                                           \
  float acc[i] = {0};                                                          \
  UNROLL_FUNC(INIT, i);                                                        \
  size_t k = 0;                                                                \
  for (; k + 4 <= K; k += 4) {                                                 \
    UNROLL_FUNC(LOADA, i);                                                     \
    GI_FLOAT32_t b0 = GiLoadFloat32(D + n * K + k);                                 \
    UNROLL_FUNC(CALADD, i);                                                    \
  }                                                                            \
  for (; k < K; k++) {                                                         \
    UNROLL_FUNC(CALOTHER, i);                                                  \
  }                                                                            \
  UNROLL_FUNC(CUMULATE, i);                                                    \
  UNROLL_FUNC(STORE, i);

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
  for(size_t n = 0;n < N;n++)
  {
    size_t m = 0;
    for (; m + 8 <= M; m += 8) {
      EXE_BLOCK(8);
    }
    if (m + 4 <= M) {
      EXE_BLOCK(4);
      m += 4;
    }
    if (m + 2 <= M) {
      EXE_BLOCK(2);
      m += 2;
    }
    if (m + 1 <= M) {
      EXE_BLOCK(1);
      m += 1;
    }
  }
#undef INIT
#undef LOADA
#undef CALADD
#undef CALOTHER
#undef STORE
#undef EXE_BLOCK
#undef UNROLL_FUNC1
#undef UNROLL_FUNC2
#undef UNROLL_FUNC3
#undef UNROLL_FUNC4
#undef UNROLL_FUNC5
#undef UNROLL_FUNC6
#undef UNROLL_FUNC7
#undef UNROLL_FUNC8
#undef UNROLL_FUNC
    )";
    writer << R"(
        return TinyNN_SUCCESS;
    })";
    // clang-format on
    return writer.str();
}

// vim: syntax=cpp.doxygen
