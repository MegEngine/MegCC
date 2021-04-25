/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/MatMulKernel/Fp32Gevm.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Fp32MatMul.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
std::string common_define(void) {
    return R"(
#define UNROLL_FUNC1(cb, a...) cb(0, ##a)
#define UNROLL_FUNC2(cb, a...) cb(0, ##a) cb(1, ##a)
#define UNROLL_FUNC3(cb, a...) UNROLL_FUNC2(cb, ##a) cb(2, ##a)
#define UNROLL_FUNC4(cb, a...) UNROLL_FUNC3(cb, ##a) cb(3, ##a)
#define UNROLL_FUNC5(cb, a...) UNROLL_FUNC4(cb, ##a) cb(4, ##a)
#define UNROLL_FUNC6(cb, a...) UNROLL_FUNC5(cb, ##a) cb(5, ##a)
#define UNROLL_FUNC7(cb, a...) UNROLL_FUNC6(cb, ##a) cb(6, ##a)
#define UNROLL_FUNC8(cb, a...) UNROLL_FUNC7(cb, ##a) cb(7, ##a)
#define UNROLL_FUNC9(cb, a...) UNROLL_FUNC8(cb, ##a) cb(8, ##a)
#define UNROLL_FUNC10(cb, a...) UNROLL_FUNC9(cb, ##a) cb(9, ##a)
#define UNROLL_FUNC11(cb, a...) UNROLL_FUNC10(cb, ##a) cb(10, ##a)
#define UNROLL_FUNC12(cb, a...) UNROLL_FUNC11(cb, ##a) cb(11, ##a)
#define UNROLL_FUNC13(cb, a...) UNROLL_FUNC12(cb, ##a) cb(12, ##a)
#define UNROLL_FUNC14(cb, a...) UNROLL_FUNC13(cb, ##a) cb(13, ##a)
#define UNROLL_FUNC15(cb, a...) UNROLL_FUNC14(cb, ##a) cb(14, ##a)
#define UNROLL_FUNC16(cb, a...) UNROLL_FUNC15(cb, ##a) cb(15, ##a) 
#define UNROLL_FUNC(cb,i,a...) UNROLL_FUNC##i(cb, ##a) 
    )";
}

std::string common_undef(void) {
    return R"(
#undef UNROLL_FUNC1
#undef UNROLL_FUNC2
#undef UNROLL_FUNC3
#undef UNROLL_FUNC4
#undef UNROLL_FUNC5
#undef UNROLL_FUNC6
#undef UNROLL_FUNC7
#undef UNROLL_FUNC8
#undef UNROLL_FUNC9
#undef UNROLL_FUNC10
#undef UNROLL_FUNC11
#undef UNROLL_FUNC12
#undef UNROLL_FUNC13
#undef UNROLL_FUNC14
#undef UNROLL_FUNC15
#undef UNROLL_FUNC16
#undef UNROLL_FUNC
    )";
}
std::string vec_mul_matrix_n_n(void) {
    return R"(
    
#define LOADB(i) GI_FLOAT32_t b##i = GiLoadFloat32(B + k * Bstride + n+4*i);
#define LOADC(i) GI_FLOAT32_t c##i = GiLoadFloat32(C + m * Cstride + n+4*i);
#define CALADD(i) c##i = GiMlaqFloat32(c##i,a0,b##i);
#define STORE(i) GiStoreFloat32(C + m * Cstride + n + 4*i,c##i);
#define EXE_BLOCK(i)\
UNROLL_FUNC(LOADB,i)\
UNROLL_FUNC(LOADC,i)\
UNROLL_FUNC(CALADD,i)\
UNROLL_FUNC(STORE,i)
    memset(C, 0, sizeof(float) * M * N);
    for (size_t m = 0; m < M; m++) {
      for (size_t k = 0; k < K; k++) {
        size_t n = 0;
        GI_FLOAT32_t a0 = GiBroadcastFloat32(A[m * Astride + k]);
        for(;n+64<=N;n+=64){
            EXE_BLOCK(16)
        }
        if(n+32<=N)
        {
            EXE_BLOCK(8)
            n +=32;
        }
        if(n+16<=N)
        {
            EXE_BLOCK(4)
            n +=16;
        }
        for (; n < N; n++) {
            C[m * Cstride + n] += A[m * Astride + k] * B[k * Bstride + n];
        }
      }
    }
#undef LOADB
#undef LOADC
#undef CALADD
#undef STORE
#undef EXE_BLOCK
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
    ss << "GI_kernel_gevm_";
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
    bool trans_b = context->getAttrBool("transposeB");
    writer << R"(
        #include <string.h>
        #include "gi_float.h"
    )";
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
    writer << common_define();
    if(trans_b)
        writer << vec_mul_matrix_n_t();
    else 
        writer << vec_mul_matrix_n_n();
    writer << common_undef();

    writer << R"(
        return TinyNN_SUCCESS;
    })";
    // clang-format on
    return writer.str();
}

// vim: syntax=cpp.doxygen
