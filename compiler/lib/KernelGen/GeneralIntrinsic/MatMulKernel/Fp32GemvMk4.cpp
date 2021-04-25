/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/MatMulKernel/Fp32GemvMk4.cpp
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

bool Fp32GemvMk4Kernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" &&
                    context->getAttrOprand("operand:1").dtype == "f32" &&
                    context->getAttrOprand("operand:2").dtype == "f32";
    bool ok_mode = context->getAttrStr("format") == "MK4";

    bool ok_layout = context->getAttrOprand("operand:0").shape.size() == 4 &&
                     context->getAttrOprand("operand:1").shape.size() == 3 &&
                     context->getAttrOprand("operand:0").shape[3] == 4 &&
                     context->getAttrOprand("operand:0").shape[2] == 4 &&
                     context->getAttrOprand("operand:1").shape[1] == 1 &&
                     context->getAttrOprand("operand:1").shape[2] == 4;

    bool ok_tran = context->getAttrBool("transposeA") == false &&
                   context->getAttrBool("transposeB") == false;
    return ok_dtype && ok_mode && ok_tran && ok_layout;
}
//! kernel gen
std::string Fp32GemvMk4Kernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "GI_kernel_gemv_MK4_nn";
    return ss.str();
}

std::string Fp32GemvMk4Kernel::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        *workspace = 0;
        return TinyNN_SUCCESS;
    })";
    ss << workspace_temp;
    return ss.str();
}

std::string Fp32GemvMk4Kernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << R"(
#include "gi_float.h"
#include "unroll_macro.h"
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
    const int Cstride = c_layout.stride[0];
    const int Bstride = b_layout.stride[0];
    const int M = c_layout.dims[0]*4;
    const int K = a_layout.dims[1]*4;
    const int N = c_layout.dims[1];

    TINYNN_ASSERT(4 == a_layout.dims[3]);
    TINYNN_ASSERT(4 == a_layout.dims[2]);
    TINYNN_ASSERT(4 == b_layout.dims[2]);
    TINYNN_ASSERT(4 == c_layout.dims[2]);
    TINYNN_ASSERT(1 == N);

    TINYNN_ASSERT(a_layout.dims[0] == c_layout.dims[0]);
    TINYNN_ASSERT(a_layout.dims[1] == b_layout.dims[0]);
    TINYNN_ASSERT(b_layout.dims[1] == b_layout.dims[1]);

    size_t PACK_SIZE = 4;
    TINYNN_ASSERT(
            N == 1 && Bstride == PACK_SIZE && M % PACK_SIZE == 0 && K % PACK_SIZE == 0);
    float* Aptr = A;
    float* Cptr = C;
    size_t m = 0;
    while (m < M) {
        float* Aptr0 = Aptr;
        float* Cptr0 = Cptr;
        GI_FLOAT32_V4_t c;
#define INIT(step) GiSetSubVectorFloat32V4(c, step, GiBroadcastFloat32(0.0f));
        UNROLL_CALL_RAW(4, INIT)
#undef INIT
        float* Bptr = B;
        size_t k = 0;
        while (k < K) {
            GI_FLOAT32_t b = GiLoadFloat32(Bptr);
            GI_FLOAT32_V4_t a;
#define LOAD_A(step) GiSetSubVectorFloat32V4(a, step, GiLoadFloat32(Aptr0 + step * 4));
            UNROLL_CALL_RAW(4, LOAD_A)
#undef LOAD_A
#define COMPT(step)                                                                \
    t = GiSimdFmaLane(                                                           \
            GiGetSubVectorFloat32V4(c, step), GiGetSubVectorFloat32V4(a, step), b, \
            step);                                                                 \
        GiSetSubVectorFloat32V4(c, step, t);

            GI_FLOAT32_t t;
            UNROLL_CALL_RAW(4, COMPT)


#undef COMPT
            Bptr += Bstride;
            Aptr0 += PACK_SIZE * PACK_SIZE;
            k += PACK_SIZE;
        }

#define ADD_C(step, stride)                             \
    t = GiAddFloat32(                                   \
            GiGetSubVectorFloat32V4(c, step),           \
            GiGetSubVectorFloat32V4(c, step + stride)); \
    GiSetSubVectorFloat32V4(c, step, t);
        GI_FLOAT32_t t;
        UNROLL_CALL_RAW(2, ADD_C, 2)
        UNROLL_CALL_RAW(1, ADD_C, 1)
#undef ADD_C
        GiStoreFloat32(Cptr0, GiGetSubVectorFloat32V4(c, 0));

        Aptr += Astride;
        Cptr += Cstride;
        m += PACK_SIZE;
    }
    return TinyNN_SUCCESS;
}
    )";
    // clang-format on
    return writer.str();
}

// vim: syntax=cpp.doxygen
