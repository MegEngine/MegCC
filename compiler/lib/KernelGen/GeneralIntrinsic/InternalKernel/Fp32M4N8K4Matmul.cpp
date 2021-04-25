/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/InternalKernel/Fp32M4N8K4Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "InternalKernel.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {

std::string GetKern4x1() {
    std::stringstream writer;
    writer << R"(
static void kern_4x1(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB = LDB - 4;
    K = K - 4;

    GI_FLOAT32_t d8d9 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d12d13 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d16d17 = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d18d19 = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d20d21 = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d22d23 = GiBroadcastFloat32(0.0f);

    GI_FLOAT32_t d0d1 = GiLoadFloat32(B);
    B = B + 4;

    d16d17 = GiSimdFmaLane(d16d17, d8d9, d0d1, 0);
    d18d19 = GiSimdFmaLane(d18d19, d10d11, d0d1, 1);

    for (; K > 0; K -= 4) {
        d8d9 = GiLoadFloat32(A);
        A = A + 4;
        d10d11 = GiLoadFloat32(A);
        A = A + 4;
        d20d21 = GiSimdFmaLane(d20d21, d12d13, d0d1, 2);
        d22d23 = GiSimdFmaLane(d22d23, d14d15, d0d1, 3);

        B = B + LDB;
        d0d1 = GiLoadFloat32(B);
        B = B + 4;
        d12d13 = GiLoadFloat32(A);
        A = A + 4;
        d14d15 = GiLoadFloat32(A);
        A = A + 4;

        d16d17 = GiSimdFmaLane(d16d17, d8d9, d0d1, 0);
        d18d19 = GiSimdFmaLane(d18d19, d10d11, d0d1, 1);
    }

    d20d21 = GiSimdFmaLane(d20d21, d12d13, d0d1, 2);
    d22d23 = GiSimdFmaLane(d22d23, d14d15, d0d1, 3);
    d16d17 = GiAddFloat32(d16d17, d20d21);
    d18d19 = GiAddFloat32(d18d19, d22d23);
    d16d17 = GiAddFloat32(d16d17, d18d19);

    GiStoreFloat32(C, d16d17);
    C = C + 4;
})";
    return writer.str();
}
std::string GetKern4x4() {
    std::stringstream writer;
    writer << R"(
static void kern_4x4(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB = (LDB - 16);
    K = K - 4;

    GI_FLOAT32_t d8d9 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d12d13 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(A);
    A = A + 4;

    GI_FLOAT32_t d0d1 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d2d3 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d4d5 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d6d7 = GiLoadFloat32(B);
    B = B + 4;

    GI_FLOAT32_t vfzero = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d16d17 = GiSimdFmaLane(vfzero, d8d9, d0d1, 0);
    GI_FLOAT32_t d18d19 = GiSimdFmaLane(vfzero, d8d9, d2d3, 0);
    GI_FLOAT32_t d20d21 = GiSimdFmaLane(vfzero, d8d9, d4d5, 0);
    GI_FLOAT32_t d22d23 = GiSimdFmaLane(vfzero, d8d9, d6d7, 0);

    d16d17 = GiSimdFmaLane(d16d17, d10d11, d0d1, 1);
    d18d19 = GiSimdFmaLane(d18d19, d10d11, d2d3, 1);
    d20d21 = GiSimdFmaLane(d20d21, d10d11, d4d5, 1);
    d22d23 = GiSimdFmaLane(d22d23, d10d11, d6d7, 1);

    for (; K > 0; K -= 4) {
        d8d9 = GiLoadFloat32(A);
        A = A + 4;
        d10d11 = GiLoadFloat32(A);
        A = A + 4;

        d16d17 = GiSimdFmaLane(d16d17, d12d13, d0d1, 2);
        d18d19 = GiSimdFmaLane(d18d19, d12d13, d2d3, 2);
        d20d21 = GiSimdFmaLane(d20d21, d12d13, d4d5, 2);
        d22d23 = GiSimdFmaLane(d22d23, d12d13, d6d7, 2);

        B = B + LDB;

        d16d17 = GiSimdFmaLane(d16d17, d14d15, d0d1, 3);
        d18d19 = GiSimdFmaLane(d18d19, d14d15, d2d3, 3);
        d0d1 = GiLoadFloat32(B);
        B = B + 4;
        d20d21 = GiSimdFmaLane(d20d21, d14d15, d4d5, 3);
        d2d3 = GiLoadFloat32(B);
        B = B + 4;
        d22d23 = GiSimdFmaLane(d22d23, d14d15, d6d7, 3);
        d4d5 = GiLoadFloat32(B);
        B = B + 4;

        d16d17 = GiSimdFmaLane(d16d17, d8d9, d0d1, 0);
        d6d7 = GiLoadFloat32(B);
        B = B + 4;
        d18d19 = GiSimdFmaLane(d18d19, d8d9, d2d3, 0);
        d20d21 = GiSimdFmaLane(d20d21, d8d9, d4d5, 0);
        d22d23 = GiSimdFmaLane(d22d23, d8d9, d6d7, 0);

        d12d13 = GiLoadFloat32(A);
        A = A + 4;
        d14d15 = GiLoadFloat32(A);
        A = A + 4;

        d16d17 = GiSimdFmaLane(d16d17, d10d11, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d10d11, d2d3, 1);
        d20d21 = GiSimdFmaLane(d20d21, d10d11, d4d5, 1);
        d22d23 = GiSimdFmaLane(d22d23, d10d11, d6d7, 1);
    }

    d16d17 = GiSimdFmaLane(d16d17, d12d13, d0d1, 2);
    d18d19 = GiSimdFmaLane(d18d19, d12d13, d2d3, 2);
    d20d21 = GiSimdFmaLane(d20d21, d12d13, d4d5, 2);
    d22d23 = GiSimdFmaLane(d22d23, d12d13, d6d7, 2);

    d16d17 = GiSimdFmaLane(d16d17, d14d15, d0d1, 3);
    d18d19 = GiSimdFmaLane(d18d19, d14d15, d2d3, 3);
    d20d21 = GiSimdFmaLane(d20d21, d14d15, d4d5, 3);
    d22d23 = GiSimdFmaLane(d22d23, d14d15, d6d7, 3);

    GiStoreFloat32(C, d16d17);
    C = C + 4;
    GiStoreFloat32(C, d18d19);
    C = C + 4;
    GiStoreFloat32(C, d20d21);
    C = C + 4;
    GiStoreFloat32(C, d22d23);
    C = C + 4;
})";
    return writer.str();
}
std::string GetKern4x8() {
    std::stringstream writer;
    writer << R"(
static void kern_4x8(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB -= 32;
    GI_FLOAT32_t d8d9 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d12d13 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(A);
    A = A + 4;

    GI_FLOAT32_t d0d1 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d2d3 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d4d5 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d6d7 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t vfzero = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d16d17 = GiSimdFmaLane(vfzero, d8d9, d0d1, 0);
    d16d17 = GiSimdFmaLane(d16d17, d10d11, d0d1, 1);
    GI_FLOAT32_t d18d19 = GiSimdFmaLane(vfzero, d8d9, d2d3, 0);
    d16d17 = GiSimdFmaLane(d16d17, d12d13, d0d1, 2);
    d18d19 = GiSimdFmaLane(d18d19, d10d11, d2d3, 1);
    d16d17 = GiSimdFmaLane(d16d17, d14d15, d0d1, 3);
    d18d19 = GiSimdFmaLane(d18d19, d12d13, d2d3, 2);
    d18d19 = GiSimdFmaLane(d18d19, d14d15, d2d3, 3);
    d0d1 = GiLoadFloat32(B);
    B = B + 4;
    d2d3 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d20d21 = GiSimdFmaLane(vfzero, d8d9, d4d5, 0);
    d20d21 = GiSimdFmaLane(d20d21, d10d11, d4d5, 1);
    GI_FLOAT32_t d22d23 = GiSimdFmaLane(vfzero, d8d9, d6d7, 0);
    d20d21 = GiSimdFmaLane(d20d21, d12d13, d4d5, 2);
    d22d23 = GiSimdFmaLane(d22d23, d10d11, d6d7, 1);
    d20d21 = GiSimdFmaLane(d20d21, d14d15, d4d5, 3);
    d22d23 = GiSimdFmaLane(d22d23, d12d13, d6d7, 2);
    d22d23 = GiSimdFmaLane(d22d23, d14d15, d6d7, 3);

    d4d5 = GiLoadFloat32(B);
    B = B + 4;
    d6d7 = GiLoadFloat32(B);
    B = B + 4;
    GI_FLOAT32_t d24d25 = GiSimdFmaLane(vfzero, d8d9, d0d1, 0);
    d24d25 = GiSimdFmaLane(d24d25, d10d11, d0d1, 1);
    GI_FLOAT32_t d26d27 = GiSimdFmaLane(vfzero, d8d9, d2d3, 0);
    d24d25 = GiSimdFmaLane(d24d25, d12d13, d0d1, 2);
    d26d27 = GiSimdFmaLane(d26d27, d10d11, d2d3, 1);
    d24d25 = GiSimdFmaLane(d24d25, d14d15, d0d1, 3);
    d26d27 = GiSimdFmaLane(d26d27, d12d13, d2d3, 2);
    d26d27 = GiSimdFmaLane(d26d27, d14d15, d2d3, 3);
    GI_FLOAT32_t d28d29 = GiSimdFmaLane(vfzero, d8d9, d4d5, 0);
    d28d29 = GiSimdFmaLane(d28d29, d10d11, d4d5, 1);
    GI_FLOAT32_t d30d31 = GiSimdFmaLane(vfzero, d8d9, d6d7, 0);
    d28d29 = GiSimdFmaLane(d28d29, d12d13, d4d5, 2);
    d30d31 = GiSimdFmaLane(d30d31, d10d11, d6d7, 1);
    d28d29 = GiSimdFmaLane(d28d29, d14d15, d4d5, 3);
    d30d31 = GiSimdFmaLane(d30d31, d12d13, d6d7, 2);
    d30d31 = GiSimdFmaLane(d30d31, d14d15, d6d7, 3);

    B = B + LDB;
    K = K - 4;
    for (; K > 0; K -= 4) {
        d8d9 = GiLoadFloat32(A);
        A = A + 4;
        d10d11 = GiLoadFloat32(A);
        A = A + 4;
        d12d13 = GiLoadFloat32(A);
        A = A + 4;
        d14d15 = GiLoadFloat32(A);
        A = A + 4;

        d0d1 = GiLoadFloat32(B);
        B = B + 4;
        d2d3 = GiLoadFloat32(B);
        B = B + 4;
        d4d5 = GiLoadFloat32(B);
        B = B + 4;
        d6d7 = GiLoadFloat32(B);
        B = B + 4;
        d16d17 = GiSimdFmaLane(d16d17, d8d9, d0d1, 0);
        d16d17 = GiSimdFmaLane(d16d17, d10d11, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d8d9, d2d3, 0);
        d16d17 = GiSimdFmaLane(d16d17, d12d13, d0d1, 2);
        d18d19 = GiSimdFmaLane(d18d19, d10d11, d2d3, 1);
        d16d17 = GiSimdFmaLane(d16d17, d14d15, d0d1, 3);
        d18d19 = GiSimdFmaLane(d18d19, d12d13, d2d3, 2);
        d18d19 = GiSimdFmaLane(d18d19, d14d15, d2d3, 3);
        d0d1 = GiLoadFloat32(B);
        B = B + 4;
        d2d3 = GiLoadFloat32(B);
        B = B + 4;
        d20d21 = GiSimdFmaLane(d20d21, d8d9, d4d5, 0);
        d20d21 = GiSimdFmaLane(d20d21, d10d11, d4d5, 1);
        d22d23 = GiSimdFmaLane(d22d23, d8d9, d6d7, 0);
        d20d21 = GiSimdFmaLane(d20d21, d12d13, d4d5, 2);
        d22d23 = GiSimdFmaLane(d22d23, d10d11, d6d7, 1);
        d20d21 = GiSimdFmaLane(d20d21, d14d15, d4d5, 3);
        d22d23 = GiSimdFmaLane(d22d23, d12d13, d6d7, 2);
        d22d23 = GiSimdFmaLane(d22d23, d14d15, d6d7, 3);

        d4d5 = GiLoadFloat32(B);
        B = B + 4;
        d6d7 = GiLoadFloat32(B);
        B = B + 4;
        d24d25 = GiSimdFmaLane(d24d25, d8d9, d0d1, 0);
        d24d25 = GiSimdFmaLane(d24d25, d10d11, d0d1, 1);
        d26d27 = GiSimdFmaLane(d26d27, d8d9, d2d3, 0);
        d24d25 = GiSimdFmaLane(d24d25, d12d13, d0d1, 2);
        d26d27 = GiSimdFmaLane(d26d27, d10d11, d2d3, 1);
        d24d25 = GiSimdFmaLane(d24d25, d14d15, d0d1, 3);
        d26d27 = GiSimdFmaLane(d26d27, d12d13, d2d3, 2);
        d26d27 = GiSimdFmaLane(d26d27, d14d15, d2d3, 3);
        d28d29 = GiSimdFmaLane(d28d29, d8d9, d4d5, 0);
        d28d29 = GiSimdFmaLane(d28d29, d10d11, d4d5, 1);
        d30d31 = GiSimdFmaLane(d30d31, d8d9, d6d7, 0);
        d28d29 = GiSimdFmaLane(d28d29, d12d13, d4d5, 2);
        d30d31 = GiSimdFmaLane(d30d31, d10d11, d6d7, 1);
        d28d29 = GiSimdFmaLane(d28d29, d14d15, d4d5, 3);
        d30d31 = GiSimdFmaLane(d30d31, d12d13, d6d7, 2);
        d30d31 = GiSimdFmaLane(d30d31, d14d15, d6d7, 3);
        B = B + LDB;
    }
    GiStoreFloat32(C, d16d17);
    C = C + 4;
    GiStoreFloat32(C, d18d19);
    C = C + 4;
    GiStoreFloat32(C, d20d21);
    C = C + 4;
    GiStoreFloat32(C, d22d23);
    C = C + 4;
    GiStoreFloat32(C, d24d25);
    C = C + 4;
    GiStoreFloat32(C, d26d27);
    C = C + 4;
    GiStoreFloat32(C, d28d29);
    C = C + 4;
    GiStoreFloat32(C, d30d31);
    C = C + 4;
})";
    return writer.str();
}
}  // namespace

std::string MatmulM4N8MK4Kernel::GetKernelSymbol(TContext*) const {
    return "GI_fp32_m4_n8_mk4_matmul";
}

std::string MatmulM4N8MK4Kernel::GetKernelSignature(TContext* ctx) const {
    std::stringstream writer;
    writer << "void " << GetKernelSymbol(ctx) << R"((const float* A, size_t LDA,
                            const float* B, size_t LDB, float* C,
                            size_t LDC, size_t M, size_t N, size_t K))";
    return writer.str();
}

std::string MatmulM4N8MK4Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include \"gi_float.h\"\n";
    writer << "#include \"stddef.h\"\n";

    writer << GetKern4x1();
    writer << "\n\n";
    writer << GetKern4x4();
    writer << "\n\n";
    writer << GetKern4x8();
    writer << "\n\n";

    writer << GetKernelSignature(ctx);
    writer << "{\n";
    writer << R"(
     const int MB=4;
    const int KB=4;
    const int NB=8;
    const int NB_HALF=4;
    //! (m/4, k/4, 4, 4) * (k/4, n, 4) = (m/4, n, 4)
    for (size_t m = 0; m < M; m += MB) {
        float* output = C + (m / MB) * LDC;
        const float* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_4x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_4x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB_HALF;
            output += MB * NB_HALF;
            n += 4;
        }
        while (n < N) {
            kern_4x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    })";
    writer << "\n}";
    return writer.str();
}

// vim: syntax=cpp.doxygen
