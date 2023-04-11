#include "../InternalKernel.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {

std::string GetKern8x1() {
    std::stringstream writer;
    writer << R"(
static void kern_8x1(
        const gi_float16_t* A, const gi_float16_t* B, size_t LDB, size_t K,
        gi_float16_t* C) {
    LDB = LDB - 8;
    K = K - 8;

    GI_FLOAT16_t d0 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d1 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d2 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d3 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d4 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d5 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d6 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d7 = GiLoadFloat16(A);
    A = A + 8;

    GI_FLOAT16_t vfzero = GiBroadcastFloat16(0.0);

    GI_FLOAT16_t d8 = MLA(vfzero, d0, *(B));
    d8 = MLA(d8, d1, *(B + 1));
    d8 = MLA(d8, d2, *(B + 2));
    d8 = MLA(d8, d3, *(B + 3));
    d8 = MLA(d8, d4, *(B + 4));
    d8 = MLA(d8, d5, *(B + 5));
    d8 = MLA(d8, d6, *(B + 6));
    d8 = MLA(d8, d7, *(B + 7));
    B += 8;

    B += LDB;

    for (; K > 0; K -= 8) {
        d0 = GiLoadFloat16(A);
        A = A + 8;
        d1 = GiLoadFloat16(A);
        A = A + 8;
        d2 = GiLoadFloat16(A);
        A = A + 8;
        d3 = GiLoadFloat16(A);
        A = A + 8;
        d4 = GiLoadFloat16(A);
        A = A + 8;
        d5 = GiLoadFloat16(A);
        A = A + 8;
        d6 = GiLoadFloat16(A);
        A = A + 8;
        d7 = GiLoadFloat16(A);
        A = A + 8;

        d8 = MLA(d8, d0, *(B));
        d8 = MLA(d8, d1, *(B + 1));
        d8 = MLA(d8, d2, *(B + 2));
        d8 = MLA(d8, d3, *(B + 3));
        d8 = MLA(d8, d4, *(B + 4));
        d8 = MLA(d8, d5, *(B + 5));
        d8 = MLA(d8, d6, *(B + 6));
        d8 = MLA(d8, d7, *(B + 7));
        B += 8;

        B += LDB;
    }

    GiStoreFloat16(C, d8);
}
)";
    return writer.str();
}
std::string GetKern8x4() {
    std::stringstream writer;
    writer << R"(
static void kern_8x4(
        const gi_float16_t* A, const gi_float16_t* B, size_t LDB, size_t K,
        gi_float16_t* C) {
    LDB = LDB - 32;
    K = K - 8;

    GI_FLOAT16_t d0 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d1 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d2 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d3 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d4 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d5 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d6 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d7 = GiLoadFloat16(A);
    A = A + 8;

    GI_FLOAT16_t vfzero = GiBroadcastFloat16(0.0);

    GI_FLOAT16_t d8 = MLA(vfzero, d0, *(B));
    d8 = MLA(d8, d1, *(B + 1));
    d8 = MLA(d8, d2, *(B + 2));
    d8 = MLA(d8, d3, *(B + 3));
    d8 = MLA(d8, d4, *(B + 4));
    d8 = MLA(d8, d5, *(B + 5));
    d8 = MLA(d8, d6, *(B + 6));
    d8 = MLA(d8, d7, *(B + 7));
    B += 8;

    GI_FLOAT16_t d9 = MLA(vfzero, d0, *(B));
    d9 = MLA(d9, d1, *(B + 1));
    d9 = MLA(d9, d2, *(B + 2));
    d9 = MLA(d9, d3, *(B + 3));
    d9 = MLA(d9, d4, *(B + 4));
    d9 = MLA(d9, d5, *(B + 5));
    d9 = MLA(d9, d6, *(B + 6));
    d9 = MLA(d9, d7, *(B + 7));
    B += 8;

    GI_FLOAT16_t d10 = MLA(vfzero, d0, *(B));
    d10 = MLA(d10, d1, *(B + 1));
    d10 = MLA(d10, d2, *(B + 2));
    d10 = MLA(d10, d3, *(B + 3));
    d10 = MLA(d10, d4, *(B + 4));
    d10 = MLA(d10, d5, *(B + 5));
    d10 = MLA(d10, d6, *(B + 6));
    d10 = MLA(d10, d7, *(B + 7));
    B += 8;

    GI_FLOAT16_t d11 = MLA(vfzero, d0, *(B));
    d11 = MLA(d11, d1, *(B + 1));
    d11 = MLA(d11, d2, *(B + 2));
    d11 = MLA(d11, d3, *(B + 3));
    d11 = MLA(d11, d4, *(B + 4));
    d11 = MLA(d11, d5, *(B + 5));
    d11 = MLA(d11, d6, *(B + 6));
    d11 = MLA(d11, d7, *(B + 7));
    B += 8;

    B += LDB;

    for (; K > 0; K -= 8) {
        d0 = GiLoadFloat16(A);
        A = A + 8;
        d1 = GiLoadFloat16(A);
        A = A + 8;
        d2 = GiLoadFloat16(A);
        A = A + 8;
        d3 = GiLoadFloat16(A);
        A = A + 8;
        d4 = GiLoadFloat16(A);
        A = A + 8;
        d5 = GiLoadFloat16(A);
        A = A + 8;
        d6 = GiLoadFloat16(A);
        A = A + 8;
        d7 = GiLoadFloat16(A);
        A = A + 8;

        d8 = MLA(d8, d0, *(B));
        d8 = MLA(d8, d1, *(B + 1));
        d8 = MLA(d8, d2, *(B + 2));
        d8 = MLA(d8, d3, *(B + 3));
        d8 = MLA(d8, d4, *(B + 4));
        d8 = MLA(d8, d5, *(B + 5));
        d8 = MLA(d8, d6, *(B + 6));
        d8 = MLA(d8, d7, *(B + 7));
        B += 8;

        d9 = MLA(d9, d0, *(B));
        d9 = MLA(d9, d1, *(B + 1));
        d9 = MLA(d9, d2, *(B + 2));
        d9 = MLA(d9, d3, *(B + 3));
        d9 = MLA(d9, d4, *(B + 4));
        d9 = MLA(d9, d5, *(B + 5));
        d9 = MLA(d9, d6, *(B + 6));
        d9 = MLA(d9, d7, *(B + 7));
        B += 8;

        d10 = MLA(d10, d0, *(B));
        d10 = MLA(d10, d1, *(B + 1));
        d10 = MLA(d10, d2, *(B + 2));
        d10 = MLA(d10, d3, *(B + 3));
        d10 = MLA(d10, d4, *(B + 4));
        d10 = MLA(d10, d5, *(B + 5));
        d10 = MLA(d10, d6, *(B + 6));
        d10 = MLA(d10, d7, *(B + 7));
        B += 8;

        d11 = MLA(d11, d0, *(B));
        d11 = MLA(d11, d1, *(B + 1));
        d11 = MLA(d11, d2, *(B + 2));
        d11 = MLA(d11, d3, *(B + 3));
        d11 = MLA(d11, d4, *(B + 4));
        d11 = MLA(d11, d5, *(B + 5));
        d11 = MLA(d11, d6, *(B + 6));
        d11 = MLA(d11, d7, *(B + 7));
        B += 8;

        B += LDB;
    }

    GiStoreFloat16(C, d8);
    C = C + 8;
    GiStoreFloat16(C, d9);
    C = C + 8;
    GiStoreFloat16(C, d10);
    C = C + 8;
    GiStoreFloat16(C, d11);
    C = C + 8;
}
)";
    return writer.str();
}
std::string GetKern8x8() {
    std::stringstream writer;
    writer << R"(
static void kern_8x8(
        const gi_float16_t* A, const gi_float16_t* B, size_t LDB, size_t K,
        gi_float16_t* C) {
    LDB -= 64;
    K = K - 8;

    GI_FLOAT16_t d0 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d1 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d2 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d3 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d4 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d5 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d6 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d7 = GiLoadFloat16(A);
    A = A + 8;

    GI_FLOAT16_t vfzero = GiZeroFloat16();

    GI_FLOAT16_t d8 = MLA(vfzero, d0, *(B));
    d8 = MLA(d8, d1, *(B + 1));
    d8 = MLA(d8, d2, *(B + 2));
    d8 = MLA(d8, d3, *(B + 3));
    d8 = MLA(d8, d4, *(B + 4));
    d8 = MLA(d8, d5, *(B + 5));
    d8 = MLA(d8, d6, *(B + 6));
    d8 = MLA(d8, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d9 = MLA(vfzero, d0, *(B));
    d9 = MLA(d9, d1, *(B + 1));
    d9 = MLA(d9, d2, *(B + 2));
    d9 = MLA(d9, d3, *(B + 3));
    d9 = MLA(d9, d4, *(B + 4));
    d9 = MLA(d9, d5, *(B + 5));
    d9 = MLA(d9, d6, *(B + 6));
    d9 = MLA(d9, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d10 = MLA(vfzero, d0, *(B));
    d10 = MLA(d10, d1, *(B + 1));
    d10 = MLA(d10, d2, *(B + 2));
    d10 = MLA(d10, d3, *(B + 3));
    d10 = MLA(d10, d4, *(B + 4));
    d10 = MLA(d10, d5, *(B + 5));
    d10 = MLA(d10, d6, *(B + 6));
    d10 = MLA(d10, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d11 = MLA(vfzero, d0, *(B));
    d11 = MLA(d11, d1, *(B + 1));
    d11 = MLA(d11, d2, *(B + 2));
    d11 = MLA(d11, d3, *(B + 3));
    d11 = MLA(d11, d4, *(B + 4));
    d11 = MLA(d11, d5, *(B + 5));
    d11 = MLA(d11, d6, *(B + 6));
    d11 = MLA(d11, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d12 = MLA(vfzero, d0, *(B));
    d12 = MLA(d12, d1, *(B + 1));
    d12 = MLA(d12, d2, *(B + 2));
    d12 = MLA(d12, d3, *(B + 3));
    d12 = MLA(d12, d4, *(B + 4));
    d12 = MLA(d12, d5, *(B + 5));
    d12 = MLA(d12, d6, *(B + 6));
    d12 = MLA(d12, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d13 = MLA(vfzero, d0, *(B));
    d13 = MLA(d13, d1, *(B + 1));
    d13 = MLA(d13, d2, *(B + 2));
    d13 = MLA(d13, d3, *(B + 3));
    d13 = MLA(d13, d4, *(B + 4));
    d13 = MLA(d13, d5, *(B + 5));
    d13 = MLA(d13, d6, *(B + 6));
    d13 = MLA(d13, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d14 = MLA(vfzero, d0, *(B));
    d14 = MLA(d14, d1, *(B + 1));
    d14 = MLA(d14, d2, *(B + 2));
    d14 = MLA(d14, d3, *(B + 3));
    d14 = MLA(d14, d4, *(B + 4));
    d14 = MLA(d14, d5, *(B + 5));
    d14 = MLA(d14, d6, *(B + 6));
    d14 = MLA(d14, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d15 = MLA(vfzero, d0, *(B));
    d15 = MLA(d15, d1, *(B + 1));
    d15 = MLA(d15, d2, *(B + 2));
    d15 = MLA(d15, d3, *(B + 3));
    d15 = MLA(d15, d4, *(B + 4));
    d15 = MLA(d15, d5, *(B + 5));
    d15 = MLA(d15, d6, *(B + 6));
    d15 = MLA(d15, d7, *(B + 7));
    B = B + 8;

    B = B + LDB;
    for (; K > 0; K -= 8) {
        d0 = GiLoadFloat16(A);
        A = A + 8;
        d1 = GiLoadFloat16(A);
        A = A + 8;
        d2 = GiLoadFloat16(A);
        A = A + 8;
        d3 = GiLoadFloat16(A);
        A = A + 8;
        d4 = GiLoadFloat16(A);
        A = A + 8;
        d5 = GiLoadFloat16(A);
        A = A + 8;
        d6 = GiLoadFloat16(A);
        A = A + 8;
        d7 = GiLoadFloat16(A);
        A = A + 8;

        d8 = MLA(d8, d0, *(B));
        d8 = MLA(d8, d1, *(B + 1));
        d8 = MLA(d8, d2, *(B + 2));
        d8 = MLA(d8, d3, *(B + 3));
        d8 = MLA(d8, d4, *(B + 4));
        d8 = MLA(d8, d5, *(B + 5));
        d8 = MLA(d8, d6, *(B + 6));
        d8 = MLA(d8, d7, *(B + 7));
        B = B + 8;

        d9 = MLA(d9, d0, *(B));
        d9 = MLA(d9, d1, *(B + 1));
        d9 = MLA(d9, d2, *(B + 2));
        d9 = MLA(d9, d3, *(B + 3));
        d9 = MLA(d9, d4, *(B + 4));
        d9 = MLA(d9, d5, *(B + 5));
        d9 = MLA(d9, d6, *(B + 6));
        d9 = MLA(d9, d7, *(B + 7));
        B = B + 8;

        d10 = MLA(d10, d0, *(B));
        d10 = MLA(d10, d1, *(B + 1));
        d10 = MLA(d10, d2, *(B + 2));
        d10 = MLA(d10, d3, *(B + 3));
        d10 = MLA(d10, d4, *(B + 4));
        d10 = MLA(d10, d5, *(B + 5));
        d10 = MLA(d10, d6, *(B + 6));
        d10 = MLA(d10, d7, *(B + 7));
        B = B + 8;

        d11 = MLA(d11, d0, *(B));
        d11 = MLA(d11, d1, *(B + 1));
        d11 = MLA(d11, d2, *(B + 2));
        d11 = MLA(d11, d3, *(B + 3));
        d11 = MLA(d11, d4, *(B + 4));
        d11 = MLA(d11, d5, *(B + 5));
        d11 = MLA(d11, d6, *(B + 6));
        d11 = MLA(d11, d7, *(B + 7));
        B = B + 8;

        d12 = MLA(d12, d0, *(B));
        d12 = MLA(d12, d1, *(B + 1));
        d12 = MLA(d12, d2, *(B + 2));
        d12 = MLA(d12, d3, *(B + 3));
        d12 = MLA(d12, d4, *(B + 4));
        d12 = MLA(d12, d5, *(B + 5));
        d12 = MLA(d12, d6, *(B + 6));
        d12 = MLA(d12, d7, *(B + 7));
        B = B + 8;

        d13 = MLA(d13, d0, *(B));
        d13 = MLA(d13, d1, *(B + 1));
        d13 = MLA(d13, d2, *(B + 2));
        d13 = MLA(d13, d3, *(B + 3));
        d13 = MLA(d13, d4, *(B + 4));
        d13 = MLA(d13, d5, *(B + 5));
        d13 = MLA(d13, d6, *(B + 6));
        d13 = MLA(d13, d7, *(B + 7));
        B = B + 8;

        d14 = MLA(d14, d0, *(B));
        d14 = MLA(d14, d1, *(B + 1));
        d14 = MLA(d14, d2, *(B + 2));
        d14 = MLA(d14, d3, *(B + 3));
        d14 = MLA(d14, d4, *(B + 4));
        d14 = MLA(d14, d5, *(B + 5));
        d14 = MLA(d14, d6, *(B + 6));
        d14 = MLA(d14, d7, *(B + 7));
        B = B + 8;

        d15 = MLA(d15, d0, *(B));
        d15 = MLA(d15, d1, *(B + 1));
        d15 = MLA(d15, d2, *(B + 2));
        d15 = MLA(d15, d3, *(B + 3));
        d15 = MLA(d15, d4, *(B + 4));
        d15 = MLA(d15, d5, *(B + 5));
        d15 = MLA(d15, d6, *(B + 6));
        d15 = MLA(d15, d7, *(B + 7));
        B = B + 8 + LDB;
    }
    GiStoreFloat16(C, d8);
    C = C + 8;
    GiStoreFloat16(C, d9);
    C = C + 8;
    GiStoreFloat16(C, d10);
    C = C + 8;
    GiStoreFloat16(C, d11);
    C = C + 8;
    GiStoreFloat16(C, d12);
    C = C + 8;
    GiStoreFloat16(C, d13);
    C = C + 8;
    GiStoreFloat16(C, d14);
    C = C + 8;
    GiStoreFloat16(C, d15);
    C = C + 8;
}
)";
    return writer.str();
}
}  // namespace

std::string Fp16MatmulM8N8MK8Kernel::GetKernelSymbol(TContext*) const {
    return "GI_fp16_m8_n8_mk8_matmul";
}

std::string Fp16MatmulM8N8MK8Kernel::GetKernelSignature(TContext* ctx) const {
    std::stringstream writer;
    writer << "void " << GetKernelSymbol(ctx) << R"((const gi_float16_t* A, size_t LDA,
                            const gi_float16_t* B, size_t LDB, gi_float16_t* C,
                            size_t LDC, size_t M, size_t N, size_t K))";
    return writer.str();
}

std::string Fp16MatmulM8N8MK8Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include \"gi_float16.h\"\n";
    writer << "#include \"stddef.h\"\n";
    writer << "#define MLA GiMultiplyAddScalarFloat16\n";

    writer << GetKern8x1();
    writer << "\n\n";
    writer << GetKern8x4();
    writer << "\n\n";
    writer << GetKern8x8();
    writer << "\n\n";

    writer << GetKernelSignature(ctx);
    writer << "{\n";
    writer << R"(
    const int MB=8;
    const int KB=8;
    const int NB=8;
    const int NB_HALF=4;
    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)

    for (size_t m = 0; m < M; m += MB) {
        gi_float16_t* output = C + (m / MB) * LDC;
        const gi_float16_t* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_8x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_8x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB_HALF;
            output += MB * NB_HALF;
            n += 4;
        }
        while (n < N) {
            kern_8x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
    )";
    writer << "\n}";
    writer << "\n#undef MLA\n";
    return writer.str();
}

// vim: syntax=cpp.doxygen
