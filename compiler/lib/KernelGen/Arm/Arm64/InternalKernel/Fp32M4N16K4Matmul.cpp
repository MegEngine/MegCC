#include "Arm/Arm64/Activation.h"
#include "InternalKernel.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;
namespace {

std::string GetKern4x1() {
    std::stringstream writer;
    writer << R"(
static void kern_4x1(const float* a_ptr, const float* b_ptr, size_t LDB, size_t K,
              float* output) {
    LDB *= sizeof(float);
    asm volatile(
            "subs %w[K], %w[K], #4\n"
            "ld1 {v4.4s, v5.4s}, [%[a_ptr]], 32\n"
            "eor v16.16b, v16.16b, v16.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "ld1 {v6.4s, v7.4s}, [%[a_ptr]], 32\n"
            "eor v18.16b, v18.16b, v18.16b\n"
            "eor v19.16b, v19.16b, v19.16b\n"
            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"
            "prfm pstl1keep, [%[b_ptr]]\n"

            "fmla v16.4s, v4.4s, v0.s[0]\n"
            "fmla v17.4s, v5.4s, v0.s[1]\n"

            "beq 2f\n"

            "1:\n"
            "ld1 {v4.4s, v5.4s}, [%[a_ptr]], 32\n"
            "fmla v18.4s, v6.4s, v0.s[2]\n"
            "fmla v19.4s, v7.4s, v0.s[3]\n"
            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"
            "prfm pstl1keep, [%[b_ptr]]\n"
            "ld1 {v6.4s, v7.4s}, [%[a_ptr]], 32\n"
            "fmla v16.4s, v4.4s, v0.s[0]\n"
            "fmla v17.4s, v5.4s, v0.s[1]\n"

            "subs %w[K], %w[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "fmla v18.4s, v6.4s, v0.s[2]\n"
            "fmla v19.4s, v7.4s, v0.s[3]\n"
            "fadd v16.4s, v16.4s, v18.4s\n"
            "fadd v17.4s, v17.4s, v19.4s\n"
            "fadd v16.4s, v16.4s, v17.4s\n"

            "st1 {v16.4s}, [%[output]], 16\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "cc",
              "memory");
})";
    return writer.str();
}
std::string GetKern4x4() {
    std::stringstream writer;
    writer << R"(
// Overview of register layout:
//
// A 4x4 block of A is stored in register v4-v7
// A 4x4 block of B is stored in register v0-v3
// A 8x4 block of accumulators store in v16-v19
//
//                    A +--------+
//                      | v4[0-3]|
//                      | v5[0-3]|
//                      | v6[0-3]|
//                      | v7[0-3]|
//                      +--------+
//      B
//  +--------+ - - - - -+--------+
//  | v0[0-3]|          |v16[0-3]|
//  | v1[0-3]|          |v17[0-3]|
//  | v2[0-3]|          |v18[0-3]|
//  | v3[0-3]|          |v19[0-3]|
//  +--------+ - - - - -+--------+
//                      Accumulator

static void kern_4x4(const float* a_ptr, const float* b_ptr, size_t LDB, size_t K,
              float* output) {
    //! As each load 16 number from B, but the pos add 12 * 4, so we minus 12
    //! here.
    LDB = (LDB - 12) * sizeof(float);
    asm volatile(
            "subs %w[K], %w[K], #4\n"
            "ld1 {v4.4s, v5.4s}, [%[a_ptr]], 32\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"

            "fmul v16.4s, v4.4s, v0.s[0]\n"
            "fmul v17.4s, v4.4s, v1.s[0]\n"
            "fmul v18.4s, v4.4s, v2.s[0]\n"
            "fmul v19.4s, v4.4s, v3.s[0]\n"
            "ld1 {v6.4s, v7.4s}, [%[a_ptr]], 32\n"

            "fmla v16.4s, v5.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v1.s[1]\n"
            "fmla v18.4s, v5.4s, v2.s[1]\n"
            "fmla v19.4s, v5.4s, v3.s[1]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v4.4s, v5.4s}, [%[a_ptr]], 32\n"

            "fmla v16.4s, v6.4s, v0.s[2]\n"
            "fmla v17.4s, v6.4s, v1.s[2]\n"
            "fmla v18.4s, v6.4s, v2.s[2]\n"
            "fmla v19.4s, v6.4s, v3.s[2]\n"

            "fmla v16.4s, v7.4s, v0.s[3]\n"
            "fmla v17.4s, v7.4s, v1.s[3]\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "fmla v18.4s, v7.4s, v2.s[3]\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "fmla v19.4s, v7.4s, v3.s[3]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"

            "fmla v16.4s, v4.4s, v0.s[0]\n"
            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"
            "fmla v17.4s, v4.4s, v1.s[0]\n"
            "fmla v18.4s, v4.4s, v2.s[0]\n"
            "fmla v19.4s, v4.4s, v3.s[0]\n"

            "ld1 {v6.4s, v7.4s}, [%[a_ptr]], 32\n"

            "fmla v16.4s, v5.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v1.s[1]\n"
            "fmla v18.4s, v5.4s, v2.s[1]\n"
            "fmla v19.4s, v5.4s, v3.s[1]\n"

            "subs %w[K], %w[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "fmla v16.4s, v6.4s, v0.s[2]\n"
            "fmla v17.4s, v6.4s, v1.s[2]\n"
            "fmla v18.4s, v6.4s, v2.s[2]\n"
            "fmla v19.4s, v6.4s, v3.s[2]\n"

            "fmla v16.4s, v7.4s, v0.s[3]\n"
            "fmla v17.4s, v7.4s, v1.s[3]\n"
            "fmla v18.4s, v7.4s, v2.s[3]\n"
            "fmla v19.4s, v7.4s, v3.s[3]\n"

            "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
              "v18", "v19", "cc", "memory");
})";
    return writer.str();
}
std::string GetKern4x8() {
    std::stringstream writer;
    writer << R"(
// Overview of register layout:
//
// A 4x4 block of A is stored in register v4-v7
// A 4x4 block of B is stored in register v0-v3, slipping until 8x4
// A 8x4 block of accumulators store in v16-v23.
//
//                    A +--------+
//                      | v4[0-3]|
//                      | v5[0-3]|
//                      | v6[0-3]|
//                      | v7[0-3]|
//                      +--------+
//      B
//  +--------+ - - - - -+--------+
//  | v0[0-3]|          |v16[0-3]|
//  | v1[0-3]|          |v17[0-3]|
//  | v2[0-3]|          |v18[0-3]|
//  | v3[0-3]|          |v19[0-3]|
//  +--------+ - - - - -+--------+
//  | v0[0-3]|          |v20[0-3]|
//  | v1[0-3]|          |v21[0-3]|
//  | v2[0-3]|          |v22[0-3]|
//  | v3[0-3]|          |v23[0-3]|
//  +--------+ - - - - -+--------+
//                      Accumulator

static void kern_4x8(const float* a_ptr, const float* b_ptr, size_t LDB, size_t K,
              float* output) {
    //! As each load 32 number from B, but the pos add 24 * 4, so we minus 24
    //! here.
    LDB = (LDB - 24) * sizeof(float);
    asm volatile(
            "subs %w[K], %w[K], #4\n"
            "ld1 {v4.4s, v5.4s}, [%[a_ptr]], 32\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "fmul v16.4s, v4.4s, v0.s[0]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "fmla v16.4s, v5.4s, v0.s[1]\n"
            "ld1 {v6.4s, v7.4s}, [%[a_ptr]], 32\n"
            "fmul v17.4s, v4.4s, v1.s[0]\n"

            "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
            "fmla v17.4s, v5.4s, v1.s[1]\n"
            "fmla v16.4s, v6.4s, v0.s[2]\n"
            "fmla v17.4s, v6.4s, v1.s[2]\n"
            "fmul v18.4s, v4.4s, v2.s[0]\n"
            "fmla v16.4s, v7.4s, v0.s[3]\n"
            "fmla v18.4s, v5.4s, v2.s[1]\n"
            "fmla v17.4s, v7.4s, v1.s[3]\n"
            "fmul v19.4s, v4.4s, v3.s[0]\n"

            "ld1 {v24.4s, v25.4s}, [%[b_ptr]], 32\n"
            "fmla v18.4s, v7.4s, v2.s[3]\n"
            "fmla v19.4s, v5.4s, v3.s[1]\n"
            "fmul v20.4s, v4.4s, v24.s[0]\n"
            "fmla v19.4s, v6.4s, v3.s[2]\n"

            "ld1 {v26.4s, v27.4s}, [%[b_ptr]], %x[LDB]\n"
            "fmla v18.4s, v6.4s, v2.s[2]\n"
            "fmla v19.4s, v7.4s, v3.s[3]\n"
            "fmul v21.4s, v4.4s, v25.s[0]\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "fmla v20.4s, v5.4s, v24.s[1]\n"
            "fmla v21.4s, v5.4s, v25.s[1]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v6.4s, v24.s[2]\n"
            "fmul v22.4s, v4.4s, v26.s[0]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmla v21.4s, v6.4s, v25.s[2]\n"
            "fmla v22.4s, v5.4s, v26.s[1]\n"

            "fmla v21.4s, v7.4s, v25.s[3]\n"
            "fmul v23.4s, v4.4s, v27.s[0]\n"
            "fmla v20.4s, v7.4s, v24.s[3]\n"
            "ld1 {v3.4s}, [%[b_ptr]], 16\n"
            "fmla v22.4s, v6.4s, v26.s[2]\n"
            "fmla v23.4s, v5.4s, v27.s[1]\n"

            "beq 2f\n"

            "1:\n"
            "ld1 {v4.4s, v5.4s}, [%[a_ptr]], 32\n"
            "fmla v22.4s, v7.4s, v26.s[3]\n"
            "fmla v23.4s, v6.4s, v27.s[2]\n"
            "fmla v16.4s, v4.4s, v0.s[0]\n"
            "fmla v17.4s, v4.4s, v1.s[0]\n"
            "fmla v23.4s, v7.4s, v27.s[3]\n"

            "ld1 {v6.4s, v7.4s}, [%[a_ptr]], 32\n"
            "fmla v16.4s, v5.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v1.s[1]\n"
            "fmla v16.4s, v6.4s, v0.s[2]\n"
            "fmla v17.4s, v6.4s, v1.s[2]\n"
            "fmla v18.4s, v4.4s, v2.s[0]\n"
            "fmla v16.4s, v7.4s, v0.s[3]\n"
            "fmla v18.4s, v5.4s, v2.s[1]\n"
            "fmla v17.4s, v7.4s, v1.s[3]\n"
            "fmla v19.4s, v4.4s, v3.s[0]\n"

            "ld1 {v24.4s, v25.4s}, [%[b_ptr]], 32\n"
            "fmla v18.4s, v6.4s, v2.s[2]\n"
            "fmla v19.4s, v5.4s, v3.s[1]\n"
            "fmla v20.4s, v4.4s, v24.s[0]\n"
            "fmla v19.4s, v6.4s, v3.s[2]\n"

            "ld1 {v26.4s, v27.4s}, [%[b_ptr]], %x[LDB]\n"
            "fmla v18.4s, v7.4s, v2.s[3]\n"
            "fmla v19.4s, v7.4s, v3.s[3]\n"
            "fmla v21.4s, v4.4s, v25.s[0]\n"
            "fmla v20.4s, v5.4s, v24.s[1]\n"
            "fmla v21.4s, v5.4s, v25.s[1]\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "fmla v20.4s, v6.4s, v24.s[2]\n"
            "fmla v22.4s, v4.4s, v26.s[0]\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "fmla v20.4s, v7.4s, v24.s[3]\n"
            "fmla v23.4s, v4.4s, v27.s[0]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmla v21.4s, v6.4s, v25.s[2]\n"
            "fmla v22.4s, v5.4s, v26.s[1]\n"
            "ld1 {v3.4s}, [%[b_ptr]], 16\n"
            "fmla v21.4s, v7.4s, v25.s[3]\n"
            "fmla v23.4s, v5.4s, v27.s[1]\n"
            "fmla v22.4s, v6.4s, v26.s[2]\n"

            "subs %w[K], %w[K], #4\n"
            "bne 1b\n"

            "2:\n"
            "st1 {v16.4s, v17.4s}, [%[output]], 32\n"
            "fmla v22.4s, v7.4s, v26.s[3]\n"
            "st1 {v18.4s, v19.4s}, [%[output]], 32\n"
            "fmla v23.4s, v6.4s, v27.s[2]\n"
            "fmla v23.4s, v7.4s, v27.s[3]\n"
            "st1 {v20.4s, v21.4s}, [%[output]], 32\n"
            "st1 {v22.4s, v23.4s}, [%[output]], 32\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17",
              "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
              "v27", "cc", "memory");
})";
    return writer.str();
}
std::string GetKern4x16() {
    std::stringstream writer;
    writer << R"(
// Overview of register layout:
//
// A 4x1 cell of Rhs is stored in 32bit in v4-v7 (v8-v11 for ping pong)
// A 4x1 cell of Lhs is stored in 32bit in v0-v3
// A 16x1 block of accumulators is stored in 32bit in v16-v31.
//
//                  Rhs +--------+
//                      | v4[0-3]|
//                      | v5[0-3]|
//                      | v6[0-3]|
//                      | v7[0-3]|
//                      +--------+
//      Lhs
//  +--------+ - - - - -+--------+
//  | v0[0-3] |         |v16[0-3]|
//  | v1[0-3] |         |v17[0-3]|
//  | v2[0-3] |         |v18[0-3]|
//  | v3[0-3] |         |v19[0-3]|
//  | v8[0-3] |         |v20[0-3]|
//  | v9[0-3] |         |v21[0-3]|
//  | v10[0-3]|         |v22[0-3]|
//  | v11[0-3]|         |v23[0-3]|
//  +--------+          |v24[0-3]|
//                      |v25[0-3]|
//                      |v26[0-3]|
//                      |v27[0-3]|
//                      |v28[0-3]|
//                      |v29[0-3]|
//                      |v30[0-3]|
//                      |v31[0-3]|
//                      +--------+
//                      Accumulator

static void kern_4x16(const float* a_ptr, const float* b_ptr, int LDB, int K,
               float* output) {
    //! As each load 64 number from B, but the pos add 56 * 4, so we minus 56
    //! here.
    LDB = (LDB - 56) * sizeof(float);

    asm volatile(
            "stp d8, d9, [sp, #-16]!\n"
            "stp d10, d11, [sp, #-16]!\n"

            "subs %w[K], %w[K], #4\n"
            "ld1 {v4.4s}, [%[a_ptr]], 16\n"
            "ld1 {v0.4s, v1.4s}, [%[b_ptr]], 32\n"

            "fmul v16.4s, v4.4s, v0.s[0]\n"
            "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
            "ld1 {v5.4s}, [%[a_ptr]], 16\n"
            "fmul v17.4s, v4.4s, v1.s[0]\n"
            "fmul v18.4s, v4.4s, v2.s[0]\n"
            "ld1 {v6.4s}, [%[a_ptr]], 16\n"
            "fmul v19.4s, v4.4s, v3.s[0]\n"

            "fmla v16.4s, v5.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v1.s[1]\n"
            "ld1 {v7.4s}, [%[a_ptr]], 16\n"
            "fmla v18.4s, v5.4s, v2.s[1]\n"
            "fmla v19.4s, v5.4s, v3.s[1]\n"

            "ld1 {v8.4s}, [%[b_ptr]], 16\n"

            "fmla v16.4s, v6.4s, v0.s[2]\n"
            "ld1 {v9.4s}, [%[b_ptr]], 16\n"
            "fmla v17.4s, v6.4s, v1.s[2]\n"
            "fmla v18.4s, v6.4s, v2.s[2]\n"
            "ld1 {v10.4s}, [%[b_ptr]], 16\n"
            "fmla v19.4s, v6.4s, v3.s[2]\n"

            "fmla v16.4s, v7.4s, v0.s[3]\n"
            "fmla v17.4s, v7.4s, v1.s[3]\n"
            "ld1 {v11.4s}, [%[b_ptr]], 16\n"
            "fmla v18.4s, v7.4s, v2.s[3]\n"
            "fmla v19.4s, v7.4s, v3.s[3]\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "fmul v20.4s, v4.4s, v8.s[0]\n"
            "fmul v21.4s, v4.4s, v9.s[0]\n"
            "fmul v22.4s, v4.4s, v10.s[0]\n"
            "fmul v23.4s, v4.4s, v11.s[0]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v5.4s, v8.s[1]\n"
            "fmla v21.4s, v5.4s, v9.s[1]\n"
            "fmla v22.4s, v5.4s, v10.s[1]\n"
            "fmla v23.4s, v5.4s, v11.s[1]\n"

            "ld1 {v2.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v6.4s, v8.s[2]\n"
            "fmla v21.4s, v6.4s, v9.s[2]\n"
            "fmla v22.4s, v6.4s, v10.s[2]\n"
            "fmla v23.4s, v6.4s, v11.s[2]\n"

            "ld1 {v3.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v7.4s, v8.s[3]\n"
            "fmla v21.4s, v7.4s, v9.s[3]\n"
            "fmla v22.4s, v7.4s, v10.s[3]\n"
            "fmla v23.4s, v7.4s, v11.s[3]\n"

            "fmul v24.4s, v4.4s, v0.s[0]\n"
            "fmul v25.4s, v4.4s, v1.s[0]\n"
            "fmul v26.4s, v4.4s, v2.s[0]\n"
            "fmul v27.4s, v4.4s, v3.s[0]\n"

            "fmla v24.4s, v5.4s, v0.s[1]\n"
            "fmla v25.4s, v5.4s, v1.s[1]\n"
            "fmla v26.4s, v5.4s, v2.s[1]\n"
            "fmla v27.4s, v5.4s, v3.s[1]\n"

            "ld1 {v8.4s, v9.4s}, [%[b_ptr]], 32\n"

            "fmla v24.4s, v6.4s, v0.s[2]\n"
            "fmla v25.4s, v6.4s, v1.s[2]\n"
            "fmla v26.4s, v6.4s, v2.s[2]\n"
            "fmla v27.4s, v6.4s, v3.s[2]\n"

            "ld1 {v10.4s, v11.4s}, [%[b_ptr]], %x[LDB]\n"

            "fmla v24.4s, v7.4s, v0.s[3]\n"
            "fmla v25.4s, v7.4s, v1.s[3]\n"
            "fmla v26.4s, v7.4s, v2.s[3]\n"
            "fmla v27.4s, v7.4s, v3.s[3]\n"

            "fmul v28.4s, v4.4s, v8.s[0]\n"
            "fmul v29.4s, v4.4s, v9.s[0]\n"
            "fmul v30.4s, v4.4s, v10.s[0]\n"
            "fmul v31.4s, v4.4s, v11.s[0]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "fmla v28.4s, v5.4s, v8.s[1]\n"
            "fmla v29.4s, v5.4s, v9.s[1]\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "fmla v30.4s, v5.4s, v10.s[1]\n"
            "fmla v31.4s, v5.4s, v11.s[1]\n"

            "ld1 {v4.4s}, [%[a_ptr]], 16\n"

            "fmla v28.4s, v6.4s, v8.s[2]\n"
            "fmla v29.4s, v6.4s, v9.s[2]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmla v30.4s, v6.4s, v10.s[2]\n"
            "fmla v31.4s, v6.4s, v11.s[2]\n"

            "ld1 {v5.4s}, [%[a_ptr]], 16\n"

            "fmla v28.4s, v7.4s, v8.s[3]\n"
            "fmla v29.4s, v7.4s, v9.s[3]\n"
            "ld1 {v3.4s}, [%[b_ptr]], 16\n"
            "fmla v30.4s, v7.4s, v10.s[3]\n"
            "fmla v31.4s, v7.4s, v11.s[3]\n"

            "ld1 {v6.4s}, [%[a_ptr]], 16\n"

            "fmla v16.4s, v4.4s, v0.s[0]\n"
            "fmla v17.4s, v4.4s, v1.s[0]\n"
            "ld1 {v8.4s}, [%[b_ptr]], 16\n"
            "fmla v18.4s, v4.4s, v2.s[0]\n"
            "fmla v19.4s, v4.4s, v3.s[0]\n"

            "ld1 {v7.4s}, [%[a_ptr]], 16\n"

            "fmla v16.4s, v5.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v1.s[1]\n"
            "ld1 {v9.4s}, [%[b_ptr]], 16\n"
            "fmla v18.4s, v5.4s, v2.s[1]\n"
            "fmla v19.4s, v5.4s, v3.s[1]\n"

            "fmla v16.4s, v6.4s, v0.s[2]\n"
            "fmla v17.4s, v6.4s, v1.s[2]\n"
            "ld1 {v10.4s}, [%[b_ptr]], 16\n"
            "fmla v18.4s, v6.4s, v2.s[2]\n"
            "fmla v19.4s, v6.4s, v3.s[2]\n"

            "ld1 {v11.4s}, [%[b_ptr]], 16\n"

            "fmla v16.4s, v7.4s, v0.s[3]\n"
            "fmla v17.4s, v7.4s, v1.s[3]\n"
            "fmla v18.4s, v7.4s, v2.s[3]\n"
            "fmla v19.4s, v7.4s, v3.s[3]\n"

            "fmla v20.4s, v4.4s, v8.s[0]\n"
            "fmla v21.4s, v4.4s, v9.s[0]\n"
            "fmla v22.4s, v4.4s, v10.s[0]\n"
            "fmla v23.4s, v4.4s, v11.s[0]\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v5.4s, v8.s[1]\n"
            "fmla v21.4s, v5.4s, v9.s[1]\n"
            "fmla v22.4s, v5.4s, v10.s[1]\n"
            "fmla v23.4s, v5.4s, v11.s[1]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v6.4s, v8.s[2]\n"
            "fmla v21.4s, v6.4s, v9.s[2]\n"
            "fmla v22.4s, v6.4s, v10.s[2]\n"
            "fmla v23.4s, v6.4s, v11.s[2]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"

            "fmla v20.4s, v7.4s, v8.s[3]\n"
            "fmla v21.4s, v7.4s, v9.s[3]\n"
            "ld1 {v3.4s}, [%[b_ptr]], 16\n"
            "fmla v22.4s, v7.4s, v10.s[3]\n"
            "fmla v23.4s, v7.4s, v11.s[3]\n"

            "fmla v24.4s, v4.4s, v0.s[0]\n"
            "fmla v25.4s, v4.4s, v1.s[0]\n"
            "fmla v26.4s, v4.4s, v2.s[0]\n"
            "fmla v27.4s, v4.4s, v3.s[0]\n"

            "fmla v24.4s, v5.4s, v0.s[1]\n"
            "fmla v25.4s, v5.4s, v1.s[1]\n"
            "fmla v26.4s, v5.4s, v2.s[1]\n"
            "fmla v27.4s, v5.4s, v3.s[1]\n"

            "ld1 {v8.4s, v9.4s}, [%[b_ptr]], 32\n"

            "fmla v24.4s, v6.4s, v0.s[2]\n"
            "fmla v25.4s, v6.4s, v1.s[2]\n"
            "fmla v26.4s, v6.4s, v2.s[2]\n"
            "fmla v27.4s, v6.4s, v3.s[2]\n"

            "ld1 {v10.4s, v11.4s}, [%[b_ptr]], %x[LDB]\n"

            "fmla v24.4s, v7.4s, v0.s[3]\n"
            "fmla v25.4s, v7.4s, v1.s[3]\n"
            "fmla v26.4s, v7.4s, v2.s[3]\n"
            "fmla v27.4s, v7.4s, v3.s[3]\n"

            "fmla v28.4s, v4.4s, v8.s[0]\n"
            "fmla v29.4s, v4.4s, v9.s[0]\n"
            "fmla v30.4s, v4.4s, v10.s[0]\n"
            "fmla v31.4s, v4.4s, v11.s[0]\n"

            "subs %w[K], %w[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "st1 {v16.4s, v17.4s}, [%[output]], 32\n"

            "fmla v28.4s, v5.4s, v8.s[1]\n"
            "fmla v29.4s, v5.4s, v9.s[1]\n"

            "st1 {v18.4s, v19.4s}, [%[output]], 32\n"

            "fmla v30.4s, v5.4s, v10.s[1]\n"
            "fmla v31.4s, v5.4s, v11.s[1]\n"

            "st1 {v20.4s, v21.4s}, [%[output]], 32\n"

            "fmla v28.4s, v6.4s, v8.s[2]\n"
            "fmla v29.4s, v6.4s, v9.s[2]\n"

            "st1 {v22.4s, v23.4s}, [%[output]], 32\n"

            "fmla v30.4s, v6.4s, v10.s[2]\n"
            "fmla v31.4s, v6.4s, v11.s[2]\n"

            "st1 {v24.4s, v25.4s}, [%[output]], 32\n"

            "fmla v28.4s, v7.4s, v8.s[3]\n"
            "fmla v29.4s, v7.4s, v9.s[3]\n"
            "st1 {v26.4s, v27.4s}, [%[output]], 32\n"
            "fmla v30.4s, v7.4s, v10.s[3]\n"
            "fmla v31.4s, v7.4s, v11.s[3]\n"

            "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output]], 64\n"

            "ldp d10, d11, [sp], #16\n"
            "ldp d8, d9, [sp], #16\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc",
              "memory");
})";
    return writer.str();
}

}  // namespace

std::string MatmulM4N16MK4Kernel::GetKernelSymbol(TContext*) const {
    return "Arm64_fp32_m4_n16_mk4_matmul";
}

std::string MatmulM4N16MK4Kernel::GetKernelSignature(TContext* ctx) const {
    std::stringstream writer;
    writer << "void " << GetKernelSymbol(ctx) << R"((const float* A, size_t LDA,
                            const float* B, size_t LDB, float* C,
                            size_t LDC, size_t M, size_t N, size_t K))";
    return writer.str();
}

std::string MatmulM4N16MK4Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include <arm_neon.h>\n";
    writer << "#include \"stddef.h\"\n";

    writer << GetKern4x1();
    writer << "\n\n";
    writer << GetKern4x4();
    writer << "\n\n";
    writer << GetKern4x8();
    writer << "\n\n";
    writer << GetKern4x16();
    writer << "\n\n";

    writer << GetKernelSignature(ctx);
    writer << "{\n";
    writer << R"(
    const int MB=4;
    const int KB=4;
    const int NB=16;
    const int CALCBLK=4;
    //! (m/4, k/4, 4, 4) * (k/4, n, 4) = (m/4, n, 4)
    for (size_t m = 0; m < M; m += MB) {
        float* output = C + (m / MB) * LDC;
        const float* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_4x16(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 8) {
            kern_4x8(A, cur_B, LDB, K, output);
            cur_B += KB * CALCBLK * 2;
            output += MB * CALCBLK * 2;
            n += 8;
        }
        if (N - n >= 4) {
            kern_4x4(A, cur_B, LDB, K, output);
            cur_B += KB * CALCBLK;
            output += MB * CALCBLK;
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
