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
using namespace Arm64;
namespace {

std::string GetKern8x1() {
    std::stringstream writer;
    writer << R"(
static void kern_8x1(
        const __fp16* a_ptr, const __fp16* b_ptr, int LDB, int K, __fp16* output) {
    LDB *= sizeof(__fp16);
    asm volatile(
            ".arch armv8.2-a+fp16\n"

            "subs %w[K], %w[K], #8\n"
            "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[a_ptr]], 64\n"
            "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[a_ptr]], 64\n"
            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"
            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"

            "fmla v24.8h, v16.8h, v0.h[0]\n"
            "fmla v25.8h, v17.8h, v0.h[1]\n"
            "fmla v26.8h, v18.8h, v0.h[2]\n"
            "fmla v27.8h, v19.8h, v0.h[3]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[a_ptr]], 64\n"
            "fmla v28.8h, v20.8h, v0.h[4]\n"
            "fmla v29.8h, v21.8h, v0.h[5]\n"
            "fmla v30.8h, v22.8h, v0.h[6]\n"
            "fmla v31.8h, v23.8h, v0.h[7]\n"

            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"

            "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[a_ptr]], 64\n"
            "fmla v24.8h, v16.8h, v0.h[0]\n"
            "fmla v25.8h, v17.8h, v0.h[1]\n"
            "fmla v26.8h, v18.8h, v0.h[2]\n"
            "fmla v27.8h, v19.8h, v0.h[3]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"

            "fmla v28.8h, v20.8h, v0.h[4]\n"
            "fmla v29.8h, v21.8h, v0.h[5]\n"
            "fmla v30.8h, v22.8h, v0.h[6]\n"
            "fmla v31.8h, v23.8h, v0.h[7]\n"

            "fadd v24.8h, v24.8h, v25.8h\n"
            "fadd v26.8h, v26.8h, v27.8h\n"
            "fadd v28.8h, v28.8h, v29.8h\n"
            "fadd v30.8h, v30.8h, v31.8h\n"
            "fadd v24.8h, v24.8h, v26.8h\n"
            "fadd v28.8h, v28.8h, v30.8h\n"
            "fadd v24.8h, v24.8h, v28.8h\n"

            "st1 {v24.4s}, [%[output]], 16\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
              "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory");
}
)";
    return writer.str();
}
std::string GetKern8x4() {
    std::stringstream writer;
    writer << R"(

static void kern_8x4(
        const __fp16* a_ptr, const __fp16* b_ptr, int LDB, int K, __fp16* output) {
    //! LDB means number of elements in one block in B. we will read 24 numbers
    //! first. so minus 24 * 2 bytes here.
    LDB = (LDB - 24) * sizeof(__fp16);

    asm volatile(
            ".arch armv8.2-a+fp16\n"

            "ld1 {v16.4s, v17.4s}, [%[a_ptr]], 32\n"

            "subs %w[K], %w[K], #8\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "fmul v24.8h, v16.8h, v0.h[0]\n"

            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "fmul v25.8h, v16.8h, v1.h[0]\n"

            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"
            "fmul v26.8h, v16.8h, v2.h[0]\n"

            "ld1 {v18.4s}, [%[a_ptr]], 16\n"
            "fmul v27.8h, v16.8h, v3.h[0]\n"

            "fmla v24.8h, v17.8h, v0.h[1]\n"
            "fmla v25.8h, v17.8h, v1.h[1]\n"
            "fmla v26.8h, v17.8h, v2.h[1]\n"
            "fmla v27.8h, v17.8h, v3.h[1]\n"

            "ld1 {v19.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v18.8h, v0.h[2]\n"
            "fmla v25.8h, v18.8h, v1.h[2]\n"
            "fmla v26.8h, v18.8h, v2.h[2]\n"
            "fmla v27.8h, v18.8h, v3.h[2]\n"

            "ld1 {v20.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v19.8h, v0.h[3]\n"
            "fmla v25.8h, v19.8h, v1.h[3]\n"
            "fmla v26.8h, v19.8h, v2.h[3]\n"
            "fmla v27.8h, v19.8h, v3.h[3]\n"

            "ld1 {v21.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v20.8h, v0.h[4]\n"
            "fmla v25.8h, v20.8h, v1.h[4]\n"
            "fmla v26.8h, v20.8h, v2.h[4]\n"
            "fmla v27.8h, v20.8h, v3.h[4]\n"

            "ld1 {v22.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v21.8h, v0.h[5]\n"
            "fmla v25.8h, v21.8h, v1.h[5]\n"
            "fmla v26.8h, v21.8h, v2.h[5]\n"
            "fmla v27.8h, v21.8h, v3.h[5]\n"

            "ld1 {v23.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v22.8h, v0.h[6]\n"
            "fmla v25.8h, v22.8h, v1.h[6]\n"
            "fmla v26.8h, v22.8h, v2.h[6]\n"
            "fmla v27.8h, v22.8h, v3.h[6]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v16.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v23.8h, v0.h[7]\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "fmla v25.8h, v23.8h, v1.h[7]\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"

            "fmla v26.8h, v23.8h, v2.h[7]\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"

            "fmla v27.8h, v23.8h, v3.h[7]\n"
            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"

            "ld1 {v17.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v16.8h, v0.h[0]\n"
            "fmla v25.8h, v16.8h, v1.h[0]\n"
            "fmla v26.8h, v16.8h, v2.h[0]\n"
            "fmla v27.8h, v16.8h, v3.h[0]\n"

            "ld1 {v18.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v17.8h, v0.h[1]\n"
            "fmla v25.8h, v17.8h, v1.h[1]\n"
            "fmla v26.8h, v17.8h, v2.h[1]\n"
            "fmla v27.8h, v17.8h, v3.h[1]\n"

            "ld1 {v19.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v18.8h, v0.h[2]\n"
            "fmla v25.8h, v18.8h, v1.h[2]\n"
            "fmla v26.8h, v18.8h, v2.h[2]\n"
            "fmla v27.8h, v18.8h, v3.h[2]\n"

            "ld1 {v20.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v19.8h, v0.h[3]\n"
            "fmla v25.8h, v19.8h, v1.h[3]\n"
            "fmla v26.8h, v19.8h, v2.h[3]\n"
            "fmla v27.8h, v19.8h, v3.h[3]\n"

            "ld1 {v21.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v20.8h, v0.h[4]\n"
            "fmla v25.8h, v20.8h, v1.h[4]\n"
            "fmla v26.8h, v20.8h, v2.h[4]\n"
            "fmla v27.8h, v20.8h, v3.h[4]\n"

            "ld1 {v22.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v21.8h, v0.h[5]\n"
            "fmla v25.8h, v21.8h, v1.h[5]\n"
            "fmla v26.8h, v21.8h, v2.h[5]\n"
            "fmla v27.8h, v21.8h, v3.h[5]\n"

            "ld1 {v23.4s}, [%[a_ptr]], 16\n"

            "fmla v24.8h, v22.8h, v0.h[6]\n"
            "fmla v25.8h, v22.8h, v1.h[6]\n"
            "fmla v26.8h, v22.8h, v2.h[6]\n"
            "fmla v27.8h, v22.8h, v3.h[6]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"

            "fmla v24.8h, v23.8h, v0.h[7]\n"
            "fmla v25.8h, v23.8h, v1.h[7]\n"
            "fmla v26.8h, v23.8h, v2.h[7]\n"
            "fmla v27.8h, v23.8h, v3.h[7]\n"

            "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "cc", "memory");
}
)";
    return writer.str();
}
std::string GetKern8x8() {
    std::stringstream writer;
    writer << R"(
static void kern_8x8(
        const __fp16* a_ptr, const __fp16* b_ptr, int LDB, int K, __fp16* output) {
    //! As each load 128 number from B, but the pos add 112 * 2, so we minus 112
    //! here.
    LDB = (LDB - 32) * sizeof(__fp16);

    asm volatile(
            ".arch armv8.2-a+fp16\n"

            "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[a_ptr]], 64\n"
            "subs %w[K], %w[K], #8\n"
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[b_ptr]], 64\n"
            "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[b_ptr]], %x[LDB]\n"

            "fmul v24.8h, v8.8h, v0.h[0]\n"
            "fmul v25.8h, v8.8h, v1.h[0]\n"
            "fmul v26.8h, v8.8h, v2.h[0]\n"
            "fmul v27.8h, v8.8h, v3.h[0]\n"
            "fmul v28.8h, v8.8h, v4.h[0]\n"
            "fmul v29.8h, v8.8h, v5.h[0]\n"
            "fmul v30.8h, v8.8h, v6.h[0]\n"
            "fmul v31.8h, v8.8h, v7.h[0]\n"

            "fmla v24.8h, v9.8h, v0.h[1]\n"
            "fmla v25.8h, v9.8h, v1.h[1]\n"
            "fmla v26.8h, v9.8h, v2.h[1]\n"
            "fmla v27.8h, v9.8h, v3.h[1]\n"
            "fmla v28.8h, v9.8h, v4.h[1]\n"
            "fmla v29.8h, v9.8h, v5.h[1]\n"
            "fmla v30.8h, v9.8h, v6.h[1]\n"
            "fmla v31.8h, v9.8h, v7.h[1]\n"

            "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[a_ptr]], 64\n"
            "fmla v24.8h, v10.8h, v0.h[2]\n"
            "fmla v25.8h, v10.8h, v1.h[2]\n"
            "fmla v26.8h, v10.8h, v2.h[2]\n"
            "fmla v27.8h, v10.8h, v3.h[2]\n"
            "fmla v28.8h, v10.8h, v4.h[2]\n"
            "fmla v29.8h, v10.8h, v5.h[2]\n"
            "fmla v30.8h, v10.8h, v6.h[2]\n"
            "fmla v31.8h, v10.8h, v7.h[2]\n"

            "fmla v24.8h, v11.8h, v0.h[3]\n"
            "fmla v25.8h, v11.8h, v1.h[3]\n"
            "fmla v26.8h, v11.8h, v2.h[3]\n"
            "fmla v27.8h, v11.8h, v3.h[3]\n"
            "fmla v28.8h, v11.8h, v4.h[3]\n"
            "fmla v29.8h, v11.8h, v5.h[3]\n"
            "fmla v30.8h, v11.8h, v6.h[3]\n"
            "fmla v31.8h, v11.8h, v7.h[3]\n"

            "fmla v24.8h, v12.8h, v0.h[4]\n"
            "fmla v25.8h, v12.8h, v1.h[4]\n"
            "fmla v26.8h, v12.8h, v2.h[4]\n"
            "fmla v27.8h, v12.8h, v3.h[4]\n"
            "fmla v24.8h, v13.8h, v0.h[5]\n"
            "fmla v25.8h, v13.8h, v1.h[5]\n"
            "fmla v26.8h, v13.8h, v2.h[5]\n"
            "fmla v27.8h, v13.8h, v3.h[5]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[a_ptr]], 64\n"
            "fmla v24.8h, v15.8h, v0.h[7]\n"
            "fmla v25.8h, v15.8h, v1.h[7]\n"
            "fmla v26.8h, v15.8h, v2.h[7]\n"
            "fmla v27.8h, v15.8h, v3.h[7]\n"
            "fmla v24.8h, v14.8h, v0.h[6]\n"
            "fmla v25.8h, v14.8h, v1.h[6]\n"
            "fmla v26.8h, v14.8h, v2.h[6]\n"
            "fmla v27.8h, v14.8h, v3.h[6]\n"
            "fmla v28.8h, v12.8h, v4.h[4]\n"
            "fmla v29.8h, v12.8h, v5.h[4]\n"
            "fmla v30.8h, v12.8h, v6.h[4]\n"
            "fmla v31.8h, v12.8h, v7.h[4]\n"

            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[b_ptr]], 64\n"
            "fmla v28.8h, v13.8h, v4.h[5]\n"
            "fmla v29.8h, v13.8h, v5.h[5]\n"
            "fmla v30.8h, v13.8h, v6.h[5]\n"
            "fmla v31.8h, v13.8h, v7.h[5]\n"
            "fmla v28.8h, v14.8h, v4.h[6]\n"
            "fmla v29.8h, v14.8h, v5.h[6]\n"
            "fmla v30.8h, v14.8h, v6.h[6]\n"
            "fmla v31.8h, v14.8h, v7.h[6]\n"
            "fmla v28.8h, v15.8h, v4.h[7]\n"
            "fmla v29.8h, v15.8h, v5.h[7]\n"
            "fmla v30.8h, v15.8h, v6.h[7]\n"
            "fmla v31.8h, v15.8h, v7.h[7]\n"
            "fmla v24.8h, v8.8h, v0.h[0]\n"
            "fmla v25.8h, v8.8h, v1.h[0]\n"
            "fmla v26.8h, v8.8h, v2.h[0]\n"
            "fmla v27.8h, v8.8h, v3.h[0]\n"

            "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[b_ptr]], %x[LDB]\n"
            "fmla v24.8h, v9.8h, v0.h[1]\n"
            "fmla v25.8h, v9.8h, v1.h[1]\n"
            "fmla v26.8h, v9.8h, v2.h[1]\n"
            "fmla v27.8h, v9.8h, v3.h[1]\n"
            "fmla v24.8h, v10.8h, v0.h[2]\n"
            "fmla v25.8h, v10.8h, v1.h[2]\n"
            "fmla v26.8h, v10.8h, v2.h[2]\n"
            "fmla v27.8h, v10.8h, v3.h[2]\n"
            "fmla v24.8h, v11.8h, v0.h[3]\n"
            "fmla v25.8h, v11.8h, v1.h[3]\n"
            "fmla v26.8h, v11.8h, v2.h[3]\n"
            "fmla v27.8h, v11.8h, v3.h[3]\n"

            "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[a_ptr]], 64\n"
            "fmla v28.8h, v10.8h, v4.h[2]\n"
            "fmla v29.8h, v10.8h, v5.h[2]\n"
            "fmla v30.8h, v10.8h, v6.h[2]\n"
            "fmla v31.8h, v10.8h, v7.h[2]\n"
            "fmla v28.8h, v8.8h, v4.h[0]\n"
            "fmla v29.8h, v8.8h, v5.h[0]\n"
            "fmla v30.8h, v8.8h, v6.h[0]\n"
            "fmla v31.8h, v8.8h, v7.h[0]\n"
            "fmla v28.8h, v9.8h, v4.h[1]\n"
            "fmla v29.8h, v9.8h, v5.h[1]\n"
            "fmla v30.8h, v9.8h, v6.h[1]\n"
            "fmla v31.8h, v9.8h, v7.h[1]\n"

            "fmla v28.8h, v11.8h, v4.h[3]\n"
            "fmla v29.8h, v11.8h, v5.h[3]\n"
            "fmla v30.8h, v11.8h, v6.h[3]\n"
            "fmla v31.8h, v11.8h, v7.h[3]\n"

            "fmla v24.8h, v12.8h, v0.h[4]\n"
            "fmla v25.8h, v12.8h, v1.h[4]\n"
            "fmla v26.8h, v12.8h, v2.h[4]\n"
            "fmla v27.8h, v12.8h, v3.h[4]\n"
            "fmla v24.8h, v13.8h, v0.h[5]\n"
            "fmla v25.8h, v13.8h, v1.h[5]\n"
            "fmla v26.8h, v13.8h, v2.h[5]\n"
            "fmla v27.8h, v13.8h, v3.h[5]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "fmla v24.8h, v14.8h, v0.h[6]\n"
            "fmla v25.8h, v14.8h, v1.h[6]\n"
            "fmla v26.8h, v14.8h, v2.h[6]\n"
            "fmla v27.8h, v14.8h, v3.h[6]\n"
            "fmla v24.8h, v15.8h, v0.h[7]\n"
            "fmla v25.8h, v15.8h, v1.h[7]\n"
            "fmla v26.8h, v15.8h, v2.h[7]\n"
            "fmla v27.8h, v15.8h, v3.h[7]\n"
            "fmla v28.8h, v12.8h, v4.h[4]\n"
            "fmla v29.8h, v12.8h, v5.h[4]\n"
            "fmla v28.8h, v13.8h, v4.h[5]\n"
            "fmla v29.8h, v13.8h, v5.h[5]\n"
            "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output]], 64\n"
            "fmla v28.8h, v14.8h, v4.h[6]\n"
            "fmla v29.8h, v14.8h, v5.h[6]\n"
            "fmla v28.8h, v15.8h, v4.h[7]\n"
            "fmla v29.8h, v15.8h, v5.h[7]\n"
            "fmla v30.8h, v12.8h, v6.h[4]\n"
            "fmla v31.8h, v12.8h, v7.h[4]\n"
            "fmla v30.8h, v13.8h, v6.h[5]\n"
            "fmla v31.8h, v13.8h, v7.h[5]\n"
            "fmla v30.8h, v14.8h, v6.h[6]\n"
            "fmla v31.8h, v14.8h, v7.h[6]\n"
            "fmla v30.8h, v15.8h, v6.h[7]\n"
            "fmla v31.8h, v15.8h, v7.h[7]\n"
            "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "v28", "v29",
              "v30", "v31", "cc", "memory");
}

)";
    return writer.str();
}
}  // namespace

std::string Fp16MatmulM8N8MK8Kernel::GetKernelSymbol(TContext*) const {
    return "arm64_fp16_m8_n8_mk8_matmul";
}

std::string Fp16MatmulM8N8MK8Kernel::GetKernelSignature(TContext* ctx) const {
    std::stringstream writer;
    writer << "void " << GetKernelSymbol(ctx) << R"((const __fp16* A, size_t LDA,
                            const __fp16* B, size_t LDB, __fp16* C,
                            size_t LDC, size_t M, size_t N, size_t K))";
    return writer.str();
}

std::string Fp16MatmulM8N8MK8Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include <arm_neon.h>\n";
    writer << "#include \"stddef.h\"\n";
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
        __fp16* output = C + (m / MB) * LDC;
        const __fp16* cur_B = B;
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
    return writer.str();
}

// vim: syntax=cpp.doxygen
