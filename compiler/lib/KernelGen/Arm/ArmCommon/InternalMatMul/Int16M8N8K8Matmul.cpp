#include "InternalMatMul.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;
namespace {
std::string GetKern8x1() {
    std::stringstream writer;
    writer << R"(
static void kern_8x1(
        const int16_t* a_ptr, const int16_t* b_ptr, int LDB, int K, int32_t* output) {
#ifdef __aarch64__
    //! As each load 32 number from B, but the pos add 24 * 2, so we minus 24
    //! here.
    LDB *= sizeof(int16_t);
    asm volatile(
            "subs %w[K], %w[K], #8\n"
            "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[a_ptr]], 64\n"
            "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[a_ptr]], 64\n"
            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"

            "smull v16.4s, v24.4h, v0.h[0]\n"
            "smull2 v17.4s, v24.8h, v0.h[0]\n"
            "smull v18.4s, v25.4h, v0.h[1]\n"
            "smull2 v19.4s, v25.8h, v0.h[1]\n"
            "smull v20.4s, v26.4h, v0.h[2]\n"
            "smull2 v21.4s, v26.8h, v0.h[2]\n"
            "smull v22.4s, v27.4h, v0.h[3]\n"
            "smull2 v23.4s, v27.8h, v0.h[3]\n"

            "beq 2f\n"

            "1:\n"
            "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[a_ptr]], 64\n"
            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v29.4h, v0.h[5]\n"
            "smlal2 v19.4s, v29.8h, v0.h[5]\n"
            "smlal v20.4s, v30.4h, v0.h[6]\n"
            "smlal2 v21.4s, v30.8h, v0.h[6]\n"
            "smlal v22.4s, v31.4h, v0.h[7]\n"
            "smlal2 v23.4s, v31.8h, v0.h[7]\n"

            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"
            "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[a_ptr]], 64\n"

            "smlal v16.4s, v24.4h, v0.h[0]\n"
            "smlal2 v17.4s, v24.8h, v0.h[0]\n"
            "smlal v18.4s, v25.4h, v0.h[1]\n"
            "smlal2 v19.4s, v25.8h, v0.h[1]\n"
            "smlal v20.4s, v26.4h, v0.h[2]\n"
            "smlal2 v21.4s, v26.8h, v0.h[2]\n"
            "smlal v22.4s, v27.4h, v0.h[3]\n"
            "smlal2 v23.4s, v27.8h, v0.h[3]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v29.4h, v0.h[5]\n"
            "smlal2 v19.4s, v29.8h, v0.h[5]\n"
            "smlal v20.4s, v30.4h, v0.h[6]\n"
            "smlal2 v21.4s, v30.8h, v0.h[6]\n"
            "smlal v22.4s, v31.4h, v0.h[7]\n"
            "smlal2 v23.4s, v31.8h, v0.h[7]\n"

            "add v16.4s, v16.4s, v18.4s\n"
            "add v20.4s, v20.4s, v22.4s\n"
            "add v17.4s, v17.4s, v19.4s\n"
            "add v21.4s, v21.4s, v23.4s\n"
            "add v16.4s, v16.4s, v20.4s\n"
            "add v17.4s, v17.4s, v21.4s\n"

            "st1 {v16.4s, v17.4s}, [%[output]], 32\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
              "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory");
#else
    //! As each load 16 number from B, but the pos add 16 * 2, so we minus 16
    //! here.
    LDB = (LDB - 4) * sizeof(int16_t);

    asm volatile(
            "subs %[K], #8\n"
            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"

            "vld1.32 {d0}, [%[b_ptr]]!\n"
            "vld1.32 {d1}, [%[b_ptr]], %[LDB]\n"

            "vmull.s16 q12, d8, d0[0]\n"
            "vmull.s16 q13, d9, d0[0]\n"
            "vmull.s16 q14, d10, d0[1]\n"
            "vmull.s16 q15, d11, d0[1]\n"

            "vmlal.s16 q12, d12, d0[2]\n"
            "vmlal.s16 q13, d13, d0[2]\n"
            "vmlal.s16 q14, d14, d0[3]\n"
            "vmlal.s16 q15, d15, d0[3]\n"

            "beq 2f\n"

            "1:\n"

            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vld1.32 {d0}, [%[b_ptr]]!\n"

            "vmlal.s16 q12, d16, d1[0]\n"
            "vmlal.s16 q13, d17, d1[0]\n"
            "vmlal.s16 q14, d18, d1[1]\n"
            "vmlal.s16 q15, d19, d1[1]\n"

            "vmlal.s16 q12, d20, d1[2]\n"
            "vmlal.s16 q13, d21, d1[2]\n"
            "vmlal.s16 q14, d22, d1[3]\n"
            "vmlal.s16 q15, d23, d1[3]\n"

            "vld1.32 {d1}, [%[b_ptr]], %[LDB]\n"
            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"

            "vmlal.s16 q12, d8, d0[0]\n"
            "vmlal.s16 q13, d9, d0[0]\n"
            "vmlal.s16 q14, d10, d0[1]\n"
            "vmlal.s16 q15, d11, d0[1]\n"

            "vmlal.s16 q12, d12, d0[2]\n"
            "vmlal.s16 q13, d13, d0[2]\n"
            "vmlal.s16 q14, d14, d0[3]\n"
            "vmlal.s16 q15, d15, d0[3]\n"

            "subs %[K], %[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "vmlal.s16 q12, d16, d1[0]\n"
            "vmlal.s16 q13, d17, d1[0]\n"
            "vmlal.s16 q14, d18, d1[1]\n"
            "vmlal.s16 q15, d19, d1[1]\n"

            "vmlal.s16 q12, d20, d1[2]\n"
            "vmlal.s16 q13, d21, d1[2]\n"
            "vmlal.s16 q14, d22, d1[3]\n"
            "vmlal.s16 q15, d23, d1[3]\n"

            "vadd.s32 q12, q12, q14\n"
            "vadd.s32 q13, q13, q15\n"

            "vst1.32 {d24, d25, d26, d27}, [%[output]]!\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "d0", "d1", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
              "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
              "d27", "d28", "d29", "d30", "d31", "cc", "memory");
#endif
}
)";
    return writer.str();
}
std::string GetKern8x4() {
    std::stringstream writer;
    writer << R"(
static void kern_8x4(
        const int16_t* a_ptr, const int16_t* b_ptr, int LDB, int K, int32_t* output) {
#ifdef __aarch64__
    //! As each load 32 number from B, but the pos add 24 * 2, so we minus 24
    //! here.
    LDB = (LDB - 24) * sizeof(int16_t);

    asm volatile(
            "subs %w[K], %w[K], #8\n"

            "ld1 {v24.4s}, [%[a_ptr]], 16\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"

            "smull v16.4s, v24.4h, v0.h[0]\n"
            "smull2 v17.4s, v24.8h, v0.h[0]\n"
            "smull v18.4s, v24.4h, v1.h[0]\n"
            "smull2 v19.4s, v24.8h, v1.h[0]\n"

            "ld1 {v25.4s}, [%[a_ptr]], 16\n"

            "smull v20.4s, v24.4h, v2.h[0]\n"
            "smull2 v21.4s, v24.8h, v2.h[0]\n"
            "smull v22.4s, v24.4h, v3.h[0]\n"
            "smull2 v23.4s, v24.8h, v3.h[0]\n"

            "smlal v16.4s, v25.4h, v0.h[1]\n"
            "smlal2 v17.4s, v25.8h, v0.h[1]\n"
            "smlal v18.4s, v25.4h, v1.h[1]\n"
            "smlal2 v19.4s, v25.8h, v1.h[1]\n"

            "ld1 {v26.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v25.4h, v2.h[1]\n"
            "smlal2 v21.4s, v25.8h, v2.h[1]\n"
            "smlal v22.4s, v25.4h, v3.h[1]\n"
            "smlal2 v23.4s, v25.8h, v3.h[1]\n"

            "smlal v16.4s, v26.4h, v0.h[2]\n"
            "smlal2 v17.4s, v26.8h, v0.h[2]\n"
            "smlal v18.4s, v26.4h, v1.h[2]\n"
            "smlal2 v19.4s, v26.8h, v1.h[2]\n"

            "ld1 {v27.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v26.4h, v2.h[2]\n"
            "smlal2 v21.4s, v26.8h, v2.h[2]\n"
            "smlal v22.4s, v26.4h, v3.h[2]\n"
            "smlal2 v23.4s, v26.8h, v3.h[2]\n"

            "smlal v16.4s, v27.4h, v0.h[3]\n"
            "smlal2 v17.4s, v27.8h, v0.h[3]\n"
            "smlal v18.4s, v27.4h, v1.h[3]\n"
            "smlal2 v19.4s, v27.8h, v1.h[3]\n"

            "ld1 {v28.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v27.4h, v2.h[3]\n"
            "smlal2 v21.4s, v27.8h, v2.h[3]\n"
            "smlal v22.4s, v27.4h, v3.h[3]\n"
            "smlal2 v23.4s, v27.8h, v3.h[3]\n"

            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v28.4h, v1.h[4]\n"
            "smlal2 v19.4s, v28.8h, v1.h[4]\n"

            "ld1 {v29.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v28.4h, v2.h[4]\n"
            "smlal2 v21.4s, v28.8h, v2.h[4]\n"
            "smlal v22.4s, v28.4h, v3.h[4]\n"
            "smlal2 v23.4s, v28.8h, v3.h[4]\n"

            "smlal v16.4s, v29.4h, v0.h[5]\n"
            "smlal2 v17.4s, v29.8h, v0.h[5]\n"
            "smlal v18.4s, v29.4h, v1.h[5]\n"
            "smlal2 v19.4s, v29.8h, v1.h[5]\n"

            "ld1 {v30.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v29.4h, v2.h[5]\n"
            "smlal2 v21.4s, v29.8h, v2.h[5]\n"
            "smlal v22.4s, v29.4h, v3.h[5]\n"
            "smlal2 v23.4s, v29.8h, v3.h[5]\n"

            "smlal v16.4s, v30.4h, v0.h[6]\n"
            "smlal2 v17.4s, v30.8h, v0.h[6]\n"
            "smlal v18.4s, v30.4h, v1.h[6]\n"
            "smlal2 v19.4s, v30.8h, v1.h[6]\n"

            "ld1 {v31.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v30.4h, v2.h[6]\n"
            "smlal2 v21.4s, v30.8h, v2.h[6]\n"
            "smlal v22.4s, v30.4h, v3.h[6]\n"
            "smlal2 v23.4s, v30.8h, v3.h[6]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v24.4s}, [%[a_ptr]], 16\n"

            "smlal v16.4s, v31.4h, v0.h[7]\n"
            "smlal2 v17.4s, v31.8h, v0.h[7]\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "smlal v18.4s, v31.4h, v1.h[7]\n"
            "smlal2 v19.4s, v31.8h, v1.h[7]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"

            "smlal v20.4s, v31.4h, v2.h[7]\n"
            "smlal2 v21.4s, v31.8h, v2.h[7]\n"

            "ld1 {v2.4s}, [%[b_ptr]], 16\n"

            "smlal v22.4s, v31.4h, v3.h[7]\n"
            "smlal2 v23.4s, v31.8h, v3.h[7]\n"

            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"

            "smlal v16.4s, v24.4h, v0.h[0]\n"
            "smlal2 v17.4s, v24.8h, v0.h[0]\n"
            "smlal v18.4s, v24.4h, v1.h[0]\n"
            "smlal2 v19.4s, v24.8h, v1.h[0]\n"

            "ld1 {v25.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v24.4h, v2.h[0]\n"
            "smlal2 v21.4s, v24.8h, v2.h[0]\n"
            "smlal v22.4s, v24.4h, v3.h[0]\n"
            "smlal2 v23.4s, v24.8h, v3.h[0]\n"

            "smlal v16.4s, v25.4h, v0.h[1]\n"
            "smlal2 v17.4s, v25.8h, v0.h[1]\n"
            "smlal v18.4s, v25.4h, v1.h[1]\n"
            "smlal2 v19.4s, v25.8h, v1.h[1]\n"

            "ld1 {v26.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v25.4h, v2.h[1]\n"
            "smlal2 v21.4s, v25.8h, v2.h[1]\n"
            "smlal v22.4s, v25.4h, v3.h[1]\n"
            "smlal2 v23.4s, v25.8h, v3.h[1]\n"

            "smlal v16.4s, v26.4h, v0.h[2]\n"
            "smlal2 v17.4s, v26.8h, v0.h[2]\n"
            "smlal v18.4s, v26.4h, v1.h[2]\n"
            "smlal2 v19.4s, v26.8h, v1.h[2]\n"

            "ld1 {v27.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v26.4h, v2.h[2]\n"
            "smlal2 v21.4s, v26.8h, v2.h[2]\n"
            "smlal v22.4s, v26.4h, v3.h[2]\n"
            "smlal2 v23.4s, v26.8h, v3.h[2]\n"

            "smlal v16.4s, v27.4h, v0.h[3]\n"
            "smlal2 v17.4s, v27.8h, v0.h[3]\n"
            "smlal v18.4s, v27.4h, v1.h[3]\n"
            "smlal2 v19.4s, v27.8h, v1.h[3]\n"

            "ld1 {v28.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v27.4h, v2.h[3]\n"
            "smlal2 v21.4s, v27.8h, v2.h[3]\n"
            "smlal v22.4s, v27.4h, v3.h[3]\n"
            "smlal2 v23.4s, v27.8h, v3.h[3]\n"

            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v28.4h, v1.h[4]\n"
            "smlal2 v19.4s, v28.8h, v1.h[4]\n"

            "ld1 {v29.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v28.4h, v2.h[4]\n"
            "smlal2 v21.4s, v28.8h, v2.h[4]\n"
            "smlal v22.4s, v28.4h, v3.h[4]\n"
            "smlal2 v23.4s, v28.8h, v3.h[4]\n"

            "smlal v16.4s, v29.4h, v0.h[5]\n"
            "smlal2 v17.4s, v29.8h, v0.h[5]\n"
            "smlal v18.4s, v29.4h, v1.h[5]\n"
            "smlal2 v19.4s, v29.8h, v1.h[5]\n"

            "ld1 {v30.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v29.4h, v2.h[5]\n"
            "smlal2 v21.4s, v29.8h, v2.h[5]\n"
            "smlal v22.4s, v29.4h, v3.h[5]\n"
            "smlal2 v23.4s, v29.8h, v3.h[5]\n"

            "smlal v16.4s, v30.4h, v0.h[6]\n"
            "smlal2 v17.4s, v30.8h, v0.h[6]\n"
            "smlal v18.4s, v30.4h, v1.h[6]\n"
            "smlal2 v19.4s, v30.8h, v1.h[6]\n"

            "ld1 {v31.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v30.4h, v2.h[6]\n"
            "smlal2 v21.4s, v30.8h, v2.h[6]\n"
            "smlal v22.4s, v30.4h, v3.h[6]\n"
            "smlal2 v23.4s, v30.8h, v3.h[6]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"

            "smlal v16.4s, v31.4h, v0.h[7]\n"
            "smlal2 v17.4s, v31.8h, v0.h[7]\n"
            "smlal v18.4s, v31.4h, v1.h[7]\n"
            "smlal2 v19.4s, v31.8h, v1.h[7]\n"
            "smlal v20.4s, v31.4h, v2.h[7]\n"
            "smlal2 v21.4s, v31.8h, v2.h[7]\n"
            "smlal v22.4s, v31.4h, v3.h[7]\n"
            "smlal2 v23.4s, v31.8h, v3.h[7]\n"

            "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output]], 64\n"
            "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc",
              "memory");
#else
    //! As each load 16 number from B, but the pos add 16 * 2, so we minus 16
    //! here.
    LDB = (LDB - 16) * sizeof(int16_t);

    asm volatile(
            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vld1.32 {d0, d1, d2, d3}, [%[b_ptr]]!\n"
            "subs %[K], #8\n"

            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]], %[LDB]\n"
            "vmull.s16 q8, d8, d0[0]\n"
            "vmull.s16 q10, d8, d2[0]\n"
            "vmull.s16 q12, d8, d4[0]\n"
            "vmull.s16 q14, d8, d6[0]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmull.s16 q9, d9, d0[0]\n"
            "vmull.s16 q11, d9, d2[0]\n"
            "vmull.s16 q13, d9, d4[0]\n"
            "vmull.s16 q15, d9, d6[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d0[1]\n"
            "vmlal.s16 q10, d10, d2[1]\n"
            "vmlal.s16 q12, d10, d4[1]\n"
            "vmlal.s16 q14, d10, d6[1]\n"
            "vmlal.s16 q9, d11, d0[1]\n"
            "vmlal.s16 q11, d11, d2[1]\n"
            "vmlal.s16 q13, d11, d4[1]\n"
            "vmlal.s16 q15, d11, d6[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d0[2]\n"
            "vmlal.s16 q10, d12, d2[2]\n"
            "vmlal.s16 q12, d12, d4[2]\n"
            "vmlal.s16 q14, d12, d6[2]\n"
            "vmlal.s16 q9, d13, d0[2]\n"
            "vmlal.s16 q11, d13, d2[2]\n"
            "vmlal.s16 q13, d13, d4[2]\n"
            "vmlal.s16 q15, d13, d6[2]\n"

            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d14, d0[3]\n"
            "vmlal.s16 q10, d14, d2[3]\n"
            "vmlal.s16 q12, d14, d4[3]\n"
            "vmlal.s16 q14, d14, d6[3]\n"
            "vmlal.s16 q9, d15, d0[3]\n"
            "vmlal.s16 q11, d15, d2[3]\n"
            "vmlal.s16 q13, d15, d4[3]\n"
            "vmlal.s16 q15, d15, d6[3]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d8, d1[0]\n"
            "vmlal.s16 q10, d8, d3[0]\n"
            "vmlal.s16 q12, d8, d5[0]\n"
            "vmlal.s16 q14, d8, d7[0]\n"
            "vmlal.s16 q9, d9, d1[0]\n"
            "vmlal.s16 q11, d9, d3[0]\n"
            "vmlal.s16 q13, d9, d5[0]\n"
            "vmlal.s16 q15, d9, d7[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d1[1]\n"
            "vmlal.s16 q10, d10, d3[1]\n"
            "vmlal.s16 q12, d10, d5[1]\n"
            "vmlal.s16 q14, d10, d7[1]\n"
            "vmlal.s16 q9, d11, d1[1]\n"
            "vmlal.s16 q11, d11, d3[1]\n"
            "vmlal.s16 q13, d11, d5[1]\n"
            "vmlal.s16 q15, d11, d7[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d1[2]\n"
            "vmlal.s16 q10, d12, d3[2]\n"
            "vmlal.s16 q12, d12, d5[2]\n"
            "vmlal.s16 q14, d12, d7[2]\n"
            "vmlal.s16 q9, d13, d1[2]\n"
            "vmlal.s16 q11, d13, d3[2]\n"
            "vmlal.s16 q13, d13, d5[2]\n"
            "vmlal.s16 q15, d13, d7[2]\n"

            "beq 2f\n"

            "1:\n"
            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d14, d1[3]\n"
            "vmlal.s16 q10, d14, d3[3]\n"
            "vmlal.s16 q9, d15, d1[3]\n"
            "vmlal.s16 q11, d15, d3[3]\n"

            "vld1.32 {d0, d1, d2, d3}, [%[b_ptr]]!\n"
            "vmlal.s16 q12, d14, d5[3]\n"
            "vmlal.s16 q14, d14, d7[3]\n"
            "vmlal.s16 q13, d15, d5[3]\n"
            "vmlal.s16 q15, d15, d7[3]\n"

            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]], %[LDB]\n"
            "vmlal.s16 q8, d8, d0[0]\n"
            "vmlal.s16 q10, d8, d2[0]\n"
            "vmlal.s16 q12, d8, d4[0]\n"
            "vmlal.s16 q14, d8, d6[0]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmlal.s16 q9, d9, d0[0]\n"
            "vmlal.s16 q11, d9, d2[0]\n"
            "vmlal.s16 q13, d9, d4[0]\n"
            "vmlal.s16 q15, d9, d6[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d0[1]\n"
            "vmlal.s16 q10, d10, d2[1]\n"
            "vmlal.s16 q12, d10, d4[1]\n"
            "vmlal.s16 q14, d10, d6[1]\n"
            "vmlal.s16 q9, d11, d0[1]\n"
            "vmlal.s16 q11, d11, d2[1]\n"
            "vmlal.s16 q13, d11, d4[1]\n"
            "vmlal.s16 q15, d11, d6[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d0[2]\n"
            "vmlal.s16 q10, d12, d2[2]\n"
            "vmlal.s16 q12, d12, d4[2]\n"
            "vmlal.s16 q14, d12, d6[2]\n"
            "vmlal.s16 q9, d13, d0[2]\n"
            "vmlal.s16 q11, d13, d2[2]\n"
            "vmlal.s16 q13, d13, d4[2]\n"
            "vmlal.s16 q15, d13, d6[2]\n"

            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d14, d0[3]\n"
            "vmlal.s16 q10, d14, d2[3]\n"
            "vmlal.s16 q12, d14, d4[3]\n"
            "vmlal.s16 q14, d14, d6[3]\n"
            "vmlal.s16 q9, d15, d0[3]\n"
            "vmlal.s16 q11, d15, d2[3]\n"
            "vmlal.s16 q13, d15, d4[3]\n"
            "vmlal.s16 q15, d15, d6[3]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d8, d1[0]\n"
            "vmlal.s16 q10, d8, d3[0]\n"
            "vmlal.s16 q12, d8, d5[0]\n"
            "vmlal.s16 q14, d8, d7[0]\n"
            "vmlal.s16 q9, d9, d1[0]\n"
            "vmlal.s16 q11, d9, d3[0]\n"
            "vmlal.s16 q13, d9, d5[0]\n"
            "vmlal.s16 q15, d9, d7[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d1[1]\n"
            "vmlal.s16 q10, d10, d3[1]\n"
            "vmlal.s16 q12, d10, d5[1]\n"
            "vmlal.s16 q14, d10, d7[1]\n"
            "vmlal.s16 q9, d11, d1[1]\n"
            "vmlal.s16 q11, d11, d3[1]\n"
            "vmlal.s16 q13, d11, d5[1]\n"
            "vmlal.s16 q15, d11, d7[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d1[2]\n"
            "vmlal.s16 q10, d12, d3[2]\n"
            "vmlal.s16 q12, d12, d5[2]\n"
            "vmlal.s16 q14, d12, d7[2]\n"
            "vmlal.s16 q9, d13, d1[2]\n"
            "vmlal.s16 q11, d13, d3[2]\n"
            "vmlal.s16 q13, d13, d5[2]\n"
            "vmlal.s16 q15, d13, d7[2]\n"

            "subs %[K], %[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "vmlal.s16 q8, d14, d1[3]\n"
            "vmlal.s16 q10, d14, d3[3]\n"
            "vmlal.s16 q9, d15, d1[3]\n"
            "vmlal.s16 q11, d15, d3[3]\n"
            "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
            "vmlal.s16 q12, d14, d5[3]\n"
            "vmlal.s16 q14, d14, d7[3]\n"
            "vmlal.s16 q13, d15, d5[3]\n"
            "vmlal.s16 q15, d15, d7[3]\n"
            "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
            "vst1.32 {d24, d25, d26, d27}, [%[output]]!\n"
            "vst1.32 {d28, d29, d30, d31}, [%[output]]!\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
              "cc", "memory");
#endif
}
)";
    return writer.str();
}
std::string GetKern8x8() {
    std::stringstream writer;
    writer << R"(
#ifdef __aarch64__
static void kern_8x8(
        const int16_t* a_ptr, const int16_t* b_ptr, int LDB, int K, int32_t* output) {
    //! As each load 64 number from B, but the pos add 48 * 2, so we minus 48
    //! here.
    LDB = (LDB - 48) * sizeof(int16_t);

    asm volatile(
            "subs %w[K], %w[K], #8\n"
            "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[a_ptr]], 64\n"
            "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[a_ptr]], 64\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "smull v16.4s, v8.4h, v0.h[0]\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "smlal v16.4s, v9.4h, v0.h[1]\n"
            "smull v18.4s, v8.4h, v1.h[0]\n"
            "smull2 v17.4s, v8.8h, v0.h[0]\n"
            "smull2 v19.4s, v8.8h, v1.h[0]\n"
            "smlal v16.4s, v10.4h, v0.h[2]\n"
            "smlal v18.4s, v9.4h, v1.h[1]\n"
            "smlal2 v17.4s, v9.8h, v0.h[1]\n"
            "smlal2 v19.4s, v9.8h, v1.h[1]\n"
            "smlal v16.4s, v11.4h, v0.h[3]\n"
            "smlal v18.4s, v10.4h, v1.h[2]\n"
            "smlal2 v17.4s, v10.8h, v0.h[2]\n"
            "smlal2 v19.4s, v10.8h, v1.h[2]\n"
            "smlal v16.4s, v12.4h, v0.h[4]\n"
            "smlal v18.4s, v11.4h, v1.h[3]\n"
            "smlal2 v17.4s, v11.8h, v0.h[3]\n"
            "smlal2 v19.4s, v11.8h, v1.h[3]\n"
            "smlal v16.4s, v13.4h, v0.h[5]\n"
            "smlal v18.4s, v12.4h, v1.h[4]\n"
            "smlal2 v17.4s, v12.8h, v0.h[4]\n"
            "smlal2 v19.4s, v12.8h, v1.h[4]\n"
            "smlal2 v17.4s, v13.8h, v0.h[5]\n"

            "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
            "smlal v16.4s, v14.4h, v0.h[6]\n"
            "smlal v18.4s, v13.4h, v1.h[5]\n"
            "smlal2 v17.4s, v14.8h, v0.h[6]\n"
            "smlal2 v19.4s, v13.8h, v1.h[5]\n"
            "smull v20.4s, v8.4h, v2.h[0]\n"
            "smull v22.4s, v8.4h, v3.h[0]\n"
            "smull2 v21.4s, v8.8h, v2.h[0]\n"
            "smull2 v23.4s, v8.8h, v3.h[0]\n"
            "smlal v16.4s, v15.4h, v0.h[7]\n"
            "smlal v18.4s, v14.4h, v1.h[6]\n"
            "smlal2 v17.4s, v15.8h, v0.h[7]\n"
            "smlal2 v19.4s, v14.8h, v1.h[6]\n"
            "smlal v20.4s, v9.4h, v2.h[1]\n"
            "smlal v22.4s, v9.4h, v3.h[1]\n"
            "smlal2 v21.4s, v9.8h, v2.h[1]\n"
            "smlal2 v23.4s, v9.8h, v3.h[1]\n"
            "smlal v18.4s, v15.4h, v1.h[7]\n"
            "smlal v20.4s, v10.4h, v2.h[2]\n"
            "smlal v22.4s, v10.4h, v3.h[2]\n"
            "smlal2 v21.4s, v10.8h, v2.h[2]\n"
            "smlal2 v23.4s, v10.8h, v3.h[2]\n"
            "smlal2 v19.4s, v15.8h, v1.h[7]\n"
            "smlal v20.4s, v11.4h, v2.h[3]\n"
            "smlal v22.4s, v11.4h, v3.h[3]\n"
            "smlal2 v21.4s, v11.8h, v2.h[3]\n"
            "smlal2 v23.4s, v11.8h, v3.h[3]\n"
            "smlal v20.4s, v12.4h, v2.h[4]\n"
            "smlal v22.4s, v12.4h, v3.h[4]\n"
            "smlal2 v21.4s, v12.8h, v2.h[4]\n"
            "smlal2 v23.4s, v12.8h, v3.h[4]\n"
            "smlal v20.4s, v13.4h, v2.h[5]\n"
            "smlal v22.4s, v13.4h, v3.h[5]\n"
            "smlal2 v21.4s, v13.8h, v2.h[5]\n"
            "smlal2 v23.4s, v13.8h, v3.h[5]\n"

            "ld1 {v4.4s, v5.4s}, [%[b_ptr]], 32\n"
            "smlal v20.4s, v14.4h, v2.h[6]\n"
            "smlal v22.4s, v14.4h, v3.h[6]\n"
            "smlal2 v21.4s, v14.8h, v2.h[6]\n"
            "smlal2 v23.4s, v14.8h, v3.h[6]\n"
            "smull v24.4s, v8.4h, v4.h[0]\n"
            "smull v26.4s, v8.4h, v5.h[0]\n"
            "smull2 v25.4s, v8.8h, v4.h[0]\n"
            "smull2 v27.4s, v8.8h, v5.h[0]\n"
            "smlal v20.4s, v15.4h, v2.h[7]\n"
            "smlal v22.4s, v15.4h, v3.h[7]\n"
            "smlal2 v21.4s, v15.8h, v2.h[7]\n"
            "smlal2 v23.4s, v15.8h, v3.h[7]\n"
            "smlal v24.4s, v9.4h, v4.h[1]\n"
            "smlal v26.4s, v9.4h, v5.h[1]\n"
            "smlal2 v25.4s, v9.8h, v4.h[1]\n"
            "smlal2 v27.4s, v9.8h, v5.h[1]\n"
            "smlal v24.4s, v10.4h, v4.h[2]\n"
            "smlal v26.4s, v10.4h, v5.h[2]\n"
            "smlal2 v25.4s, v10.8h, v4.h[2]\n"
            "smlal2 v27.4s, v10.8h, v5.h[2]\n"
            "smlal v24.4s, v11.4h, v4.h[3]\n"
            "smlal v26.4s, v11.4h, v5.h[3]\n"
            "smlal2 v25.4s, v11.8h, v4.h[3]\n"
            "smlal2 v27.4s, v11.8h, v5.h[3]\n"
            "smlal v24.4s, v12.4h, v4.h[4]\n"
            "smlal v26.4s, v12.4h, v5.h[4]\n"
            "smlal2 v25.4s, v12.8h, v4.h[4]\n"
            "smlal2 v27.4s, v12.8h, v5.h[4]\n"
            "smlal v24.4s, v13.4h, v4.h[5]\n"
            "smlal v26.4s, v13.4h, v5.h[5]\n"
            "smlal2 v25.4s, v13.8h, v4.h[5]\n"
            "smlal2 v27.4s, v13.8h, v5.h[5]\n"

            "ld1 {v6.4s, v7.4s}, [%[b_ptr]], %x[LDB]\n"
            "smlal v24.4s, v14.4h, v4.h[6]\n"
            "smlal v26.4s, v14.4h, v5.h[6]\n"
            "smlal2 v25.4s, v14.8h, v4.h[6]\n"
            "smlal2 v27.4s, v14.8h, v5.h[6]\n"
            "smull v28.4s, v8.4h, v6.h[0]\n"
            "smull v30.4s, v8.4h, v7.h[0]\n"
            "smull2 v29.4s, v8.8h, v6.h[0]\n"
            "smull2 v31.4s, v8.8h, v7.h[0]\n"
            "smlal v28.4s, v9.4h, v6.h[1]\n"
            "smlal v30.4s, v9.4h, v7.h[1]\n"
            "smlal2 v29.4s, v9.8h, v6.h[1]\n"
            "smlal2 v31.4s, v9.8h, v7.h[1]\n"
            "smlal v28.4s, v10.4h, v6.h[2]\n"
            "smlal v30.4s, v10.4h, v7.h[2]\n"
            "smlal2 v29.4s, v10.8h, v6.h[2]\n"
            "smlal2 v31.4s, v10.8h, v7.h[2]\n"
            "smlal v28.4s, v11.4h, v6.h[3]\n"
            "smlal v30.4s, v11.4h, v7.h[3]\n"
            "smlal2 v29.4s, v11.8h, v6.h[3]\n"
            "smlal2 v31.4s, v11.8h, v7.h[3]\n"
            "smlal v28.4s, v12.4h, v6.h[4]\n"
            "smlal v30.4s, v12.4h, v7.h[4]\n"
            "smlal2 v29.4s, v12.8h, v6.h[4]\n"
            "smlal2 v31.4s, v12.8h, v7.h[4]\n"
            "smlal v28.4s, v13.4h, v6.h[5]\n"
            "smlal v30.4s, v13.4h, v7.h[5]\n"
            "smlal2 v29.4s, v13.8h, v6.h[5]\n"
            "smlal2 v31.4s, v13.8h, v7.h[5]\n"

            "beq 2f\n"

            "1:\n"

            "smlal v24.4s, v15.4h, v4.h[7]\n"
            "smlal v26.4s, v15.4h, v5.h[7]\n"
            "smlal2 v25.4s, v15.8h, v4.h[7]\n"

            "ld1 {v8.4s, v9.4s}, [%[a_ptr]], 32\n"
            "smlal2 v27.4s, v15.8h, v5.h[7]\n"
            "smlal v28.4s, v14.4h, v6.h[6]\n"
            "smlal v30.4s, v14.4h, v7.h[6]\n"

            "ld1 {v10.4s, v11.4s}, [%[a_ptr]], 32\n"
            "smlal2 v29.4s, v15.8h, v6.h[7]\n"
            "smlal2 v31.4s, v14.8h, v7.h[6]\n"
            "smlal v28.4s, v15.4h, v6.h[7]\n"

            "ld1 {v12.4s, v13.4s}, [%[a_ptr]], 32\n"
            "smlal v30.4s, v15.4h, v7.h[7]\n"
            "smlal2 v29.4s, v14.8h, v6.h[6]\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "smlal2 v31.4s, v15.8h, v7.h[7]\n"
            "smlal v16.4s, v8.4h, v0.h[0]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "smlal v16.4s, v9.4h, v0.h[1]\n"
            "smlal2 v17.4s, v8.8h, v0.h[0]\n"
            "smlal v16.4s, v10.4h, v0.h[2]\n"
            "smlal v18.4s, v8.4h, v1.h[0]\n"
            "smlal2 v17.4s, v9.8h, v0.h[1]\n"
            "smlal2 v19.4s, v8.8h, v1.h[0]\n"

            "ld1 {v14.4s, v15.4s}, [%[a_ptr]], 32\n"
            "smlal v16.4s, v11.4h, v0.h[3]\n"
            "smlal v18.4s, v9.4h, v1.h[1]\n"
            "smlal2 v17.4s, v10.8h, v0.h[2]\n"
            "smlal2 v19.4s, v9.8h, v1.h[1]\n"
            "smlal v16.4s, v12.4h, v0.h[4]\n"
            "smlal v18.4s, v10.4h, v1.h[2]\n"
            "smlal2 v17.4s, v11.8h, v0.h[3]\n"
            "smlal2 v19.4s, v10.8h, v1.h[2]\n"
            "smlal v16.4s, v13.4h, v0.h[5]\n"
            "smlal v18.4s, v11.4h, v1.h[3]\n"
            "smlal2 v17.4s, v12.8h, v0.h[4]\n"
            "smlal2 v19.4s, v11.8h, v1.h[3]\n"
            "smlal v16.4s, v14.4h, v0.h[6]\n"
            "smlal v18.4s, v12.4h, v1.h[4]\n"
            "smlal2 v17.4s, v13.8h, v0.h[5]\n"
            "smlal2 v19.4s, v12.8h, v1.h[4]\n"
            "smlal v16.4s, v15.4h, v0.h[7]\n"
            "smlal v18.4s, v13.4h, v1.h[5]\n"
            "smlal2 v17.4s, v14.8h, v0.h[6]\n"
            "smlal2 v19.4s, v13.8h, v1.h[5]\n"

            "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
            "smlal v18.4s, v14.4h, v1.h[6]\n"
            "smlal2 v17.4s, v15.8h, v0.h[7]\n"
            "smlal2 v19.4s, v14.8h, v1.h[6]\n"
            "smlal v20.4s, v8.4h, v2.h[0]\n"
            "smlal v22.4s, v8.4h, v3.h[0]\n"
            "smlal2 v21.4s, v8.8h, v2.h[0]\n"
            "smlal2 v23.4s, v8.8h, v3.h[0]\n"
            "smlal v18.4s, v15.4h, v1.h[7]\n"
            "smlal v20.4s, v9.4h, v2.h[1]\n"
            "smlal v22.4s, v9.4h, v3.h[1]\n"
            "smlal2 v21.4s, v9.8h, v2.h[1]\n"
            "smlal2 v23.4s, v9.8h, v3.h[1]\n"
            "smlal2 v19.4s, v15.8h, v1.h[7]\n"
            "smlal v20.4s, v10.4h, v2.h[2]\n"
            "smlal v22.4s, v10.4h, v3.h[2]\n"
            "smlal2 v21.4s, v10.8h, v2.h[2]\n"
            "smlal2 v23.4s, v10.8h, v3.h[2]\n"
            "smlal v20.4s, v11.4h, v2.h[3]\n"
            "smlal v22.4s, v11.4h, v3.h[3]\n"
            "smlal2 v21.4s, v11.8h, v2.h[3]\n"
            "smlal2 v23.4s, v11.8h, v3.h[3]\n"
            "smlal v20.4s, v12.4h, v2.h[4]\n"
            "smlal v22.4s, v12.4h, v3.h[4]\n"
            "smlal2 v21.4s, v12.8h, v2.h[4]\n"
            "smlal2 v23.4s, v12.8h, v3.h[4]\n"
            "smlal v20.4s, v13.4h, v2.h[5]\n"
            "smlal v22.4s, v13.4h, v3.h[5]\n"
            "smlal2 v21.4s, v13.8h, v2.h[5]\n"
            "smlal2 v23.4s, v13.8h, v3.h[5]\n"

            "ld1 {v4.4s, v5.4s}, [%[b_ptr]], 32\n"
            "smlal v20.4s, v14.4h, v2.h[6]\n"
            "smlal v22.4s, v14.4h, v3.h[6]\n"
            "smlal2 v21.4s, v14.8h, v2.h[6]\n"
            "smlal2 v23.4s, v14.8h, v3.h[6]\n"
            "smlal v24.4s, v8.4h, v4.h[0]\n"
            "smlal v26.4s, v8.4h, v5.h[0]\n"
            "smlal2 v25.4s, v8.8h, v4.h[0]\n"
            "smlal2 v27.4s, v8.8h, v5.h[0]\n"
            "smlal v20.4s, v15.4h, v2.h[7]\n"
            "smlal2 v21.4s, v15.8h, v2.h[7]\n"
            "smlal v22.4s, v15.4h, v3.h[7]\n"
            "smlal2 v23.4s, v15.8h, v3.h[7]\n"
            "smlal v24.4s, v9.4h, v4.h[1]\n"
            "smlal v26.4s, v9.4h, v5.h[1]\n"
            "smlal2 v25.4s, v9.8h, v4.h[1]\n"
            "smlal2 v27.4s, v9.8h, v5.h[1]\n"
            "smlal v24.4s, v10.4h, v4.h[2]\n"
            "smlal v26.4s, v10.4h, v5.h[2]\n"
            "smlal2 v25.4s, v10.8h, v4.h[2]\n"
            "smlal2 v27.4s, v10.8h, v5.h[2]\n"
            "smlal v24.4s, v11.4h, v4.h[3]\n"
            "smlal v26.4s, v11.4h, v5.h[3]\n"
            "smlal2 v25.4s, v11.8h, v4.h[3]\n"
            "smlal2 v27.4s, v11.8h, v5.h[3]\n"
            "smlal v24.4s, v12.4h, v4.h[4]\n"
            "smlal v26.4s, v12.4h, v5.h[4]\n"
            "smlal2 v25.4s, v12.8h, v4.h[4]\n"
            "smlal2 v27.4s, v12.8h, v5.h[4]\n"
            "smlal v24.4s, v13.4h, v4.h[5]\n"
            "smlal v26.4s, v13.4h, v5.h[5]\n"
            "smlal2 v25.4s, v13.8h, v4.h[5]\n"
            "smlal2 v27.4s, v13.8h, v5.h[5]\n"

            "ld1 {v6.4s, v7.4s}, [%[b_ptr]], %x[LDB]\n"
            "smlal v24.4s, v14.4h, v4.h[6]\n"
            "smlal v26.4s, v14.4h, v5.h[6]\n"
            "smlal2 v25.4s, v14.8h, v4.h[6]\n"
            "smlal2 v27.4s, v14.8h, v5.h[6]\n"
            "smlal v28.4s, v8.4h, v6.h[0]\n"
            "smlal v30.4s, v8.4h, v7.h[0]\n"
            "smlal2 v29.4s, v8.8h, v6.h[0]\n"
            "smlal2 v31.4s, v8.8h, v7.h[0]\n"
            "smlal v28.4s, v9.4h, v6.h[1]\n"
            "smlal v30.4s, v9.4h, v7.h[1]\n"
            "smlal2 v29.4s, v9.8h, v6.h[1]\n"
            "smlal2 v31.4s, v9.8h, v7.h[1]\n"
            "smlal v28.4s, v10.4h, v6.h[2]\n"
            "smlal v30.4s, v10.4h, v7.h[2]\n"
            "smlal2 v29.4s, v10.8h, v6.h[2]\n"
            "smlal2 v31.4s, v10.8h, v7.h[2]\n"
            "smlal v28.4s, v11.4h, v6.h[3]\n"
            "smlal v30.4s, v11.4h, v7.h[3]\n"
            "smlal2 v29.4s, v11.8h, v6.h[3]\n"
            "smlal2 v31.4s, v11.8h, v7.h[3]\n"
            "smlal v28.4s, v12.4h, v6.h[4]\n"
            "smlal v30.4s, v12.4h, v7.h[4]\n"
            "smlal2 v29.4s, v12.8h, v6.h[4]\n"
            "smlal2 v31.4s, v12.8h, v7.h[4]\n"
            "smlal v28.4s, v13.4h, v6.h[5]\n"
            "smlal v30.4s, v13.4h, v7.h[5]\n"
            "smlal2 v29.4s, v13.8h, v6.h[5]\n"
            "smlal2 v31.4s, v13.8h, v7.h[5]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output]], 64\n"
            "smlal v24.4s, v15.4h, v4.h[7]\n"
            "smlal v28.4s, v14.4h, v6.h[6]\n"
            "smlal v30.4s, v14.4h, v7.h[6]\n"
            "smlal v26.4s, v15.4h, v5.h[7]\n"
            "smlal2 v25.4s, v15.8h, v4.h[7]\n"
            "smlal2 v27.4s, v15.8h, v5.h[7]\n"
            "smlal2 v29.4s, v14.8h, v6.h[6]\n"
            "smlal2 v31.4s, v14.8h, v7.h[6]\n"
            "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[output]], 64\n"
            "smlal v28.4s, v15.4h, v6.h[7]\n"
            "smlal v30.4s, v15.4h, v7.h[7]\n"
            "smlal2 v29.4s, v15.8h, v6.h[7]\n"
            "smlal2 v31.4s, v15.8h, v7.h[7]\n"
            "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output]], 64\n"
            "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "cc", "memory");
}
#endif
)";
    return writer.str();
}
}  // namespace

std::string Int16M8N8K8MatMulKernel::GetKernelSymbol(TContext* context) const {
    return "ArmCommon_int16_m8_n8_k8_matmul";
}

std::string Int16M8N8K8MatMulKernel::GetKernelSignature(TContext* context) const {
    std::stringstream writer;
    writer << "void " << GetKernelSymbol(context) << R"((const int16_t* A, size_t LDA,
                            const int16_t* B, size_t LDB, int32_t* C,
                            size_t LDC, size_t M, size_t N, size_t K))";
    return writer.str();
}

std::string Int16M8N8K8MatMulKernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    writer << "#include <marm_neon.h>\n";
    writer << GetKern8x1();
    writer << "\n\n";
    writer << GetKern8x4();
    writer << "\n\n";
    writer << GetKern8x8();
    writer << "\n\n";

    writer << GetKernelSignature(context);
    writer << "{\n";
    writer << R"(
    const size_t MB=8;
    const size_t KB=8;

#ifdef __aarch64__
    const size_t NB=8;
    const size_t CALCBLK=4;
    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        int32_t* output = C + (m / MB) * LDC;
        const int16_t* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_8x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_8x4(A, cur_B, LDB, K, output);
            cur_B += KB * CALCBLK;
            output += MB * CALCBLK;
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
#else 
    const size_t NB=4;
    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        int32_t* output = C + (m / MB) * LDC;
        const int16_t* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_8x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        while (n < N) {
            kern_8x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
#endif
)";
    writer << "\n}";
    return writer.str();
}