#pragma once
#include <sstream>
#include <string>

namespace megcc {
namespace KernelGen {
namespace Armv7 {
namespace {

static inline std::string gen_armv7_transpose_interleave_4x4_4_b_int8() {
    return std::string{R"(
static inline void transpose_interleave_4x4_4_b(
        const int8_t** inptr0, const int8_t** inptr1, const int8_t** inptr2, const int8_t** inptr3,
        int8_t* outptr, int stride) {
    //! pack form {1, 4(icb), 4(ic), 4(oc)} to {1, 1, 4(oc), 16(ic)}
    asm volatile(
            "add r1, %[outptr], %[stride]\n"
            "vld4.8 {d0-d3},[%[inptr0]]!\n"
            "vld4.8 {d4-d7},[%[inptr0]]!\n"
            "add r2, r1, %[stride]\n"
            "vld4.8 {d8-d11},[%[inptr1]]!\n"
            "vld4.8 {d12-d15},[%[inptr1]]!\n"
            "vld4.8 {d16-d19},[%[inptr2]]!\n"
            "add r3, r2, %[stride]\n"
            "vld4.8 {d20-d23},[%[inptr2]]!\n"
            "vld4.8 {d24-d27},[%[inptr3]]!\n"
            "vld4.8 {d28-d31},[%[inptr3]]!\n"

            "vst1.8 d0, [%[outptr]]!\n"
            "vst1.8 d4, [%[outptr]]!\n"
            "vst1.8 d1, [%[outptr]]!\n"
            "vst1.8 d5, [%[outptr]]!\n"
            "vst1.8 d2, [%[outptr]]!\n"
            "vst1.8 d6, [%[outptr]]!\n"
            "vst1.8 d3, [%[outptr]]!\n"
            "vst1.8 d7, [%[outptr]]!\n"

            "vst1.8 d8, [r1]!\n"
            "vst1.8 d12,[r1]!\n"
            "vst1.8 d9, [r1]!\n"
            "vst1.8 d13,[r1]!\n"
            "vst1.8 d10,[r1]!\n"
            "vst1.8 d14,[r1]!\n"
            "vst1.8 d11,[r1]!\n"
            "vst1.8 d15,[r1]!\n"

            "vst1.8 d16,[r2]!\n"
            "vst1.8 d20,[r2]!\n"
            "vst1.8 d17,[r2]!\n"
            "vst1.8 d21,[r2]!\n"
            "vst1.8 d18,[r2]!\n"
            "vst1.8 d22,[r2]!\n"
            "vst1.8 d19,[r2]!\n"
            "vst1.8 d23,[r2]!\n"

            "vst1.8 d24,[r3]!\n"
            "vst1.8 d28,[r3]!\n"
            "vst1.8 d25,[r3]!\n"
            "vst1.8 d29,[r3]!\n"
            "vst1.8 d26,[r3]!\n"
            "vst1.8 d30,[r3]!\n"
            "vst1.8 d27,[r3]!\n"
            "vst1.8 d31,[r3]!\n"
            : [inptr0] "+r"(*inptr0), [inptr1] "+r"(*inptr1), [inptr2] "+r"(*inptr2),
              [inptr3] "+r"(*inptr3), [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "r1", "r2", "r3", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10", "q11", "q12", "q14", "q15", "memory");
}
)"};
}

static inline std::string gen_armv7_transpose_interleave_1x4_4_b_int8() {
    return std::string{R"(
static inline void transpose_interleave_1x4_4_b(
        const int8_t** inptr0, int8_t* outptr, int stride) {
    asm volatile(
            "vld4.8 {d0-d3},[%[inptr0]]!\n"
            "vld4.8 {d4-d7},[%[inptr0]]!\n"

            "vst1.8 d0, [%[outptr]]!\n"
            "vst1.8 d4, [%[outptr]]!\n"
            "vst1.8 d1, [%[outptr]]!\n"
            "vst1.8 d5, [%[outptr]]!\n"
            "vst1.8 d2, [%[outptr]]!\n"
            "vst1.8 d6, [%[outptr]]!\n"
            "vst1.8 d3, [%[outptr]]!\n"
            "vst1.8 d7, [%[outptr]]!\n"
            : [inptr0] "+r"(*inptr0), [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "q0", "q1", "q2", "q3", "memory");
}
)"};
}

static inline std::string gen_armv7_transpose_4x2_1_s_int32() {
    return std::string{R"(
static inline void transpose_4x2_1_s(
        const int32_t** inptr0, const int32_t** inptr1, const int32_t** inptr2, const int32_t** inptr3,
        int32_t* outptr, int stride) {
    stride -= 8;
    asm volatile(
            "vld1.32 {d0},  [%[inptr0]]!\n"  // A0A1
            "vld1.32 {d1},  [%[inptr1]]!\n"  // B0B1
            "vld1.32 {d2},  [%[inptr2]]!\n"  // C0C1
            "vld1.32 {d3},  [%[inptr3]]!\n"  // D0D1
            "vtrn.32 d0, d1\n"               // A0B0 A1B1
            "vtrn.32 d2, d3\n"               // C0D0 C1D1
            "vst1.32 {d0},  [%[outptr]]!\n"
            "vst1.32 {d2},  [%[outptr]]!\n"
            "vst1.32 {d1},  [%[outptr]]!\n"
            "vst1.32 {d3},  [%[outptr]]!\n"
            : [inptr0] "+r"(*inptr0), [inptr1] "+r"(*inptr1), [inptr2] "+r"(*inptr2),
              [inptr3] "+r"(*inptr3), [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "d0", "d1", "d2", "d3", "memory");
}
)"};
}
}  // namespace

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc