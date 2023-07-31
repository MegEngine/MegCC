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

static inline std::string gen_armv7_interleave_8x4_8_b_int8() {
    return std::string{R"(
static inline void interleave_6x4_8_b_int8(
        const int8_t** inptr0, const int8_t** inptr1, const int8_t** inptr2, const int8_t** inptr3,
        const int8_t** inptr4, const int8_t** inptr5, int8_t* outptr) {
    asm volatile(
            "vld4.32  {d0-d3}, [%[inptr0]]! \n"  // q0,q1=r00,r04,r01,r05,r02,r06,r03,r07
            "vld4.32  {d4-d7}, [%[inptr1]]! \n"  // q2,q3=r10,r14,r11,r15,r12,r16,r13,r17
            "vld4.32  {d8-d11}, [%[inptr2]]!\n"  // q4,q5=r20,r24,r21,r25,r22,r26,r23,r27
            "vld4.32  {d12-d15}, [%[inptr3]]!\n"  // q6,q7=r30,r34,r31,r35,r32,r36,r33,r37
            "vld4.32  {d16-d19}, [%[inptr4]]!\n"  // q8,q9=r40,r44,r41,r45,r42,r46,r43,r47
            "vld4.32  {d20-d23}, [%[inptr5]]!\n"  // q10,q11=r50,r54,r51,r55,r52,r56,r53,r5

            "vtrn.32  q0, q2    \n"  // q0=r00,r10,r01,r11 q2=r04,r14,r05,r15
            "vtrn.32  q4, q6    \n"  // q4=r20,r30,r21,r31 q6=r24,r34,r25,r35
            "vtrn.32  q8, q10   \n"  // q8=r40,r50,r41,r51 q10=r44,r54,r45,r55
            "vswp     d1, d8    \n"  // q0=r00,r10,r20,r30 q4=r01,r11,r21,r31
            "vtrn.32  q1, q3    \n"  // q1=r02,r12,r03,r13 q3=r06,r16,r07,r17
            "vtrn.32  q5, q7    \n"  // q5=r22,r32,r23,r33 q7=r26,r36,r27,r37
            "vtrn.32  q9, q11   \n"  // q9=r42,r52,r43,r53 q11=r46,r56,r47,r57
            "vst1.32  {d0-d1},  [%[outptr]]! \n"
            "vst1.32  {d16},    [%[outptr]]! \n"
            "vswp     d3, d10   \n"  //  q1=r02,r12,r22,r32 q5=r03,r13,r23,r33
            "vst1.32  {d8-d9},  [%[outptr]]! \n"
            "vst1.32  {d17},    [%[outptr]]! \n"
            "vst1.32  {d2-d3},  [%[outptr]]!\n"
            "vst1.32  {d18},    [%[outptr]]!\n"
            "vswp     d5, d12   \n"  // q2=r04,r14,r24,r34 q6=r05,r15,r25,r35
            "vst1.32  {d10-d11},[%[outptr]]!\n"
            "vst1.32  {d19},    [%[outptr]]!\n"
            "vst1.32  {d4-d5},  [%[outptr]]! \n"
            "vst1.32  {d20},    [%[outptr]]! \n"
            "vswp     d7, d14   \n"  // q3=r06,r16,r26,r36 q7=r07,r17,r27,r37
            "vst1.32  {d12-d13},[%[outptr]]! \n"
            "vst1.32  {d21},    [%[outptr]]! \n"
            "vst1.32  {d6-d7},  [%[outptr]]! \n"
            "vst1.32  {d22},    [%[outptr]]! \n"
            "vst1.32  {d14-d15},[%[outptr]]! \n"
            "vst1.32  {d23},    [%[outptr]]! \n"
            : [inptr0] "+r"(*inptr0), [inptr1] "+r"(*inptr1), [inptr2] "+r"(*inptr2),
              [inptr3] "+r"(*inptr3), [inptr4] "+r"(*inptr4), [inptr5] "+r"(*inptr5),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
              "cc", "memory");
}
)"};
}

static inline std::string gen_armv7_interleave_8x4_4_b_int8() {
    return std::string{R"(
static inline void interleave_6x4_4_b_int8(
        const int8_t** inptr0, const int8_t** inptr1, const int8_t** inptr2, const int8_t** inptr3,
        const int8_t** inptr4, const int8_t** inptr5, int8_t* outptr) {
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"    // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"    // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"    // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"    // D0D1D2D3
            "vld1.32 {d8, d9},  [%[inptr4]]!\n"    // E0E1E2E3
            "vld1.32 {d10, d11},  [%[inptr5]]!\n"  // F0F1F2F3
            "vtrn.32 q0, q1\n"                     // A0B0A2B2 A1B1A3B3
            "vtrn.32 q2, q3\n"                     // C0D0C2D2 C1D1C3D3
            "vtrn.32 q4, q5\n"                     // E0F0E2F2 E1F1E3F3
            "vswp     d1, d4    \n"                // q0=A0,B0,C0,D0 q2=A2,B2,C2,D2
            "vswp     d3, d6    \n"                // q1=A1,B1,C1,D1 q3=A3,B3,C3,D3
            "vst1.32 {d0-d1},[%[outptr]]!\n"
            "vst1.32 {d8},   [%[outptr]]!\n"

            "vst1.32 {d2-d3},[%[outptr]]!\n"
            "vst1.32 {d10},  [%[outptr]]!\n"

            "vst1.32 {d4-d5},[%[outptr]]!\n"
            "vst1.32 {d9},   [%[outptr]]!\n"

            "vst1.32 {d6-d7},[%[outptr]]!\n"
            "vst1.32 {d11},  [%[outptr]]!\n"
            : [inptr0] "+r"(*inptr0), [inptr1] "+r"(*inptr1), [inptr2] "+r"(*inptr2),
              [inptr3] "+r"(*inptr3), [inptr4] "+r"(*inptr4), [inptr5] "+r"(*inptr5),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5",
              "cc", "memory");
}
)"};
}

static inline std::string gen_armv7_interleave_helper_int8() {
    return R"(
  static inline void interleave_helper_int8(
        const int8_t** inptr, int8_t* outptr, int unroll_k, int ksize, int8_t val) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = **inptr;
        *inptr = *inptr + 1;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
  }
  )";
}

static inline std::string gen_armv7_interleave_6_int8() {
    return R"(
  static inline void interleave_6_int8(
        const int8_t** inptr0, const int8_t** inptr1, const int8_t** inptr2, const int8_t** inptr3,
        const int8_t** inptr4, const int8_t** inptr5, int8_t* outptr, int unroll_k, int ksize,
        int8_t val) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = unroll_k < ksize - k ? unroll_k : ksize - k;
        interleave_helper_int8(inptr0, outptr, unroll_k, size, val);
        outptr += unroll_k;
        interleave_helper_int8(inptr1, outptr, unroll_k, size, val);
        outptr += unroll_k;
        interleave_helper_int8(inptr2, outptr, unroll_k, size, val);
        outptr += unroll_k;
        interleave_helper_int8(inptr3, outptr, unroll_k, size, val);
        outptr += unroll_k;
        interleave_helper_int8(inptr4, outptr, unroll_k, size, val);
        outptr += unroll_k;
        interleave_helper_int8(inptr5, outptr, unroll_k, size, val);
        outptr += unroll_k;
    }
  }
  )";
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

static inline std::string gen_armv7_transpose_8x4_1_b_int8() {
    return std::string{R"(
static inline void transpose_8x4_1_b_int8(
        const int8_t** inptr0, const int8_t** inptr1, const int8_t** inptr2, const int8_t** inptr3,
        int8_t* outptr) {
    asm volatile(
            "vld1.32 {d0},  [%[inptr0]]!\n"  // A1A2A3A4A5A6A7A8
            "vld1.32 {d1},  [%[inptr1]]!\n"  // B1B2B3B4B5B6B7B8
            "vld1.32 {d2},  [%[inptr2]]!\n"  // C1C2C3C4C5C6C7C8
            "vld1.32 {d3},  [%[inptr3]]!\n"  // D1D2D3D4D5D6D7D8

            "vtrn.8 d0, d1\n"  // A1B1A3B3A5B5A7B7 A2B2A4B4A6B6A8B8
            "vtrn.8 d2, d3\n"  // C1D1C3D3C5D5C7D7 C2D2C4D4C6D6C8D8

            "vtrn.16 d0, d2\n"  // A1B1C1D1A5B5C5D5 A3B3C3D3A7B7C7D7
            "vtrn.16 d1, d3\n"  // A2B2C2D2A6B6C6D6 A4B4C4D4A8B8C8D8

            //! ABCD=E then
            //! d0: E1E5 d1: E2E6 d2: E3E7 d3: E4E8
            "vzip.32 d0, d1\n"  // E1E2 E5E6
            "vzip.32 d2, d3\n"  // E3E4 E7E8

            "vst1.32 {d0}, [%[outptr]]!\n"
            "vst1.32 {d2}, [%[outptr]]!\n"
            "vst1.32 {d1}, [%[outptr]]!\n"
            "vst1.32 {d3}, [%[outptr]]!\n"
            : [inptr0] "+r"(*inptr0), [inptr1] "+r"(*inptr1), [inptr2] "+r"(*inptr2),
              [inptr3] "+r"(*inptr3), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "memory");
}
)"};
}

static inline std::string gen_armv7_transpose_4_int8() {
    return std::string{R"(
  static inline void transpose_4_int8(
        const int8_t** inptr0, const int8_t** inptr1, const int8_t** inptr2, const int8_t** inptr3,
        int8_t* outptr, int interleave, int size, int8_t val) {
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = **inptr0;
        *inptr0 = *inptr0 + 1;
        *outptr++ = **inptr1;
        *inptr1 = *inptr1 + 1;
        *outptr++ = **inptr2;
        *inptr2 = *inptr2 + 1;
        *outptr++ = **inptr3;
        *inptr3 = *inptr3 + 1;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
    }
  }
)"};
}
}  // namespace

}  // namespace Armv7
}  // namespace KernelGen
}  // namespace megcc