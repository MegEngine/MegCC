#pragma once
#include <sstream>
#include <string>
#include "Common/PoolingKernel.h"
#include "Utils/SymbolHelper.h"
namespace megcc {
namespace KernelGen {
namespace ArmCommon {
namespace {
static inline std::string gen_common_interleve_f32() {
    return R"(

static inline void interleave_helper(
        const float* inptr, float* outptr, int unroll_k, int ksize, float val) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}
static inline void interleave_1(
        const float* inptr0, float* outptr, int unroll_k, int ksize, float val) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        inptr0 += size;outptr+=unroll_k;
    }
}

static inline void interleave_4(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr, int unroll_k, int ksize, float val) {
     for (int k = 0; k < ksize; k += unroll_k) {
        int size = min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        inptr0 += size;outptr+=unroll_k;
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        inptr1 += size;outptr+=unroll_k;
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        inptr2 += size;outptr+=unroll_k;
        interleave_helper(inptr3, outptr, unroll_k, size, val);
        inptr3 += size;outptr+=unroll_k;
    }
}

    )";
}
static inline std::string gen_common_prefetch_2x_f32() {
    return R"(
static inline void prefetch_2x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
})";
}
static inline std::string gen_common_prefetch_3x_f32() {
    return R"(
static inline void prefetch_3x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
})";
}

static inline std::string gen_common_prefetch_4x_f32() {
    return R"(
static inline void prefetch_4x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
})";
}
}  // namespace

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc
