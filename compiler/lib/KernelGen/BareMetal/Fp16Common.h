#pragma once

#include <string>

static std::string gen_fp16_define() {
    std::string res = R"(
#if defined(__arm__) || defined(__aarch64__)
#define GI_TARGET_ARM
#endif

#if defined(GI_TARGET_ARM) && defined(__ARM_NEON)
#define GI_NEON_INTRINSICS
#endif

#if defined(__riscv_vector)
#define GI_RVV_INTRINSICS
#include <riscv_vector.h>
#endif

#if defined(GI_RVV_INTRINSICS)
typedef float16_t gi_float16_t;
#elif defined(GI_NEON_INTRINSICS) && __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
typedef __fp16 gi_float16_t;
#endif

    )";

    return res;
}