#pragma once
#include <sstream>
#include <string>
#include "Common/PoolingKernel.h"
#include "Utils/SymbolHelper.h"
namespace megcc {
namespace KernelGen {
namespace ArmCommon {
namespace {
std::string gen_neon_intrin_compat() {
    return R"(
#if defined(__aarch64__)
    
#else
    static inline float32x4_t vdivq_f32(float32x4_t x, float32x4_t y) {
        float32x4_t recp = vrecpeq_f32(y);
        recp = vmulq_f32(vrecpsq_f32(y, recp), recp);
        return vmulq_f32(x, recp);
    }
    static inline float32_t vaddvq_f32(float32x4_t a) {
        return vgetq_lane_f32(a, 0) + vgetq_lane_f32(a, 1) + vgetq_lane_f32(a, 2) +
            vgetq_lane_f32(a, 3);
    }
    static inline float32x4_t vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v, const int lane){
        if(lane == 0){
            return vmlaq_lane_f32(a, b, vget_low_f32(v), 0);
        }else if(lane == 1){
            return vmlaq_lane_f32(a, b, vget_low_f32(v), 1);
        }else if(lane == 2){
            return vmlaq_lane_f32(a, b, vget_high_f32(v), 0);
        }else if(lane == 3){
            return vmlaq_lane_f32(a, b, vget_high_f32(v), 1);
        }
        return a;
    }
    static inline float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c){
        return vmlaq_f32(a, b, c);
    }
#endif

#if __ARM_ARCH >= 8

#else
    static inline int32x4_t vcvtaq_s32_f32(float32x4_t a) {
        float32x4_t temp = vbslq_f32(
            vcgeq_f32(a, vdupq_n_f32(0.f)), vdupq_n_f32(0.5f),
                    vdupq_n_f32(-0.5f));
        return vcvtq_s32_f32(vaddq_f32(a, temp));
    }
#endif

    )";
}
}  // namespace

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc
