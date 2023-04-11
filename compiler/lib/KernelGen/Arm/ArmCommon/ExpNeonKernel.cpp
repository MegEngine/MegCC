#include "InternalKernel.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

std::string ExpNeonKernel::GetKernelSymbol(TContext*) const {
    return "ArmCommon_internel_exp_ps_f32";
}

std::string ExpNeonKernel::GetKernelSignature(TContext*) const {
    return R"(float32x4_t exp_ps_f32(float32x4_t x))";
}

std::string ExpNeonKernel::GetKernelBody(TContext*) const {
    std::stringstream writer;

    writer << R"(
    #include<arm_neon.h>
    #include<math.h>

    typedef float32x4_t v4sf;  // vector of 4 float
    typedef uint32x4_t v4su;   // vector of 4 uint32
    typedef int32x4_t v4si;    // vector of 4 int32

    #define c_exp_hi 88.3762626647949f
    #define c_exp_lo -88.3762626647949f

    #define c_cephes_LOG2EF 1.44269504088896341
    #define c_cephes_exp_C1 0.693359375
    #define c_cephes_exp_C2 -2.12194440e-4

    #define c_cephes_exp_p0 1.9875691500E-4
    #define c_cephes_exp_p1 1.3981999507E-3
    #define c_cephes_exp_p2 8.3334519073E-3
    #define c_cephes_exp_p3 4.1665795894E-2
    #define c_cephes_exp_p4 1.6666665459E-1
    #define c_cephes_exp_p5 5.0000001201E-1

    /* exp() computed for 4 float at once */
    float32x4_t exp_ps_f32(float32x4_t x) {
        v4sf tmp, fx;

        v4sf one = vdupq_n_f32(1);
        x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
        x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

        /* express exp(x) as exp(g + n*log(2)) */
        fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

        /* perform a floorf */
        tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

        /* if greater, subtract 1 */
        v4su mask = vcgtq_f32(tmp, fx);
        mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

        fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

        tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
        v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
        x = vsubq_f32(x, tmp);
        x = vsubq_f32(x, z);

        static const float cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1,
                                              c_cephes_exp_p2, c_cephes_exp_p3,
                                              c_cephes_exp_p4, c_cephes_exp_p5};
        v4sf y = vld1q_dup_f32(cephes_exp_p + 0);
        v4sf c1 = vld1q_dup_f32(cephes_exp_p + 1);
        v4sf c2 = vld1q_dup_f32(cephes_exp_p + 2);
        v4sf c3 = vld1q_dup_f32(cephes_exp_p + 3);
        v4sf c4 = vld1q_dup_f32(cephes_exp_p + 4);
        v4sf c5 = vld1q_dup_f32(cephes_exp_p + 5);

        y = vmulq_f32(y, x);
        z = vmulq_f32(x, x);
        y = vaddq_f32(y, c1);
        y = vmulq_f32(y, x);
        y = vaddq_f32(y, c2);
        y = vmulq_f32(y, x);
        y = vaddq_f32(y, c3);
        y = vmulq_f32(y, x);
        y = vaddq_f32(y, c4);
        y = vmulq_f32(y, x);
        y = vaddq_f32(y, c5);

        y = vmulq_f32(y, z);
        y = vaddq_f32(y, x);
        y = vaddq_f32(y, one);

        /* build 2^n */
        int32x4_t mm;
        mm = vcvtq_s32_f32(fx);
        mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
        mm = vshlq_n_s32(mm, 23);
        v4sf pow2n = vreinterpretq_f32_s32(mm);

        y = vmulq_f32(y, pow2n);
        return y;
    })";
    return writer.str();
}

// vim: syntax=cpp.doxygen
