#include <sstream>
#include <string>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/Armv7/ConvKernel/ConvKernel.h"
#include "Int8DirectNchwBase.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
using namespace ArmCommon;

namespace {

std::string gen_5x5_s1_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode,
        const std::string& func_name) {
    std::stringstream ss;
    std::string kernel_impl = R"(
#define ACC_S16_S32(dst0, dst1, src)           \
    dst0 = vaddw_s16(dst0, vget_low_s16(src)); \
    dst1 = vaddw_s16(dst1, vget_high_s16(src));

static void ${func_name}(
        const int8_t* src, const int8_t* filter, const int32_t* bias, int32_t* temp,
        int8_t* dst, const size_t IH, const size_t IW, const size_t OH, const size_t OW, const int first_ic, const int last_ic,
        const float bias_scale, const float inv_dst_scale) {
    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k03 = vdup_n_s8(filter[3]);
    int8x8_t k04 = vdup_n_s8(filter[4]);
    int8x8_t k10 = vdup_n_s8(filter[5]);
    int8x8_t k11 = vdup_n_s8(filter[6]);
    int8x8_t k12 = vdup_n_s8(filter[7]);
    int8x8_t k13 = vdup_n_s8(filter[8]);
    int8x8_t k14 = vdup_n_s8(filter[9]);
    int8x8_t k20 = vdup_n_s8(filter[10]);
    int8x8_t k21 = vdup_n_s8(filter[11]);
    int8x8_t k22 = vdup_n_s8(filter[12]);
    int8x8_t k23 = vdup_n_s8(filter[13]);
    int8x8_t k24 = vdup_n_s8(filter[14]);
    int8x8_t k30 = vdup_n_s8(filter[15]);
    int8x8_t k31 = vdup_n_s8(filter[16]);
    int8x8_t k32 = vdup_n_s8(filter[17]);
    int8x8_t k33 = vdup_n_s8(filter[18]);
    int8x8_t k34 = vdup_n_s8(filter[19]);
    int8x8_t k40 = vdup_n_s8(filter[20]);
    int8x8_t k41 = vdup_n_s8(filter[21]);
    int8x8_t k42 = vdup_n_s8(filter[22]);
    int8x8_t k43 = vdup_n_s8(filter[23]);
    int8x8_t k44 = vdup_n_s8(filter[24]);

    ${init_store}
    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01, sum10, sum11;

            if (!first_ic) {
                sum00 = vld1q_s32(tptr + 0 * OW);
                sum01 = vld1q_s32(tptr + 0 * OW + 4);
                sum10 = vld1q_s32(tptr + 1 * OW);
                sum11 = vld1q_s32(tptr + 1 * OW + 4);
            } else {
                sum00 = ${init_bias};
                sum01 = sum00;
                sum10 = sum00;
                sum11 = sum00;
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
            int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
            int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
            int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r13, k13);
            d0 = vmlal_s8(d0, _r14, k14);
            ACC_S16_S32(sum00, sum01, d0);
            int16x8_t d1 = vmull_s8(_r10, k00);
            d1 = vmlal_s8(d1, _r11, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r12, k02);
            d1 = vmlal_s8(d1, _r13, k03);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r14, k04);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
            int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            d0 = vmlal_s8(d0, _r23, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r24, k24);
            d1 = vmlal_s8(d1, _r20, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r21, k11);
            d1 = vmlal_s8(d1, _r22, k12);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r23, k13);
            d1 = vmlal_s8(d1, _r24, k14);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
            int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
            d0 = vmlal_s8(d0, _r30, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r31, k31);
            d0 = vmlal_s8(d0, _r32, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r33, k33);
            d0 = vmlal_s8(d0, _r34, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmull_s8(_r30, k20);
            d1 = vmlal_s8(d1, _r31, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r32, k22);
            d1 = vmlal_s8(d1, _r33, k23);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r34, k24);

            int8x8_t _r40 = vld1_s8(sptr + 4 * IW);
            int8x8_t _r4n = vld1_s8(sptr + 4 * IW + 8);
            int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
            int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
            int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
            int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
            d0 = vmull_s8(_r40, k40);
            d0 = vmlal_s8(d0, _r41, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r42, k42);
            d0 = vmlal_s8(d0, _r43, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r44, k44);
            ACC_S16_S32(sum00, sum01, d0);
            if(last_ic) {
                ${store_func(sum00, dptr)}
                ${store_func(sum01, dptr + 4)}
            } else {
                vst1q_s32(tptr, sum00);
                vst1q_s32(tptr + 4, sum01);
            }
            d1 = vmlal_s8(d1, _r40, k30);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r41, k31);
            d1 = vmlal_s8(d1, _r42, k32);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r43, k33);
            d1 = vmlal_s8(d1, _r44, k34);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r50 = vld1_s8(sptr + 5 * IW);
            int8x8_t _r5n = vld1_s8(sptr + 5 * IW + 8);
            int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
            int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
            int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
            int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
            d1 = vmull_s8(_r50, k40);
            d1 = vmlal_s8(d1, _r51, k41);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r52, k42);
            d1 = vmlal_s8(d1, _r53, k43);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r54, k44);
            ACC_S16_S32(sum10, sum11, d1);
            if(last_ic) {
                ${store_func(sum10, dptr + OW)}
                ${store_func(sum11, dptr + OW + 4)}
            } else {
                vst1q_s32(tptr + OW, sum10);
                vst1q_s32(tptr + OW + 4, sum11);
            }
        }
    }

    if (oh < OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01;

            if (!first_ic) {
                sum00 = vld1q_s32(tptr + 0 * OW);
                sum01 = vld1q_s32(tptr + 0 * OW + 4);
            } else {
                sum00 = ${init_bias};
                sum01 = sum00;
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
            int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
            int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
            int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r13, k13);
            d0 = vmlal_s8(d0, _r14, k14);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
            int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            d0 = vmlal_s8(d0, _r23, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r24, k24);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
            int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
            d0 = vmlal_s8(d0, _r30, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r31, k31);
            d0 = vmlal_s8(d0, _r32, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r33, k33);
            d0 = vmlal_s8(d0, _r34, k34);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r40 = vld1_s8(sptr + 4 * IW);
            int8x8_t _r4n = vld1_s8(sptr + 4 * IW + 8);
            int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
            int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
            int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
            int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
            d0 = vmull_s8(_r40, k40);
            d0 = vmlal_s8(d0, _r41, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r42, k42);
            d0 = vmlal_s8(d0, _r43, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r44, k44);
            ACC_S16_S32(sum00, sum01, d0);
            if(last_ic) {
                ${store_func(sum00, dptr)}
                ${store_func(sum01, dptr + 4)}
            } else {
                vst1q_s32(tptr, sum00);
                vst1q_s32(tptr + 4, sum01);
            }
        }
    }
}
#undef ACC_S16_S32
)";
    std::string init_bias = with_bias ? "vdupq_n_s32(bptr[0])" : "vdupq_n_s32(0)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("init_bias", init_bias)
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("func_name", func_name)
                    .add("store_func",
                         [=](const std::string& reg, const std::string& dst) {
                             return Int8DirectNchwHelperBase::gen_res_store_code(
                                     reg, dst, *activate_gen.get());
                         })
                    .render(kernel_impl);
    return ss.str();
}

class Int8DirectNchwS1Helper : public Int8DirectNchwHelperBase {
public:
    std::string gen_need_copy_padding() const override {
        return R"(
        static inline int need_dst_copy_padding(const size_t ow) {
            return ow % 8;
        }

        static inline int need_src_copy_padding(const size_t ph, const size_t pw, const size_t ow) {
            if (ph || pw) {
                return 1;
            }
            return need_dst_copy_padding(ow);
        }
    )";
    }

    std::string gen_get_rectified_size(TContext* ctx) const override {
        return StringTemplate::StringTemplateArgs(ctx)
                .add_ctx_int("stride_w")
                .add_ctx_int("kernel_h")
                .add_ctx_int("kernel_w")
                .render(R"(
        static inline void get_rectified_size(
                const size_t IH, const size_t IW, const size_t OH, const size_t OW, size_t* IH2,
                size_t* IW2, size_t* OH2, size_t* OW2) {
            size_t SW = ${stride_w};
            size_t FH = ${kernel_h};
            size_t FW = ${kernel_w};

            *OH2 = OH;
            *OW2 = (OW + 7) & ~7;
            *IH2 = SW * OH + FH - SW;
            *IW2 = SW * *OW2 + FW - SW;
        }
    )");
    }

private:
    std::string gen_kern(
            TContext* ctx, bool with_bias, std::string nonline_mode,
            std::string func_name) const override {
        int fh = ctx->getAttrInt("kernel_h");
        switch (fh) {
            case 5:
                return gen_5x5_s1_kern(ctx, with_bias, nonline_mode, func_name);
            default:
                CC_ASSERT(0) << "Unsupport kernel size" << fh;
                return "";
        }
    }
};
}  // namespace

Int8Conv5x5S1DirectNCHW::Int8Conv5x5S1DirectNCHW()
        : Int8DirectNchwBase(new Int8DirectNchwS1Helper) {}

bool Int8Conv5x5S1DirectNCHW::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("kernel_h") == 5 &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 1 && ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (ctx->getAttrStr("sparse") == "DENSE" ||
                          ctx->getAttrStr("sparse") == "GROUP") &&
                         ctx->getAttrStr("format") == "NCHW" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";
    bool type_ok = is_qint8_conv_dtype(ctx);

    return param_value_ok && param_mode_ok && type_ok && noline_ok;
}

std::string Int8Conv5x5S1DirectNCHW::GetKernelSymbol(TContext* ctx) const {
    return "Armv7_int8_direct_" + ConvImpl::GetKernelSymbol(ctx);
}
