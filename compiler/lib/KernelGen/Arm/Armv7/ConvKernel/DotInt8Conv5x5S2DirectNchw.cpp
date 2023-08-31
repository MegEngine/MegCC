#include <sstream>
#include <string>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/Armv7/ConvKernel/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
using namespace ArmCommon;

namespace {

std::string gen_5x5_s2_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode,
        const std::string& func_name) {
    std::stringstream ss;
    std::string kernel_impl = R"(
static inline int8x16_t vqtbl1q_s8_v7(int8x16_t a, uint8x16_t index) {
    int8x8x2_t src;
    src.val[0] = vget_low_s8(a);
    src.val[1] = vget_high_s8(a);
    uint8x8_t index_low = vget_low_u8(index);
    uint8x8_t index_high = vget_high_u8(index);
    int8x8_t r00 = vtbl2_s8(src, vreinterpret_s8_u8(index_low));
    int8x8_t r01 = vtbl2_s8(src, vreinterpret_s8_u8(index_high));
    int8x16_t r = vcombine_s8(r00, r01);
    return r;
}

#define CALC_0(_k00_idx, _k01_idx, _c_idx)                         \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##0);                  \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k00_idx, _elem); \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##1);                  \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k01_idx, _elem);

#define CALC_1(_k00_idx, _k01_idx, _c_idx)                         \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##0);                  \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k00_idx, _elem); \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##1);                  \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k01_idx, _elem);

#define CALC_2(_k00_idx, _k01_idx, _k10_idx, _k11_idx, _c_idx)     \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##0);                  \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k00_idx, _elem); \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k10_idx, _elem); \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##1);                  \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k01_idx, _elem); \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k11_idx, _elem);

static void ${func_name}(
        const int8_t* src, const int8_t* filter, const int32_t* bias, int32_t* temp,
        int8_t* dst, const size_t IH, const size_t IW, const size_t OH, const size_t OW, const int first_ic, const int last_ic,
        const float bias_scale, const float inv_dst_scale) {
    const size_t tail_step = IW - 2 * OW + IW;

    const uint8x16_t _idx00 = {0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9};
    const uint8x16_t _idx01 = {4, 16, 16, 16, 6,  16, 16, 16,
                               8, 16, 16, 16, 10, 16, 16, 16};
    //! start from 8
    const uint8x16_t _idx10 = _idx00;
    const uint8x16_t _idx11 = _idx01;

    int8x16_t _tmp, _elem;
    int32_t* outptr = temp;
    int32_t* outptr2 = outptr + OW;
    int8_t* dstptr = dst;
    int8_t* dstptr2 = dstptr + OW;
    const int32_t* __restrict bptr = bias;

    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + IW * 2;
    const int8_t* r3 = src + IW * 3;
    const int8_t* r4 = src + IW * 4;
    const int8_t* r5 = src + IW * 5;
    const int8_t* r6 = src + IW * 6;

    const int8_t* k0 = filter;

    int8x16_t _k = vld1q_s8(k0);
    //! filter row 1
    uint8x16_t _idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    int8x16_t _k123 = vqtbl1q_s8_v7(_k, _idx);
    uint8x16_t _idx1 = {4, 16, 16, 16, 4, 16, 16, 16, 4, 16, 16, 16, 4, 16, 16, 16};
    int8x16_t _k4 = vqtbl1q_s8_v7(_k, _idx1);
    //! filter row 2
    uint8x16_t _idx2 = {5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    int8x16_t _k5678 = vqtbl1q_s8_v7(_k, _idx2);
    uint8x16_t _idx3 = {9, 16, 16, 16, 9, 16, 16, 16, 9, 16, 16, 16, 9, 16, 16, 16};
    int8x16_t _k9 = vqtbl1q_s8_v7(_k, _idx3);
    //! filter row 3
    uint8x16_t _idx4 = {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13};
    int8x16_t _k10111213 = vqtbl1q_s8_v7(_k, _idx4);
    uint8x16_t _idx5 = {14, 16, 16, 16, 14, 16, 16, 16, 14, 16, 16, 16, 14, 16, 16, 16};
    int8x16_t _k14 = vqtbl1q_s8_v7(_k, _idx5);
    //! 9 10 11 12 -> 13 14 15 16 -> 17 18 19 20 -> 21 22 23 24
    _k = vld1q_s8(k0 + 9);
    //! filter row 4
    uint8x16_t _idx6 = {6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9};
    int8x16_t _k15161718 = vqtbl1q_s8_v7(_k, _idx6);
    uint8x16_t _idx7 = {10, 16, 16, 16, 10, 16, 16, 16, 10, 16, 16, 16, 10, 16, 16, 16};
    int8x16_t _k19 = vqtbl1q_s8_v7(_k, _idx7);
    //! filter row 5
    uint8x16_t _idx8 = {11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14};
    int8x16_t _k20212223 = vqtbl1q_s8_v7(_k, _idx8);
    uint8x16_t _idx9 = {15, 16, 16, 16, 15, 16, 16, 16, 15, 16, 16, 16, 15, 16, 16, 16};
    int8x16_t _k24 = vqtbl1q_s8_v7(_k, _idx9);

    ${init_store}
    const int width = OW >> 2;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int w = 0;
        for (; w + 1 < width; w += 2) {
            int32x4_t _sum00, _sum01, _sum10, _sum11;
            if (!first_ic) {
                _sum00 = vld1q_s32(outptr);
                _sum01 = vld1q_s32(outptr + 4);
                _sum10 = vld1q_s32(outptr2);
                _sum11 = vld1q_s32(outptr2 + 4);
            } else {
                _sum00 = ${init_bias};
                _sum01 = _sum00;
                _sum10 = _sum00;
                _sum11 = _sum00;
            }

            _tmp = vld1q_s8(r0);
            CALC_0(123, 4, 0);
            _tmp = vld1q_s8(r0 + 8);
            CALC_0(123, 4, 1);

            _tmp = vld1q_s8(r1);
            CALC_0(5678, 9, 0);
            _tmp = vld1q_s8(r1 + 8);
            CALC_0(5678, 9, 1);

            _tmp = vld1q_s8(r2);
            CALC_2(10111213, 14, 123, 4, 0);
            _tmp = vld1q_s8(r2 + 8);
            CALC_2(10111213, 14, 123, 4, 1);

            _tmp = vld1q_s8(r3);
            CALC_2(15161718, 19, 5678, 9, 0);
            _tmp = vld1q_s8(r3 + 8);
            CALC_2(15161718, 19, 5678, 9, 1);

            _tmp = vld1q_s8(r4);
            CALC_2(20212223, 24, 10111213, 14, 0);
            _tmp = vld1q_s8(r4 + 8);
            CALC_2(20212223, 24, 10111213, 14, 1);

            _tmp = vld1q_s8(r5);
            CALC_1(15161718, 19, 0);
            _tmp = vld1q_s8(r5 + 8);
            CALC_1(15161718, 19, 1);

            _tmp = vld1q_s8(r6);
            CALC_1(20212223, 24, 0);
            _tmp = vld1q_s8(r6 + 8);
            CALC_1(20212223, 24, 1);

            if(last_ic) {
                ${store_func(_sum00, dstptr)}
                ${store_func(_sum01, dstptr + 4)}
                ${store_func(_sum10, dstptr2)}
                ${store_func(_sum11, dstptr2 + 4)}
            } else {
                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr2, _sum10);
                vst1q_s32(outptr2 + 4, _sum11);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            r6 += 16;
            outptr += 8;
            outptr2 += 8;
            dstptr += 8;
            dstptr2 += 8;
        }
        for (; w < width; w++) {
            int32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_s32(outptr);
                _sum10 = vld1q_s32(outptr2);
            } else {
                _sum00 = ${init_bias};
                _sum10 = _sum00;
            }

            _tmp = vld1q_s8(r0);
            CALC_0(123, 4, 0);

            _tmp = vld1q_s8(r1);
            CALC_0(5678, 9, 0);

            _tmp = vld1q_s8(r2);
            CALC_2(10111213, 14, 123, 4, 0);

            _tmp = vld1q_s8(r3);
            CALC_2(15161718, 19, 5678, 9, 0);

            _tmp = vld1q_s8(r4);
            CALC_2(20212223, 24, 10111213, 14, 0);

            _tmp = vld1q_s8(r5);
            CALC_1(15161718, 19, 0);

            _tmp = vld1q_s8(r6);
            CALC_1(20212223, 24, 0);

            if(last_ic) {
                ${store_func(_sum00, dstptr)}
                ${store_func(_sum10, dstptr2)}
            } else {
                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr2, _sum10);
            }

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            r5 += 8;
            r6 += 8;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW * 2;
        r1 += tail_step + IW * 2;
        r2 += tail_step + IW * 2;
        r3 += tail_step + IW * 2;
        r4 += tail_step + IW * 2;
        r5 += tail_step + IW * 2;
        r6 += tail_step + IW * 2;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int w = 0;
        for (; w + 1 < width; w += 2) {
            int32x4_t _sum00, _sum01;
            if (!first_ic) {
                _sum00 = vld1q_s32(outptr);
                _sum01 = vld1q_s32(outptr + 4);
            } else {
                _sum00 = ${init_bias};
                _sum01 = _sum00;
            }

            _tmp = vld1q_s8(r0);
            CALC_0(123, 4, 0);
            _tmp = vld1q_s8(r0 + 8);
            CALC_0(123, 4, 1);

            _tmp = vld1q_s8(r1);
            CALC_0(5678, 9, 0);
            _tmp = vld1q_s8(r1 + 8);
            CALC_0(5678, 9, 1);

            _tmp = vld1q_s8(r2);
            CALC_0(10111213, 14, 0);
            _tmp = vld1q_s8(r2 + 8);
            CALC_0(10111213, 14, 1);

            _tmp = vld1q_s8(r3);
            CALC_0(15161718, 19, 0);
            _tmp = vld1q_s8(r3 + 8);
            CALC_0(15161718, 19, 1);

            _tmp = vld1q_s8(r4);
            CALC_0(20212223, 24, 0);
            _tmp = vld1q_s8(r4 + 8);
            CALC_0(20212223, 24, 1);

            if(last_ic) {
                ${store_func(_sum00, dstptr)}
                ${store_func(_sum01, dstptr + 4)}
            } else {
                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            outptr += 8;
            dstptr += 8;
        }
        for (; w < width; w++) {
            int32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_s32(outptr);
            } else {
                _sum00 = ${init_bias};
            }

            _tmp = vld1q_s8(r0);
            CALC_0(123, 4, 0);

            _tmp = vld1q_s8(r1);
            CALC_0(5678, 9, 0);

            _tmp = vld1q_s8(r2);
            CALC_0(10111213, 14, 0);

            _tmp = vld1q_s8(r3);
            CALC_0(15161718, 19, 0);

            _tmp = vld1q_s8(r4);
            CALC_0(20212223, 24, 0);

            if(last_ic) {
                ${store_func(_sum00, dstptr)}
            } else {
                vst1q_s32(outptr, _sum00);
            }

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
    }
}

#undef CALC_0
#undef CALC_1
#undef CALC_2
)";
    std::string init_bias = with_bias ? "vdupq_n_s32(bptr[0])" : "vdupq_n_s32(0)";
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << StringTemplate::StringTemplateArgs()
                    .add("init_bias", init_bias)
                    .add("func_name", func_name)
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("store_func",
                         [=](const std::string& reg, const std::string& dst) {
                             return Int8DirectNchwHelperBase::gen_res_store_code(
                                     reg, dst, *activate_gen.get());
                         })
                    .render(kernel_impl);
    return ss.str();
}

class DotInt8DirectNchwS2Helper : public Int8DirectNchwHelperBase {
public:
    std::string gen_need_copy_padding() const override {
        return R"(
        static inline int need_dst_copy_padding(const size_t ow) {
            return ow % 4;
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
            *OW2 = (OW + 3) & ~3;
            *IH2 = SW * OH + FH - SW;
            *IW2 = SW * *OW2 + FW - SW;
            // Because stride is 2, sometimes IW == IW2+1. Do a max update to
            // handle this case.
            *IH2 = *IH2 > IH ? *IH2 : IH;
            *IW2 = *IW2 > IW ? *IW2 : IW;
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
                return gen_5x5_s2_kern(ctx, with_bias, nonline_mode, func_name);
            default:
                CC_ASSERT(0) << "Unsupport kernel size" << fh;
                return "";
        }
    }
};
}  // namespace

DotInt8Conv5x5S2DirectNCHW::DotInt8Conv5x5S2DirectNCHW()
        : Int8DirectNchwBase(new DotInt8DirectNchwS2Helper) {}

bool DotInt8Conv5x5S2DirectNCHW::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("kernel_h") == 5 &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 2 && ctx->getAttrUInt("dilate_h") == 1 &&
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

    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && bias_ok;
}

std::string DotInt8Conv5x5S2DirectNCHW::GetKernelSymbol(TContext* ctx) const {
    return "Armv7_dot_int8_direct_" + ConvImpl::GetKernelSymbol(ctx);
}
