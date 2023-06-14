#include <sstream>

#include "Arm/ArmCommon/Pooling.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

namespace {
std::string gen_handle_padding_code(TContext* ctx, bool is_max_mode) {
    std::stringstream ss;
    std::string body_temp = R"(
static inline void handle_padding(const int8_t* src, int8_t* ws_sptr, size_t IH, size_t IW, size_t IH2, size_t IW2){
    int8_t padding_value = ${padding_value};
    memset(ws_sptr, padding_value, sizeof(int8_t) * IH2 * IW2 * 4);
    rep(ih, IH) {
        memcpy(
                ws_sptr + (ih + ${pad_h}) * IW2 * 4 + ${pad_w} * 4, src + ih * IW * 4,
                sizeof(int8_t) * IW * 4);
    }
}
)";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add("padding_value", is_max_mode ? "INT8_MIN" : "0")
                    .render(body_temp);
    return ss.str();
}

std::string gen_common_func_signature(std::string mode, size_t window, size_t stride) {
    std::string signature = R"(
static inline void do_${mode}_pooling_${window}x${window}_stride${stride}_int8_nchw44_NEON(const int8_t* sptr, int8_t* dst, size_t IH, size_t IW, size_t OH, size_t OW, size_t PH, size_t PW, size_t IW2)
)";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("mode", mode)
                    .add("window", std::to_string(window))
                    .add("stride", std::to_string(stride))
                    .render(signature);
    return ss.str();
}

std::string gen_max_pooling_2x2_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 2, 1);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123 = vld1q_s8(sptr0);
            int8x16_t src1234 = vld1q_s8(sptr0 + 4);
            int8x16_t max0 = vmaxq_s8(src0123, src1234);

            src0123 = vld1q_s8(sptr1);
            src1234 = vld1q_s8(sptr1 + 4);
            int8x16_t max1 = vmaxq_s8(src0123, src1234);

            int8x16_t max_out = vmaxq_s8(max0, max1);

            vst1q_s8(dptr, max_out);

            sptr0 += 16;
            sptr1 += 16;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);

            int8x8_t max_out = vmax_s8(src001, src101);
#define store(i) *(dptr + i) = MAX(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
            sptr0 += 4;
            sptr1 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_2x2_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 2, 1);
    ss << R"({
    int16_t filter_size = 4;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123, src1234;
            int16x8_t src01, src23, src12, src34;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                      \
    src0123 = vld1q_s8(sptr##i);             \
    src1234 = vld1q_s8(sptr##i + 4);         \
    src01 = vmovl_s8(vget_low_s8(src0123));  \
    src23 = vmovl_s8(vget_high_s8(src0123)); \
    src12 = vmovl_s8(vget_low_s8(src1234));  \
    src34 = vmovl_s8(vget_high_s8(src1234)); \
    sum01 = vaddq_s16(sum01, src01);         \
    sum01 = vaddq_s16(sum01, src12);         \
    sum23 = vaddq_s16(sum23, src23);         \
    sum23 = vaddq_s16(sum23, src34);

            UNROLL_CALL_NOWRAPPER(2, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;

#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 16;
            sptr1 += 16;
            dptr += 16;
#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);
            int16x8_t src00 = vmovl_s8(src001);
            int16x8_t src10 = vmovl_s8(src101);
            int16x8_t max_tmp = vaddq_s16(src00, src10);
#define do_acc(i) \
    int16_t sum##i = vgetq_lane_s16(max_tmp, i) + vgetq_lane_s16(max_tmp, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);
            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef do_avg
#undef do_acc
            sptr0 += 4;
            sptr1 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_2x2_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 2, 2);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src04 = vld1q_s8(sptr0 + 4 * 4);
            int32x4x2_t src_tmp =
                    vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04));
            int32x4_t src0246 = src_tmp.val[0];
            int32x4_t src1357 = src_tmp.val[1];
            int8x16_t max0 = vmaxq_s8(
                    vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357));

            src00 = vld1q_s8(sptr1);
            src04 = vld1q_s8(sptr1 + 4 * 4);
            src_tmp =
                    vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04));
            src0246 = src_tmp.val[0];
            src1357 = src_tmp.val[1];
            int8x16_t max1 = vmaxq_s8(
                    vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357));

            int8x16_t max_out = vmaxq_s8(max0, max1);

            vst1q_s8(dptr, max_out);

            sptr0 += 32;
            sptr1 += 32;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);

            int8x8_t max_out = vmax_s8(src001, src101);
#define store(i) *(dptr + i) = MAX(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
            sptr0 += 8;
            sptr1 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_2x2_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 2, 2);
    ss << R"({
    int16_t filter_size = 4;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t src_tmp;
            int8x16_t src00, src04;
            int32x4_t src0246, src1357;
            int16x8_t src02, src13, src46, src57;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                                            \
    src00 = vld1q_s8(sptr##i);                                                     \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                             \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04)); \
    src0246 = src_tmp.val[0];                                                      \
    src1357 = src_tmp.val[1];                                                      \
    src02 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src0246)));                  \
    src46 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src0246)));                 \
    src13 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src1357)));                  \
    src57 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src1357)));                 \
    sum01 = vaddq_s16(sum01, src02);                                               \
    sum01 = vaddq_s16(sum01, src13);                                               \
    sum23 = vaddq_s16(sum23, src46);                                               \
    sum23 = vaddq_s16(sum23, src57);

            UNROLL_CALL_NOWRAPPER(2, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 32;
            sptr1 += 32;
            dptr += 16;
#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);
            int16x8_t src00 = vmovl_s8(src001);
            int16x8_t src10 = vmovl_s8(src101);
            int16x8_t max_tmp = vaddq_s16(src00, src10);

#define do_acc(i) \
    int16_t sum##i = vgetq_lane_s16(max_tmp, i) + vgetq_lane_s16(max_tmp, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);
            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef do_avg
#undef do_acc
#undef store
            sptr0 += 8;
            sptr1 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_3x3_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 3, 1);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123 = vld1q_s8(sptr0);
            int8x16_t src1234 = vld1q_s8(sptr0 + 4);
            int8x16_t src2345 = vld1q_s8(sptr0 + 8);
            int8x16_t max0 = vmaxq_s8(src0123, src1234);
            max0 = vmaxq_s8(max0, src2345);

            src0123 = vld1q_s8(sptr1);
            src1234 = vld1q_s8(sptr1 + 4);
            src2345 = vld1q_s8(sptr1 + 8);
            int8x16_t max1 = vmaxq_s8(src0123, src1234);
            max1 = vmaxq_s8(max1, src2345);

            src0123 = vld1q_s8(sptr2);
            src1234 = vld1q_s8(sptr2 + 4);
            src2345 = vld1q_s8(sptr2 + 8);
            int8x16_t max2 = vmaxq_s8(src0123, src1234);
            max2 = vmaxq_s8(max2, src2345);

            int8x16_t max_out = vmaxq_s8(max0, max1);
            max_out = vmaxq_s8(max_out, max2);

            vst1q_s8(dptr, max_out);

            sptr0 += 16;
            sptr1 += 16;
            sptr2 += 16;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src012 = vld1_s8(sptr0 + 4);

            int8x8_t src101 = vld1_s8(sptr1);
            int8x8_t src112 = vld1_s8(sptr1 + 4);

            int8x8_t src201 = vld1_s8(sptr2);
            int8x8_t src212 = vld1_s8(sptr2 + 4);
            int8x8_t max01_tmp = vmax_s8(src001, src101);
            max01_tmp = vmax_s8(max01_tmp, src201);

            int8x8_t max12_tmp = vmax_s8(src012, src112);
            max12_tmp = vmax_s8(max12_tmp, src212);
#define cb(i)       \
    int8_t dst##i = \
            MAX(MAX(max01_tmp[i], max01_tmp[i + 4]), max12_tmp[i + 4]);
#define store(i) *(dptr + i) = dst##i;
            UNROLL_CALL_NOWRAPPER(4, cb)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef cb
            sptr0 += 4;
            sptr1 += 4;
            sptr2 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_3x3_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 3, 1);
    ss << R"({
    int16_t filter_size = 9;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123, src1234, src2345;
            int16x8_t src01, src23, src12, src34, src45;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                      \
    src0123 = vld1q_s8(sptr##i);             \
    src1234 = vld1q_s8(sptr##i + 4);         \
    src2345 = vld1q_s8(sptr##i + 8);         \
    src01 = vmovl_s8(vget_low_s8(src0123));  \
    src23 = vmovl_s8(vget_high_s8(src0123)); \
    src12 = vmovl_s8(vget_low_s8(src1234));  \
    src34 = vmovl_s8(vget_high_s8(src1234)); \
    src45 = vmovl_s8(vget_high_s8(src2345)); \
    sum01 = vaddq_s16(sum01, src01);         \
    sum01 = vaddq_s16(sum01, src12);         \
    sum01 = vaddq_s16(sum01, src23);         \
    sum23 = vaddq_s16(sum23, src23);         \
    sum23 = vaddq_s16(sum23, src34);         \
    sum23 = vaddq_s16(sum23, src45);

            UNROLL_CALL_NOWRAPPER(3, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 16;
            sptr1 += 16;
            sptr2 += 16;
            dptr += 16;
#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001, src012;
            int16x8_t src01, src12, sum01, sum02;
            sum01 = vdupq_n_s16(0);
            sum02 = vdupq_n_s16(0);

#define CACULATE_ROW(i)              \
    src001 = vld1_s8(sptr##i);       \
    src012 = vld1_s8(sptr##i + 4);   \
    src01 = vmovl_s8(src001);        \
    src12 = vmovl_s8(src012);        \
    sum01 = vaddq_s16(sum01, src01); \
    sum02 = vaddq_s16(sum02, src12);

            UNROLL_CALL_NOWRAPPER(3, CACULATE_ROW)

#define do_acc(i)                                                              \
    int16_t sum##i = vgetq_lane_s16(sum01, i) + vgetq_lane_s16(sum01, i + 4) + \
                     vgetq_lane_s16(sum02, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);
            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef do_avg
#undef do_acc
#undef CACULATE_ROW
            sptr0 += 4;
            sptr1 += 4;
            sptr2 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_3x3_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 3, 2);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src04 = vld1q_s8(sptr0 + 4 * 4);
            int32x4_t src08 =
                    vld1q_dup_s32((const int32_t*)(sptr0 + 4 * 8));
            int32x4x2_t src_tmp =
                    vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04));
            int32x4_t src0246 = src_tmp.val[0];
            int32x4_t src1357 = src_tmp.val[1];
            int32x4_t src2468 = vextq_s32(src0246, src08, 1);
            int8x16_t max_tmp = vmaxq_s8(
                    vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357));
            int8x16_t max0 = vmaxq_s8(max_tmp, vreinterpretq_s8_s32(src2468));

            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src14 = vld1q_s8(sptr1 + 4 * 4);
            int32x4_t src18 =
                    vld1q_dup_s32((const int32_t*)(sptr1 + 4 * 8));

            src_tmp =
                    vuzpq_s32(vreinterpretq_s32_s8(src10), vreinterpretq_s32_s8(src14));
            src0246 = src_tmp.val[0];
            src1357 = src_tmp.val[1];
            src2468 = vextq_s32(src0246, src18, 1);
            max_tmp = vmaxq_s8(
                    vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357));
            int8x16_t max1 = vmaxq_s8(max_tmp, vreinterpretq_s8_s32(src2468));

            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src24 = vld1q_s8(sptr2 + 4 * 4);
            int32x4_t src28 =
                    vld1q_dup_s32((const int32_t*)(sptr2 + 4 * 8));

            src_tmp =
                    vuzpq_s32(vreinterpretq_s32_s8(src20), vreinterpretq_s32_s8(src24));
            src0246 = src_tmp.val[0];
            src1357 = src_tmp.val[1];
            src2468 = vextq_s32(src0246, src28, 1);

            max_tmp = vmaxq_s8(
                    vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357));
            int8x16_t max2 = vmaxq_s8(max_tmp, vreinterpretq_s8_s32(src2468));
            max_tmp = vmaxq_s8(max0, max1);
            int8x16_t max_out = vmaxq_s8(max_tmp, max2);

            vst1q_s8(dptr, max_out);

            sptr0 += 32;
            sptr1 += 32;
            sptr2 += 32;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src012 = vld1_s8(sptr0 + 4);

            int8x8_t src101 = vld1_s8(sptr1);
            int8x8_t src112 = vld1_s8(sptr1 + 4);

            int8x8_t src201 = vld1_s8(sptr2);
            int8x8_t src212 = vld1_s8(sptr2 + 4);
            int8x8_t max01_tmp = vmax_s8(src001, src101);
            max01_tmp = vmax_s8(max01_tmp, src201);

            int8x8_t max12_tmp = vmax_s8(src012, src112);
            max12_tmp = vmax_s8(max12_tmp, src212);
#define cb(i)       \
    int8_t dst##i = \
            MAX(MAX(max01_tmp[i], max01_tmp[i + 4]), max12_tmp[i + 4]);
#define store(i) *(dptr + i) = dst##i;
            UNROLL_CALL_NOWRAPPER(4, cb)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef cb
            sptr0 += 8;
            sptr1 += 8;
            sptr2 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_3x3_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 3, 2);
    ss << R"({
    int16_t filter_size = 9;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t src_tmp;
            int8x16_t src00, src04;
            int32x4_t src0246, src1357, src2468, src08;
            int16x8_t src02, src46, src13, src57, src24, src68;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                                            \
    src00 = vld1q_s8(sptr##i);                                                     \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                             \
    src08 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 8));      \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04)); \
    src0246 = src_tmp.val[0];                                                      \
    src1357 = src_tmp.val[1];                                                      \
    src2468 = vextq_s32(src0246, src08, 1);                                        \
    src02 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src0246)));                  \
    src46 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src0246)));                 \
    src13 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src1357)));                  \
    src57 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src1357)));                 \
    src24 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src2468)));                  \
    src68 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src2468)));                 \
    sum01 = vaddq_s16(sum01, src02);                                               \
    sum01 = vaddq_s16(sum01, src13);                                               \
    sum01 = vaddq_s16(sum01, src24);                                               \
    sum23 = vaddq_s16(sum23, src46);                                               \
    sum23 = vaddq_s16(sum23, src57);                                               \
    sum23 = vaddq_s16(sum23, src68);

            UNROLL_CALL_NOWRAPPER(3, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 32;
            sptr1 += 32;
            sptr2 += 32;
            dptr += 16;
#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001, src012;
            int16x8_t src01, src12, sum01, sum02;
            sum01 = vdupq_n_s16(0);
            sum02 = vdupq_n_s16(0);

#define CACULATE_ROW(i)              \
    src001 = vld1_s8(sptr##i);       \
    src012 = vld1_s8(sptr##i + 4);   \
    src01 = vmovl_s8(src001);        \
    src12 = vmovl_s8(src012);        \
    sum01 = vaddq_s16(sum01, src01); \
    sum02 = vaddq_s16(sum02, src12);

            UNROLL_CALL_NOWRAPPER(3, CACULATE_ROW)

#define do_acc(i)                                                              \
    int16_t sum##i = vgetq_lane_s16(sum01, i) + vgetq_lane_s16(sum01, i + 4) + \
                     vgetq_lane_s16(sum02, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef do_avg
#undef do_acc
            sptr0 += 8;
            sptr1 += 8;
            sptr2 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_4x4_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 4, 1);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00, src04, max_out, max_tmp0, max_tmp1, max_tmp2, max_tmp3;
            int32x4_t src1234, src2345, src3456;

#define CACULATE_ROW(i)                                                               \
    src00 = vld1q_s8(sptr##i);                                                        \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                                \
    src1234 = vextq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04), 1); \
    src2345 = vextq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04), 2); \
    src3456 = vextq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04), 3); \
    max_tmp##i = vmaxq_s8(src00, vreinterpretq_s8_s32(src1234));                      \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src2345));                 \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src3456));

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)
            max_out = vmaxq_s8(max_tmp0, max_tmp1);
            max_out = vmaxq_s8(max_out, max_tmp2);
            max_out = vmaxq_s8(max_out, max_tmp3);

            vst1q_s8(dptr, max_out);

            sptr0 += 16;
            sptr1 += 16;
            sptr2 += 16;
            sptr3 += 16;
            dptr += 16;
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src01, src23, max_out;

#define CACULATE_ROW(i)           \
    src01 = vld1_s8(sptr##i);     \
    src23 = vld1_s8(sptr##i + 8); \
    int8x8_t max_tmp##i = vmax_s8(src01, src23);

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

            max_out = vmax_s8(max_tmp0, max_tmp1);
            max_out = vmax_s8(max_out, max_tmp2);
            max_out = vmax_s8(max_out, max_tmp3);

#define store(i) *(dptr + i) = MAX(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef CACULATE_ROW
            sptr0 += 4;
            sptr1 += 4;
            sptr2 += 4;
            sptr3 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_4x4_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 4, 1);
    ss << R"({
    int16_t filter_size = 16;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int16x8_t src01, src23, src12, src34, src45, src56;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                      \
    src01 = vmovl_s8(vld1_s8(sptr##i));      \
    src23 = vmovl_s8(vld1_s8(sptr##i + 8));  \
    src12 = vmovl_s8(vld1_s8(sptr##i + 4));  \
    src34 = vmovl_s8(vld1_s8(sptr##i + 12)); \
    src45 = vmovl_s8(vld1_s8(sptr##i + 16)); \
    src56 = vmovl_s8(vld1_s8(sptr##i + 20)); \
    sum01 = vaddq_s16(sum01, src01);         \
    sum01 = vaddq_s16(sum01, src12);         \
    sum01 = vaddq_s16(sum01, src23);         \
    sum01 = vaddq_s16(sum01, src34);         \
    sum23 = vaddq_s16(sum23, src23);         \
    sum23 = vaddq_s16(sum23, src34);         \
    sum23 = vaddq_s16(sum23, src45);         \
    sum23 = vaddq_s16(sum23, src56);

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 16;
            sptr1 += 16;
            sptr2 += 16;
            sptr3 += 16;
            dptr += 16;

#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int16x8_t src01, src23, sum01;
            sum01 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                     \
    src01 = vmovl_s8(vld1_s8(sptr##i));     \
    src23 = vmovl_s8(vld1_s8(sptr##i + 8)); \
    sum01 = vaddq_s16(sum01, src01);        \
    sum01 = vaddq_s16(sum01, src23);

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

#define do_acc(i) \
    int16_t sum##i = vgetq_lane_s16(sum01, i) + vgetq_lane_s16(sum01, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)

#undef store
#undef do_avg
#undef do_acc
#undef CACULATE_ROW

            sptr0 += 4;
            sptr1 += 4;
            sptr2 += 4;
            sptr3 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_4x4_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 4, 2);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00, src04, max_tmp0, max_tmp1, max_tmp2, max_tmp3;
            int32x4_t src0246, src1357, src2468, src3579, src08, src09;
            int32x4x2_t src_tmp;
#define CACULATE_ROW(i)                                                             \
    src00 = vld1q_s8(sptr##i);                                                      \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                              \
    src08 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 8));       \
    src09 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 9));       \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04));  \
    src0246 = src_tmp.val[0];                                                       \
    src1357 = src_tmp.val[1];                                                       \
    src2468 = vextq_s32(src0246, src08, 1);                                         \
    src3579 = vextq_s32(src1357, src09, 1);                                         \
    max_tmp##i =                                                                    \
            vmaxq_s8(vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357)); \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src2468));               \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src3579));

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

            int8x16_t max_out = vmaxq_s8(max_tmp0, max_tmp1);
            max_out = vmaxq_s8(max_out, max_tmp2);
            max_out = vmaxq_s8(max_out, max_tmp3);

            vst1q_s8(dptr, max_out);

            sptr0 += 32;
            sptr1 += 32;
            sptr2 += 32;
            sptr3 += 32;
            dptr += 16;
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src01, src23, max_out;

#define CACULATE_ROW(i)           \
    src01 = vld1_s8(sptr##i);     \
    src23 = vld1_s8(sptr##i + 8); \
    int8x8_t max_tmp##i = vmax_s8(src01, src23);

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

            max_out = vmax_s8(max_tmp0, max_tmp1);
            max_out = vmax_s8(max_out, max_tmp2);
            max_out = vmax_s8(max_out, max_tmp3);

#define store(i) *(dptr + i) = MAX(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef CACULATE_ROW
            sptr0 += 8;
            sptr1 += 8;
            sptr2 += 8;
            sptr3 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_4x4_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 4, 2);
    ss << R"({
    int16_t filter_size = 16;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t src_tmp;
            int8x16_t src00, src04;
            int16x8_t src02, src13, src57, src24, src68, src35, src79, src46;
            int32x4_t src08, src09, src0246, src1357, src2468, src3579;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                                            \
    src00 = vld1q_s8(sptr##i);                                                     \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                             \
    src08 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 8));      \
    src09 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 9));      \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04)); \
    src0246 = src_tmp.val[0];                                                      \
    src1357 = src_tmp.val[1];                                                      \
    src2468 = vextq_s32(src0246, src08, 1);                                        \
    src3579 = vextq_s32(src1357, src09, 1);                                        \
    src02 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src0246)));                  \
    src46 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src0246)));                 \
    src13 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src1357)));                  \
    src57 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src1357)));                 \
    src24 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src2468)));                  \
    src68 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src2468)));                 \
    src35 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src3579)));                  \
    src79 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src3579)));                 \
    sum01 = vaddq_s16(sum01, src02);                                               \
    sum01 = vaddq_s16(sum01, src13);                                               \
    sum01 = vaddq_s16(sum01, src24);                                               \
    sum01 = vaddq_s16(sum01, src35);                                               \
    sum23 = vaddq_s16(sum23, src46);                                               \
    sum23 = vaddq_s16(sum23, src57);                                               \
    sum23 = vaddq_s16(sum23, src68);                                               \
    sum23 = vaddq_s16(sum23, src79);

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 32;
            sptr1 += 32;
            sptr2 += 32;
            sptr3 += 32;
            dptr += 16;

#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001, src023;
            int16x8_t src01, src23, sum01;
            sum01 = vdupq_n_s16(0);

#define CACULATE_ROW(i)              \
    src001 = vld1_s8(sptr##i);       \
    src023 = vld1_s8(sptr##i + 8);   \
    src01 = vmovl_s8(src001);        \
    src23 = vmovl_s8(src023);        \
    sum01 = vaddq_s16(sum01, src01); \
    sum01 = vaddq_s16(sum01, src23);

            UNROLL_CALL_NOWRAPPER(4, CACULATE_ROW)

#define do_acc(i) \
    int16_t sum##i = vgetq_lane_s16(sum01, i) + vgetq_lane_s16(sum01, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)

#undef store
#undef do_avg
#undef do_acc
#undef CACULATE_ROW
            sptr0 += 8;
            sptr1 += 8;
            sptr2 += 8;
            sptr3 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_5x5_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 5, 1);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        const int8_t* restrict sptr4 = sptr + (ih + 4) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00, src04, max_out, max_tmp0, max_tmp1, max_tmp2, max_tmp3,
                    max_tmp4;
            int32x4_t src1234, src2345, src3456;

#define CACULATE_ROW(i)                                                               \
    src00 = vld1q_s8(sptr##i);                                                        \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                                \
    src1234 = vextq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04), 1); \
    src2345 = vextq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04), 2); \
    src3456 = vextq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04), 3); \
    max_tmp##i = vmaxq_s8(src00, vreinterpretq_s8_s32(src1234));                      \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src2345));                 \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src3456));                 \
    max_tmp##i = vmaxq_s8(max_tmp##i, src04);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)
            max_out = vmaxq_s8(max_tmp0, max_tmp1);
            max_out = vmaxq_s8(max_out, max_tmp2);
            max_out = vmaxq_s8(max_out, max_tmp3);
            max_out = vmaxq_s8(max_out, max_tmp4);

            vst1q_s8(dptr, max_out);

            sptr0 += 16;
            sptr1 += 16;
            sptr2 += 16;
            sptr3 += 16;
            sptr4 += 16;
            dptr += 16;
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src01, src23, max_out;

#define CACULATE_ROW(i)           \
    src01 = vld1_s8(sptr##i);     \
    src23 = vld1_s8(sptr##i + 8); \
    int8x8_t max_tmp##i = vmax_s8(src01, src23);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

            max_out = vmax_s8(max_tmp0, max_tmp1);
            max_out = vmax_s8(max_out, max_tmp2);
            max_out = vmax_s8(max_out, max_tmp3);
            max_out = vmax_s8(max_out, max_tmp4);

#define COMPARE_SRC45(i)    \
    int32x2_t src##i##_45 = \
            vld1_dup_s32((const int32_t*)(sptr##i + 4 * 4));
            UNROLL_CALL_NOWRAPPER(5, COMPARE_SRC45)
            int8x8_t max_45 =
                    vmax_s8(vreinterpret_s8_s32(src0_45), vreinterpret_s8_s32(src1_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src1_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src2_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src3_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src4_45));

#define store(i) \
    *(dptr + i) = MAX(MAX(max_out[i], max_out[i + 4]), max_45[i]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef COMPARE_SRC45
#undef CACULATE_ROW
            sptr0 += 4;
            sptr1 += 4;
            sptr2 += 4;
            sptr3 += 4;
            sptr4 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_5x5_stride1_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 5, 1);
    ss << R"({
    int16_t filter_size = 25;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        const int8_t* restrict sptr4 = sptr + (ih + 4) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int16x8_t src01, src23, src12, src34, src45, src56, src67;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                      \
    src01 = vmovl_s8(vld1_s8(sptr##i));      \
    src23 = vmovl_s8(vld1_s8(sptr##i + 8));  \
    src12 = vmovl_s8(vld1_s8(sptr##i + 4));  \
    src34 = vmovl_s8(vld1_s8(sptr##i + 12)); \
    src45 = vmovl_s8(vld1_s8(sptr##i + 16)); \
    src56 = vmovl_s8(vld1_s8(sptr##i + 20)); \
    src67 = vmovl_s8(vld1_s8(sptr##i + 24)); \
    sum01 = vaddq_s16(sum01, src01);         \
    sum01 = vaddq_s16(sum01, src12);         \
    sum01 = vaddq_s16(sum01, src23);         \
    sum01 = vaddq_s16(sum01, src34);         \
    sum01 = vaddq_s16(sum01, src45);         \
    sum23 = vaddq_s16(sum23, src23);         \
    sum23 = vaddq_s16(sum23, src34);         \
    sum23 = vaddq_s16(sum23, src45);         \
    sum23 = vaddq_s16(sum23, src56);         \
    sum23 = vaddq_s16(sum23, src67);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 16;
            sptr1 += 16;
            sptr2 += 16;
            sptr3 += 16;
            sptr4 += 16;
            dptr += 16;

#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int32x2_t src004;
            int8x8_t src001, src023;
            int16x8_t src01, src23, src04, sum01, sum02;
            sum01 = vdupq_n_s16(0);
            sum02 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                                       \
    src001 = vld1_s8(sptr##i);                                                \
    src023 = vld1_s8(sptr##i + 8);                                            \
    src004 = vld1_dup_s32((const int32_t*)(sptr##i + 4 * 4)); \
    src01 = vmovl_s8(src001);                                                 \
    src23 = vmovl_s8(src023);                                                 \
    src04 = vmovl_s8(vreinterpret_s8_s32(src004));                            \
    sum01 = vaddq_s16(sum01, src01);                                          \
    sum01 = vaddq_s16(sum01, src23);                                          \
    sum02 = vaddq_s16(sum02, src04);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

#define do_acc(i)                                                              \
    int16_t sum##i = vgetq_lane_s16(sum01, i) + vgetq_lane_s16(sum01, i + 4) + \
                     vgetq_lane_s16(sum02, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)

#undef store
#undef do_avg
#undef do_acc
#undef CACULATE_ROW
            sptr0 += 4;
            sptr1 += 4;
            sptr2 += 4;
            sptr3 += 4;
            sptr4 += 4;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_max_pooling_5x5_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("max", 5, 2);
    ss << R"({
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        const int8_t* restrict sptr4 = sptr + (ih + 4) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00, src04, max_tmp0, max_tmp1, max_tmp2, max_tmp3, max_tmp4;
            int32x4_t src0246, src1357, src2468, src3579, src46810, src10, src09, src08;
            int32x4x2_t src_tmp;
#define CACULATE_ROW(i)                                                             \
    src00 = vld1q_s8(sptr##i);                                                      \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                              \
    src08 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 8));       \
    src09 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 9));       \
    src10 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 10));      \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04));  \
    src0246 = src_tmp.val[0];                                                       \
    src1357 = src_tmp.val[1];                                                       \
    src2468 = vextq_s32(src0246, src08, 1);                                         \
    src3579 = vextq_s32(src1357, src09, 1);                                         \
    src46810 = vextq_s32(src2468, src10, 1);                                        \
    max_tmp##i =                                                                    \
            vmaxq_s8(vreinterpretq_s8_s32(src0246), vreinterpretq_s8_s32(src1357)); \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src2468));               \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src3579));               \
    max_tmp##i = vmaxq_s8(max_tmp##i, vreinterpretq_s8_s32(src46810));

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

            int8x16_t max_out = vmaxq_s8(max_tmp0, max_tmp1);
            max_out = vmaxq_s8(max_out, max_tmp2);
            max_out = vmaxq_s8(max_out, max_tmp3);
            max_out = vmaxq_s8(max_out, max_tmp4);

            vst1q_s8(dptr, max_out);

            sptr0 += 32;
            sptr1 += 32;
            sptr2 += 32;
            sptr3 += 32;
            sptr4 += 32;
            dptr += 16;
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src01, src23, max_out;

#define CACULATE_ROW(i)           \
    src01 = vld1_s8(sptr##i);     \
    src23 = vld1_s8(sptr##i + 8); \
    int8x8_t max_tmp##i = vmax_s8(src01, src23);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

            max_out = vmax_s8(max_tmp0, max_tmp1);
            max_out = vmax_s8(max_out, max_tmp2);
            max_out = vmax_s8(max_out, max_tmp3);
            max_out = vmax_s8(max_out, max_tmp4);

#define COMPARE_SRC45(i)    \
    int32x2_t src##i##_45 = \
            vld1_dup_s32((const int32_t*)(sptr##i + 4 * 4));
            UNROLL_CALL_NOWRAPPER(5, COMPARE_SRC45)
            int8x8_t max_45 =
                    vmax_s8(vreinterpret_s8_s32(src0_45), vreinterpret_s8_s32(src1_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src1_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src2_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src3_45));
            max_45 = vmax_s8(max_45, vreinterpret_s8_s32(src4_45));

#define store(i) \
    *(dptr + i) = MAX(MAX(max_out[i], max_out[i + 4]), max_45[i]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef COMPARE_SRC45
#undef CACULATE_ROW
            sptr0 += 8;
            sptr1 += 8;
            sptr2 += 8;
            sptr3 += 8;
            sptr4 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

std::string gen_avg_pooling_5x5_stride2_code() {
    std::stringstream ss;
    ss << gen_common_func_signature("avg", 5, 2);
    ss << R"({
    int16_t filter_size = 25;
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        const int8_t* restrict sptr2 = sptr + (ih + 2) * IW2 * 4;
        const int8_t* restrict sptr3 = sptr + (ih + 3) * IW2 * 4;
        const int8_t* restrict sptr4 = sptr + (ih + 4) * IW2 * 4;
        int8_t* restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t src_tmp;
            int8x16_t src00, src04;
            int16x8_t src02, src13, src57, src24, src68, src35, src79, src46, src810;
            int32x4_t src08, src09, src10, src0246, src1357, src2468, src3579, src46810;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                                            \
    src00 = vld1q_s8(sptr##i);                                                     \
    src04 = vld1q_s8(sptr##i + 4 * 4);                                             \
    src08 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 8));      \
    src09 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 9));      \
    src10 = vld1q_dup_s32((const int32_t*)(sptr##i + 4 * 10));     \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00), vreinterpretq_s32_s8(src04)); \
    src0246 = src_tmp.val[0];                                                      \
    src1357 = src_tmp.val[1];                                                      \
    src2468 = vextq_s32(src0246, src08, 1);                                        \
    src3579 = vextq_s32(src1357, src09, 1);                                        \
    src46810 = vextq_s32(src2468, src10, 1);                                       \
    src02 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src0246)));                  \
    src46 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src0246)));                 \
    src13 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src1357)));                  \
    src57 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src1357)));                 \
    src24 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src2468)));                  \
    src68 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src2468)));                 \
    src35 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src3579)));                  \
    src79 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src3579)));                 \
    src46 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src46810)));                 \
    src810 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src46810)));               \
    sum01 = vaddq_s16(sum01, src02);                                               \
    sum01 = vaddq_s16(sum01, src13);                                               \
    sum01 = vaddq_s16(sum01, src24);                                               \
    sum01 = vaddq_s16(sum01, src35);                                               \
    sum01 = vaddq_s16(sum01, src46);                                               \
    sum23 = vaddq_s16(sum23, src46);                                               \
    sum23 = vaddq_s16(sum23, src57);                                               \
    sum23 = vaddq_s16(sum23, src68);                                               \
    sum23 = vaddq_s16(sum23, src79);                                               \
    sum23 = vaddq_s16(sum23, src810);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum01, i) > 0                                       \
                   ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum01, i) - filter_size / 2) / filter_size;
#define sum23_avg(i)                                                            \
    sum##i = vgetq_lane_s16(sum23, i) > 0                                       \
                   ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / filter_size \
                   : (vgetq_lane_s16(sum23, i) - filter_size / 2) / filter_size;
#define store_sum01(i) *(dptr + i) = (int8_t)(sum##i);
#define store_sum23(i) *(dptr + i + 8) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 32;
            sptr1 += 32;
            sptr2 += 32;
            sptr3 += 32;
            sptr4 += 32;
            dptr += 16;

#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int32x2_t src004;
            int8x8_t src001, src023;
            int16x8_t src01, src23, src04, sum01, sum02;
            sum01 = vdupq_n_s16(0);
            sum02 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                                       \
    src001 = vld1_s8(sptr##i);                                                \
    src023 = vld1_s8(sptr##i + 8);                                            \
    src004 = vld1_dup_s32((const int32_t*)(sptr##i + 4 * 4)); \
    src01 = vmovl_s8(src001);                                                 \
    src23 = vmovl_s8(src023);                                                 \
    src04 = vmovl_s8(vreinterpret_s8_s32(src004));                            \
    sum01 = vaddq_s16(sum01, src01);                                          \
    sum01 = vaddq_s16(sum01, src23);                                          \
    sum02 = vaddq_s16(sum02, src04);

            UNROLL_CALL_NOWRAPPER(5, CACULATE_ROW)

#define do_acc(i)                                                              \
    int16_t sum##i = vgetq_lane_s16(sum01, i) + vgetq_lane_s16(sum01, i + 4) + \
                     vgetq_lane_s16(sum02, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = (int8_t)(sum##i);

            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)

#undef store
#undef do_avg
#undef do_acc
#undef CACULATE_ROW
            sptr0 += 8;
            sptr1 += 8;
            sptr2 += 8;
            sptr3 += 8;
            sptr4 += 8;
            dptr += 4;
        }
    }
}
)";
    return ss.str();
}

}  // namespace

bool PoolingNchw44Int8::IsAvailable(TContext* context) const {
    bool format_ok = context->getAttrStr("format") == "NCHW44";
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto dst_dtype = context->getAttrOprand("operand:1").dtype;
    bool dtype_ok =
            (src_dtype == dst_dtype) && (src_dtype == "i8" || src_dtype == "si8" ||
                                         Utils::is_quant_dtype(src_dtype, 8));
    bool mode_ok = (context->getAttrStr("mode") == "MAX" ||
                    context->getAttrStr("mode") == "AVERAGE") &&
                   !(context->getAttrStr("mode") == "AVERAGE" &&
                     (src_dtype == "i8" || src_dtype == "si8"));
    if (Utils::is_quant_dtype(src_dtype, 8)) {
        CC_ASSERT(
                context->getAttrOprand("operand:0").scale ==
                context->getAttrOprand("operand:1").scale)
                << "quant pooling only support same scale\n";
    }
    auto SW = context->getAttrUInt("stride_w");
    auto SH = context->getAttrUInt("stride_h");
    auto FW = context->getAttrUInt("window_w");
    auto FH = context->getAttrUInt("window_h");
    bool param_ok = SH == SW && (SW == 1 || SW == 2) && FH == FW &&
                    (FH == 2 || FH == 3 || FH == 4 || FH == 5);
    return format_ok && dtype_ok && mode_ok && param_ok;
}

std::string PoolingNchw44Int8::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        uint32_t ph = ${pad_h};
        uint32_t pw = ${pad_w};
        size_t padding_size = 0;
        if ((ph != 0) || (pw != 0)) {
            padding_size = (iw + 2 * pw) * (ih + 2 * ph) * 4 * sizeof(int8_t) + 16;
        }
        *workspace = padding_size;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .render(workspace_temp);
    return ss.str();
}

std::string PoolingNchw44Int8::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "ArmCommon_FilterX_modeX_";
    ss << PoolingImpl::GetKernelSymbol(context);
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string PoolingNchw44Int8::GetKernelBody(TContext* context) const {
    auto format_str = context->getAttrStr("format");
    auto mode_str = context->getAttrStr("mode");
    auto mode_type = mode_str == "MAX" ? 0 : 1;
    std::stringstream ss;
    const uint32_t window_h = context->getAttrInt("window_h");
    const uint32_t sw = context->getAttrInt("stride_w");
    const uint32_t ph = context->getAttrInt("pad_h");
    const uint32_t pw = context->getAttrInt("pad_w");
    bool need_pad = ((ph != 0) || (pw != 0)) ? true : false;
    ss << R"(
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include "unroll_macro.h"
    )";
    ss << "\n";
    ss << R"(
#define rep(i, n)            for (size_t i = 0; i < (n); ++i)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
    )";
    ss << "\n";
    if (need_pad) {
        ss << gen_handle_padding_code(context, mode_str == "MAX");
    }

#define DISPATCH_FUNC(window, i, mode) \
    ss << gen_##mode##_pooling_##window##x##window##_stride##i##_code();

#define DISPATCH_MODE(window, stride)                                      \
    switch (mode_type) {                                                   \
        case 0: {                                                          \
            DISPATCH_FUNC(window, stride, max);                            \
            break;                                                         \
        }                                                                  \
        case 1: {                                                          \
            DISPATCH_FUNC(window, stride, avg);                            \
            break;                                                         \
        }                                                                  \
        default:                                                           \
            CC_ABORT << "Unsupported pooling mode: " << mode_str << ".\n"; \
    }

#define DISPATCH_STRIDE(window)                                  \
    switch (sw) {                                                \
        case 1: {                                                \
            DISPATCH_MODE(window, 1);                            \
            break;                                               \
        }                                                        \
        case 2: {                                                \
            DISPATCH_MODE(window, 2);                            \
            break;                                               \
        }                                                        \
        default:                                                 \
            CC_ABORT << "Unsupport stride size:" << sw << ".\n"; \
    }

    switch (window_h) {
        case 2:
            DISPATCH_STRIDE(2);
            break;
        case 3:
            DISPATCH_STRIDE(3);
            break;
        case 4:
            DISPATCH_STRIDE(4);
            break;
        case 5:
            DISPATCH_STRIDE(5);
            break;
        default:
            CC_ABORT << "Unsupported pooling window size: " << window_h << ".\n";
    }

#undef DISPATCH_STRIDE
#undef DISPATCH_MODE
#undef DISPATCH_FUNC

    ss << GenCommonRet() << " " << GetKernelSignature(context) << "{\n";
    std::string body_temp = R"(
    const uint32_t window_h = ${window_h};
    const uint32_t window_w = ${window_w};
    const uint32_t ph = ${pad_h};
    const uint32_t pw = ${pad_w};
    int8_t* input_data = (int8_t*)inputs[0]->ptr;
    TINYNN_ASSERT(input_data);
    int8_t* workspace_ptr = (int8_t*)workspace->ptr;
    TINYNN_ASSERT(workspace_ptr);
    int8_t* output_data = (int8_t*)outputs[0]->ptr;
    TINYNN_ASSERT(output_data);
    const Layout src_layout = inputs[0]->layout;
    const Layout dst_layout = outputs[0]->layout;
    const uint32_t batch = src_layout.dims[0];
    const uint32_t icb = src_layout.dims[1];
    const uint32_t ih = src_layout.dims[2];
    const uint32_t iw = src_layout.dims[3];
    const uint32_t oh = dst_layout.dims[2];
    const uint32_t ow = dst_layout.dims[3];

    size_t IH2, IW2;
    ${get_padded_size_func}
    for(uint32_t n_idx=0; n_idx < batch; ++n_idx){
        for(uint32_t c_idx = 0; c_idx < icb; c_idx++){
            int8_t* input_ptr = input_data + n_idx * icb * ih * iw * 4 + c_idx * ih * iw * 4;
            int8_t* output_ptr = output_data + n_idx * icb * oh * ow * 4 + c_idx * oh * ow * 4;
            ${do_pad_func}
            do_${mode_str}_pooling_${window_h}x${window_w}_stride${stride_h}_int8_nchw44_NEON(${target_inptr}, output_ptr, ih, iw, oh, ow, ph, pw, IW2);
        }
    }
    return TinyNN_SUCCESS;
})";
    std::string get_padded_size_func = need_pad ? R"(
        IH2 = ih + 2 * ph;
        IW2 = iw + 2 * pw;
    )"
                                                : R"(
        IH2 = ih;
        IW2 = iw;
    )";
    std::string do_pad_func = need_pad ? R"(
        handle_padding(input_ptr, workspace_ptr, ih, iw, IH2, IW2);
    )"
                                       : "";

    ss << StringTemplate::StringTemplateArgs(context)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("window_h")
                    .add_ctx_int("window_w")
                    .add_ctx_int("stride_h")
                    .add("mode_str", mode_str == "MAX" ? "max" : "avg")
                    .add("do_pad_func", do_pad_func)
                    .add("get_padded_size_func", get_padded_size_func)
                    .add("target_inptr", need_pad ? "workspace_ptr" : "input_ptr")
                    .render(body_temp);

    return ss.str();
}
