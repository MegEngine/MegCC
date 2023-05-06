#include <string>
#include "Arm/Arm64/Activation.h"
#include "Arm/Arm64/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;

namespace {

std::string gen_common_code() {
    std::stringstream ss;
    ss << R"(
#define rep(i, n)            for (int i = 0; i < (n); ++i)
#define UNROLL_RAW1(cb, v0, a...) cb(0, ##a)
#define UNROLL_RAW2(cb, v0, a...) cb(0, ##a) cb(1, ##a)
#define UNROLL_RAW3(cb, v0, a...) UNROLL_RAW2(cb, v0, ##a) cb(2, ##a)
#define UNROLL_RAW4(cb, v0, a...) \
    UNROLL_RAW2(cb, v0, ##a)      \
    cb(2, ##a) cb(3, ##a)
#define UNROLL_RAW5(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a)
#define UNROLL_RAW6(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a) cb(5, ##a)
#define UNROLL_RAW7(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a) cb(5, ##a) cb(6, ##a)
#define UNROLL_RAW8(cb, v0, a...) \
    UNROLL_RAW4(cb, v0, ##a)      \
    cb(4, ##a) cb(5, ##a) cb(6, ##a) cb(7, ##a)
#define UNROLL_RAW9(cb, v0, a...) \
    UNROLL_RAW8(cb, v0, ##a)      \
    cb(8, ##a)
#define UNROLL_RAW10(cb, v0, a...) \
    UNROLL_RAW9(cb, v0, ##a)       \
    cb(9, ##a)
#define UNROLL_RAW11(cb, v0, a...) \
    UNROLL_RAW10(cb, v0, ##a)      \
    cb(10, ##a)
#define UNROLL_RAW12(cb, v0, a...) \
    UNROLL_RAW11(cb, v0, ##a)      \
    cb(11, ##a)
#define UNROLL_RAW13(cb, v0, a...) \
    UNROLL_RAW12(cb, v0, ##a)      \
    cb(12, ##a)
#define UNROLL_RAW14(cb, v0, a...) \
    UNROLL_RAW13(cb, v0, ##a)      \
    cb(13, ##a)
#define UNROLL_RAW15(cb, v0, a...) \
    UNROLL_RAW14(cb, v0, ##a)      \
    cb(14, ##a)

// clang-format off
#define UNROLL_RAW16(cb, v0, a...)                                        \
    UNROLL_RAW8(cb, v0, ##a)                                              \
    cb(8, ##a) cb(9, ##a) cb(10, ##a) cb(11, ##a) cb(12, ##a) cb(13, ##a) \
            cb(14, ##a) cb(15, ##a)
#define UNROLL_RAW17(cb, v0, a...) \
    UNROLL_RAW16(cb, v0, ##a)      \
    cb(16, ##a)
#define UNROLL_RAW24(cb, v0, a...)                                          \
    UNROLL_RAW16(cb, v0, ##a)                                               \
    cb(16, ##a) cb(17, ##a) cb(18, ##a) cb(19, ##a) cb(20, ##a) cb(21, ##a) \
            cb(22, ##a) cb(23, ##a)
#define UNROLL_RAW25(cb, v0, a...) \
    UNROLL_RAW24(cb, v0, ##a)      \
    cb(24, ##a)
#define UNROLL_RAW49(cb, v0, a...)                                          \
    UNROLL_RAW25(cb, v0, ##a)                                               \
    cb(25, ##a) cb(26, ##a) cb(27, ##a) cb(28, ##a) cb(29, ##a) cb(30, ##a) \
    cb(31, ##a) cb(32, ##a) cb(33, ##a) cb(34, ##a) cb(35, ##a) cb(36, ##a) \
    cb(37, ##a) cb(38, ##a) cb(39, ##a) cb(40, ##a) cb(41, ##a) cb(42, ##a) \
    cb(43, ##a) cb(44, ##a) cb(45, ##a) cb(46, ##a) cb(47, ##a) cb(48, ##a)

#define UNROLL_CALL0(step, cb, v...) UNROLL_RAW##step(cb, 0, ##v)
#define UNROLL_CALL1(step, cb, v...) UNROLL_CALL0(step, cb, ##v)
#define UNROLL_CALL(step, cb, v...)  \
    do {                             \
        UNROLL_CALL1(step, cb, ##v); \
    } while (0)

#define UNROLL_CALL_RAW(step, cb, v...) UNROLL_CALL1(step, cb, ##v);
#define UNROLL_CALL_NOWRAPPER(step, cb) UNROLL_CALL_RAW(step, cb)

#define UNROLL_CALL0(step, cb, v...) UNROLL_RAW##step(cb, 0, ##v)
#define UNROLL_CALL1(step, cb, v...) UNROLL_CALL0(step, cb, ##v)
#define UNROLL_CALL(step, cb, v...)  \
    do {                             \
        UNROLL_CALL1(step, cb, ##v); \
    } while (0)
)";

    ss << R"(
static inline void accumulate_2_q_vector(
        int8x16_t src0, int8x16_t kern0, int8x16_t src1, int8x16_t kern1,
        int32x4_t* sum) {
    int16x8_t tmp_sum0 = vmull_s8(vget_low_s8(src0), vget_low_s8(kern0));
    int16x8_t tmp_sum1 = vmull_high_s8(src0, kern0);
    tmp_sum0 = vmlal_s8(tmp_sum0, vget_low_s8(src1), vget_low_s8(kern1));
    tmp_sum1 = vmlal_high_s8(tmp_sum1, src1, kern1);
    sum[0] = vaddw_s16(sum[0], vget_low_s16(tmp_sum0));
    sum[1] = vaddw_s16(sum[1], vget_high_s16(tmp_sum0));
    sum[2] = vaddw_s16(sum[2], vget_low_s16(tmp_sum1));
    sum[3] = vaddw_s16(sum[3], vget_high_s16(tmp_sum1));
}

static inline void accumulate_1_q_vector(
        int8x16_t src0, int8x16_t kern0, int32x4_t* sum) {
    int16x8_t tmp_sum0 = vmull_s8(vget_low_s8(src0), vget_low_s8(kern0));
    int16x8_t tmp_sum1 = vmull_high_s8(src0, kern0);
    sum[0] = vaddw_s16(sum[0], vget_low_s16(tmp_sum0));
    sum[1] = vaddw_s16(sum[1], vget_high_s16(tmp_sum0));
    sum[2] = vaddw_s16(sum[2], vget_low_s16(tmp_sum1));
    sum[3] = vaddw_s16(sum[3], vget_high_s16(tmp_sum1));
}

static inline void accumulate_1_line_horizon(
        const int8x8_t src0, const int8x8_t kern0, const int8x8_t src1,
        const int8x8_t kern1, int32x4_t* sum) {
    int16x8_t tmp_sum = vmull_s8(src0, kern0);
    tmp_sum = vmlal_s8(tmp_sum, src1, kern1);
    *sum = vaddw_s16(*sum, vget_low_s16(tmp_sum));
    *sum = vaddw_s16(*sum, vget_high_s16(tmp_sum));
}

#define ACC_S16_S32(sum, tmp_sum)                \
    sum = vaddw_s16(sum, vget_low_s16(tmp_sum)); \
    sum = vaddw_s16(sum, vget_high_s16(tmp_sum));

    

)";
    return ss.str();
}
std::string render_store_1_line(
        std::string dst, std::string oh, std::string ow, std::string OW,
        std::string reg_name, std::string oh_idx,
        const ActivationGenIntrinsicBase& act) {
    std::stringstream ss;
    std::string store_temp = R"({
        int8_t* store_ptr = ((int8_t*)${dst}) + (${oh} + ${oh_idx}) * ${OW} * 4 + ${ow} * 4;
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("dst", dst)
                    .add("oh", oh)
                    .add("oh_idx", oh_idx)
                    .add("ow", ow)
                    .add("OW", OW)
                    .render(store_temp);
    for (int i = 0; i < 4; ++i) {
        ss << act.GenIntrinsicQuantStore(
                reg_name + "[" + std::to_string(i) + "]",
                "store_ptr + " + std::to_string(i) + " * 4", "scale");
    }
    ss << "\n}";
    return ss.str();
}

std::string render_store_1_line_remain(
        std::string dst, std::string oh, std::string ow, std::string OW,
        std::string reg_name, std::string oh_idx, std::string remain,
        const ActivationGenIntrinsicBase& act) {
    std::stringstream ss;
    std::string store_temp = R"({
        int8_t* store_ptr = ((int8_t*)${dst}) + (${oh} + ${oh_idx}) * ${OW} * 4 + ${ow} * 4;
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("dst", dst)
                    .add("oh", oh)
                    .add("ow", ow)
                    .add("OW", OW)
                    .add("oh_idx", oh_idx)
                    .render(store_temp);
    for (int remain_val = 1; remain_val <= 3; remain_val++) {
        ss << "if (" << remain << "==" << remain_val << "){\n";
        for (int i = 0; i < remain_val; ++i) {
            ss << act.GenIntrinsicQuantStore(
                    reg_name + "[" + std::to_string(i) + "]",
                    "store_ptr + " + std::to_string(i) + " * 4", "scale");
        }
        ss << "\n}";
    }
    ss << "\n}";
    return ss.str();
}

std::string gen_3x3_s1_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    std::string bias_str = with_bias ? "vld1q_s32(bias)" : "vdupq_n_s32(0)";
    std::string body_temp = R"(
static inline void nchw44_chanwise_3x3_int8(const int8_t* sptr, const int8_t* fptr, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW, float scale){
    int32x4_t init_v = ${bias_str};
    const int* filter = (int*)(fptr);
    int8x16_t kern[9];
#define cb(i) kern[i] = (int8x16_t)vld1q_dup_s32(filter + i);
    UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb

#define LOAD_2_LINE_SRC(sptr0, sptr1)              \
    src[0][0] = vld1q_s8(sptr0);                   \
    src[0][2] = vld1q_s8(sptr0 + 16);              \
    src[1][0] = vld1q_s8(sptr1);                   \
    src[1][2] = vld1q_s8(sptr1 + 16);              \
    src[0][1] = vextq_s8(src[0][0], src[0][2], 4); \
    src[1][1] = vextq_s8(src[1][0], src[1][2], 4); \
    src[0][2] = vextq_s8(src[0][0], src[0][2], 8); \
    src[1][2] = vextq_s8(src[1][0], src[1][2], 8);

#define LOAD_1_LINE_SRC(sptr0, src)       \
    src[0] = vld1q_s8(sptr0);             \
    src[2] = vld1q_s8(sptr0 + 16);        \
    src[1] = vextq_s8(src[0], src[2], 4); \
    src[2] = vextq_s8(src[0], src[2], 8);

#define ACC_1_LINE(src, kern0, kern1, kern2, sum)             \
    accumulate_2_q_vector(src[0], kern0, src[1], kern1, sum); \
    accumulate_1_q_vector(src[2], kern2, sum);

    size_t oh = 0;
    for (; oh + 3 <= OH; oh += 3) {
        size_t ih = oh;
        size_t ow = 0;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum0[4], sum1[4], sum2[4];
#define cb(j)         \
    sum0[j] = init_v; \
    sum1[j] = init_v; \
    sum2[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
//! gcc will report error of "more than 30 operands in 'asm'"
#if defined(__clang__)
            asm volatile(
                    //! load src 0,1
                    "ldr q21, [%[sptr0]]\n"
                    "ldr q24, [%[sptr1]]\n"

                    //! sum0 line<0,1>
                    "smull  v27.8h, v21.8b, %[k0].8b\n"
                    "ldr q23, [%[sptr0], #16]\n"
                    "smull2 v28.8h, v21.16b, %[k0].16b\n"
                    "ldr q26, [%[sptr1], #16]\n"
                    "smlal  v27.8h, v24.8b, %[k3].8b\n"
                    "ext v22.16b, v21.16b, v23.16b, #4\n"
                    "smlal2 v28.8h, v24.16b, %[k3].16b\n"
                    "ext v23.16b, v21.16b, v23.16b, #8\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v27.4h\n"
                    "ext v25.16b, v24.16b, v26.16b, #4\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v27.8h\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v28.4h\n"
                    "ext v26.16b, v24.16b, v26.16b, #8\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v28.8h\n"

                    "ldr q21, [%[sptr2]]\n"
                    "smull  v29.8h, v22.8b, %[k1].8b\n"
                    "smull2 v30.8h, v22.16b, %[k1].16b\n"
                    "ldr q31, [%[sptr2], #16]\n"
                    "smull  v27.8h, v23.8b, %[k2].8b\n"
                    "ext v22.16b, v21.16b, v31.16b, #4\n"
                    "smull2 v28.8h, v23.16b, %[k2].16b\n"
                    "ext v23.16b, v21.16b, v31.16b, #8\n"
                    "smlal  v29.8h, v25.8b, %[k4].8b\n"
                    "smlal2 v30.8h, v25.16b, %[k4].16b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v29.4h\n"
                    "smlal  v27.8h, v26.8b, %[k5].8b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v29.8h\n"
                    "smlal2 v28.8h, v26.16b, %[k5].16b\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v30.4h\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v30.8h\n"
                    //! load src 2

                    //! sum0 line<2>
                    "smull  v29.8h, v21.8b, %[k6].8b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v27.4h\n"
                    "smull2 v30.8h, v21.16b, %[k6].16b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v27.8h\n"
                    "smull  v27.8h, v23.8b, %[k8].8b\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v28.4h\n"
                    "smlal  v29.8h, v22.8b, %[k7].8b\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v28.8h\n"
                    "smlal2 v30.8h, v22.16b, %[k7].16b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v29.4h\n"
                    "smull2 v28.8h, v23.16b, %[k8].16b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v29.8h\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v30.4h\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v30.8h\n"

                    //! sum1 line<0,1>
                    "saddw2  %[sum03].4s, %[sum03].4s, v28.8h\n"
                    "smull  v29.8h, v24.8b, %[k0].8b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v27.4h\n"
                    "smull2 v30.8h, v24.16b, %[k0].16b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v27.8h\n"
                    "smull  v27.8h, v25.8b, %[k1].8b\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v28.4h\n"
                    "smull2 v28.8h, v25.16b, %[k1].16b\n"
                    "smlal  v29.8h, v21.8b, %[k3].8b\n"
                    "smlal2 v30.8h, v21.16b, %[k3].16b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v29.4h\n"
                    "smlal  v27.8h, v22.8b, %[k4].8b\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v29.8h\n"
                    "smlal2 v28.8h, v22.16b, %[k4].16b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v30.4h\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v30.8h\n"

                    "ldr q24, [%[sptr3]]\n"
                    "smull  v29.8h, v26.8b, %[k2].8b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v27.4h\n"
                    "smull2 v30.8h, v26.16b, %[k2].16b\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v27.8h\n"
                    "smlal  v29.8h, v23.8b, %[k5].8b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v28.4h\n"
                    "smlal2 v30.8h, v23.16b, %[k5].16b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v28.8h\n"
                    "ldr q26, [%[sptr3], #16]\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v29.4h\n"
                    "ext v25.16b, v24.16b, v26.16b, #4\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v29.8h\n"
                    "ext v26.16b, v24.16b, v26.16b, #8\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v30.4h\n"
                    //! src line 3

                    //! sum1 line<2>
                    "smull  v27.8h, v24.8b, %[k6].8b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v30.8h\n"
                    "smull2 v28.8h, v24.16b, %[k6].16b\n"
                    "smlal  v27.8h, v25.8b, %[k7].8b\n"
                    "smlal2 v28.8h, v25.16b, %[k7].16b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v27.4h\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v27.8h\n"

                    "smull  v29.8h, v26.8b, %[k8].8b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v28.4h\n"
                    "smull2 v30.8h, v26.16b, %[k8].16b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v28.8h\n"

                    //! sum2 line<0,1>
                    "smull  v27.8h, v21.8b, %[k0].8b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v29.4h\n"
                    "smull2 v28.8h, v21.16b, %[k0].16b\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v29.8h\n"
                    "smull  v29.8h, v22.8b, %[k1].8b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v30.4h\n"
                    "smlal  v27.8h, v24.8b, %[k3].8b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v30.8h\n"
                    "smull2 v30.8h, v22.16b, %[k1].16b\n"
                    "ldr q21, [%[sptr4]]\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v27.4h\n"
                    "smlal2 v28.8h, v24.16b, %[k3].16b\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v27.8h\n"
                    "smlal  v29.8h, v25.8b, %[k4].8b\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v28.4h\n"
                    "smlal2 v30.8h, v25.16b, %[k4].16b\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v28.8h\n"

                    "smull  v27.8h, v23.8b, %[k2].8b\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v29.4h\n"
                    "smull2 v28.8h, v23.16b, %[k2].16b\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v29.8h\n"
                    "ldr q23, [%[sptr4], #16]\n"
                    "smlal  v27.8h, v26.8b, %[k5].8b\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v30.4h\n"
                    "smlal2 v28.8h, v26.16b, %[k5].16b\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v30.8h\n"
                    "ext v22.16b, v21.16b, v23.16b, #4\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v27.4h\n"
                    "ext v23.16b, v21.16b, v23.16b, #8\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v27.8h\n"
                    //! src line 3

                    //! sum2 line<2>
                    "smull  v29.8h, v21.8b, %[k6].8b\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v28.4h\n"
                    "smull2 v30.8h, v21.16b, %[k6].16b\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v28.8h\n"
                    "smull  v27.8h, v23.8b, %[k8].8b\n"
                    "smull2 v28.8h, v23.16b, %[k8].16b\n"
                    "smlal  v29.8h, v22.8b, %[k7].8b\n"
                    "smlal2 v30.8h, v22.16b, %[k7].16b\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v29.4h\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v29.8h\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v30.4h\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v30.8h\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v27.4h\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v27.8h\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v28.4h\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v28.8h\n"
                    :
                    [k0] "+w"(kern[0]), [k1] "+w"(kern[1]), [k2] "+w"(kern[2]),
                    [k3] "+w"(kern[3]), [k4] "+w"(kern[4]), [k5] "+w"(kern[5]),
                    [k6] "+w"(kern[6]), [k7] "+w"(kern[7]), [k8] "+w"(kern[8]),
                    [sum00] "+w"(sum0[0]), [sum01] "+w"(sum0[1]), [sum02] "+w"(sum0[2]),
                    [sum03] "+w"(sum0[3]), [sum10] "+w"(sum1[0]), [sum11] "+w"(sum1[1]),
                    [sum12] "+w"(sum1[2]), [sum13] "+w"(sum1[3]), [sum20] "+w"(sum2[0]),
                    [sum21] "+w"(sum2[1]), [sum22] "+w"(sum2[2]), [sum23] "+w"(sum2[3]),
                    [sptr0] "+r"(sptr0), [sptr1] "+r"(sptr1), [sptr2] "+r"(sptr2),
                    [sptr3] "+r"(sptr3), [sptr4] "+r"(sptr4)
                    :
                    : "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                      "v30", "v31", "cc", "memory");
            ${init_store}
            ${render_store_1_line(dst, 0, sum0)};
            ${render_store_1_line(dst, 1, sum1)};
            ${render_store_1_line(dst, 2, sum2)};
#else
            int8x16_t src[2][3];
            LOAD_2_LINE_SRC(sptr0, sptr1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);

            LOAD_1_LINE_SRC(sptr2, src[0]);

            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);
            accumulate_2_q_vector(src[1][0], kern[0], src[0][0], kern[3], sum1);
            accumulate_2_q_vector(src[1][1], kern[1], src[0][1], kern[4], sum1);
            accumulate_2_q_vector(src[1][2], kern[2], src[0][2], kern[5], sum1);
            ${init_store}
            ${render_store_1_line(dst, 0, sum0)};

            LOAD_1_LINE_SRC(sptr3, src[1]);
            ACC_1_LINE(src[1], kern[6], kern[7], kern[8], sum1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum2);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum2);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum2);

            ${render_store_1_line(dst, 1, sum1)};
            LOAD_1_LINE_SRC(sptr4, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum2);

            ${render_store_1_line(dst, 2, sum2)};
#endif
        }
        if (ow < OW) {
            size_t iw = ow;
            size_t remain = OW - ow;
            const int8_t* sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum0[4], sum1[4], sum2[4];
            int8x16_t src[2][3];
#define cb(j)         \
    sum0[j] = init_v; \
    sum1[j] = init_v; \
    sum2[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_2_LINE_SRC(sptr0, sptr1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);

            LOAD_1_LINE_SRC(sptr2, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);
            accumulate_2_q_vector(src[1][0], kern[0], src[0][0], kern[3], sum1);
            accumulate_2_q_vector(src[1][1], kern[1], src[0][1], kern[4], sum1);
            accumulate_2_q_vector(src[1][2], kern[2], src[0][2], kern[5], sum1);
            ${init_store}
            ${render_store_1_line_remain(dst, 0, sum0, remain)};

            LOAD_1_LINE_SRC(sptr3, src[1]);
            ACC_1_LINE(src[1], kern[6], kern[7], kern[8], sum1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum2);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum2);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum2);

            ${render_store_1_line_remain(dst, 1, sum1, remain)};
            LOAD_1_LINE_SRC(sptr4, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum2);

            ${render_store_1_line_remain(dst, 2, sum2, remain)};
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0;
        ${init_store}
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum0[4];
            int8x16_t src[2][3];
#define cb(i) sum0[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_2_LINE_SRC(sptr0, sptr1);
            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);
            LOAD_1_LINE_SRC(sptr2, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);
            ${render_store_1_line(dst, 0, sum0)};
        }
        if (ow < OW) {
            size_t iw = ow;
            const int8_t* sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum0[4];
            int8x16_t src[2][3];
#define cb(i) sum0[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_2_LINE_SRC(sptr0, sptr1);
            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);
            LOAD_1_LINE_SRC(sptr2, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);
            size_t remain = (OW - ow);
            ${render_store_1_line_remain(dst, 0, sum0, remain)};
        }
    }
#undef LOAD_1_LINE_SRC
#undef LOAD_2_LINE_SRC
#undef ACC_1_LINE
}
    )";

    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("render_store_1_line",
                         [=](const std::string& dst, const std::string& idx,
                             const std::string& sum) -> std::string {
                             return render_store_1_line(
                                     dst, "oh", "ow", "OW", sum, idx,
                                     *activate_gen.get());
                         })
                    .add("render_store_1_line_remain",
                         [=](const std::string& dst, const std::string& idx,
                             const std::string& sum,
                             const std::string& remain) -> std::string {
                             return render_store_1_line_remain(
                                     dst, "oh", "ow", "OW", sum, idx, remain,
                                     *activate_gen.get());
                         })
                    .add("bias_str", bias_str)
                    .add_ctx_int("pad_w")
                    .render(body_temp);
    return ss.str();
}

std::string gen_3x3_s2_kern(
        TContext* ctx, bool with_bias, const std::string& nonline_mode) {
    std::stringstream ss;
    auto activate_gen = create_activation_gener_instrinsic(nonline_mode);
    std::string bias_str = with_bias ? "vld1q_s32(bias)" : "vdupq_n_s32(0)";
    std::string body_temp = R"(
static inline void nchw44_chanwise_3x3_int8(const int8_t* src, const int8_t* filter, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW, float scale){

    int32x4_t init_v = ${bias_str};
    int32x2_t zero = vdup_n_s32(0);
    int8x8_t kern01 = vld1_s8(filter);
    int8x8_t kern20 = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 8)), zero).val[0]);
    int8x8_t kern34 = vld1_s8(filter + 12);
    int8x8_t kern50 = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 20)), zero).val[0]);
    int8x8_t kern67 = vld1_s8(filter + 24);
    //! in case of illegal read
    int8x8_t kern80 = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 28)), zero).val[1]);

#define COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum)             \
    accumulate_1_line_horizon(vget_low_s8(src00), kern01, vget_high_s8(src00), \
                              kern20, &sum[0]);                                 \
    accumulate_1_line_horizon(vget_high_s8(src00), kern01, vget_low_s8(src01), \
                              kern20, &sum[1]);                                 \
    accumulate_1_line_horizon(vget_low_s8(src01), kern01, vget_high_s8(src01), \
                              kern20, &sum[2]);                                 \
    accumulate_1_line_horizon(vget_high_s8(src01), kern01, src02, kern20,      \
                              &sum[3]);

    size_t oh = 0;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum[0]);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum[0]);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum[0]);
            //! sum1
            COMPUTE_ONE_LINE(src20, src21, src22, kern01, kern20, sum[1]);

            //! line 3
            int8x16_t src30 = vld1q_s8(sptr3);
            int8x16_t src31 = vld1q_s8(sptr3 + 16);
            int8x8_t src32 = vld1_s8(sptr3 + 32);
            COMPUTE_ONE_LINE(src30, src31, src32, kern34, kern50, sum[1]);

            //! line 4
            int8x16_t src40 = vld1q_s8(sptr4);
            int8x16_t src41 = vld1q_s8(sptr4 + 16);
            int8x8_t src42 = vld1_s8(sptr4 + 32);
            COMPUTE_ONE_LINE(src40, src41, src42, kern67, kern80, sum[1]);
            ${init_store}
            ${render_store_1_line(dst, 0, sum[0])};
            ${render_store_1_line(dst, 1, sum[1])};
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* sptr4 = src + (ih + 4) * IW * 4 + iw * 4;

            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum[0]);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum[0]);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum[0]);
            //! sum1
            COMPUTE_ONE_LINE(src20, src21, src22, kern01, kern20, sum[1]);

            //! line 3
            int8x16_t src30 = vld1q_s8(sptr3);
            int8x16_t src31 = vld1q_s8(sptr3 + 16);
            int8x8_t src32 = vld1_s8(sptr3 + 32);
            COMPUTE_ONE_LINE(src30, src31, src32, kern34, kern50, sum[1]);

            //! line 4
            int8x16_t src40 = vld1q_s8(sptr4);
            int8x16_t src41 = vld1q_s8(sptr4 + 16);
            int8x8_t src42 = vld1_s8(sptr4 + 32);
            COMPUTE_ONE_LINE(src40, src41, src42, kern67, kern80, sum[1]);
            ${init_store}
            ${render_store_1_line_remain(dst, 0, sum[0], remain)};
            ${render_store_1_line_remain(dst, 1, sum[1], remain)};
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum[4];
#define cb(i) sum[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum);
            ${init_store}
            ${render_store_1_line(dst, 0, sum)};
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum[4];
#define cb(i) sum[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum);
            ${init_store}
            ${render_store_1_line_remain(dst, 0, sum, remain)};
        }
    }
#undef COMPUTE_ONE_LINE
}
    )";

    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add("init_store", activate_gen->GenIntrinsicInitFloat())
                    .add("render_store_1_line",
                         [=](const std::string& dst, const std::string& idx,
                             const std::string& sum) -> std::string {
                             return render_store_1_line(
                                     dst, "oh", "ow", "OW", sum, idx,
                                     *activate_gen.get());
                         })
                    .add("render_store_1_line_remain",
                         [=](const std::string& dst, const std::string& idx,
                             const std::string& sum,
                             const std::string& remain) -> std::string {
                             return render_store_1_line_remain(
                                     dst, "oh", "ow", "OW", sum, idx, remain,
                                     *activate_gen.get());
                         })
                    .add("bias_str", bias_str)
                    .add_ctx_int("pad_w")
                    .render(body_temp);
    return ss.str();
}

}  // namespace

bool ChannelWiseInt8Mk4K3::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == 3 && ctx->getAttrUInt("kernel_w") == 3 &&
            //! stride_h == stride_w and stride == 1 or stride == 2
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            (ctx->getAttrUInt("stride_h") == 1 || ctx->getAttrUInt("stride_h") == 2) &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;

    bool param_mode_ok = ctx->getAttrStr("sparse") == "GROUP" &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU";

    bool type_ok = is_qint8_conv_dtype(ctx);
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;
    bool channel_wise_ok = ctx->getAttrOprand("operand:1").shape.size() == 6 &&
                           ctx->getAttrOprand("operand:1").shape[5] == 4 &&
                           ctx->getAttrOprand("operand:1").shape[1] == 1 &&
                           ctx->getAttrOprand("operand:1").shape[2] == 1;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           channel_wise_ok;
}
std::string ChannelWiseInt8Mk4K3::GetKernelSymbol(TContext* context) const {
    return "Arm64_chanwise" + ConvImpl::GetKernelSymbol(context);
}

std::string ChannelWiseInt8Mk4K3::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const int pack_c_size = ${pack_c_size};
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        uint32_t kernel_h = ${kernel_h};
        uint32_t kernel_w = ${kernel_w};
        uint32_t stride_h = ${stride_h};
        uint32_t stride_w = ${stride_w};

        const uint32_t oh = (ih - kernel_h + 2 * ${pad_h}) / stride_h + 1;
        const uint32_t ow = (iw - kernel_w + 2 * ${pad_w}) / stride_w + 1;        
        const uint32_t ih2 = stride_h * oh + kernel_h - stride_h;
        const uint32_t ow_round4 = (ow + 3) & ~3;
        const uint32_t iw2 = stride_w * ow_round4 + kernel_w - stride_w;
        size_t res = ih2 * iw2 * pack_c_size + 16;

        *workspace = res;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .add("pack_c_size", 4)
                    .render(workspace_temp);
    return ss.str();
}

std::string ChannelWiseInt8Mk4K3::GetKernelBody(TContext* ctx) const {
    int stride = ctx->getAttrUInt("stride_h");
    bool with_bias = ConvImpl::is_bias(ctx);

    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::stringstream writer;
    writer << R"(
        #include <arm_neon.h>
        #include <string.h>
    )";
    writer << gen_common_code();
    writer << "\n\n";

    if (1 == stride) {
        writer << gen_3x3_s1_kern(ctx, with_bias, nonline_mode);
    } else if (2 == stride) {
        writer << gen_3x3_s2_kern(ctx, with_bias, nonline_mode);
    } else {
        CC_ABORT << "unsupport stride in mk4 channel wise kernel.\n";
    }
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    std::string bias_str = with_bias ? "inputs[2]->ptr" : "0";
    std::string body_temp = R"(
        const int pack_ic_size = 4;
        const int N = inputs[0]->layout.dims[0];
        const int ICB = inputs[0]->layout.dims[1];
        const int IH = inputs[0]->layout.dims[2];
        const int IW = inputs[0]->layout.dims[3];
        const int OH = outputs[0]->layout.dims[2];
        const int OW = outputs[0]->layout.dims[3];
        const int ICB_stride = IH * IW * pack_ic_size;
        const int N_stride = ICB * ICB_stride;
        const int OCB_stride = OH * OW * pack_ic_size;
        const int ON_stride = ICB * OCB_stride;
        
        const int PH = ${pad_h};
        const int PW = ${pad_w};
        const int IH2 = ${stride_h} * OH + ${kernel_h} - ${stride_h};
        const int OW_ROUND4 = (OW + 3) & ~3;
        const int IW2 = ${stride_w} * OW_ROUND4 + ${kernel_w} - ${stride_w};
        const int min_ih = IH2 < IH? IH2:IH;
        const int min_iw = IW2 < IW? IW2:IW;

        const float src_scale = inputs[0]->dtype.param.scale;
        const float flt_scale = inputs[1]->dtype.param.scale;
        const float dst_scale = outputs[0]->dtype.param.scale;
        const float scale = src_scale * flt_scale / dst_scale;


        int8_t* input_data = inputs[0]->ptr;
        int8_t* output_data = outputs[0]->ptr;
        int8_t* weight_data = inputs[1]->ptr;
        int8_t* padding_src = workspace->ptr;
        int32_t* bias_data = ${bias_str};

        rep(n, N){
            rep(icb, ICB){
                int8_t* src_ptr = input_data + icb * ICB_stride;
                int8_t* dst_ptr = output_data + icb * OCB_stride;
                int8_t* weight_ptr = weight_data + icb * ${kernel_h} * ${kernel_w} * pack_ic_size;
                int32_t* bias_ptr = bias_data + icb * pack_ic_size;
                memset(padding_src, 0, sizeof(int8_t) * IH2 * IW2 * pack_ic_size);
                rep(ih, min_ih) {
                    memcpy(padding_src + ((ih + PH) * IW2 + PW) * pack_ic_size,
                           src_ptr + ih * IW * pack_ic_size, sizeof(int8_t) * min_iw * pack_ic_size);
                }
                nchw44_chanwise_3x3_int8(padding_src, weight_ptr, bias_ptr, dst_ptr, IH2, IW2, OH, OW, scale);
            }
            input_data += N_stride;
            output_data += ON_stride;
        }
        return TinyNN_SUCCESS;
    })";
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add("bias_str", bias_str)
                      .render(body_temp);
    return writer.str();
}

// vim: syntax=cpp.doxygen
