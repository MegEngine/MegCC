#include "WinogradF23Strategy8x8Nchw44MK8Int8.h"
#include <string>
#include "Arm/ArmCommon/Activation.h"
#include "Arm/ArmCommon/InternalMatMul/InternalMatMul.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

std::string WinogradF23Strategy8x8Nchw44MK8Int8::WeightTrans(
        const std::vector<std::string>& strs) {
    auto inptr = strs[0];
    auto outptr = strs[1];
    auto OC = strs[2];
    auto IC = strs[3];
    std::string filter_process = R"(
        const int alpha = 2 + 3 - 1;

        /**
         * origin: (4x3) * (3 x 3) * (3 x 4)
         */
        //! 1      0    0    v00 v01 v02   1 0.5  0.5 0
        //! 0.5  0.5  0.5    v10 v11 v12   0 0.5 -0.5 0
        //! 0.5 -0.5  0.5    v20 v21 v22   0 0.5  0.5 1
        //! 0      0    1

        //! 2   0  0    v00 v01 v02   2 1  1 0
        //! 1   1  1    v10 v11 v12   0 1 -1 0
        //! 1  -1  1    v20 v21 v22   0 1  1 2
        //! 0   0  2
        //! G * g * GT

        TINYNN_ASSERT_MSG(
                    ${IC} % 8 == 0 && ${OC} % 8 == 0,
                "Winograd filter transform input param is not times of 8!");
        size_t OCB = ${OC} / 8;
        size_t ICB = ${IC} / 8;
        size_t ICB4 = ${IC} / 4;
        for (size_t ocb = 0; ocb < ${OC} / 4; ocb++) {
            size_t tmp_ocb = ocb / 2;
            size_t index = ((ocb & 1) == 0) ? 0 : 1;
            for (size_t icb = 0; icb < ICB4; icb++) {
                for (size_t ic_inner = 0; ic_inner < 4; ic_inner++) {
                    const int8_t* fptr = ${filter} +
                                         (ocb * ICB4 + icb) * 3 * 3 * 4 * 4 +
                                         ic_inner * 4;

                    int16x4_t g00 = vget_low_s16(vmovl_s8(vld1_s8(fptr)));
                    int16x4_t g01 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 4 * 4)));
                    int16x4_t g02 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 2 * 4 * 4)));
                    int16x4_t g10 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 3 * 4 * 4)));
                    int16x4_t g11 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 4 * 4 * 4)));
                    int16x4_t g12 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 5 * 4 * 4)));
                    int16x4_t g20 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 6 * 4 * 4)));
                    int16x4_t g21 = vget_low_s16(vmovl_s8(vld1_s8(fptr + 7 * 4 * 4)));
                    int16x4_t g22 = vget_high_s16(vmovl_s8(vld1_s8(fptr + 8 * 4 * 4 - 4)));

#define FILTER_TRANSFORM(n, wd, g)                 \
    int16x4_t wd##n##0 = vmul_n_s16(g##0##n, 2);   \
    v_tmp = vadd_s16(g##0##n, g##2##n);            \
    int16x4_t wd##n##1 = vadd_s16(v_tmp, g##1##n); \
    int16x4_t wd##n##2 = vsub_s16(v_tmp, g##1##n); \
    int16x4_t wd##n##3 = vmul_n_s16(g##2##n, 2);

                    int16x4_t v_tmp;
                    UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                    UNROLL_CALL_RAW(4, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM

#define cb(m, n)                                              \
    vst1_s16(                                                 \
            ${outptr} + (m * alpha + n) * OCB * ICB * 8 * 8 + \
            tmp_ocb * ICB * 8 * 8 + icb * 4 * 8 + ic_inner * 8 + index * 4, ret##m##n);

                    UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
                }
            }
        }
)";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("OC", OC)
                    .add("IC", IC)
                    .add("filter", inptr)
                    .add("outptr", outptr)
                    .render(filter_process);
    return ss.str();
}

std::string WinogradF23Strategy8x8Nchw44MK8Int8::InputFeatureTrans(
        const std::vector<std::string>& strs) {
    auto InputTransformF23NCHW44Int8 = []() {
        std::string kernel = R"(
        int16x8_t d[4][4];
#define cb(m, n) \
    d[m][n] = vdupq_n_s16(0);

        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
        //! NCHW4 --> NCHW8
        if (inner) {
            const int8_t* input_ptr =
                    input + ic * IH_ * IW_ + ih_start * IW_ * 4 + iw_start * 4;
            for (size_t ico = 0; ico < alpha; ++ico) {
                int8x16_t v_input0 = vld1q_s8(input_ptr);                // c0123
                int8x16_t v_input1 = vld1q_s8(input_ptr + IH_ * IW_ * 4);  // c4567
                int32x4_t v32_00 = vreinterpretq_s32_s8(v_input0);
                int32x4_t v32_01 = vreinterpretq_s32_s8(v_input1);

                int32x4x2_t v_trn = vtrnq_s32(v32_00, v32_01);  // c04261537

                v_input0 = vreinterpretq_s8_s32(v_trn.val[0]);
                v_input1 = vreinterpretq_s8_s32(v_trn.val[1]);

                d[ico][0] = vmovl_s8(vget_low_s8(v_input0));
                d[ico][2] = vmovl_s8(vget_high_s8(v_input0));
                d[ico][1] = vmovl_s8(vget_low_s8(v_input1));
                d[ico][3] = vmovl_s8(vget_high_s8(v_input1));

                input_ptr += IW_ * 4;  // next row
            }
        } else {
            const int8_t* input_ptr = input + ic * IH_ * IW_;
            int ih0_act = ih_start > 0 ? ih_start : 0,
                ih1_act = ih_start + alpha < (int)IH_ ? ih_start + alpha : (int)IH_,
                iw0_act = iw_start > 0 ? iw_start : 0,
                iw1_act = iw_start + alpha < (int)IW_ ? iw_start + alpha : (int)IW_;
            // partial copy
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    d[iho][iwo] = vcombine_s16(
                            vget_low_s16(vmovl_s8(vld1_s8(input_ptr + ih * IW_ * 4 + iw * 4))),
                            vget_high_s16(vmovl_s8(vld1_s8(input_ptr + IH_ * IW_ * 4 + ih * IW_ * 4 +
                                    iw * 4 - 4))));
                }
            }
        }

        // BT * d * B

        //! 1   0 -1 0    d00 d01 d02 d03     1 0  0  0
        //! 0   1  1 0    d10 d11 d12 d13     0 1 -1 -1
        //! 0  -1  1 0    d20 d21 d22 d23    -1 1  1  0
        //! 0  -1  0 1    d30 d31 d32 d33     0 0  0  1
#define cb(m)                                      \
    int16x8_t t0##m = vsubq_s16(d[0][m], d[2][m]); \
    int16x8_t t1##m = vaddq_s16(d[1][m], d[2][m]); \
    int16x8_t t2##m = vsubq_s16(d[2][m], d[1][m]); \
    int16x8_t t3##m = vsubq_s16(d[3][m], d[1][m]);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m)                              \
    d[m][0] = vsubq_s16(t##m##0, t##m##2); \
    d[m][1] = vaddq_s16(t##m##1, t##m##2); \
    d[m][2] = vsubq_s16(t##m##2, t##m##1); \
    d[m][3] = vsubq_s16(t##m##3, t##m##1);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        size_t ICB = IC_ / 8;
        size_t icb = ic / 8;
#define cb(m, n)                                                 \
    vst1q_s16(                                                   \
            dst + (m * alpha + n) * ICB * nr_tiles_in_loop * 8 + \
            icb * nr_tiles_in_loop * 8 + tile_idx * 8, d[m][n]);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
)";
        return kernel;
    };
    std::string input_process = R"(
    const int alpha = 3 + 2 - 1;
    const uint32_t OUTPUT_BLOCK_SIZE = 2;
    const uint32_t KS = 3;

    int16_t* dst = ${transform_input_ptr};
    const int8_t* input = ${inptr};
    uint32_t IH_ = ${IH};
    uint32_t IW_ = ${IW};
    uint32_t IC_ = ${IC};
    uint32_t PH_ = ${PH};
    uint32_t PW_ = ${PW};
    uint32_t nr_tiles_in_loop_ = ${nr_tiles_in_loop};
    uint32_t tile_id_ = ${tile_id};

    uint32_t OW = IW_ + 2 * PW_ - KS + 1;
    uint32_t tiles_w = (OW + OUTPUT_BLOCK_SIZE -1)/ OUTPUT_BLOCK_SIZE;

    for (uint32_t ic = 0; ic < IC_; ic += 8) {
        uint32_t tile_start_id = tile_id_;
        for(uint32_t tile_idx = 0; tile_idx < nr_tiles_in_loop_; tile_idx++) {
            uint32_t index = tile_start_id + tile_idx;
            uint32_t nh = index / tiles_w;
            uint32_t nw = index % tiles_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH_;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW_;
            int inner = (ih_start >= 0 && iw_start >= 0 &&
                        ih_start + alpha <= (int)IH_ &&
                        iw_start + alpha <= (int)IW_);
            ${InputTransformF23NCHW44Int8()}
        }
    }
    )";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("inptr", strs[0])
                    .add("transform_input_ptr", strs[1])
                    .add("IH", strs[2])
                    .add("IW", strs[3])
                    .add("IC", strs[4])
                    .add("PH", strs[5])
                    .add("PW", strs[6])
                    .add("tile_id", strs[7])
                    .add("nr_tiles_in_loop", strs[8])
                    .add("InputTransformF23NCHW44Int8", InputTransformF23NCHW44Int8)
                    .render(input_process);
    return ss.str();
}

std::string WinogradF23Strategy8x8Nchw44MK8Int8::DependMatmulSymbol() {
    return Int16M8N8K8MatMulKernel().GetKernelSymbol(nullptr);
}

std::string WinogradF23Strategy8x8Nchw44MK8Int8::BatchedMatMul(
        const std::vector<std::string>& strs) {
    std::string matmul_compute = R"(
    for(uint32_t i =0; i< Alpha; i++){
        for(uint32_t j=0; j<Alpha; j++){
            const int16_t* a_ptr = ${A_ptr} +
                (i * Alpha + j) * ${OC} * ${IC};
            int16_t* b_ptr = ${B_ptr} +
                (i * Alpha + j) * ${nr_tiles_in_loop} * ${IC};
            int32_t* c_ptr = ${C_ptr} +
                (i * Alpha + j) * ${nr_tiles_in_loop} * ${OC};
            ${MatMul}(a_ptr, ${LDA}, b_ptr, ${LDB}, c_ptr, ${LDC}, ${OC}, 
                    ${nr_tiles_in_loop}, ${IC});
        }
    })";

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("MatMul", DependMatmulSymbol())
                    .add("A_ptr", strs[0])
                    .add("LDA", strs[1])
                    .add("B_ptr", strs[2])
                    .add("LDB", strs[3])
                    .add("C_ptr", strs[4])
                    .add("LDC", strs[5])
                    .add("OC", strs[6])
                    .add("IC", strs[7])
                    .add("nr_tiles_in_loop", strs[8])
                    .render(matmul_compute);
    return ss.str();
}

std::string WinogradF23Strategy8x8Nchw44MK8Int8::OutputFeatureTrans(
        const std::vector<std::string>& strs, TContext* ctx) {
    std::string ouput_trans = R"(
    int32_t* transform_output_ptr_ = ${transform_output_ptr};
    int8_t* outptr_ = ${outptr};
    const int32_t* bias_ptr_ = ${bias_ptr};
    uint32_t OH_ = ${OH};
    uint32_t OW_ = ${OW};
    uint32_t OC_ = ${OC};
    uint32_t tile_id_ = ${tile_id};
    uint32_t nr_tiles_in_loop_ = ${nr_tiles_in_loop};

    const uint32_t OutputBlockSize = 2;
    uint32_t tiles_w_ = (OW_ + OutputBlockSize -1) / OutputBlockSize;

    for (uint32_t oc = 0; oc < OC_; oc += 8) {
        for(uint32_t tile_idx = 0; tile_idx < nr_tiles_in_loop_; tile_idx++) {
            uint32_t index = tile_id_ + tile_idx;
            uint32_t nh = index / tiles_w_;
            uint32_t nw = index % tiles_w_;
            uint32_t oh_start = nh * OutputBlockSize;
            uint32_t ow_start = nw * OutputBlockSize;

            //! AT * m * A
            uint32_t OCB = OC_ / 8;

            for(uint32_t oc_inner = 0; oc_inner < 8; oc_inner += 4) {
                int32x4_t src[4][4];

#define LOAD_V(m, n)                                               \
    src[m][n] = vld1q_s32(transform_output_ptr_ +                  \
            (m * Alpha + n) * OCB * nr_tiles_in_loop_ * 8 +        \
            oc * nr_tiles_in_loop_ + tile_idx * 8 + oc_inner);

                UNROLL_CALL_NOWRAPPER_D2(4, 4, LOAD_V);
#undef LOAD_V

                //! 1  1  1 0  v00 v01 v02 v03    1  0
                //! 0  1 -1 1  v10 v11 v12 v13    1  1
                //!            v20 v21 v22 v23    1 -1
                //!            v30 v31 v32 v33    0  1

                int32x4_t mid[2][4];
#define MULTI_ONE(m)                                                   \
    mid[0][m] = vaddq_s32(vaddq_s32(src[0][m], src[1][m]), src[2][m]); \
    mid[1][m] = vaddq_s32(vsubq_s32(src[1][m], src[2][m]), src[3][m]);

                UNROLL_CALL_NOWRAPPER(4, MULTI_ONE);
#undef MULTI_ONE

                int32x4_t dst_v[2][2];
#define MULTI_TWO(m)                                                             \
            dst_v[m][0] = vaddq_s32(vaddq_s32(mid[m][0], mid[m][1]), mid[m][2]); \
            dst_v[m][1] = vaddq_s32(vsubq_s32(mid[m][1], mid[m][2]), mid[m][3]);

                UNROLL_CALL_NOWRAPPER(2, MULTI_TWO);
#undef MULTI_TWO

                if (bias_ptr_) {
                    int32x4_t vbias = vmulq_n_s32(vld1q_s32(bias_ptr_ + oc + oc_inner), 4);
                    dst_v[0][0]= vaddq_s32(dst_v[0][0], vbias);
                    dst_v[0][1]= vaddq_s32(dst_v[0][1], vbias);
                    dst_v[1][0]= vaddq_s32(dst_v[1][0], vbias);
                    dst_v[1][1]= vaddq_s32(dst_v[1][1], vbias);
                }

                //! fuse activation
                ${nonline_gen_init()}
                for(int oho = 0; oho < 2 && oh_start + oho < OH_; ++oho)
                    for(int owo = 0; owo < 2 && ow_start + owo < OW_; ++owo){
                        ${nonline_gen_func(dst_v[oho][owo], outptr_ + (oc + oc_inner) * OH_ * OW_ + (oh_start + oho) * OW_ * 4 + (ow_start + owo) * 4, bias_scale, dst_scale)}
                    }
            }
        }
    })";
    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        return nonline_gen->GenIntrinsicQuantStore(str[0], str[1], str[2], str[3]);
    };
    auto nonline_gen_init = [&]() -> std::string {
        return nonline_gen->GenIntrinsicInitFloat();
    };

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("nonline_gen_func", nonline_gen_func)
                    .add("nonline_gen_init", nonline_gen_init)
                    .add("transform_output_ptr", strs[0])
                    .add("outptr", strs[1])
                    .add("bias_ptr", strs[2])
                    .add("OH", strs[3])
                    .add("OW", strs[4])
                    .add("OC", strs[5])
                    .add("tile_id", strs[6])
                    .add("nr_tiles_in_loop", strs[7])
                    .render(ouput_trans);
    return ss.str();
}

// vim: syntax=cpp.doxygen
