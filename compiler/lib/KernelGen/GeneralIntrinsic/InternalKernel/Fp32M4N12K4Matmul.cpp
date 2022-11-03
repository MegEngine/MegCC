/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/InternalKernel/Fp32M8N12K4Matmul.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ElemwiseHelper/ElemwiseHelper.h"
#include "GeneralIntrinsic/GIMathHelper.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {

static inline std::pair<std::string, std::string> gen_postprocess_inline(
        TContext* ctx, bool need_postprocess = true) {
    std::string call_str;
    std::stringstream declare_ss;
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    if ((nonline_mode == "SIGMOID") && need_postprocess) {
        std::vector<CCOperand> operands;
        operands.resize(2);
        auto dtype = ctx->getAttrStr("dtype");
        auto create_elem = [=](std::string src_dtype, std::string dst_dtype)
                -> std::shared_ptr<ElemwiseGenUnary> {
            return std::make_shared<ElemwiseGenUnarySigmoid>(src_dtype,
                                                             dst_dtype, true);
        };

        std::shared_ptr<ElemwiseGenUnary> ElemwiseImpl =
                create_elem("f32", "f32");

        if (Utils::is_quant_dtype(dtype)) {
            ElemwiseImpl = create_elem("si32", "si8");
        }

        auto ImpleGen = [=](std::vector<std::string> strs) {
            return ElemwiseImpl->GenCodeBody(strs);
        };
        std::string post_process_temp = R"(
            if (LDC == N){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                for(int m_idx = 0; m_idx < M; ++m_idx){
                    ${ElemwiseImplName}(C + m_idx * LDC, C + m_idx * LDC, N);
                }
            }
        )";
        if (ctx->getAttrStr("format") == "MK4") {
            post_process_temp = R"(
            if (LDC == (4 * N)){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < M; m_idx += 4){
                    ${ElemwiseImplName}(C + cnt * LDC, C + cnt * LDC, 4 * N);
                    ++cnt;
                }
            }
        )";
        } else {
            CC_ASSERT(ctx->getAttrStr("format") == "MK4_DOT");
            post_process_temp = R"(
            if (LDC == (4 * N)){
                ${ElemwiseImplName}(gemm_output, C, M * N, temp_scale, dst_scale_inv);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < M; m_idx += 4){
                    ${ElemwiseImplName}(gemm_output + cnt * LDC, C + cnt * LDC, 4 * N, temp_scale, dst_scale_inv);
                    ++cnt;
                }
            }
        )";
        }
        call_str =
                StringTemplate::StringTemplateArgs()
                        .add("ElemwiseImplName", ElemwiseImpl->GenInlineName())
                        .render(post_process_temp);
        GIMathHelper gi_math;
        declare_ss << R"(
#include "gi_int.h"
            )";
        declare_ss << gi_math.GiExpPsFloat32() << "\n";
        declare_ss << gi_math.GiSigmoidPsFloat32() << "\n";
        declare_ss << ElemwiseImpl->GenCodeBody({});
    }
    return {declare_ss.str(), call_str};
}

std::string transpose_1x12_4_s() {
    return R"(
static inline void transpose_1x12_4_s(const float* inptr0, float* outptr) {
    GI_FLOAT32_t tmp_a, tmp_b;
#define LOAD()                     \
    tmp_a = GiLoadFloat32(inptr0); \
    inptr0 += 4;                   \
    tmp_b = GiLoadFloat32(inptr0); \
    inptr0 += 4;
    LOAD();
    GI_FLOAT32_V2_t d0d1d2d3 = GiZipqFloat32(tmp_a, tmp_b);
    LOAD();
    GI_FLOAT32_V2_t d4d5d6d7 = GiZipqFloat32(tmp_a, tmp_b);
    LOAD();
    GI_FLOAT32_V2_t d8d9d10d11 = GiZipqFloat32(tmp_a, tmp_b);
    LOAD();
    GI_FLOAT32_V2_t d12d13d14d15 = GiZipqFloat32(tmp_a, tmp_b);
    LOAD();
    GI_FLOAT32_V2_t d16d17d18d19 = GiZipqFloat32(tmp_a, tmp_b);
    LOAD();
    GI_FLOAT32_V2_t d20d21d22d23 = GiZipqFloat32(tmp_a, tmp_b);
#undef LOAD
    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 0)));
    GiSt1Float32(outptr + 1 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 0)));
    GiSt1Float32(
            outptr + 2 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d8d9d10d11, 0)));
    GiSt1Float32(
            outptr + 3 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d12d13d14d15, 0)));
    GiSt1Float32(
            outptr + 4 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d16d17d18d19, 0)));
    GiSt1Float32(
            outptr + 5 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d20d21d22d23, 0)));
    GiSt1Float32(
            outptr + 6 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 0)));
    GiSt1Float32(
            outptr + 7 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 0)));
    GiSt1Float32(
            outptr + 8 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d8d9d10d11, 0)));
    GiSt1Float32(
            outptr + 9 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d12d13d14d15, 0)));
    GiSt1Float32(
            outptr + 10 * 2,
            GiGetHighFloat32(GiGetSubVectorFloat32V2(d16d17d18d19, 0)));
    GiSt1Float32(
            outptr + 11 * 2,
            GiGetHighFloat32(GiGetSubVectorFloat32V2(d20d21d22d23, 0)));
    GiSt1Float32(
            outptr + 12 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 1)));
    GiSt1Float32(
            outptr + 13 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 1)));
    GiSt1Float32(
            outptr + 14 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d8d9d10d11, 1)));
    GiSt1Float32(
            outptr + 15 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d12d13d14d15, 1)));
    GiSt1Float32(
            outptr + 16 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d16d17d18d19, 1)));
    GiSt1Float32(
            outptr + 17 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d20d21d22d23, 1)));
    GiSt1Float32(
            outptr + 18 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 1)));
    GiSt1Float32(
            outptr + 19 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 1)));
    GiSt1Float32(
            outptr + 20 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d8d9d10d11, 1)));
    GiSt1Float32(
            outptr + 21 * 2,
            GiGetHighFloat32(GiGetSubVectorFloat32V2(d12d13d14d15, 1)));
    GiSt1Float32(
            outptr + 22 * 2,
            GiGetHighFloat32(GiGetSubVectorFloat32V2(d16d17d18d19, 1)));
    GiSt1Float32(
            outptr + 23 * 2,
            GiGetHighFloat32(GiGetSubVectorFloat32V2(d20d21d22d23, 1)));
    outptr += 24 * 2;

    
}
)";
}

std::string transpose_1x4_4_s() {
    return R"(
static inline void transpose_1x4_4_s(const float* inptr0, float* outptr) {
     GI_FLOAT32_t tmp_a, tmp_b;
#define LOAD()                     \
    tmp_a = GiLoadFloat32(inptr0); \
    inptr0 += 4;                   \
    tmp_b = GiLoadFloat32(inptr0); \
    inptr0 += 4;

    LOAD();
    GI_FLOAT32_V2_t d0d1d2d3 = GiZipqFloat32(tmp_a, tmp_b);
    LOAD();
    GI_FLOAT32_V2_t d4d5d6d7 = GiZipqFloat32(tmp_a, tmp_b);
#undef LOAD
    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 0)));
    GiSt1Float32(outptr + 1 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 0)));
    GiSt1Float32(
            outptr + 2 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 0)));
    GiSt1Float32(
            outptr + 3 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 0)));
    GiSt1Float32(outptr + 4 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 1)));
    GiSt1Float32(outptr + 5 * 2, GiGetLowFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 1)));
    GiSt1Float32(
            outptr + 6 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d0d1d2d3, 1)));
    GiSt1Float32(
            outptr + 7 * 2, GiGetHighFloat32(GiGetSubVectorFloat32V2(d4d5d6d7, 1)));
    outptr += 8 * 2;
}
)";
}

static std::string kern_4x12(TContext* ctx) {
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    auto activation_gen = create_activation_gener_instrinsic(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::stringstream writer;
    writer << R"(
static inline void kern_4x12_bias_relu(const float* packA, const float* packB, int K,
                          float* output, int LDC, const float* bias_ptr) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    float* r1 = output;

    GI_FLOAT32_t d0d1, d2d3, d4d5, d6d7, d8d9, d10d11, d12d13, d14d15, d16d17, d18d19,
    d20d21, d22d23, d24d25, d26d27, d28d29, d30d31;
    )";
    if (with_bias) {
        writer << R"(
    d8d9 = GiLoadFloat32(bias_ptr);
       )";
    } else {
        writer << R"(
    d8d9 = GiBroadcastFloat32(0.0f);
        )";
    }
    writer << R"(
    d10d11 = d8d9;
    d12d13 = d8d9;
    d14d15 = d8d9;
    d16d17 = d8d9;
    d18d19 = d8d9;
    d20d21 = d8d9;
    d22d23 = d8d9;
    d24d25 = d8d9;
    d26d27 = d8d9;
    d28d29 = d8d9;
    d30d31 = d8d9;
       )";
    std::string body_temp = R"(
    d0d1 = GiLoadFloat32(a_ptr);
    a_ptr = a_ptr + 4;
    d4d5 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;
    d6d7 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;
    for (; K > 0; K--) {
        d8d9 = GiSimdFmaLane(d8d9, d0d1, d4d5, 0);
        d10d11 = GiSimdFmaLane(d10d11, d0d1, d4d5, 1);
        d12d13 = GiSimdFmaLane(d12d13, d0d1, d4d5, 2);
        d14d15 = GiSimdFmaLane(d14d15, d0d1, d4d5, 3);
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d16d17 = GiSimdFmaLane(d16d17, d0d1, d6d7, 0);
        d18d19 = GiSimdFmaLane(d18d19, d0d1, d6d7, 1);
        d20d21 = GiSimdFmaLane(d20d21, d0d1, d6d7, 2);
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d22d23 = GiSimdFmaLane(d22d23, d0d1, d6d7, 3);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d24d25 = GiSimdFmaLane(d24d25, d0d1, d4d5, 0);
        d26d27 = GiSimdFmaLane(d26d27, d0d1, d4d5, 1);
        d28d29 = GiSimdFmaLane(d28d29, d0d1, d4d5, 2);
        d30d31 = GiSimdFmaLane(d30d31, d0d1, d4d5, 3);
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d8d9 = GiSimdFmaLane(d8d9, d2d3, d6d7, 0);
        d10d11 = GiSimdFmaLane(d10d11, d2d3, d6d7, 1);
        d12d13 = GiSimdFmaLane(d12d13, d2d3, d6d7, 2);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d6d7, 3);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d16d17 = GiSimdFmaLane(d16d17, d2d3, d4d5, 0);
        d18d19 = GiSimdFmaLane(d18d19, d2d3, d4d5, 1);
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d4d5, 2);
        d22d23 = GiSimdFmaLane(d22d23, d2d3, d4d5, 3);
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d24d25 = GiSimdFmaLane(d24d25, d2d3, d6d7, 0);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d6d7, 1);
        d28d29 = GiSimdFmaLane(d28d29, d2d3, d6d7, 2);
        d30d31 = GiSimdFmaLane(d30d31, d2d3, d6d7, 3);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
    }

    if (1 == oddk) {
        d8d9 = GiSimdFmaLane(d8d9, d0d1, d4d5, 0);
        d10d11 = GiSimdFmaLane(d10d11, d0d1, d4d5, 1);
        d12d13 = GiSimdFmaLane(d12d13, d0d1, d4d5, 2);
        d14d15 = GiSimdFmaLane(d14d15, d0d1, d4d5, 3);
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d16d17 = GiSimdFmaLane(d16d17, d0d1, d6d7, 0);
        ${GenActivate(d8d9, d8d9)}
        GiStoreFloat32(output0, d8d9);
        output0 = output0 + 4;
        ${GenActivate(d10d11, d10d11)}
        GiStoreFloat32(output0, d10d11);
        output0 = output0 + 4;
        d18d19 = GiSimdFmaLane(d18d19, d0d1, d6d7, 1);
        d20d21 = GiSimdFmaLane(d20d21, d0d1, d6d7, 2);
        ${GenActivate(d12d13, d12d13)}
        GiStoreFloat32(output0, d12d13);
        output0 = output0 + 4;
        ${GenActivate(d14d15, d14d15)}
        GiStoreFloat32(output0, d14d15);
        output0 = output0 + 4;
        d22d23 = GiSimdFmaLane(d22d23, d0d1, d6d7, 3);
        d24d25 = GiSimdFmaLane(d24d25, d0d1, d4d5, 0);
        ${GenActivate(d16d17, d16d17)}
        GiStoreFloat32(output0, d16d17);
        output0 = output0 + 4;
        ${GenActivate(d18d19, d18d19)}
        GiStoreFloat32(output0, d18d19);
        output0 = output0 + 4;
        d26d27 = GiSimdFmaLane(d26d27, d0d1, d4d5, 1);
        ${GenActivate(d20d21, d20d21)}
        GiStoreFloat32(output0, d20d21);
        output0 = output0 + 4;
        ${GenActivate(d22d23, d22d23)}
        GiStoreFloat32(output0, d22d23);
        output0 = output0 + 4;
        d28d29 = GiSimdFmaLane(d28d29, d0d1, d4d5, 2);
        ${GenActivate(d24d25, d24d25)}
        GiStoreFloat32(output0, d24d25);
        output0 = output0 + 4;
        ${GenActivate(d26d27, d26d27)}
        GiStoreFloat32(output0, d26d27);
        output0 = output0 + 4;
        d30d31 = GiSimdFmaLane(d30d31, d0d1, d4d5, 3);
        ${GenActivate(d28d29, d28d29)}
        GiStoreFloat32(output0, d28d29);
        output0 = output0 + 4;
        ${GenActivate(d30d31, d30d31)}
        GiStoreFloat32(output0, d30d31);
        output0 = output0 + 4;

    } else {
        d8d9 = GiSimdFmaLane(d8d9, d0d1, d4d5, 0);
        d10d11 = GiSimdFmaLane(d10d11, d0d1, d4d5, 1);
        d12d13 = GiSimdFmaLane(d12d13, d0d1, d4d5, 2);
        d14d15 = GiSimdFmaLane(d14d15, d0d1, d4d5, 3);
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d16d17 = GiSimdFmaLane(d16d17, d0d1, d6d7, 0);
        d18d19 = GiSimdFmaLane(d18d19, d0d1, d6d7, 1);
        d20d21 = GiSimdFmaLane(d20d21, d0d1, d6d7, 2);
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d22d23 = GiSimdFmaLane(d22d23, d0d1, d6d7, 3);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d24d25 = GiSimdFmaLane(d24d25, d0d1, d4d5, 0);
        d26d27 = GiSimdFmaLane(d26d27, d0d1, d4d5, 1);
        d28d29 = GiSimdFmaLane(d28d29, d0d1, d4d5, 2);
        d30d31 = GiSimdFmaLane(d30d31, d0d1, d4d5, 3);
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d6d7, 0);
        d10d11 = GiSimdFmaLane(d10d11, d2d3, d6d7, 1);
        d12d13 = GiSimdFmaLane(d12d13, d2d3, d6d7, 2);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d6d7, 3);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d16d17 = GiSimdFmaLane(d16d17, d2d3, d4d5, 0);
        d18d19 = GiSimdFmaLane(d18d19, d2d3, d4d5, 1);
        ${GenActivate(d8d9, d8d9)}
        GiStoreFloat32(output0, d8d9);
        output0 = output0 + 4;
        ${GenActivate(d10d11, d10d11)}
        GiStoreFloat32(output0, d10d11);
        output0 = output0 + 4;
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d4d5, 2);
        d22d23 = GiSimdFmaLane(d22d23, d2d3, d4d5, 3);
        ${GenActivate(d12d13, d12d13)}
        GiStoreFloat32(output0, d12d13);
        output0 = output0 + 4;
        ${GenActivate(d14d15, d14d15)}
        GiStoreFloat32(output0, d14d15);
        output0 = output0 + 4;
        d24d25 = GiSimdFmaLane(d24d25, d2d3, d6d7, 0);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d6d7, 1);
        ${GenActivate(d16d17, d16d17)}
        GiStoreFloat32(output0, d16d17);
        output0 = output0 + 4;
        ${GenActivate(d18d19, d18d19)}
        GiStoreFloat32(output0, d18d19);
        output0 = output0 + 4;
        d28d29 = GiSimdFmaLane(d28d29, d2d3, d6d7, 2);
        d30d31 = GiSimdFmaLane(d30d31, d2d3, d6d7, 3);
        ${GenActivate(d20d21, d20d21)}
        GiStoreFloat32(output0, d20d21);
        output0 = output0 + 4;
        ${GenActivate(d22d23, d22d23)}
        GiStoreFloat32(output0, d22d23);
        output0 = output0 + 4;
        ${GenActivate(d24d25, d24d25)}
        GiStoreFloat32(output0, d24d25);
        output0 = output0 + 4;
        ${GenActivate(d26d27, d26d27)}
        GiStoreFloat32(output0, d26d27);
        output0 = output0 + 4;
        ${GenActivate(d28d29, d28d29)}
        GiStoreFloat32(output0, d28d29);
        output0 = output0 + 4;
        ${GenActivate(d30d31, d30d31)}
        GiStoreFloat32(output0, d30d31);
        output0 = output0 + 4;
    }
}
              )";
    writer << activation_gen->GenIntrinsicInitFloat();
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenActivate",
                           [=](std::vector<std::string> args) {
                               return activation_gen->GenIntrinsicFloat(
                                       args[0], args[1]);
                           })
                      .render(body_temp);
    return writer.str();
}

static std::string kern_4x4(TContext* ctx) {
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    auto activation_gen = create_activation_gener_instrinsic(nonline_mode);
    bool with_bias = ctx->getAttrBool("with_bias");
    std::stringstream writer;
    writer << R"(
static inline void kern_4x4_bias_relu(const float* packA, const float* packB, int K,
                                           float* output, int LDC, const float* bias_ptr,
                                           int n_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    float* r1 = output;

    GI_FLOAT32_t d0d1, d2d3, d4d5, d6d7, d8d9, d10d11, d12d13, d14d15;
    )";
    if (with_bias) {
        writer << R"(
    d8d9 = GiLoadFloat32(bias_ptr);
         )";
    } else {
        writer << R"(
    d8d9 = GiBroadcastFloat32(0.0f);
        )";
    }
    writer << R"(
    d10d11 = d8d9;
    d12d13 = d8d9;
    d14d15 = d8d9;
         )";
    std::string body_temp = R"(
    d0d1 = GiLoadFloat32(a_ptr);
    a_ptr = a_ptr + 4;
    d4d5 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;

      for (; K > 0; K--) {
        d8d9 = GiSimdFmaLane(d8d9, d0d1, d4d5, 0);
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d10d11 = GiSimdFmaLane(d10d11, d0d1, d4d5, 1);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d12d13 = GiSimdFmaLane(d12d13, d0d1, d4d5, 2);
        d14d15 = GiSimdFmaLane(d14d15, d0d1, d4d5, 3);

        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d8d9 = GiSimdFmaLane(d8d9, d2d3, d6d7, 0);
        d10d11 = GiSimdFmaLane(d10d11, d2d3, d6d7, 1);
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d12d13 = GiSimdFmaLane(d12d13, d2d3, d6d7, 2);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d6d7, 3);
    }

    if (1 == oddk) {
        d8d9 = GiSimdFmaLane(d8d9, d0d1, d4d5, 0);
        d10d11 = GiSimdFmaLane(d10d11, d0d1, d4d5, 1);
        d12d13 = GiSimdFmaLane(d12d13, d0d1, d4d5, 2);
        d14d15 = GiSimdFmaLane(d14d15, d0d1, d4d5, 3);
    } else {
        d8d9 = GiSimdFmaLane(d8d9, d0d1, d4d5, 0);
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d10d11 = GiSimdFmaLane(d10d11, d0d1, d4d5, 1);
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d12d13 = GiSimdFmaLane(d12d13, d0d1, d4d5, 2);
        d14d15 = GiSimdFmaLane(d14d15, d0d1, d4d5, 3);

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d6d7, 0);
        d10d11 = GiSimdFmaLane(d10d11, d2d3, d6d7, 1);
        d12d13 = GiSimdFmaLane(d12d13, d2d3, d6d7, 2);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d6d7, 3);
    }
    ${GenActivate(d8d9, d8d9)}
    ${GenActivate(d10d11, d10d11)}
    ${GenActivate(d12d13, d12d13)}
    ${GenActivate(d14d15, d14d15)}
    if (n_remain == 4) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
        GiStoreFloat32(output, d10d11);
        output = output + 4;
        GiStoreFloat32(output, d12d13);
        output = output + 4;
        GiStoreFloat32(output, d14d15);
        output = output + 4;
    } else if (n_remain == 3) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
        GiStoreFloat32(output, d10d11);
        output = output + 4;
        GiStoreFloat32(output, d12d13);
        output = output + 4;
    } else if (n_remain == 2) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
        GiStoreFloat32(output, d10d11);
        output = output + 4;
    } else if (n_remain == 1) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
    }
}
)";
    writer << activation_gen->GenIntrinsicInitFloat();
    writer << StringTemplate::StringTemplateArgs()
                      .add("GenActivate",
                           [=](std::vector<std::string> args) {
                               return activation_gen->GenIntrinsicFloat(
                                       args[0], args[1]);
                           })
                      .render(body_temp);
    return writer.str();
}

std::string gen_pack_a(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    const int PACK_C_SIZE = 4;
    size_t cp_length = (kmax - k0) * PACK_C_SIZE;
    for (int m = y0; m < ymax; m += 4) {
        const float* src = inptr + (m / PACK_C_SIZE) * ldin + k0 * PACK_C_SIZE;
        memcpy(outptr, src, cp_length * sizeof(float));
        outptr += cp_length;
    }
}

    )";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    float tmpbuff[16] = {0.0f};
    const int PACK_C_SIZE = 4;
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = outptr;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr0 = inptr + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            float*  outptr_interleave = access_outptr;
            transpose_1x12_4_s(inptr0, outptr_interleave);
            inptr0 += 48;
            access_outptr += ksize12;
        }
        access_outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            float*  outptr_interleave = access_outptr;
            transpose_1x4_4_s(inptr0, outptr_interleave);
            inptr0 += 16;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            memcpy(tmpbuff, inptr0, sizeof(float) * (xmax - x) * PACK_C_SIZE);
            float*  outptr_interleave = access_outptr;
            const float* tmp_ptr = &tmpbuff[0];
            transpose_1x4_4_s(tmp_ptr, outptr_interleave);
            access_outptr += ksize4;
        }
        outptr_base += 12 * PACK_C_SIZE;
        outptr_base4 += 4 * PACK_C_SIZE;
    }
}

    )";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)(kmax - k0) * (ymax - y0) * sizeof(float);
        return res;
}

    )";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_n = 12;
        const size_t packed_hw = (xmax - x0 + packed_n - 1) / packed_n * packed_n;
        size_t res = (size_t)(kmax - k0) * packed_hw * sizeof(float);
        return res;
}

    )";
    return ss.str();
}

std::string gen_kernel(const std::string& sig, TContext* ctx,
                       const std::string& postprocess_call,
                       const std::string& preset_str = "") {
    std::string keren_body =
            R"(
    ${kernel_sig}{
        ${preset_str}
        const int m_block = 4;
        const int n_block = 12;
        const int pack_mk = 4;
        const int K12 = K * 12;
        const int K4 = K * 4;
        size_t m = 0;        
        for (; m + m_block <= M; m += m_block) {
            float* output = C + (m / pack_mk * LDC);

            size_t n = 0;
            const float* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_4x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr);
                output += n_block * pack_mk;
                cur_pack_b += K12;
            }

            for (; n < N; n += 4) {                
                kern_4x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, N - n > 4 ? 4 : N - n);
                output += 4 * pack_mk;
                cur_pack_b += K4;
            }
            pack_a += K4;
            bias_ptr += m_block;
        }        
        ${postprocess_call}
    }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("postprocess_call", postprocess_call)
            .add("preset_str", preset_str)
            .add("kernel_sig", sig)
            .render(keren_body);
}

}  // namespace

std::string MatmulM4N12MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "GI_fp32_m4_n12_k4_matmul";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") &&
        ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    return ss.str();
}

std::string MatmulM4N12MK4Kernel::GetKernelBody(TContext* ctx) const {
    auto postprocess_pair = gen_postprocess_inline(ctx);
    std::stringstream writer;
    writer << R"(
#include <math.h>
#include <string.h>
#include "gi_float.h"
)";
    writer << transpose_1x12_4_s();
    writer << transpose_1x4_4_s();
    writer << kern_4x12(ctx);
    writer << kern_4x4(ctx);
    writer << gen_pack_a(GetPackASignature(ctx));
    writer << gen_pack_b(GetPackBSignature(ctx));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << postprocess_pair.first;
    writer << gen_kernel(GetNakedKernelSignature(ctx), ctx,
                         postprocess_pair.second);

    std::string preset_temp = R"(
        size_t pack_a_size = ${packa_workspace_sym}(0, M, 0, K);
        float* pack_a = workspace;
        float* pack_b = workspace + pack_a_size;
        ${packa_sym}(pack_a, A, LDA, 0, M, 0, K);
        ${packb_sym}(pack_b, B, LDB, 0, N, 0, K);
    )";
    std::string preset_str =
            StringTemplate::StringTemplateArgs()
                    .add("packa_workspace_sym", GetPackAWorkspaceSymbol(ctx))
                    .add("packa_sym", GetPackASymbol(ctx))
                    .add("packb_sym", GetPackBSymbol(ctx))
                    .render(preset_temp);
    writer << gen_kernel(GetKernelSignature(ctx), ctx, postprocess_pair.second,
                         preset_str);
    return writer.str();
}

std::string MatmulM4N12MK4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string MatmulM4N12MK4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
