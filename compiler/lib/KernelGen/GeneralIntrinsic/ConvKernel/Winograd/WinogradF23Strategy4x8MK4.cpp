/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Winograd/WinogradF23Strategy4x8MK4.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "WinogradF23Strategy4x8MK4.h"
#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ConvKernel/ConvKernel.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

std::string WinogradF23Strategy4x8MK4::WeightTrans(
        const std::vector<std::string>& strs) {
    auto inptr = strs[0];
    auto outptr = strs[1];
    auto OC = strs[2];
    auto IC = strs[3];
    std::string filter_process = R"(
    //! 1      0    0    v00 v01 v02   1 0.5  0.5 0
    //! 0.5  0.5  0.5    v10 v11 v12   0 0.5 -0.5 0
    //! 0.5 -0.5  0.5    v20 v21 v22   0 0.5  0.5 1
    //! 0      0    1
    const uint32_t PACK_SIZE= 4;
    const uint32_t KERNEL_SIZE = 3;
    size_t OCB = ${OC} / PACK_SIZE;
    size_t ICB = ${IC} / PACK_SIZE;

    for (size_t ocb = 0; ocb < OCB; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < PACK_SIZE; ic_inner++) {
                const float* fptr = ${filter} + (ocb * ICB + icb) * KERNEL_SIZE *
                      KERNEL_SIZE * PACK_SIZE * PACK_SIZE +
                      ic_inner * PACK_SIZE;
                //! read 4OC 1IC filter
                GI_FLOAT32_t g00 = GiLoadFloat32(fptr + 0* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g01 = GiLoadFloat32(fptr + 1* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g02 = GiLoadFloat32(fptr + 2* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g10 = GiLoadFloat32(fptr + 3* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g11 = GiLoadFloat32(fptr + 4* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g12 = GiLoadFloat32(fptr + 5* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g20 = GiLoadFloat32(fptr + 6* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g21 = GiLoadFloat32(fptr + 7* PACK_SIZE * PACK_SIZE);
                GI_FLOAT32_t g22 = GiLoadFloat32(fptr + 8* PACK_SIZE * PACK_SIZE);

                //! twice matmul
                GI_FLOAT32_t tmp0, tmp1;
                ${FilterTransUnroll(3, midle, g, tmp0, tmp1)}
                ${FilterTransUnroll(4, ret, midle, tmp0, tmp1)}

                //! write to the dst
                float* dst = ${outptr};
                ${StoreRet2D(4, 4, ret)};
            }
        }
    })";
    auto FilterTransUnroll = [](const std::vector<std::string>& strs) {
        int times = std::stoi(strs[0]);
        std::string dst = strs[1];
        std::string src = strs[2];
        std::string tmp0 = strs[3];
        std::string tmp1 = strs[4];
        std::stringstream ss;
        for (int i = 0; i < times; i++) {
            ss << "GI_FLOAT32_t " << dst << i << "0 = " << src << "0" << i
               << ";\n";
            ss << tmp0 << " = GiMultiplyScalerFloat32(GiAddFloat32(" << src
               << "0" << i << ", " << src << "2" << i << "), 0.5f);\n";
            ss << tmp1 << " = GiMultiplyScalerFloat32(" << src << "1" << i
               << ", 0.5f);\n";
            ss << "GI_FLOAT32_t " << dst << i << "1 = GiAddFloat32(" << tmp0
               << ", " << tmp1 << ");\n";
            ss << "GI_FLOAT32_t " << dst << i << "2 = GiSubtractFloat32("
               << tmp0 << ", " << tmp1 << ");\n";
            ss << "GI_FLOAT32_t " << dst << i << "3 = " << src << "2" << i
               << ";\n";
        }
        return ss.str();
    };

    auto StoreRet2D = [](const std::vector<std::string>& strs) {
        int times_out = std::stoi(strs[0]);
        int times_inner = std::stoi(strs[1]);
        std::string src = strs[2];
        std::stringstream ss;
        for (int out = 0; out < times_out; out++) {
            for (int inner = 0; inner < times_inner; inner++) {
                ss << "GiStoreFloat32(dst + (" << out << " * 4 + " << inner
                   << ") * OCB * ICB * 4*4 + ocb * ICB *4 *4 + icb* 4*4 + "
                      "ic_inner*4, "
                   << src << out << inner << ");\n";
            }
        }
        return ss.str();
    };
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("StoreRet2D", StoreRet2D)
                    .add("FilterTransUnroll", FilterTransUnroll)
                    .add("OC", OC)
                    .add("IC", IC)
                    .add("filter", inptr)
                    .add("outptr", outptr)
                    .render(filter_process);
    return ss.str();
}

std::string WinogradF23Strategy4x8MK4::InputFeatureTrans(
        const std::vector<std::string>& strs) {
    auto InputTransformF23NCHW44 = [](std::vector<std::string>) {
        std::stringstream ss;
        std::string kernel = R"(
        size_t IW4 = IW_ * 4;
        size_t icb = ic / 4;
        size_t iw4_start = iw_start * 4;
        size_t ICB = IC_ / 4;

        GI_FLOAT32_FIXLEN_t dv[4][4];
        if (inner) {
            const float* input_ptr =
                    source + icb * IH_ * IW4 + ih_start * IW4 + iw4_start;
#define LOAD_V(m) \
            dv[m][0]=GiFloat32Type2FixLenType(GiLoadFloat32(input_ptr));        \
            dv[m][1]=GiFloat32Type2FixLenType(GiLoadFloat32(input_ptr + 1 * 4));\
            dv[m][2]=GiFloat32Type2FixLenType(GiLoadFloat32(input_ptr + 2 * 4));\
            dv[m][3]=GiFloat32Type2FixLenType(GiLoadFloat32(input_ptr + 3 * 4));

            LOAD_V(0)
            input_ptr += IW4;
            LOAD_V(1)
            input_ptr += IW4;
            LOAD_V(2)
            input_ptr += IW4;
            LOAD_V(3)
#undef LOAD_V
        } else {
#define SET_ZERO_V(m)                 \
            dv[m][0]=GiFloat32Type2FixLenType(GiBroadcastFloat32(0.f));\
            dv[m][1]=GiFloat32Type2FixLenType(GiBroadcastFloat32(0.f));\
            dv[m][2]=GiFloat32Type2FixLenType(GiBroadcastFloat32(0.f));\
            dv[m][3]=GiFloat32Type2FixLenType(GiBroadcastFloat32(0.f));
            SET_ZERO_V(0);
            SET_ZERO_V(1);
            SET_ZERO_V(2);
            SET_ZERO_V(3);
#undef SET_ZERO_V
#define MAX(a, b) (a) > (b)? (a):(b);
#define MIN(a, b) (a) > (b)? (b):(a);
            int ih0_act = MAX(ih_start, 0);
            int ih1_act = MIN(ih_start + Alpha, IH_);
            int iw0_act = MAX(iw_start, 0);
            int iw1_act = MIN(iw_start + Alpha, IW_);
            const float* input_ptr = source + icb * IH_ * IW4;
            // partial read
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    uint32_t iho = ih - ih_start, iwo = iw - iw_start;
                    dv[iho][iwo] = GiFloat32Type2FixLenType(GiLoadFloat32(input_ptr + ih * IW4 + iw * 4));
                }
            }
        }
        //! 1   0 -1 0    d00 d01 d02 d03     1 0  0  0
        //! 0   1  1 0    d10 d11 d12 d13     0 1 -1 -1
        //! 0  -1  1 0    d20 d21 d22 d23    -1 1  1  0
        //! 0  -1  0 1    d30 d31 d32 d33     0 0  0  1
        GI_FLOAT32_FIXLEN_t tmp[4][4];
#define MULTI_ONE(m)                           \
    tmp[0][m] = GiFloat32Type2FixLenType(GiSubtractFloat32(GiFixLenType2GiFloat32Type(dv[0][m]), GiFixLenType2GiFloat32Type(dv[2][m]))); \
    tmp[1][m] = GiFloat32Type2FixLenType(GiAddFloat32(GiFixLenType2GiFloat32Type(dv[1][m]),      GiFixLenType2GiFloat32Type(dv[2][m]))); \
    tmp[2][m] = GiFloat32Type2FixLenType(GiSubtractFloat32(GiFixLenType2GiFloat32Type(dv[2][m]), GiFixLenType2GiFloat32Type(dv[1][m]))); \
    tmp[3][m] = GiFloat32Type2FixLenType(GiSubtractFloat32(GiFixLenType2GiFloat32Type(dv[3][m]), GiFixLenType2GiFloat32Type(dv[1][m])));

    MULTI_ONE(0)
    MULTI_ONE(1)
    MULTI_ONE(2)
    MULTI_ONE(3)
#undef MULTI_ONE

#define MULTI_TWO(m)                           \
    dv[m][0] = GiFloat32Type2FixLenType(GiSubtractFloat32(GiFixLenType2GiFloat32Type(tmp[m][0]), GiFixLenType2GiFloat32Type(tmp[m][2]))); \
    dv[m][1] = GiFloat32Type2FixLenType(GiAddFloat32(GiFixLenType2GiFloat32Type(tmp[m][1]), GiFixLenType2GiFloat32Type(tmp[m][2]))); \
    dv[m][2] = GiFloat32Type2FixLenType(GiSubtractFloat32(GiFixLenType2GiFloat32Type(tmp[m][2]), GiFixLenType2GiFloat32Type(tmp[m][1]))); \
    dv[m][3] = GiFloat32Type2FixLenType(GiSubtractFloat32(GiFixLenType2GiFloat32Type(tmp[m][3]), GiFixLenType2GiFloat32Type(tmp[m][1])));

    MULTI_TWO(0)
    MULTI_TWO(1)
    MULTI_TWO(2)
    MULTI_TWO(3)
#undef MULTI_TWO

#define STORE(m, n)                                                 \
    GiStoreFloat32(dst + (m * Alpha + n) * ICB * nr_tiles_in_loop_ * 4 + \
                 icb * nr_tiles_in_loop_ * 4 + tile_idx * 4, GiFixLenType2GiFloat32Type(dv[m][n]));

        STORE(0,0);STORE(0,1);STORE(0,2);STORE(0,3);
        STORE(1,0);STORE(1,1);STORE(1,2);STORE(1,3);
        STORE(2,0);STORE(2,1);STORE(2,2);STORE(2,3);
        STORE(3,0);STORE(3,1);STORE(3,2);STORE(3,3);
#undef STORE)";
        return kernel;
    };
    std::string input_process = R"(
    const uint32_t Alpha = 3 + 2 - 1;
    const uint32_t OUTPUT_BLOCK_SIZE = 2;
    const uint32_t KS = 3;

    float* dst = ${transform_input_ptr};
    const float* source = ${inptr};
    uint32_t IH_ = ${IH};
    uint32_t IW_ = ${IW};
    uint32_t IC_ = ${IC};
    uint32_t PH_ = ${PH};
    uint32_t PW_ = ${PW};
    uint32_t nr_tiles_in_loop_ = ${nr_tiles_in_loop};
    uint32_t tile_id_ = ${tile_id};

    uint32_t OW = IW_ + 2 * PW_ - KS + 1;
    uint32_t tiles_w = (OW + OUTPUT_BLOCK_SIZE -1)/ OUTPUT_BLOCK_SIZE;

    for (uint32_t ic = 0; ic < IC_; ic += 4) {
        uint32_t tile_start_id = tile_id_;
        for(uint32_t tile_idx = 0; tile_idx < nr_tiles_in_loop_; tile_idx++) {
            uint32_t index = tile_start_id + tile_idx;
            uint32_t nh = index / tiles_w;
            uint32_t nw = index % tiles_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH_;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW_;
            int inner = (ih_start >= 0 && iw_start >= 0 &&
                        ih_start + Alpha <= (int)IH_ &&
                        iw_start + Alpha <= (int)IW_);
            ${InputTransformF23NCHW44()}
        }
    })";
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
                    .add("InputTransformF23NCHW44", InputTransformF23NCHW44)
                    .render(input_process);
    return ss.str();
}

std::string WinogradF23Strategy4x8MK4::DependMatmulSymbol() {
    return MatmulM4N8MK4Kernel().GetKernelSymbol(nullptr);
}

std::string WinogradF23Strategy4x8MK4::BatchedMatMul(
        const std::vector<std::string>& strs) {
    std::string matmul_compute = R"(
    for(uint32_t i =0; i< Alpha; i++){
        for(uint32_t j=0; j<Alpha; j++){
            const float* a_ptr = ${A_ptr} +
                (i * Alpha + j) * ${OC} * ${IC};
            float* b_ptr = ${B_ptr} +
                (i * Alpha + j) * ${nr_tiles_in_loop} * ${IC};
            float* c_ptr = ${C_ptr} +
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

std::string WinogradF23Strategy4x8MK4::OutputFeatureTrans(
        const std::vector<std::string>& strs, TContext* ctx) {
    std::string ouput_trans = R"(
    float* transform_output_ptr_ = ${transform_output_ptr};
    float* outptr_ = ${outptr};
    const float* bias_ptr_ = ${bias_ptr};
    uint32_t OH_ = ${OH};
    uint32_t OW_ = ${OW};
    uint32_t OC_ = ${OC};
    uint32_t tile_id_ = ${tile_id};
    uint32_t nr_tiles_in_loop_ = ${nr_tiles_in_loop};

    const uint32_t OutputBlockSize = 2;
    uint32_t tiles_w_ = (OW_ + OutputBlockSize -1) / OutputBlockSize;

    for (uint32_t oc = 0; oc < OC_; oc += 4) {
        for(uint32_t tile_idx = 0; tile_idx < nr_tiles_in_loop_; tile_idx++) {
            uint32_t index = tile_id_ + tile_idx;
            uint32_t nh = index / tiles_w_;
            uint32_t nw = index % tiles_w_;
            uint32_t oh_start = nh * OutputBlockSize;
            uint32_t ow_start = nw * OutputBlockSize;

            //! AT * m * A
            uint32_t OCB = OC_ / 4;

            GI_FLOAT32_FIXLEN_t src[4][4];

#define LOAD_V(m, n)                                               \
    src[m][n] = GiFloat32Type2FixLenType(GiLoadFloat32(transform_output_ptr_ +                  \
            (m * Alpha + n) * OCB * nr_tiles_in_loop_ * 4 +        \
            oc * nr_tiles_in_loop_ + tile_idx * 4));

            LOAD_V(0,0);LOAD_V(0,1);LOAD_V(0,2);LOAD_V(0,3);
            LOAD_V(1,0);LOAD_V(1,1);LOAD_V(1,2);LOAD_V(1,3);
            LOAD_V(2,0);LOAD_V(2,1);LOAD_V(2,2);LOAD_V(2,3);
            LOAD_V(3,0);LOAD_V(3,1);LOAD_V(3,2);LOAD_V(3,3);
#undef LOAD_V

            //! 1  1  1 0  v00 v01 v02 v03    1  0
            //! 0  1 -1 1  v10 v11 v12 v13    1  1
            //!            v20 v21 v22 v23    1 -1
            //!            v30 v31 v32 v33    0  1

            GI_FLOAT32_FIXLEN_t mid[2][4];
#define MULTI_ONE(m)                                                   \
    mid[0][m] = GiFloat32Type2FixLenType(GiAddFloat32(GiAddFloat32(     GiFixLenType2GiFloat32Type(src[0][m]), GiFixLenType2GiFloat32Type(src[1][m])), GiFixLenType2GiFloat32Type(src[2][m]))); \
    mid[1][m] = GiFloat32Type2FixLenType(GiAddFloat32(GiSubtractFloat32(GiFixLenType2GiFloat32Type(src[1][m]), GiFixLenType2GiFloat32Type(src[2][m])), GiFixLenType2GiFloat32Type(src[3][m])));

            MULTI_ONE(0);
            MULTI_ONE(1);
            MULTI_ONE(2);
            MULTI_ONE(3);
#undef MULTI_ONE

            GI_FLOAT32_FIXLEN_t dst_v[2][2];
#define MULTI_TWO(m)                                                             \
            dst_v[m][0] = GiFloat32Type2FixLenType(GiAddFloat32(GiAddFloat32(     GiFixLenType2GiFloat32Type(mid[m][0]), GiFixLenType2GiFloat32Type(mid[m][1])), GiFixLenType2GiFloat32Type(mid[m][2]))); \
            dst_v[m][1] = GiFloat32Type2FixLenType(GiAddFloat32(GiSubtractFloat32(GiFixLenType2GiFloat32Type(mid[m][1]), GiFixLenType2GiFloat32Type(mid[m][2])), GiFixLenType2GiFloat32Type(mid[m][3])));

            MULTI_TWO(0);
            MULTI_TWO(1);
#undef MULTI_TWO

            if (bias_ptr) {
                GI_FLOAT32_t vbias = GiLoadFloat32(bptr + oc);
                dst_v[0][0]= GiFloat32Type2FixLenType(GiAddFloat32(GiFixLenType2GiFloat32Type(dst_v[0][0]), vbias));
                dst_v[0][1]= GiFloat32Type2FixLenType(GiAddFloat32(GiFixLenType2GiFloat32Type(dst_v[0][1]), vbias));
                dst_v[1][0]= GiFloat32Type2FixLenType(GiAddFloat32(GiFixLenType2GiFloat32Type(dst_v[1][0]), vbias));
                dst_v[1][1]= GiFloat32Type2FixLenType(GiAddFloat32(GiFixLenType2GiFloat32Type(dst_v[1][1]), vbias));
            }

            //! fuse activation
            GI_FLOAT32_t dst_ans0, dst_ans1, dst_ans2, dst_ans3; 
            ${nonline_gen_init()}
            ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][0]), dst_ans0)}
            ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][1]), dst_ans1)}
            ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][0]), dst_ans2)}
            ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][1]), dst_ans3)}
            dst_v[0][0]= GiFloat32Type2FixLenType(dst_ans0);
            dst_v[0][1]= GiFloat32Type2FixLenType(dst_ans1);
            dst_v[1][0]= GiFloat32Type2FixLenType(dst_ans2);
            dst_v[1][1]= GiFloat32Type2FixLenType(dst_ans3);


#define STORE(oho, owo)                                        \
    do {                                                       \
        uint32_t oh = oh_start + oho;                          \
        uint32_t ow = ow_start + owo;                          \
        if (oh < OH_ && ow < OW_) {                            \
            GiStoreFloat32(outptr + oc * OH_ * OW_ + oh * OW_ * 4 + \
                             ow * 4, GiFixLenType2GiFloat32Type(dst_v[oho][owo]));         \
        }                                                      \
    } while (0);

              STORE(0, 0);
              STORE(0, 1);
              STORE(1, 0);
              STORE(1, 1);
#undef STORE
        }
    })";
    std::string nonline_mode = ctx->haveAttr("nonlineMode")
                                       ? ctx->getAttrStr("nonlineMode")
                                       : "IDENTITY";
    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        return nonline_gen->GenIntrinsicFloat(str[0], str[1]);
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
