#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ConvKernel/ConvKernel.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

std::string WinogradF23Strategy8x8MK8::WeightTrans(
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
    const uint32_t KERNEL_SIZE = 3;
    size_t OCB = ${OC} / PACK_C_SIZE ;
    size_t ICB = ${IC} / PACK_C_SIZE ;

    for (size_t ocb = 0; ocb < OCB; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < PACK_C_SIZE ; ic_inner++) {
                const gi_float16_t* fptr = ${filter} + (ocb * ICB + icb) * KERNEL_SIZE *
                      KERNEL_SIZE * PACK_C_SIZE  * PACK_C_SIZE  +
                      ic_inner * PACK_C_SIZE ;
                //! read 4OC 1IC filter
                GI_FLOAT16_t g00 = GiLoadFloat16(fptr + 0* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g01 = GiLoadFloat16(fptr + 1* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g02 = GiLoadFloat16(fptr + 2* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g10 = GiLoadFloat16(fptr + 3* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g11 = GiLoadFloat16(fptr + 4* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g12 = GiLoadFloat16(fptr + 5* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g20 = GiLoadFloat16(fptr + 6* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g21 = GiLoadFloat16(fptr + 7* PACK_C_SIZE  * PACK_C_SIZE );
                GI_FLOAT16_t g22 = GiLoadFloat16(fptr + 8* PACK_C_SIZE  * PACK_C_SIZE );

                //! twice matmul
                GI_FLOAT16_t tmp0, tmp1;
                ${FilterTransUnroll(3, midle, g, tmp0, tmp1)}
                ${FilterTransUnroll(4, ret, midle, tmp0, tmp1)}

                //! write to the dst
                gi_float16_t* dst = ${outptr};
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
            ss << "GI_FLOAT16_t " << dst << i << "0 = " << src << "0" << i << ";\n";
            ss << tmp0 << " = GiMultiplyScalerFloat16(GiAddFloat16(" << src << "0" << i
               << ", " << src << "2" << i << "), 0.5);\n";
            ss << tmp1 << " = GiMultiplyScalerFloat16(" << src << "1" << i
               << ", 0.5f);\n";
            ss << "GI_FLOAT16_t " << dst << i << "1 = GiAddFloat16(" << tmp0 << ", "
               << tmp1 << ");\n";
            ss << "GI_FLOAT16_t " << dst << i << "2 = GiSubtractFloat16(" << tmp0
               << ", " << tmp1 << ");\n";
            ss << "GI_FLOAT16_t " << dst << i << "3 = " << src << "2" << i << ";\n";
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
                ss << "GiStoreFloat16(dst + (" << out << " * Alpha + " << inner
                   << ") * OCB * ICB * PACK_C_SIZE*PACK_C_SIZE + ocb * ICB "
                      "*PACK_C_SIZE *PACK_C_SIZE + icb* "
                      "PACK_C_SIZE*PACK_C_SIZE + "
                      "ic_inner*PACK_C_SIZE, "
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

std::string WinogradF23Strategy8x8MK8::InputFeatureTrans(
        const std::vector<std::string>& strs) {
    auto InputTransformF23NCHW88 = [](std::vector<std::string>) {
        std::stringstream ss;
        std::string kernel = R"(
        size_t PACK_C_SIZE = 8;
        size_t IW8 = IW * PACK_C_SIZE;
        size_t icb = ic / PACK_C_SIZE;
        size_t iw8_start = iw_start * PACK_C_SIZE;
        size_t ICB = IC / PACK_C_SIZE;

#define cb(m, n) GI_FLOAT16_t d##m##n;
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb

        if (!(inner && ic + PACK_C_SIZE < IC)) {
            memset(patchT, 0, sizeof(gi_float16_t) * PACK_C_SIZE * Alpha * Alpha);
        }
        if (inner) { 
            const gi_float16_t* input_ptr_ =
                    src + icb * IH * IW8 + ih_start * IW8 + iw8_start;
#define cb(n, m) d##m##n = GiLoadFloat16(input_ptr_ + PACK_C_SIZE * n);

            UNROLL_CALL_RAW(4, cb, 0);
            input_ptr_ += IW8;
            UNROLL_CALL_RAW(4, cb, 1);
            input_ptr_ += IW8;
            UNROLL_CALL_RAW(4, cb, 2);
            input_ptr_ += IW8;
            UNROLL_CALL_RAW(4, cb, 3);
#undef cb
        } else {
            int ih0_act = ih_start> 0?ih_start:0;
            int ih1_act = (ih_start + Alpha) < IH?(ih_start + Alpha):IH;
            int iw0_act = iw_start> 0?iw_start:0;
            int iw1_act = (iw_start + Alpha) < IW? (iw_start + Alpha) :IW;
            const gi_float16_t* input_ptr_ = src + icb * IH * IW8;            
            // partial copy
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    GI_FLOAT16_t tmp = GiLoadFloat16(input_ptr_ + ih * IW8 + iw * PACK_C_SIZE);
                    GiStoreFloat16(
                            patchT + iho * Alpha * PACK_C_SIZE + iwo * PACK_C_SIZE, tmp);
                }
            }
#define cb(m, n) \
    d##m##n = GiLoadFloat16(patchT + m * Alpha * PACK_C_SIZE + n * PACK_C_SIZE);
            UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
        }

        //! 1   0 -1 0    d00 d01 d02 d03     1 0  0  0
        //! 0   1  1 0    d10 d11 d12 d13     0 1 -1 -1
        //! 0  -1  1 0    d20 d21 d22 d23    -1 1  1  0
        //! 0  -1  0 1    d30 d31 d32 d33     0 0  0  1
#define cb(m)                          \
    GI_FLOAT16_t t0##m = GiSubtractFloat16 (d0##m, d2##m); \
    GI_FLOAT16_t t1##m = GiAddFloat16 (d1##m, d2##m); \
    GI_FLOAT16_t t2##m = GiSubtractFloat16 (d2##m, d1##m); \
    GI_FLOAT16_t t3##m = GiSubtractFloat16 (d3##m, d1##m);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m)                           \
    d##m##0 = GiSubtractFloat16 (t##m##0, t##m##2); \
    d##m##1 = GiAddFloat16 (t##m##1, t##m##2); \
    d##m##2 = GiSubtractFloat16 (t##m##2, t##m##1); \
    d##m##3 = GiSubtractFloat16 (t##m##3, t##m##1);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#define cb(m, n)                                                               \
    GiStoreFloat16(                                                            \
            dst + (m * Alpha + n) * ICB * nr_tiles_in_loop_ * PACK_C_SIZE +     \
                    icb * nr_tiles_in_loop_ * PACK_C_SIZE + tile_idx * PACK_C_SIZE, \
            d##m##n);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb)";
        return kernel;
    };
    std::string input_process = R"(
    const uint32_t Alpha = 3 + 2 - 1;
    const uint32_t OutputBlockSize = 2;
    const uint32_t KS = 3;

    gi_float16_t* dst = ${transform_input_ptr_};
    const gi_float16_t* src = ${inptr};
    uint32_t IH_ = ${IH};
    uint32_t IW_ = ${IW};
    uint32_t IC_ = ${IC};
    uint32_t PH_ = ${PH};
    uint32_t PW_ = ${PW};
    uint32_t nr_tiles_in_loop_ = ${nr_tiles_in_loop};
    uint32_t tile_id_ = ${tile_id};

    uint32_t OW = IW_ + 2 * PW_ - KS + 1;
    uint32_t tiles_w = (OW + OutputBlockSize -1)/ OutputBlockSize;
    gi_float16_t* patchT = transform_mid_ptr + PACK_C_SIZE * Alpha * Alpha;

    for (uint32_t ic = 0; ic < IC_; ic += PACK_C_SIZE) {
        uint32_t tile_start_id = tile_id_;
        for(uint32_t tile_idx = 0; tile_idx < nr_tiles_in_loop_; tile_idx++) {
            uint32_t index = tile_start_id + tile_idx;
            uint32_t nh = index / tiles_w;
            uint32_t nw = index % tiles_w;
            int ih_start = nh * OutputBlockSize - PH_;
            int iw_start = nw * OutputBlockSize - PW_;
            int inner = (ih_start >= 0 && iw_start >= 0 &&
                        ih_start + Alpha <= (int)IH_ &&
                        iw_start + Alpha <= (int)IW_);
            ${InputTransformF23NCHW88()}
        }
    })";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("inptr", strs[0])
                    .add("transform_input_ptr_", strs[1])
                    .add("IH", strs[2])
                    .add("IW", strs[3])
                    .add("IC", strs[4])
                    .add("PH", strs[5])
                    .add("PW", strs[6])
                    .add("tile_id", strs[7])
                    .add("nr_tiles_in_loop", strs[8])
                    .add("InputTransformF23NCHW88", InputTransformF23NCHW88)
                    .render(input_process);
    return ss.str();
}

std::string WinogradF23Strategy8x8MK8::DependMatmulSymbol() {
    return Fp16MatmulM8N8MK8Kernel().GetKernelSymbol(nullptr);
}

std::string WinogradF23Strategy8x8MK8::OutputFeatureTrans(
        const std::vector<std::string>& strs, TContext* ctx) {
    std::string ouput_trans = R"(
    gi_float16_t* transform_output_ptr_ = ${transform_output_ptr};
    gi_float16_t* outptr_ = ${outptr};
    const gi_float16_t* bias_ptr_ = ${bias_ptr};
    uint32_t OH_ = ${OH};
    uint32_t OW_ = ${OW};
    uint32_t OC_ = ${OC};
    uint32_t tile_id_ = ${tile_id};
    uint32_t nr_tiles_in_loop_ = ${nr_tiles_in_loop};

    const uint32_t OutputBlockSize = 2;
    uint32_t tiles_w_ = (OW_ + OutputBlockSize -1) / OutputBlockSize;
    size_t OCB =OC_/ PACK_C_SIZE;
   for (size_t oc = 0; oc < OC_; oc += PACK_C_SIZE) {                                                         
        for(size_t tile_idx = 0 ;tile_idx < nr_tiles_in_loop_;++tile_idx) {                                         
            size_t index = tile_id_+tile_idx;                             
            size_t nh = index / tiles_w_;                                            
            size_t nw = index % tiles_w_;                                            
            size_t oh_start = nh * OutputBlockSize;                             
            size_t ow_start = nw * OutputBlockSize;                          
            //! AT * m * A
            size_t ocb = oc / PACK_C_SIZE;

    #define cb(m, n)                                                   \
        GI_FLOAT16_t v##m##n = GiLoadFloat16(                                  \
                transform_output_ptr_ +                                 \
                (m * Alpha + n) * OCB * nr_tiles_in_loop_ * PACK_C_SIZE + \
                ocb * nr_tiles_in_loop_ * PACK_C_SIZE + tile_idx * PACK_C_SIZE);
            UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
    #undef cb

            //! 1  1  1 0  v00 v01 v02 v03    1  0
            //! 0  1 -1 1  v10 v11 v12 v13    1  1
            //!            v20 v21 v22 v23    1 -1
            //!            v30 v31 v32 v33    0  1

    #define cb(m)                                         \
        GI_FLOAT16_t t0##m = GiAddFloat16 (GiAddFloat16 (v0##m, v1##m), v2##m); \
        GI_FLOAT16_t t1##m = GiAddFloat16 (GiSubtractFloat16 (v1##m, v2##m), v3##m);

            UNROLL_CALL_NOWRAPPER(4, cb);
    #undef cb

    #define cb(m)                                            \
        v##m##0 = GiAddFloat16 (GiAddFloat16 (t##m##0, t##m##1), t##m##2); \
        v##m##1 = GiAddFloat16 (GiSubtractFloat16 (t##m##1, t##m##2), t##m##3);

            UNROLL_CALL_NOWRAPPER(2, cb);
    #undef cb
            ${nonline_gen_init()}
            GI_FLOAT16_t vbias;
            if (bias_ptr_) {
                vbias = GiLoadFloat16(bias_ptr_ + oc);

    #define cb(m, n) v##m##n = GiAddFloat16 (v##m##n, vbias);
                UNROLL_CALL_RAW_D2(2, 2, cb);
    #undef cb
            }
    ${nonline_gen_func(v00, vbias)};v00=vbias;
    ${nonline_gen_func(v01, vbias)};v01=vbias;
    ${nonline_gen_func(v10, vbias)};v10=vbias;
    ${nonline_gen_func(v11, vbias)};v11=vbias;

            
    #define out_save(oho, owo)                                                           \
        do {                                                                             \
            size_t oh = oh_start + oho;                                                  \
            size_t ow = ow_start + owo;                                                  \
            if (oh < OH && ow < OW) {                                                    \
                GiStoreFloat16(                                                          \
                        outptr_ + oc * OH * OW + oh * OW * PACK_C_SIZE + ow * PACK_C_SIZE,    \
                        v##oho##owo);                                                    \
            }                                                                            \
        } while (0);
            UNROLL_CALL_RAW_D2(2, 2, out_save);
    #undef out_save                                                                                               
        }
    })";
    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode, "f16");
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
