/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/ConvKernel/Int8/Int8DotConvNchwNchw44.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <sstream>
#include <string>
#include "Arm/ARMBackend.h"
#include "Arm/Arm64/Activation.h"
#include "Arm/Arm64/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;

bool ConvDotNCHWNCHW44::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 2 &&
            ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW44_DOT" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU";

    bool type_ok = is_qint8_conv_dtype(ctx);
    bool layout_ok =
            ctx->getAttrOprand("operand:0").shape.size() == 4 &&
            ctx->getAttrOprand(
                       "operand:" +
                       std::to_string(ctx->getAttrInt("nr_operands") - 1))
                            .shape.size() == 5;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}
std::string ConvDotNCHWNCHW44::GetKernelSymbol(TContext* ctx) const {
    auto src_tensor = ctx->getAttrOprand("operand:0");
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    uint32_t oc = dst_tensor.shape[1] * 4;
    std::string name_temp =
            "${base_kernel_sym}_dot_nchw_nchw44_oc${oc}_ic${ic}";
    return StringTemplate::StringTemplateArgs(ctx)
            .add("base_kernel_sym", Arm64ConvImpl::GetKernelSymbol(ctx))
            .add("oc", oc)
            .add("ic", ic)
            .render(name_temp);
}

std::string ConvDotNCHWNCHW44::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto src_tensor = ctx->getAttrOprand("operand:0");
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    uint32_t oc = dst_tensor.shape[1] * 4;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    writer << R"(
        #include <arm_neon.h>
        #include <stdint.h>
        #include <string.h>
    )";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    Tensor* in_weights = inputs[1];
                      )";
    std::string fill_weight_attr = R"(
    out_weights->layout.nr_dim = 1;
    int oc = in_weights->layout.dims[0] * 4;
    int kh = in_weights->layout.dims[1];
    int kw = in_weights->layout.dims[2];
    int ic = in_weights->layout.dims[3];
    out_weights->layout.dims[0] = oc * ic * kh * ((kw + 3) / 4 * 4);
    out_weights->layout.stride[0] = 1;
    out_weights->dtype.type_enum = TinyNN_QINT8;
    out_weights->name = in_weights->name;
    out_weights->dtype.param.scale = in_weights->dtype.param.scale;
 )";
    std::string fill_weight_transform = StringTemplate::StringTemplateArgs()
                                                .add("ic", ic)
                                                .add("oc", oc)
                                                .add("kh", filter_size)
                                                .add("kw", filter_size)
                                                .render(R"(
#define rep_step(i, n, step) for (int i = 0; i < (n); i += (step)) 
#define rep(i, n) for (int i = 0; i < (n); ++i) 
    const int ic = ${ic};
    const int fh = ${kh};
    const int fw = ${kw};
    const int oc = ${oc};
    const int oc_step = 4;
    const int fw2 = (fw + 3) / 4 * 4;
    int8_t* dst_ptr = out_weights->ptr;
    const int8_t* src_ptr = in_weights->ptr;
    const int fw_remain = fw2 - fw;
    const int dst_ic_stride = fh * fw2;
    const int oc_step_stride = fh * fw2 * ic * oc_step;
    static const uint8_t transpose_4x4_idx[16] = {0, 4, 8,  12, 1, 5, 9,  13,
                                                  2, 6, 10, 14, 3, 7, 11, 15};
    uint8x16_t tbl_transpose_4x4 = vld1q_u8(&transpose_4x4_idx[0]);
    rep_step(oc_idx, oc, oc_step) {
        int32_t* dst_temp_ptr =
                (int32_t*)(dst_ptr + oc_idx * ic * fh * fw2);
        const int32_t* src_temp_ptr = (int32_t*)(
                src_ptr + oc_idx * ic * fh * fw);
        // transpose ic and pad
        rep(fh_idx, fh) {
            rep(fw_idx, fw) {
                rep(ic_idx, ic) {
                    *(dst_temp_ptr + ic_idx * dst_ic_stride) = *src_temp_ptr;
                    src_temp_ptr++;
                }
                dst_temp_ptr++;
            }
            rep(ic_idx, ic) {
                memset(dst_temp_ptr + ic_idx * dst_ic_stride, 0,
                       sizeof(int8_t) * oc_step * fw_remain);
            }
            dst_temp_ptr += fw_remain;
        }
        // transpose fw oc
        int8_t* trans_dst_temp_ptr =
                (int8_t*)(dst_ptr + oc_idx * ic * fh * fw2);

        rep_step(idx, oc_step_stride, 16) {
            int8x16_t temp = vld1q_s8(trans_dst_temp_ptr + idx);
            vst1q_s8(trans_dst_temp_ptr + idx,
                     vqtbl1q_s8(temp, tbl_transpose_4x4));
        }
    }
#undef rep_step
#undef rep
)");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string ConvDotNCHWNCHW44::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << R"(
        static inline int round_up(int x, int d){
            TINYNN_ASSERT(d > 0);
            return (x + d - 1) / d * d;
        }
    )";
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1];
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const uint32_t pad_h = ${pad_h};
        const uint32_t pad_w = ${pad_w};
        const uint32_t packed_iw = iw + 2 * pad_w;
        const uint32_t packed_ih = ih + 2 * pad_h;
        const uint32_t border = 2 * ${cacheline_byte};
        *workspace = (size_t)ic * packed_ih * packed_iw * sizeof(int8_t) + border;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add("cacheline_byte", ARMBackend::cacheline_byte)
                    .render(workspace_temp);
    return ss.str();
}

namespace {
std::string render_init(int c_idx, int nr_ow, bool with_bias) {
    std::stringstream ss;
    for (int src_idx = 0; src_idx < nr_ow; ++src_idx) {
        if (with_bias) {
            ss << "c[" << c_idx << "][" << src_idx
               << "] = vld1q_s32(bias_ptr + " << c_idx << " * 4);";
        } else {
            ss << "c[" << c_idx << "][" << src_idx << "] = vdupq_n_s32(0);";
        }
    }
    return ss.str();
}
std::string render_core(int src_reg_size, int filter_reg_size, int filter_size,
                        bool is_big_oc, int ld_weight_fw, int ld_weight_oc,
                        int simd_len, int nr_ow, int stride, bool remain_n) {
    using Rendor = StringTemplate::StringTemplateArgs;
    std::stringstream fw_ss;
    if (remain_n) {
        fw_ss << R"(
        rep(src_idx, read_src_reg_size){
            src[0][src_idx] = vld1q_s8(src_ptr + ${fh_idx} * packed_iw + src_idx * ${simd_len});
            src[1][src_idx] = vld1q_s8(src_ptr + ${fh_idx} * packed_iw + src_idx * ${simd_len} + 2);
        })";
    } else {
        for (int src_idx = 0; src_idx < src_reg_size; ++src_idx) {
            fw_ss << "src[0][" << src_idx
                  << "] = vld1q_s8(src_ptr + ${fh_idx} * packed_iw + "
                  << src_idx << "* ${simd_len});\n";
            fw_ss << "src[1][" << src_idx
                  << "] = vld1q_s8(src_ptr + ${fh_idx} * packed_iw + "
                  << src_idx << "* ${simd_len} + 2);\n";
        }
    }

    for (int fw_idx = 0; fw_idx < filter_reg_size; ++fw_idx) {
        fw_ss << "weight[0][" << fw_idx
              << "] = vld1q_s8(filter_ptr + ${fh_idx} * ${ld_weight_fw} + "
              << fw_idx << " * ${simd_len});\n";
    }
    if (is_big_oc) {
        for (int fw_idx = 0; fw_idx < filter_reg_size; ++fw_idx) {
            fw_ss << "weight[1][" << fw_idx
                  << "] = vld1q_s8(filter_ptr + ${ld_weight_oc} + ${fh_idx} * "
                     "${ld_weight_fw} + "
                  << fw_idx << " * ${simd_len});\n";
        }
    }
    for (int fw_idx = 0; fw_idx < filter_reg_size; ++fw_idx) {
        auto src_idx = fw_idx;
        auto weight_idx = fw_idx;
        for (int i = 0; i < nr_ow / 2; ++i) {
            fw_ss << "c[0][" << (i * 2) << "] = vdotq_laneq_s32(c[0]["
                  << (i * 2) << "], weight[0][" << weight_idx << "],  src[0][("
                  << i << " + " << src_idx << ") / 4], (" << i << " + "
                  << src_idx << ") % 4);\n";
            if (is_big_oc) {
                fw_ss << "c[1][" << (i * 2) << "] = vdotq_laneq_s32(c[1]["
                      << (i * 2) << "], weight[1][" << weight_idx
                      << "],  src[0][(" << i << " + " << src_idx << ") / 4], ("
                      << i << " + " << src_idx << ") % 4);\n";
            }
            fw_ss << "c[0][" << (i * 2 + 1) << "] = vdotq_laneq_s32(c[0]["
                  << (i * 2 + 1) << "], weight[0][" << weight_idx
                  << "],  src[1][(" << i << " + " << src_idx << ") / 4], (" << i
                  << " + " << src_idx << ") % 4);\n";
            if (is_big_oc) {
                fw_ss << "c[1][" << (i * 2 + 1) << "] = vdotq_laneq_s32(c[1]["
                      << (i * 2 + 1) << "], weight[1][" << weight_idx
                      << "],  src[1][(" << i << " + " << src_idx << ") / 4], ("
                      << i << " + " << src_idx << ") % 4);\n";
            }
        }
    }

    auto base_render = Rendor().add("simd_len", simd_len)
                               .add("ld_weight_fw", ld_weight_fw)
                               .add("stride", stride)
                               .add("ld_weight_oc", ld_weight_oc);
    auto temp_str = fw_ss.str();
    std::stringstream res_ss;
    for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
        res_ss << base_render.add("fh_idx", fh_idx).render(temp_str);
    }
    return res_ss.str();
}
std::string render_store(int nr_ow, int c_idx, const std::string& store_offset,
                         const ActivationGenIntrinsicBase& act) {
    std::stringstream ss;
    for (int ow_idx = 0; ow_idx < nr_ow; ++ow_idx) {
        ss << act.GenIntrinsicQuantStore("c[" + std::to_string(c_idx) + "][" +
                                                 std::to_string(ow_idx) + "]",
                                         "dst_ptr + " + store_offset + " + " +
                                                 std::to_string(ow_idx) +
                                                 " * 4",
                                         "scale");
    }
    return ss.str();
}

std::string render_kernel(TContext* ctx) {
    std::stringstream ss;
    ss << R"(typedef void kern_func (const int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr,
                                        int8_t* dst_ptr, const int packed_iw, const int packed_ic_stride, const int ld_dst_oc, const float scale);
    )";

    auto mode = ctx->getAttrStr("nonlineMode");
    auto activate_gen = create_activation_gener_instrinsic(mode);

    auto src_tensor = ctx->getAttrOprand("operand:0");
    int ic = src_tensor.shape[1];
    auto dst_tensor = Utils::get_last_operand(ctx);
    constexpr int big_oc_step = 8;
    constexpr int packed_oc = 4;
    constexpr int packed_fw = 4;
    constexpr int simd_len = 16;
    int oc = dst_tensor.shape[1] * packed_oc;
    int filter_size = ctx->getAttrUInt("kernel_h");
    int stride = ctx->getAttrUInt("stride_h");
    constexpr int ow_step = 8;
    const int ld_weight_oc = packed_oc * filter_size *
                             Utils::round_up(filter_size, packed_fw) * ic;
    const int ld_weight_ic =
            packed_oc * filter_size * Utils::round_up(filter_size, packed_fw);
    const int src_reg_size =
            ((ow_step)*stride + filter_size - stride + simd_len - 1) / simd_len;
    const int flt_reg_size = Utils::round_up(filter_size, 4) / 4;
    const int ld_weight_fw =
            packed_oc * Utils::round_up(filter_size, packed_fw);
    bool with_bias = ConvImpl::is_bias(ctx);

    auto kernel_render = StringTemplate::StringTemplateArgs(ctx);
    kernel_render.add_ctx_int("pad_h")
            .add_ctx_int("pad_w")
            .add("stride", 2)
            .add("oc_step", big_oc_step)
            .add("simd_len", simd_len)
            .add("ow_step", ow_step)
            .add("ic", ic)
            .add("oc", oc)
            .add("nr_ow", ow_step)
            .add("ld_weight_oc", ld_weight_oc)
            .add("ld_weight_ic", ld_weight_ic)
            .add("ld_weight_fw", ld_weight_fw)
            .add("src_reg_size", src_reg_size)
            .add("filter_size", filter_size)
            .add("flt_reg_size", flt_reg_size)
            .add("activate_init",
                 [=](std::vector<std::string> args) {
                     CC_ASSERT(args.size() == 0)
                             << "args size = " << args.size();

                     auto str = activate_gen->GenIntrinsicInitFloat();
                     return str;
                 })
            .add("render_init",
                 [=](const std::string& cidx_str) {
                     int c_idx = std::atoi(cidx_str.c_str());
                     return render_init(c_idx, ow_step, with_bias);
                 })
            .add("render_core",
                 [=]() {
                     return render_core(src_reg_size, flt_reg_size, filter_size,
                                        true, ld_weight_fw, ld_weight_oc,
                                        simd_len, ow_step, stride, false);
                 })
            .add("render_store",
                 [=](const std::string& cidx_str,
                     const std::string& store_offset) {
                     int c_idx = std::atoi(cidx_str.c_str());
                     return render_store(ow_step, c_idx, store_offset,
                                         *activate_gen);
                 })
            .add("activate", [=](std::vector<std::string> args) {
                CC_ASSERT(args.size() == 2) << "args size = " << args.size();
                auto str = activate_gen->GenIntrinsicQuantStore(
                        args[0], args[1], "scale");
                return str;
            });
    std::string kernel_temp = R"(  
    __attribute__((target("dotprod")))
    static inline void nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_${nr_ow}_oc${oc_step}(const int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr,
                                            int8_t* dst_ptr, const int packed_iw, const int packed_ic_stride, const int ld_dst_oc, const float scale){
            const int simd_len = ${simd_len};
            int32x4_t c[${oc_step}/4][${ow_step}];
            ${render_init(0)}
            ${render_init(1)}

            for (int ic_idx = 0; ic_idx < ${ic}; ++ic_idx) {
                int8x16_t src[2][${src_reg_size}];
                int8x16_t weight[${oc_step}/4][${flt_reg_size}];                
                ${render_core()}
                src_ptr += packed_ic_stride;
                filter_ptr += ${ld_weight_ic};
            }
            ${activate_init()}            
            ${render_store(0, 0)}
            ${render_store(1, ld_dst_oc)}
        }
    )";
    //! render oc % 8 == 0
    ss << kernel_render.render(kernel_temp);

    std::string kernel_remain_temp = R"(
    __attribute__((target("dotprod")))
    static inline void
    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_remain_oc${oc_step}(const
    int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr,
                                            int8_t* dst_ptr, const int
                                            packed_iw, const int
                                            packed_ic_stride, const int
                                            ld_dst_oc, const int dst_nr_ow, const float scale){
            const int pad_h = ${pad_h};
            const int pad_w = ${pad_w};
            const int simd_len = ${simd_len};
            const int is_even = (dst_nr_ow & 0x1) == 0? 1:0;
            const int read_src_reg_size =
                ((dst_nr_ow - is_even) * ${stride} + ${filter_size} - ${stride} +
                simd_len - 1) / simd_len;
            int32x4_t c[${oc_step}/4][${ow_step}];
            ${render_init(0)}
            ${render_init(1)}

            for (int ic_idx = 0; ic_idx < ${ic}; ++ic_idx) {
                int8x16_t src[2][${src_reg_size}];
                int8x16_t weight[${oc_step}/4][${flt_reg_size}];
                ${render_core()}
                src_ptr += packed_ic_stride;
                filter_ptr += ${ld_weight_ic};
            }
            ${activate_init()}
            rep(step, dst_nr_ow){
                ${activate(c[0][step], dst_ptr + step * 4)};
            }
            rep(step, dst_nr_ow){
                ${activate(c[1][step], dst_ptr + ld_dst_oc + step * 4)};
            }
        }
    )";

    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(src_reg_size, flt_reg_size,
                                                filter_size, true, ld_weight_fw,
                                                ld_weight_oc, simd_len, ow_step,
                                                stride, true);
                         })
                    .render(kernel_remain_temp);

    //! render small oc
    kernel_render.add("oc_step", packed_oc);
    std::string small_oc_kernel_temp = R"(
    __attribute__((target("dotprod")))
    static inline void
    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_${nr_ow}_oc${oc_step}(const
    int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr,
                                        int8_t* dst_ptr, const int packed_iw,
                                        const int packed_ic_stride, const int
                                        ld_dst_oc, const float scale){
        const int pad_h = ${pad_h};
        const int pad_w = ${pad_w};
        const int simd_len = ${simd_len};
        int32x4_t c[${oc_step}/4][${ow_step}];
        ${render_init(0)}

        for (int ic_idx = 0; ic_idx < ${ic}; ++ic_idx) {
            int8x16_t src[2][${src_reg_size}];
            int8x16_t weight[${oc_step}/4][${flt_reg_size}];                
            ${render_core()}
            src_ptr += packed_ic_stride;
            filter_ptr += ${ld_weight_ic};
        }
        ${activate_init()}            
        ${render_store(0, 0)}
    }
    )";
    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(
                                     src_reg_size, flt_reg_size, filter_size,
                                     false, ld_weight_fw, ld_weight_oc,
                                     simd_len, ow_step, stride, false);
                         })
                    .render(small_oc_kernel_temp);

    std::string small_oc_kernel_remain_temp = R"(
    __attribute__((target("dotprod")))
    static inline void
    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_remain_oc${oc_step}(const
    int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr,
                                        int8_t* dst_ptr, const int packed_iw,
                                        const int packed_ic_stride, const int
                                        ld_dst_oc, const int dst_nr_ow, const float scale){
        const int pad_h = ${pad_h};
        const int pad_w = ${pad_w};
        const int simd_len = ${simd_len};
        const int is_even = (dst_nr_ow & 0x1) == 0? 1:0;
        const int read_src_reg_size =
                ((dst_nr_ow - is_even) * ${stride} + ${filter_size} - ${stride} +
                simd_len - 1) / simd_len;
        int32x4_t c[${oc_step}/4][${ow_step}];
        ${render_init(0)}

        for (int ic_idx = 0; ic_idx < ${ic}; ++ic_idx) {
            int8x16_t src[2][${src_reg_size}];
            int8x16_t weight[${oc_step}/4][${flt_reg_size}];

            ${render_core()}

            src_ptr += packed_ic_stride;
            filter_ptr += ${ld_weight_ic};
        }
        ${activate_init()}
        rep(step, dst_nr_ow){
            ${activate(c[0][step], dst_ptr + step * 4)};
        }
    }
    )";
    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(
                                     src_reg_size, flt_reg_size, filter_size,
                                     false, ld_weight_fw, ld_weight_oc,
                                     simd_len, ow_step, stride, true);
                         })
                    .render(small_oc_kernel_remain_temp);

    return ss.str();
}
}  // namespace

std::string ConvDotNCHWNCHW44::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto src_tensor = ctx->getAttrOprand("operand:0");
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = Utils::get_last_operand(ctx);
    constexpr int packed_oc = 4;
    uint32_t oc = dst_tensor.shape[1] * packed_oc;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    writer << R"(
        #if defined(__ARM_FEATURE_DOTPROD)
            #undef __ARM_FEATURE_DOTPROD
        #endif
        #define __ARM_FEATURE_DOTPROD 1
        
        #include <arm_neon.h>
        #include <math.h>
        #include <string.h>
    )";
    writer << R"(
#define rep(i, n) for (int i = 0; i < (n); ++i)
static inline void copy_pad_src(int8_t* sptr_base, const int8_t* sptr_origin,
                        int ic, int ih, int iw, int ph, int pw) {
    const int ic_stride = ih * iw;
    const int ih2 = ih + 2 * ph;
    const int iw2 = iw + 2 * pw;
    rep(ic_idx, ic) {
        const int8_t* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_base, 0, sizeof(int8_t) * iw2 * ih2);
        sptr_base += iw2 * ph;
        rep(ih_idx, ih) {
            memcpy(sptr_base + pw, sptr, sizeof(int8_t) * iw);
            sptr_base += iw2;
            sptr += iw;
        }
        sptr_base += iw2 * ph;
    }
}
)";
    writer << render_kernel(ctx);
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{";
    std::string body_temp = R"(
    const int pad_h = ${pad_h};
    const int pad_w = ${pad_w};
    const int stride_h = ${stride_h};
    const int stride_w = ${stride_w};
    const int8_t* input_ptr = inputs[0]->ptr;
    const int8_t* weight_ptr = inputs[1]->ptr;
    const int32_t* bias_ptr = inputs[2]->ptr;
    int8_t* output_ptr = outputs[0]->ptr;
    
    Layout in_layout = inputs[0]->layout;
    Layout out_layout = outputs[0]->layout;
    const int batch_size = in_layout.dims[0];
    const int ic = ${ic};
    const int ih = in_layout.dims[2];
    const int iw = in_layout.dims[3];
    const int src_batch_stride = ic * ih * iw;
    const int pack_oc_size = 4;
    const int big_oc_step = 8;
    const int oh_step = 1;
    const int ow_step = ${ow_step};
    const int fh = ${filter_size};
    const int fw = (${filter_size} + 3) / 4 * 4;
    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    const float scale = src_scale * flt_scale / dst_scale;

    const int oc = ${oc};
    const int oh = out_layout.dims[2];
    const int ow = out_layout.dims[3];
    const int oc_stride = oh * ow;
    const int ld_dst_oc = oc_stride * pack_oc_size;

    const int packed_iw = iw + 2 * pad_w;
    const int packed_ih = ih + 2 * pad_h;
    const int packed_ic_stride = packed_iw * packed_ih;
    int8_t* workspace_ptr = workspace->ptr;

    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const int oc_remain = oc - oc_end;
    const int8_t* src_ptr = workspace_ptr;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx){

        copy_pad_src(workspace_ptr, input_ptr + batch_idx * src_batch_stride, ic, ih, iw, pad_h, pad_w);
        for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const int weight_offset = oc_idx * ic * fh * fw;
            for (int oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_idx * stride_w;
                    const int dst_offset = oc_idx * oc_stride +
                                            (oh_idx * ow + ow_idx) * pack_oc_size;
                    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_${ow_step}_oc8(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_idx, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, scale);
                }
                if (ow_remain) {                    
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_end * stride_w;
                    const int dst_offset = oc_idx * oc_stride +
                                            (oh_idx * ow + ow_end) * pack_oc_size;
                    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_remain_oc8(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_idx, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, ow_remain, scale);                    
                }
            }
        }
        if (oc_remain){
            const int weight_offset = oc_end * ic * fh * fw;
            for (int oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_idx * stride_w;
                    const int dst_offset = oc_end * oc_stride +
                                            (oh_idx * ow + ow_idx) * pack_oc_size;
                    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_${ow_step}_oc4(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_end, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, scale);
                }
                if (ow_remain) {                    
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_end * stride_w;
                    const int dst_offset = oc_end * oc_stride +
                                            (oh_idx * ow + ow_end) * pack_oc_size;
                    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_remain_oc4(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_end, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, ow_remain, scale);
                }
            }
        }
        output_ptr += oc * oc_stride;
    }
    
    return TinyNN_SUCCESS;
}
)";
    constexpr int ow_step = 8;
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add("ow_step", ow_step)
                      .add("ic", ic)
                      .add("oc", oc)
                      .add("filter_size", filter_size)
                      .render(body_temp);
    return writer.str();
}
