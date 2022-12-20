/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/ConvKernel/F32ConvNCHWNCHW443x3s2.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>
#include <string>
#include "Arm/ARMBackend.h"
#include "Arm/ArmCommon/NeonIntrinCompat.h"
#include "Arm/Armv7/Activation.h"
#include "ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
using namespace ArmCommon;

bool ConvFloatNCHWNCHW443x3s2::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == 2 &&
            ctx->getAttrUInt("kernel_w") == 3 &&
            ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU";

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f32" &&
                   ctx->getAttrOprand("operand:1").dtype == "f32" &&
                   ctx->getAttrOprand("operand:2").dtype == "f32";
    bool layout_ok =
            ctx->getAttrOprand("operand:0").shape.size() == 4 &&
            ctx->getAttrOprand(
                       "operand:" +
                       std::to_string(ctx->getAttrInt("nr_operands") - 1))
                            .shape.size() == 5;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}
std::string ConvFloatNCHWNCHW443x3s2::GetKernelSymbol(TContext* ctx) const {
    auto src_tensor = ctx->getAttrOprand("operand:0");
    CC_ASSERT(src_tensor.shape.size() > 0)
            << "src_tensor size should > 0, now" << src_tensor.shape.size();
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    uint32_t oc = dst_tensor.shape[1] * 4;
    std::string name_temp = "${base_kernel_sym}_nchw_nchw44_oc${oc}_ic${ic}";
    return StringTemplate::StringTemplateArgs(ctx)
            .add("base_kernel_sym", Armv7ConvImpl::GetKernelSymbol(ctx))
            .add("oc", oc)
            .add("ic", ic)
            .render(name_temp);
}

std::string ConvFloatNCHWNCHW443x3s2::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto src_tensor = ctx->getAttrOprand("operand:0");
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    uint32_t oc = dst_tensor.shape[1] * 4;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    writer << "#include <arm_neon.h>\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    Tensor* in_weights = inputs[1];
                      )";
    std::string fill_weight_attr = R"(
    out_weights->layout.nr_dim = 5;
    for(size_t i = 0; i < out_weights->layout.nr_dim; ++i){
        out_weights->layout.dims[i] = in_weights->layout.dims[i];
        out_weights->layout.stride[i] = in_weights->layout.stride[i];
    }
    out_weights->dtype.type_enum = TinyNN_FLOAT;
    out_weights->name = in_weights->name;
 )";
    std::string fill_weight_transform = StringTemplate::StringTemplateArgs()
                                                .add("ic", ic)
                                                .add("oc", oc)
                                                .add("kh", filter_size)
                                                .add("kw", filter_size)
                                                .render(R"(
    const int oc_step = 4;
    const int filter_oc_stride = ${kh} * ${kw} * ${ic};
    const int filter_ic_stride = ${kh} * ${kw} * oc_step;
    float* outptr = out_weights->ptr;
    const float* inptr = in_weights->ptr;
    for (int oc_idx = 0; oc_idx < ${oc}; oc_idx += oc_step) {
        const float* in_ptr_oc = inptr + oc_idx * filter_oc_stride;
        float* dst_ptr_oc = outptr + oc_idx * filter_oc_stride;
        for (int kh_idx = 0; kh_idx < ${kh}; ++kh_idx) {
            for (int kw_idx = 0; kw_idx < ${kw}; ++kw_idx) {
                for (int ic_idx = 0; ic_idx < ${ic}; ++ic_idx) {
                    float32x4_t vsrc = vld1q_f32(in_ptr_oc);
                    vst1q_f32(dst_ptr_oc + ic_idx * filter_ic_stride, vsrc);
                    in_ptr_oc += oc_step;
                }
                dst_ptr_oc += oc_step;
            }
        }
    }
)");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string ConvFloatNCHWNCHW443x3s2::GetWorkspaceBody(
        TContext* context) const {
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
        const uint32_t packed_iw =
        round_up(iw + 2 * pad_w,
                     ${cacheline_byte} / (uint32_t)sizeof(float));
        const uint32_t packed_ih = ih + 2 * pad_h;        
        *workspace = (size_t)ic * packed_ih * packed_iw * sizeof(float);
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
               << "] = vld1q_f32(bias_ptr + " << c_idx << " * 4);";
        } else {
            ss << "c[" << c_idx << "][" << src_idx << "] = vdupq_n_f32(0.f);";
        }
    }
    return ss.str();
}
std::string render_core(int src_reg_size, int filter_size, bool is_big_oc,
                        int ld_weight_fw, int ld_weight_oc, int simd_len,
                        int nr_ow, int stride, bool remain_n) {
    using Rendor = StringTemplate::StringTemplateArgs;
    std::stringstream fw_ss;
    if (remain_n) {
        fw_ss << R"(
        rep(src_idx, read_src_reg_size){
            src[src_idx] = vld1q_f32(src_ptr + ${fh_idx} * packed_iw + src_idx * ${simd_len});
        })";
    } else {
        for (int src_idx = 0; src_idx < src_reg_size; ++src_idx) {
            fw_ss << "src[" << src_idx
                  << "] = vld1q_f32(src_ptr + ${fh_idx} * packed_iw + "
                  << src_idx << "* ${simd_len});\n";
        }
    }

    for (int fw_idx = 0; fw_idx < filter_size; ++fw_idx) {
        fw_ss << "weight[0][" << fw_idx
              << "] = vld1q_f32(filter_ptr + ${fh_idx} * ${ld_weight_fw} + "
              << fw_idx << " * ${simd_len});\n";
    }
    if (is_big_oc) {
        for (int fw_idx = 0; fw_idx < filter_size; ++fw_idx) {
            fw_ss << "weight[1][" << fw_idx
                  << "] = vld1q_f32(filter_ptr + ${ld_weight_oc} + ${fh_idx} * "
                     "${ld_weight_fw} + "
                  << fw_idx << " * ${simd_len});\n";
        }
    }
    for (int fw_idx = 0; fw_idx < filter_size; ++fw_idx) {
        auto src_idx = fw_idx;
        auto weight_idx = fw_idx;
        for (int i = 0; i < nr_ow; ++i) {
            fw_ss << "c[0][" << i << "] = vfmaq_laneq_f32(c[0][" << i
                  << "], weight[0][" << weight_idx << "],  src[(" << i
                  << " * ${stride} + " << src_idx << ") / 4], (" << i
                  << " * ${stride} + " << src_idx << ") % 4);";
            if (is_big_oc) {
                fw_ss << "c[1][" << i << "] = vfmaq_laneq_f32(c[1][" << i
                      << "], weight[1][" << weight_idx << "],  src[(" << i
                      << " * ${stride} + " << src_idx << ") / 4], (" << i
                      << " * ${stride} + " << src_idx << ") % 4);";
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
        ss << act.GenIntrinsicFloatStore("c[" + std::to_string(c_idx) + "][" +
                                                 std::to_string(ow_idx) + "]",
                                         "dst_ptr + " + store_offset + " + " +
                                                 std::to_string(ow_idx) +
                                                 " * simd_len");
    }
    return ss.str();
}

std::string render_kernel(TContext* ctx) {
    std::stringstream ss;
    ss << R"(typedef void kern_func (const float* src_ptr, const float* filter_ptr, const float* bias_ptr,
                                        float* dst_ptr, const int packed_iw, const int packed_ic_stride, const int ld_dst_oc);   
    )";

    std::string mode = ctx->haveAttr("nonlineMode")
                               ? ctx->getAttrStr("nonlineMode")
                               : "IDENTITY";

    auto activate_gen = create_activation_gener_instrinsic(mode);

    auto src_tensor = ctx->getAttrOprand("operand:0");
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    constexpr int big_oc_step = 4;
    constexpr int packed_oc = 4;
    constexpr int simd_len = 4;
    uint32_t oc = dst_tensor.shape[1] * packed_oc;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    uint32_t stride = ctx->getAttrUInt("stride_h");
    constexpr int ow_step = 8;
    const int ld_weight_oc = packed_oc * filter_size * filter_size * ic;
    const int ld_weight_ic = packed_oc * filter_size * filter_size;
    const int src_reg_size =
            (ow_step * stride + filter_size - stride + simd_len - 1) / simd_len;
    const int ld_weight_fw = packed_oc * filter_size;
    bool with_bias = ConvImpl::is_bias(ctx);

    auto kernel_render = StringTemplate::StringTemplateArgs(ctx);
    kernel_render.add_ctx_int("pad_h")
            .add_ctx_int("pad_w")
            .add("stride", 2)
            .add("oc_step", big_oc_step)
            .add("simd_len", 4)
            .add("ow_step", ow_step)
            .add("ic", ic)
            .add("oc", oc)
            .add("nr_ow", ow_step)
            .add("ld_weight_oc", ld_weight_oc)
            .add("ld_weight_ic", ld_weight_ic)
            .add("ld_weight_fw", ld_weight_fw)
            .add("src_reg_size", src_reg_size)
            .add("filter_size", filter_size)
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
            .add("render_store",
                 [=](const std::string& cidx_str,
                     const std::string& store_offset) {
                     int c_idx = std::atoi(cidx_str.c_str());
                     return render_store(ow_step, c_idx, store_offset,
                                         *activate_gen);
                 })
            .add("activate", [=](std::vector<std::string> args) {
                CC_ASSERT(args.size() == 2) << "args size = " << args.size();
                auto str =
                        activate_gen->GenIntrinsicFloatStore(args[0], args[1]);
                return str;
            });
    std::string kernel_temp = R"(    
        static inline int round_up(int x, int d){
            TINYNN_ASSERT(d > 0);
            return (x + d - 1) / d * d;
        } 
        static inline void prefetch_2x(const void* pfp) {
            // clang-format off
            asm volatile("PLD [%[pfp]]\n"
                        "PLD [%[pfp], #64]\n"
                        :
                        : [pfp] "r"(pfp)
                        : "memory");
            // clang-format on
        }   
        static inline void armv7_nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_${nr_ow}_oc${oc_step}(const float* src_ptr, const float* filter_ptr, const float* bias_ptr,
                                                float* dst_ptr, const int packed_iw, const int packed_ic_stride, const int ld_dst_oc){
        const int simd_len = ${simd_len};
        const int oc_block = 4;
        const int stride = 2;
        const int remain_w = 8;
        const int ow_block = 8;
        const int loop_ic_step = 1;
        const int filter_size = 3;
        const int oc_step = 4;
        const int src_line_block = ow_block * stride + filter_size - stride;

        const int iw_skip_bytes =(packed_iw - round_up(src_line_block, 2)) * sizeof(float);
        const int ld_src_ic_skip_bytes = (packed_ic_stride-packed_iw*filter_size) * sizeof(float)+iw_skip_bytes;
        float32x4_t c[1][8];
        ${render_init(0)}

        for (int ic_idx = 0; ic_idx < ${ic}; ic_idx += loop_ic_step) {
            asm volatile(

                    "2:\n"
                    //! row 0
                    "vld1.32 {d10, d11}, [%[filter_ptr]]!\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vld1.32 {d12, d13}, [%[filter_ptr]]!\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vld1.32 {d14, d15}, [%[filter_ptr]]!\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "add %[src_ptr], %[src_ptr], %[iw_skip_bytes]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"
                    "vld1.32 {d10, d11}, [%[filter_ptr]]!\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"
                    "vld1.32 {d12, d13}, [%[filter_ptr]]!\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"
                    "vld1.32 {d14, d15}, [%[filter_ptr]]!\n"
                    //! row 1

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "add %[src_ptr], %[src_ptr], %[iw_skip_bytes]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"
                    "vld1.32 {d10, d11}, [%[filter_ptr]]!\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"
                    "vld1.32 {d12, d13}, [%[filter_ptr]]!\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"
                    "vld1.32 {d14, d15}, [%[filter_ptr]]!\n"
                    //! row 2

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "add %[src_ptr], %[src_ptr], %[ld_src_ic_skip_bytes]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"

                    "6:\n"
                    : [c0] "+w"(c[0][0]), [c1] "+w"(c[0][1]), [c2] "+w"(c[0][2]),
                      [c3] "+w"(c[0][3]), [c4] "+w"(c[0][4]), [c5] "+w"(c[0][5]),
                      [c6] "+w"(c[0][6]), [c7] "+w"(c[0][7]), [src_ptr] "+r"(src_ptr),
                      [filter_ptr] "+r"(filter_ptr)

                    : [ld_src_ic_skip_bytes] "r"(ld_src_ic_skip_bytes),
                      [iw_skip_bytes] "r"(iw_skip_bytes)
                    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
                      "d11", "d12", "d13", "d14", "d15", "r1", "r2", "cc", "memory");
        }
                ${activate_init()}            
                ${render_store(0, 0)}
            }
        )";
    //! render oc % 4 == 0
    ss << kernel_render.render(kernel_temp);

    std::string kernel_remain_temp = R"(
    static inline void
    nchw_nchw44_s2_${filter_size}x${filter_size}_kernel_remain_oc${oc_step}(const
    float* src_ptr, const float* filter_ptr, const float* bias_ptr,
                                            float* dst_ptr, const int
                                            packed_iw, const int
                                            packed_ic_stride, const int
                                            ld_dst_oc, const int dst_nr_ow){
            const int pad_h = ${pad_h};
            const int pad_w = ${pad_w};
            const int simd_len = ${simd_len};
            const int read_src_reg_size =
                (dst_nr_ow * ${stride} + ${filter_size} - ${stride} +
                simd_len - 1) / simd_len;
            float32x4_t c[${oc_step}/4][${ow_step}];
            ${render_init(0)}

            for (int ic_idx = 0; ic_idx < ${ic}; ++ic_idx) {
                float32x4_t src[${src_reg_size}];
                float32x4_t weight[${oc_step}/4][${filter_size}];
                ${render_core()}
                src_ptr += packed_ic_stride;
                filter_ptr += ${ld_weight_ic};
            }
            ${activate_init()}
            rep(step, dst_nr_ow){
                ${activate(c[0][step], dst_ptr + step * simd_len)};
            }
        }
    )";

    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(src_reg_size, filter_size,
                                                false, ld_weight_fw,
                                                ld_weight_oc, simd_len, ow_step,
                                                stride, true);
                         })
                    .render(kernel_remain_temp);
    return ss.str();
}
}  // namespace

std::string ConvFloatNCHWNCHW443x3s2::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto src_tensor = ctx->getAttrOprand("operand:0");
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    constexpr int packed_oc = 4;
    constexpr int simd_len = 4;
    uint32_t oc = dst_tensor.shape[1] * packed_oc;
    uint32_t filter_size = ctx->getAttrUInt("kernel_h");
    uint32_t stride = ctx->getAttrUInt("stride_h");
    writer << R"(
        #include <arm_neon.h>
        #include <math.h>
        #include <string.h>
    )";
    writer << gen_neon_intrin_compat();
    writer << R"(
#define rep(i, n) for (int i = 0; i < (n); ++i)
static inline void copy_pad_src(float* sptr_dst, const float* sptr_origin,
                        int pw, int ih, int iw,
                        int packed_iw, int pad_top, int pad_bottom, int ic) {
    const int pad_right = packed_iw - iw - pw;
    const int ic_stride = ih * iw;
    rep(ic_idx, ic) {
        const float* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_dst, 0, sizeof(float) * packed_iw * pad_top);
        sptr_dst += packed_iw * pad_top;
        rep(ih_idx, ih) {
            rep(pad_idx, pw){
                *sptr_dst = 0.f;
                sptr_dst++;
            }
            memcpy(sptr_dst, sptr, sizeof(float) * iw);
            sptr_dst += iw;
            sptr += iw;
            rep(pad_idx, pad_right){
                *sptr_dst = 0.f;
                sptr_dst++;
            }
        }
        memset(sptr_dst, 0, sizeof(float) * packed_iw * pad_bottom);
        sptr_dst += packed_iw * pad_bottom;
    }
}

)";
    writer << render_kernel(ctx);
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{";
    std::string body_temp = R"(
    const int pad_h = ${pad_h};
    const int pad_w = ${pad_w};
    const int stride_h = 2;
    const int stride_w = 2;
    const float* input_ptr = inputs[0]->ptr;
    const float* weight_ptr = inputs[1]->ptr;
    const float* bias_ptr = inputs[2]->ptr;
    float* output_ptr = outputs[0]->ptr;
    
    Layout in_layout = inputs[0]->layout;
    Layout out_layout = outputs[0]->layout;
    const int batch_size = in_layout.dims[0];
    const int ic = ${ic};
    const int ih = in_layout.dims[2];
    const int iw = in_layout.dims[3];
    const int src_batch_stride = ic * ih * iw;
    const int pack_oc_size = 4;
    const int big_oc_step = 4;
    const int oh_step = 1;
    const int ow_step = 8;

    const int oc = ${oc};
    const int oh = out_layout.dims[2];
    const int ow = out_layout.dims[3];
    const int oc_stride = oh * ow;
    const int ld_dst_oc = oc_stride * pack_oc_size;

    const int packed_iw = ((iw + 2 * pad_w + 15) >> 4) << 4;
    const int packed_ih = ih + 2 * pad_h;
    const int packed_ic_stride = packed_iw * packed_ih;
    float* workspace_ptr = workspace->ptr;

    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const float* src_ptr = workspace_ptr;       
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx){
        copy_pad_src(workspace_ptr, input_ptr + batch_idx * src_batch_stride, pad_w, ih, iw, packed_iw, pad_h, pad_h, ic);
        for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const int weight_offset = oc_idx * ic * ${filter_size} * ${filter_size};
            for (int oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_idx * stride_w;
                    const int dst_offset = oc_idx * oc_stride +
                                            (oh_idx * ow + ow_idx) * pack_oc_size;
                    ${main_kern}(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_idx, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc);
                }
                if (ow_remain) {             
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_end * stride_w;
                    const int dst_offset = oc_idx * oc_stride +
                                            (oh_idx * ow + ow_end) * pack_oc_size;
                    ${main_kern_remain}(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_idx, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, ow_remain);                    
                }
            }
        }
        output_ptr += oc * oc_stride;
    }
    
    return TinyNN_SUCCESS;
}
)";
    const int big_oc_step = 4;
    constexpr int ow_step = 8;
    const int ld_weight_oc = packed_oc * filter_size * filter_size * ic;
    const int ld_weight_ic = packed_oc * filter_size * filter_size;
    const int src_reg_size =
            (ow_step * stride + filter_size - stride + simd_len - 1) / simd_len;
    const int ld_weight_fw = packed_oc * filter_size;

    std::string main_kern =
            "armv7_nchw_nchw44_s2_" + std::to_string(filter_size) + "x" +
            std::to_string(filter_size) + "_kernel_" + std::to_string(ow_step) +
            "_oc" + std::to_string(big_oc_step);
    std::string main_kern_remain =
            "nchw_nchw44_s2_" + std::to_string(filter_size) + "x" +
            std::to_string(filter_size) + "_kernel_remain_oc" +
            std::to_string(big_oc_step);
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add("ow_step", ow_step)
                      .add("big_oc_step", big_oc_step)
                      .add("simd_len", 4)
                      .add("ic", ic)
                      .add("oc", oc)
                      .add("ld_weight_oc", ld_weight_oc)
                      .add("ld_weight_ic", ld_weight_ic)
                      .add("ld_weight_fw", ld_weight_fw)
                      .add("src_reg_size", src_reg_size)
                      .add("filter_size", filter_size)
                      .add("main_kern", main_kern.c_str())
                      .add("main_kern_remain", main_kern_remain.c_str())
                      .render(body_temp);
    return writer.str();
}
