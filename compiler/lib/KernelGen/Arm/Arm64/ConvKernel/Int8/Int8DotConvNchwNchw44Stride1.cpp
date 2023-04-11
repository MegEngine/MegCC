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

bool ConvDotNCHWNCHW44Stride1::IsAvailable(TContext* ctx) const {
    return ConvDotNCHWNCHW44Common::IsAvailableCommon(ctx, 1);
}

namespace {
std::string render_init(int c_idx, int nr_ow, bool with_bias) {
    std::stringstream ss;
    for (int src_idx = 0; src_idx < nr_ow; ++src_idx) {
        if (with_bias) {
            ss << "c[" << c_idx << "][" << src_idx << "] = vld1q_s32(bias_ptr + "
               << c_idx << " * 4);";
        } else {
            ss << "c[" << c_idx << "][" << src_idx << "] = vdupq_n_s32(0);";
        }
    }
    return ss.str();
}
std::string render_core(
        int filter_size_round_up, int ow_step, int filter_size, int oc_step,
        int ld_weight_oc, bool is_ow_remain) {
    constexpr int SIMD_LEN_S32 = 4;
    constexpr int OC_PACK_SIZE = 4;
    using Rendor = StringTemplate::StringTemplateArgs;
    std::stringstream res, temp_ss;
    auto base_rendor = Rendor();
    base_rendor.add("ld_weight_oc", ld_weight_oc)
            .add("filter_size_round_up", filter_size_round_up)
            .add("i", "${i}");
    std::string temp_str;
    if (is_ow_remain == false) {
        temp_str =
                R"(src[${j}] = vld1q_s8(src_ptr_base + ${i} * packed_iw + ${j} * simd_len);
            )";
        for (int j = 0;
             j < ow_step / SIMD_LEN_S32 + filter_size_round_up / SIMD_LEN_S32 - 1;
             ++j) {
            temp_ss << base_rendor.add("j", j).render(temp_str);
        }
    } else {
        temp_str =
                R"(for(int j = 0; j < ow_remain_round_up / 4 + ${filter_size_round_up} / 4 - 1; ++j){
            src[j] = vld1q_s8(src_ptr_base + ${i} * packed_iw + j * simd_len);
        }
        )";
        temp_ss << base_rendor.render(temp_str);
    }
    temp_str =
            R"(weight[${j}][${k}] = vld1q_s8(filter_ptr_base + ${j} * ${ld_weight_oc} + ${i} * 4 * ${filter_size_round_up} + ${k} * simd_len);
            )";
    for (int j = 0; j < oc_step / OC_PACK_SIZE; ++j) {
        for (int k = 0; k < filter_size_round_up / SIMD_LEN_S32; ++k) {
            temp_ss << base_rendor.add("j", j).add("k", k).render(temp_str);
        }
    }
    if (is_ow_remain == false) {
        temp_str =
                R"(c[${j}][${k}] = vdotq_laneq_s32(c[${j}][${k}], weight[${j}][${l}], src[${k} / 4 + ${l}], ${k} % 4);
            )";
        for (int j = 0; j < oc_step / OC_PACK_SIZE; ++j) {
            for (int k = 0; k < ow_step; ++k) {
                for (int l = 0; l < filter_size_round_up / SIMD_LEN_S32; ++l) {
                    temp_ss << base_rendor.add("j", j).add("k", k).add("l", l).render(
                            temp_str);
                }
            }
        }
    } else {
        temp_str = R"(switch(ow_remain){
            case 7:
                c[${j}][6] = vdotq_laneq_s32(c[${j}][6], weight[${j}][${k}], src[1 + ${k}], 2);
            case 6:
                c[${j}][5] = vdotq_laneq_s32(c[${j}][5], weight[${j}][${k}], src[1 + ${k}], 1);
            case 5:
                c[${j}][4] = vdotq_laneq_s32(c[${j}][4], weight[${j}][${k}], src[1 + ${k}], 0);
            case 4:
                c[${j}][3] = vdotq_laneq_s32(c[${j}][3], weight[${j}][${k}], src[${k}], 3);
            case 3:
                c[${j}][2] = vdotq_laneq_s32(c[${j}][2], weight[${j}][${k}], src[${k}], 2);
            case 2:
                c[${j}][1] = vdotq_laneq_s32(c[${j}][1], weight[${j}][${k}], src[${k}], 1);
            case 1:
                c[${j}][0] = vdotq_laneq_s32(c[${j}][0], weight[${j}][${k}], src[${k}], 0);
                break;
            default:
                TINYNN_ASSERT(0);
        }
        )";
        for (int j = 0; j < oc_step / OC_PACK_SIZE; ++j) {
            for (int k = 0; k < filter_size_round_up / SIMD_LEN_S32; ++k) {
                temp_ss << base_rendor.add("j", j).add("k", k).render(temp_str);
            }
        }
    }
    for (int i = 0; i < filter_size; ++i) {
        res << base_rendor.add("i", i).render(temp_ss.str());
    }
    return res.str();
}
std::string render_store(
        int nr_ow, int c_idx, const std::string& store_offset,
        const ActivationGenIntrinsicBase& act) {
    std::stringstream ss;
    for (int ow_idx = 0; ow_idx < nr_ow; ++ow_idx) {
        ss << act.GenIntrinsicQuantStore(
                "c[" + std::to_string(c_idx) + "][" + std::to_string(ow_idx) + "]",
                "dst_ptr + " + store_offset + " + " + std::to_string(ow_idx) + " * 4",
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
    constexpr int ow_step = 8;
    const int ld_weight_oc =
            packed_oc * filter_size * Utils::round_up(filter_size, packed_fw) * ic;
    const int filter_size_round_up =
            (filter_size + (packed_fw - 1)) / packed_fw * packed_fw;
    bool with_bias = ConvImpl::is_bias(ctx);

    auto kernel_render = StringTemplate::StringTemplateArgs(ctx);
    kernel_render.add("oc_step", big_oc_step)
            .add("simd_len", simd_len)
            .add("ow_step", ow_step)
            .add("ic", ic)
            .add("oc", oc)
            .add("nr_ow", ow_step)
            .add("ld_weight_oc", ld_weight_oc)
            .add("filter_size", filter_size)
            .add("filter_size_round_up", filter_size_round_up)
            .add("packed_oc", packed_oc)
            .add("packed_fw", packed_fw)
            .add("activate_init",
                 [=](std::vector<std::string> args) {
                     CC_ASSERT(args.size() == 0) << "args size = " << args.size();

                     auto str = activate_gen->GenIntrinsicInitFloat();
                     return str;
                 })
            .add("render_init",
                 [=](const std::string& cidx_str) {
                     int c_idx = std::atoi(cidx_str.c_str());
                     return render_init(c_idx, ow_step, with_bias);
                 })
            .add("render_store",
                 [=](const std::string& cidx_str, const std::string& store_offset) {
                     int c_idx = std::atoi(cidx_str.c_str());
                     return render_store(ow_step, c_idx, store_offset, *activate_gen);
                 })
            .add("activate", [=](std::vector<std::string> args) {
                CC_ASSERT(args.size() == 2) << "args size = " << args.size();
                auto str =
                        activate_gen->GenIntrinsicQuantStore(args[0], args[1], "scale");
                return str;
            });

    std::string kernel_temp_stride1 = R"(
        __attribute__((target("dotprod")))
        static inline void nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_${nr_ow}_oc${oc_step}(
            const int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
            const int packed_iw, const int packed_ic_stride, const int ld_dst_oc, const float scale
        ){
            const int simd_len = ${simd_len};
            int32x4_t c[${oc_step} / ${packed_oc}][${ow_step}];
            ${render_init(0)}
            ${render_init(1)}
            int8x16_t src[${filter_size_round_up} / ${packed_fw} - 1 + ${ow_step} / 4];
            int8x16_t weight[${oc_step} / ${packed_oc}][${filter_size_round_up} / ${packed_fw}];

            for(int ic_idx = 0; ic_idx < ${ic}; ++ic_idx){
                const int8_t* src_ptr_base = src_ptr + ic_idx * packed_ic_stride;
                const int8_t* filter_ptr_base = filter_ptr + ic_idx * ${packed_oc} * ${filter_size} * ${filter_size_round_up};
                ${render_core()}
            }
            ${activate_init()}
            ${render_store(0, 0)}
            ${render_store(1, ld_dst_oc)}
        }
    )";
    //! render ow % 8 == 0
    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(
                                     filter_size_round_up, ow_step, filter_size,
                                     big_oc_step, ld_weight_oc, false);
                         })
                    .render(kernel_temp_stride1);

    std::string kernel_temp_remain_stride1 = R"(
        __attribute__((target("dotprod")))
        static inline void nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_remain_oc${oc_step}(
            const int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
            const int packed_iw, const int packed_ic_stride, const int ld_dst_oc, const int ow_remain, const float scale
        ){
            const int simd_len = ${simd_len};
            int32x4_t c[${oc_step} / ${packed_oc}][${ow_step}];
            const int ow_remain_round_up = (ow_remain + 3) / 4 * 4;
            ${render_init(0)}
            ${render_init(1)}
            int8x16_t src[${filter_size_round_up} / ${packed_fw} - 1 + ow_remain_round_up / 4];
            int8x16_t weight[${oc_step} / ${packed_oc}][${filter_size_round_up} / ${packed_fw}];

            for(int ic_idx = 0; ic_idx < ${ic}; ++ic_idx){
                const int8_t* src_ptr_base = src_ptr + ic_idx * packed_ic_stride;
                const int8_t* filter_ptr_base = filter_ptr + ic_idx * ${packed_oc} * ${filter_size} * ${filter_size_round_up};
                ${render_core()}
            }
            ${activate_init()}
            for(int step = 0; step < ow_remain; ++step){
                ${activate(c[0][step], dst_ptr + step * 4)};
            }
            for(int step = 0; step < ow_remain; ++step){
                ${activate(c[1][step], dst_ptr + ld_dst_oc + step * 4)};
            }
        }
    )";
    //! render ow % 8 != 0
    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(
                                     filter_size_round_up, ow_step, filter_size,
                                     big_oc_step, ld_weight_oc, true);
                         })
                    .render(kernel_temp_remain_stride1);

    kernel_render.add("oc_step", packed_oc);
    std::string small_oc_kernel_temp_stride1 = R"(
        __attribute__((target("dotprod")))
        static inline void
        nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_${nr_ow}_oc${oc_step}(
            const int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr, int8_t* dst_ptr, 
            const int packed_iw, const int packed_ic_stride, const int ld_dst_oc, const float scale
        ){
            const int simd_len = ${simd_len};
            int32x4_t c[${oc_step} / ${packed_oc}][${ow_step}];
            ${render_init(0)}
            int8x16_t src[${filter_size_round_up} / ${packed_fw} - 1 + ${ow_step} / 4];
            int8x16_t weight[${oc_step} / ${packed_oc}][${filter_size_round_up} / ${packed_fw}];

            for(int ic_idx = 0; ic_idx < ${ic}; ++ic_idx){
                const int8_t* src_ptr_base = src_ptr + ic_idx * packed_ic_stride;
                const int8_t* filter_ptr_base = filter_ptr + ic_idx * ${packed_oc} * ${filter_size} * ${filter_size_round_up};
                ${render_core()}
            }
            ${activate_init()}
            ${render_store(0, 0)}
        }
    )";
    //! render ow % 8 == 0
    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(
                                     filter_size_round_up, ow_step, filter_size,
                                     packed_oc, ld_weight_oc, false);
                         })
                    .render(small_oc_kernel_temp_stride1);

    std::string small_oc_kernel_temp_remain_stride1 = R"(
        __attribute__((target("dotprod")))
        static inline void nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_remain_oc${oc_step}(
            const int8_t* src_ptr, const int8_t* filter_ptr, const int32_t* bias_ptr, int8_t* dst_ptr,
            const int packed_iw, const int packed_ic_stride, const int ld_dst_oc, const int ow_remain, const float scale
        ){
            const int simd_len = ${simd_len};
            int32x4_t c[${oc_step} / ${packed_oc}][${ow_step}];
            const int ow_remain_round_up = (ow_remain + 3) / 4 * 4;
            ${render_init(0)}
            int8x16_t src[${filter_size_round_up} / ${packed_fw} - 1 + ow_remain_round_up / 4];
            int8x16_t weight[${oc_step} / ${packed_oc}][${filter_size_round_up} / ${packed_fw}];

            for(int ic_idx = 0; ic_idx < ${ic}; ++ic_idx){
                const int8_t* src_ptr_base = src_ptr + ic_idx * packed_ic_stride;
                const int8_t* filter_ptr_base = filter_ptr + ic_idx * ${packed_oc} * ${filter_size} * ${filter_size_round_up};
                ${render_core()}
            }
            ${activate_init()}
            for(int step = 0; step < ow_remain; ++step){
                ${activate(c[0][step], dst_ptr + step * 4)};
            }
        }
    )";
    //! render ow % 8 != 0
    ss << kernel_render
                    .add("render_core",
                         [=]() {
                             return render_core(
                                     filter_size_round_up, ow_step, filter_size,
                                     packed_oc, ld_weight_oc, true);
                         })
                    .render(small_oc_kernel_temp_remain_stride1);

    return ss.str();
}
}  // namespace

std::string ConvDotNCHWNCHW44Stride1::GetKernelBody(TContext* ctx) const {
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
                        int ic, int ih, int iw, int ph, int pw, int stride) {
    const int ic_stride = ih * iw;
    const int ih2 = ih + 2 * ph;
    const int iw2 = iw + 2 * pw;
    const int boundary = 16 - 4;
    const int iw_pack = 4;
    const int sptr_base_stride = ih2 * iw2 * iw_pack;
    int8_t tmp_line[iw2 + boundary];
    uint8_t reorder_idx[] = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
    uint8x16_t tbl = vld1q_u8(reorder_idx);
    int8x16_t src[4], reordered[4];
    rep(ic_idx, ic){
        memset(sptr_base, 0, sizeof(int8_t) * ph * iw2 * iw_pack);
        sptr_base += ph * iw2 * iw_pack;
        rep(ih_idx, ih){
            memset(tmp_line, 0, sizeof(int8_t) * (iw2 + boundary));
            memcpy(tmp_line + pw, sptr_origin, iw * sizeof(int8_t));
            int iw2_idx = 0;
            for(; iw2_idx + 15 < iw2; iw2_idx += 16){
                src[0] = vld1q_s8(tmp_line + iw2_idx);
                src[1] = vld1q_s8(tmp_line + iw2_idx + 4);
                src[2] = vld1q_s8(tmp_line + iw2_idx + 8);
                src[3] = vld1q_s8(tmp_line + iw2_idx + 12);
                reordered[0] = vqtbl1q_s8(src[0], tbl);
                reordered[1] = vqtbl1q_s8(src[1], tbl);
                reordered[2] = vqtbl1q_s8(src[2], tbl);
                reordered[3] = vqtbl1q_s8(src[3], tbl);
                vst1q_s8(sptr_base, reordered[0]);
                vst1q_s8(sptr_base + 16, reordered[1]);
                vst1q_s8(sptr_base + 32, reordered[2]);
                vst1q_s8(sptr_base + 48, reordered[3]);
                sptr_base += 64;
            }
            for(; iw2_idx < iw2; ++iw2_idx){
                sptr_base[0] = tmp_line[iw2_idx];
                sptr_base[1] = tmp_line[iw2_idx + 1];
                sptr_base[2] = tmp_line[iw2_idx + 2];
                sptr_base[3] = tmp_line[iw2_idx + 3];
                sptr_base += 4;
            }
            sptr_origin += iw;
        }
        memset(sptr_base, 0, sizeof(int8_t) * ph * iw2 * iw_pack);
        sptr_base += ph * iw2 * iw_pack;
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
    const int iw_pack_size = 4;
    const int packed_iw = (iw + 2 * pad_w) * iw_pack_size;
    const int packed_ih = ih + 2 * pad_h;
    const int packed_ic_stride = packed_iw * packed_ih;
    int8_t* workspace_ptr = workspace->ptr;

    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const int oc_remain = oc - oc_end;
    const int8_t* src_ptr = workspace_ptr;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx){
        copy_pad_src(workspace_ptr, input_ptr + batch_idx * src_batch_stride, ic, ih, iw, pad_h, pad_w, stride_h);
        for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const int weight_offset = oc_idx * ic * fh * fw;
            for (int oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_idx * stride_w * iw_pack_size;
                    const int dst_offset = oc_idx * oc_stride +
                                            (oh_idx * ow + ow_idx) * pack_oc_size;
                    nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_${ow_step}_oc8(
                        src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_idx,
                        output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, scale);
                }
                if (ow_remain) {
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_end * stride_w * iw_pack_size;
                    const int dst_offset = oc_idx * oc_stride +
                                            (oh_idx * ow + ow_end) * pack_oc_size;
                    nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_remain_oc8(src_ptr + src_offset,
                     weight_ptr + weight_offset, bias_ptr + oc_idx, output_ptr + dst_offset, 
                     packed_iw, packed_ic_stride, ld_dst_oc, ow_remain, scale);                    
                }
            }
        }
        if (oc_remain){
            const int weight_offset = oc_end * ic * fh * fw;
            for (int oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_idx * stride_w * iw_pack_size;
                    const int dst_offset = oc_end * oc_stride +
                                            (oh_idx * ow + ow_idx) * pack_oc_size;
                    nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_${ow_step}_oc4(
                        src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_end, 
                        output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, scale);
                }
                if (ow_remain) {                    
                    const int src_offset =  oh_idx * stride_h * packed_iw +
                                            ow_end * stride_w * iw_pack_size;
                    const int dst_offset = oc_end * oc_stride +
                                            (oh_idx * ow + ow_end) * pack_oc_size;
                    nchw_nchw44_s1_${filter_size}x${filter_size}_kernel_remain_oc4(src_ptr + src_offset, weight_ptr + weight_offset, bias_ptr + oc_end, output_ptr + dst_offset, packed_iw, packed_ic_stride, ld_dst_oc, ow_remain, scale);
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
