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

bool ConvDotNCHWNCHW44Common::IsAvailableCommon(
        TContext* ctx, const uint32_t valid_stride) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_w") == valid_stride &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
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
                       "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1))
                            .shape.size() == 5;
    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           bias_ok;
}
std::string ConvDotNCHWNCHW44Common::GetKernelSymbol(TContext* ctx) const {
    auto src_tensor = ctx->getAttrOprand("operand:0");
    CC_ASSERT((src_tensor.shape.size()) > 0) << "src_tensor.shape.size > 0";
    uint32_t ic = src_tensor.shape[1];
    auto dst_tensor = ctx->getAttrOprand(
            "operand:" + std::to_string(ctx->getAttrInt("nr_operands") - 1));
    constexpr uint32_t oc_pack_size = 4;
    uint32_t oc = dst_tensor.shape[1] * oc_pack_size;
    std::string name_temp = "${base_kernel_sym}_dot_nchw_nchw44_oc${oc}_ic${ic}";
    return StringTemplate::StringTemplateArgs(ctx)
            .add("base_kernel_sym", Arm64ConvImpl::GetKernelSymbol(ctx))
            .add("oc", oc)
            .add("ic", ic)
            .render(name_temp);
}

std::string ConvDotNCHWNCHW44Common::GetInitBody(TContext* ctx) const {
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

std::string ConvDotNCHWNCHW44Common::GetWorkspaceBody(TContext* context) const {
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
        const uint32_t iw_pack_size = (${stride_w} == 1 ? 4 : 1);
        const uint32_t packed_iw = (iw + 2 * pad_w) * iw_pack_size;
        const uint32_t packed_ih = ih + 2 * pad_h;
        const uint32_t border = 2 * ${cacheline_byte};
        *workspace = (size_t)ic * packed_ih * packed_iw * sizeof(int8_t) + border;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("stride_w")
                    .add("cacheline_byte", ARMBackend::cacheline_byte)
                    .render(workspace_temp);
    return ss.str();
}