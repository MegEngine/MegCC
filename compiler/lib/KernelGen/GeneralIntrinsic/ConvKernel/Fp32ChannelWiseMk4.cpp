#include <string>
#include "ConvKernel.h"
#include "GeneralIntrinsic/Activation.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

namespace {
struct ComputPadBorder {
    int m_kernel;
    int m_pad_h;
    int m_pad_w;
    int m_stride;
    std::string m_nonline_mode;
    bool m_flt_GI;

    ComputPadBorder(
            int kernel, int stride, int pad_h, int pad_w, std::string nonline_mode,
            bool flt_GI)
            : m_kernel(kernel),
              m_pad_h(pad_h),
              m_pad_w(pad_w),
              m_stride(stride),
              m_nonline_mode(nonline_mode),
              m_flt_GI(flt_GI) {}

    //! Gen the PaddingCompute function signature
    std::string GenPaddingComputeSignature() {
        std::string signature = R"(
        static void PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode} (
                                    const float* src, const float* bias,
                                    float* dst, const size_t stride,
                                    const size_t IH, const size_t IW,
                                    const size_t OH, const size_t OW,
                                    const float* kernel,
                                    const GI_FLOAT32_t* init))";
        if (m_flt_GI) {
            signature = R"(
        static void PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode} (
                                    const float* src, const float* bias,
                                    float* dst, const size_t stride,
                                    const size_t IH, const size_t IW,
                                    const size_t OH, const size_t OW,
                                    const GI_FLOAT32_FIXLEN_t* kernel,
                                    const GI_FLOAT32_t* init))";
        }
        return signature;
    }
    std::string GenComputeCodeGeneral(
            std::shared_ptr<ActivationGenIntrinsicBase> nonline_gen) {
        //! the compute code
        auto compute = [&]() -> std::string {
            std::string flt_read =
                    R"(GiLoadFloat32(kernel + (kh * ${kernel} + kw) * 4))";
            if (m_flt_GI) {
                flt_read = R"(GiFixLenType2GiFloat32Type(kernel[kh * ${kernel} + kw]))";
            }
            std::string body = R"(
            int iw = ow * ${stride} - ${pad_w};
            GI_FLOAT32_t result = *init;
            for (int kh = 0; kh < ${kernel}; kh++) {
                if (kh + ih < 0 || kh + ih >= (int)(IH))
                    continue;
                for (int kw = 0; kw < ${kernel}; kw++) {
                    if (kw + iw < 0 || kw + iw >= (int)(IW))
                        continue;
                    const float* sptr = src + (kh + ih) * IW * 4 + (kw + iw) * 4;
                    result = GiMlaqFloat32(result, ${flt_read},
                                      GiLoadFloat32(sptr));
                }
            }
            float* output = dst + oh * OW * 4 + ow * 4;)";
            std::stringstream ss;
            ss << StringTemplate::StringTemplateArgs()
                            .add("kernel", m_kernel)
                            .add("pad_h", m_pad_h)
                            .add("pad_w", m_pad_w)
                            .add("stride", m_stride)
                            .add("flt_read", flt_read)
                            .render(body);
            return ss.str();
        };

        auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
            return nonline_gen->GenIntrinsicFloat(str[0], str[1]);
        };

        std::string comp_body = R"(
        for (size_t oh = 0; oh < oh_start; oh++) {
            int ih = oh * stride - ${pad_h};
            for (size_t ow = 0; ow < OW; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
        }
        for (size_t oh = oh_start; oh < oh_end; oh++) {
            int ih = oh * stride - ${pad_h};
            for (size_t ow = 0; ow < ow_start; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
            for (size_t ow = ow_end; ow < OW; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);

            }
        }
        for (size_t oh = oh_end; oh < OH; oh++) {
            int ih = oh * stride - ${pad_h};
            for (size_t ow = 0; ow < OW; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);

            }
        })";

        std::stringstream ss;
        ss << StringTemplate::StringTemplateArgs()
                        .add("kernel", m_kernel)
                        .add("pad_h", m_pad_h)
                        .add("pad_w", m_pad_w)
                        .add("stride", m_stride)
                        .add("compute", compute)
                        .add("nonline_gen", nonline_gen_func)
                        .render(comp_body);
        return ss.str();
    }

    std::string GenComputeCodeK3P1(
            std::shared_ptr<ActivationGenIntrinsicBase> nonline_gen) {
        std::string compute_code = R"(
        // line one left
        {
            GI_FLOAT32_t result = *init;
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(src));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[5]), GiLoadFloat32(src + 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[7]), GiLoadFloat32(src + IW * 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[8]), GiLoadFloat32(src + IW * 4 + 4));
            float* output = dst;
            ${nonline_gen(result, result)}
            GiStoreFloat32(output, result);
        }
        // line one mid
        for (size_t ow = ow_start; ow < ow_end; ow++) {
            int iw = ow * ${stride} - ${pad_w};
            GI_FLOAT32_t result = *init;
            const float* sptr = src + iw * 4;
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[3]), GiLoadFloat32(sptr));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[5]), GiLoadFloat32(sptr + 8));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[6]), GiLoadFloat32(sptr + IW * 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[7]), GiLoadFloat32(sptr + IW * 4 + 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[8]), GiLoadFloat32(sptr + IW * 4 + 8));
            float* output = dst + ow * 4;
            ${nonline_gen(result, result)}
            GiStoreFloat32(output, result);
        }
        // line one right
        if (OW != ow_end) {
            GI_FLOAT32_t result = *init;
            const float* sptr = src + (ow_end * ${stride} - ${pad_w}) * 4;
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[3]), GiLoadFloat32(sptr));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[6]), GiLoadFloat32(sptr + IW * 4));
            result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[7]), GiLoadFloat32(sptr + IW * 4 + 4));
            float* output = dst + ow_end * 4;
            ${nonline_gen(result, result)}
            GiStoreFloat32(output, result);
        }
        // mid line
        for (size_t oh = oh_start; oh < oh_end; oh++) {
            int ih = oh * stride - ${pad_h};
            // left
            {
                GI_FLOAT32_t result = *init;
                const float* sptr = src + ih * IW * 4;
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[1]), GiLoadFloat32(sptr));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[2]), GiLoadFloat32(sptr + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[5]), GiLoadFloat32(sptr + IW * 4 + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[7]), GiLoadFloat32(sptr + 2 * IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[8]),
                                   GiLoadFloat32(sptr + 2 * IW * 4 + 4));
                float* output = dst + oh * OW * 4;
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
            // right
            if (OW != ow_end) {
                GI_FLOAT32_t result = *init;
                const float* sptr = src + ih * IW * 4 + (ow_end * ${stride} - ${pad_w}) * 4;
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[0]), GiLoadFloat32(sptr));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[1]), GiLoadFloat32(sptr + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[3]), GiLoadFloat32(sptr + IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + IW * 4 + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[6]), GiLoadFloat32(sptr + 2 * IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[7]),
                                   GiLoadFloat32(sptr + 2 * IW * 4 + 4));
                float* output = dst + oh * OW * 4 + ow_end * 4;
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
        }
        // last line left
        if (OH != oh_end) {
            size_t oh = OH - 1;
            {
                GI_FLOAT32_t result = *init;
                const float* sptr = src + (oh_end * ${stride} - ${pad_h}) * IW * 4;
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[1]), GiLoadFloat32(sptr));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[2]), GiLoadFloat32(sptr + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[5]), GiLoadFloat32(sptr + IW * 4 + 4));
                float* output = dst + oh_end * OW * 4;
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
            // last line mid
            for (size_t ow = ow_start; ow < ow_end; ow++) {
                int iw = ow * ${stride} - ${pad_w};
                GI_FLOAT32_t result = *init;
                const float* sptr = src + (oh_end * ${stride} - ${pad_h}) * IW * 4 + iw * 4;
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[0]), GiLoadFloat32(sptr));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[1]), GiLoadFloat32(sptr + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[2]), GiLoadFloat32(sptr + 8));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[3]), GiLoadFloat32(sptr + IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + IW * 4 + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[5]), GiLoadFloat32(sptr + IW * 4 + 8));
                float* output = dst + oh_end * OW * 4 + ow * 4;
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
            // last line right
            if (OW != ow_end) {
                GI_FLOAT32_t result = *init;
                const float* sptr = src + (oh_end * ${stride} - ${pad_h}) * IW * 4 +
                                    (ow_end * ${stride} - ${pad_w}) * 4;
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[0]), GiLoadFloat32(sptr));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[1]), GiLoadFloat32(sptr + 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[3]), GiLoadFloat32(sptr + IW * 4));
                result = GiMlaqFloat32(result, GiFixLenType2GiFloat32Type(kernel[4]), GiLoadFloat32(sptr + IW * 4 + 4));
                float* output = dst + oh_end * OW * 4 + ow_end * 4;
                ${nonline_gen(result, result)}
                GiStoreFloat32(output, result);
            }
        })";

        auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
            return nonline_gen->GenIntrinsicFloat(str[0], str[1]);
        };

        std::stringstream ss;
        ss << StringTemplate::StringTemplateArgs()
                        .add("kernel", m_kernel)
                        .add("pad_h", m_pad_h)
                        .add("pad_w", m_pad_w)
                        .add("stride", m_stride)
                        .add("nonline_gen", nonline_gen_func)
                        .render(compute_code);
        return ss.str();
    }

    std::string GenWholeFunction() {
        auto nonline_gen = create_activation_gener_instrinsic(m_nonline_mode);
        std::string body_string = GenPaddingComputeSignature();
        body_string += "{\n";
        body_string += R"(
        size_t oh_start = (${pad_h} + ${stride} - 1) / ${stride};
        size_t ow_start = (${pad_w} + ${stride} - 1) / ${stride};
        size_t oh_end = (IH + ${pad_h} - ${kernel}) / ${stride} + 1;
        size_t ow_end = (IW + ${pad_w} - ${kernel}) / ${stride} + 1;)";
        body_string += nonline_gen->GenIntrinsicInitFloat();
        if (m_kernel == 3 && m_pad_h == 1 && m_pad_w == 1 && m_stride == 1) {
            body_string += GenComputeCodeK3P1(nonline_gen);
        } else {
            body_string += GenComputeCodeGeneral(nonline_gen);
        }
        body_string += "\n}";
        std::stringstream ss;
        ss << StringTemplate::StringTemplateArgs()
                        .add("kernel", m_kernel)
                        .add("pad_h", m_pad_h)
                        .add("pad_w", m_pad_w)
                        .add("stride", m_stride)
                        .add("nonline_mode", m_nonline_mode)
                        .render(body_string);
        return ss.str();
    }
};
template <int compute_size>
static std::string compute_vec(std::vector<std::string> str) {
    std::string dst = str[0];
    std::string src1 = str[1];
    std::string src2 = str[2];
    std::stringstream ss;
    for (int i = 0; i < compute_size; ++i) {
        ss << dst
           << " = GiFloat32Type2FixLenType(GiMlaqFloat32(GiFixLenType2GiFloat32Type("
           << dst << "), GiFixLenType2GiFloat32Type((" << src1 << ")[" << i
           << "]), GiFixLenType2GiFloat32Type((" << src2 << ")[" << i << "])));\n";
    }
    return ss.str();
}

static std::string load_vec_x(std::vector<std::string> str) {
    int times = std::stoi(str[0]);
    std::string dst = str[1];
    std::string src = str[2];
    std::stringstream ss;
    for (int i = 0; i < times; i++) {
        ss << "(" << dst << ")[" << i << "] = GiFloat32Type2FixLenType(GiLoadFloat32(("
           << src << ")+ 4 *" << i << "));\n";
    }
    return ss.str();
}

}  // namespace

bool ChannelWiseFloatMk4::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            (ctx->getAttrUInt("kernel_w") == 3 || ctx->getAttrUInt("kernel_w") == 5) &&
            //! stride_h == stride_w and stride == 1 or stride == 2
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            (ctx->getAttrUInt("stride_h") == 1 || ctx->getAttrUInt("stride_h") == 2) &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;

    bool param_mode_ok = ctx->getAttrStr("sparse") == "GROUP" &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f32" &&
                   ctx->getAttrOprand("operand:1").dtype == "f32" &&
                   ctx->getAttrOprand("operand:2").dtype == "f32";
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;
    bool channel_wise_ok = ctx->getAttrOprand("operand:1").shape.size() == 6 &&
                           ctx->getAttrOprand("operand:1").shape[5] == 4 &&
                           ctx->getAttrOprand("operand:1").shape[1] == 1 &&
                           ctx->getAttrOprand("operand:1").shape[2] == 1;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           channel_wise_ok;
}

std::string ChannelWiseFloatMk4::GetKernelBody(TContext* ctx) const {
    int kernel = ctx->getAttrUInt("kernel_h");
    int stride = ctx->getAttrUInt("stride_h");
    int pad_h = ctx->getAttrUInt("pad_h");
    int pad_w = ctx->getAttrUInt("pad_w");
    auto kern_size = ctx->getAttrUInt("kernel_w");

    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    bool flt_all_in_reg = (5 == kern_size) ? false : true;
    auto border_compute =
            ComputPadBorder(kernel, stride, pad_h, pad_w, nonline_mode, flt_all_in_reg);
    std::stringstream writer;
    writer << "#include\"gi_float.h\"\n";
    writer << "#include<string.h>\n";
    writer << border_compute.GenWholeFunction();
    writer << "\n\n";

    writer << GenCommonRet() << " " << GetKernelSignature(ctx);

    if (3 == kern_size && 1 == stride) {
        writer << GenBodyMk4K3S1(ctx);
    } else if (3 == kern_size && 2 == stride) {
        writer << GenBodyMk4K3S2(ctx);
    } else if (5 == kern_size && 1 == stride) {
        writer << GenBodyMk4K5S1(ctx);
    } else if (5 == kern_size && 2 == stride) {
        writer << GenBodyMk4K5S2(ctx);
    } else {
        CC_ABORT << "unsupported stride in mk4 channel wise kernel.\n";
    }
    return writer.str();
}

std::string ChannelWiseFloatMk4::GenBodyMk4K3S1(TContext* ctx) const {
    int kernel = ctx->getAttrUInt("kernel_h");
    int pad_h = ctx->getAttrUInt("pad_h");
    int pad_w = ctx->getAttrUInt("pad_w");
    bool with_bias = is_bias(ctx);
    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::stringstream writer;
    writer << " {\n";
    writer << R"(
    size_t N = inputs[0]->layout.dims[0];
    size_t ICB = inputs[0]->layout.dims[1];
    size_t IH = inputs[0]->layout.dims[2];
    size_t IW = inputs[0]->layout.dims[3];
    size_t IHW = IH * IW;

    size_t OCB = outputs[0]->layout.dims[1];
    size_t OH = outputs[0]->layout.dims[2];
    size_t OW = outputs[0]->layout.dims[3];
    size_t OHW = OH * OW;

    TINYNN_ASSERT(ICB == OCB);

    float* input_data = inputs[0]->ptr;
    float* output_data = outputs[0]->ptr;
    float* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n < N; n++) {
        for (int g = 0; g < ICB; g++) {
            float* filter = weight_data + g * 9 * 4;
            float* src = input_data + g * IHW * 4;
            float* dst = output_data + g * OHW * 4;
            ${nonline_gen_init()}

            GI_FLOAT32_FIXLEN_t kernel[9];
            for (int i = 0; i< 9; i++)
                kernel[i] = GiFloat32Type2FixLenType(GiLoadFloat32(filter + i * 4));
            ${gen_bias_load()}
            GI_FLOAT32_FIXLEN_t bias_fix = GiFloat32Type2FixLenType(bias);

            size_t oh_start = ${pad_h};
            size_t ow_start = ${pad_w};
            size_t oh_end = IH + ${pad_h} - 2;
            size_t ow_end = IW + ${pad_w} - 2;

            PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode}(src, NULL, dst, 1, IH, IW, OH, OW, kernel, &bias);
            size_t oh = oh_start;
            for (; oh + 1 < oh_end; oh += 2) {
                size_t ih = oh - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 3 < ow_end; ow += 4) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2][4] = {bias_fix, bias_fix, bias_fix, bias_fix, 
                                               bias_fix, bias_fix, bias_fix,bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][6];

                    ${load_vec_x(6, src_v[0], input)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], &kernel[0])}
                    ${compute_vec(dst_v[0][2], &src_v[0][2], &kernel[0])}
                    ${compute_vec(dst_v[0][3], &src_v[0][3], &kernel[0])}
                    ${load_vec_x(6, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[0][1], &src_v[1][1], &kernel[3])}
                    ${compute_vec(dst_v[0][2], &src_v[1][2], &kernel[3])}
                    ${compute_vec(dst_v[0][3], &src_v[1][3], &kernel[3])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], &kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], &kernel[0])}
                    ${compute_vec(dst_v[1][2], &src_v[1][2], &kernel[0])}
                    ${compute_vec(dst_v[1][3], &src_v[1][3], &kernel[0])}
                    ${load_vec_x(6, src_v[0], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], &kernel[6])}
                    ${compute_vec(dst_v[0][2], &src_v[0][2], &kernel[6])}
                    ${compute_vec(dst_v[0][3], &src_v[0][3], &kernel[6])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], &kernel[3])}
                    ${compute_vec(dst_v[1][1], &src_v[0][1], &kernel[3])}
                    ${compute_vec(dst_v[1][2], &src_v[0][2], &kernel[3])}
                    ${compute_vec(dst_v[1][3], &src_v[0][3], &kernel[3])}
                    ${load_vec_x(6, src_v[1], input + 3 * IW * 4)}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], &kernel[6])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], &kernel[6])}
                    ${compute_vec(dst_v[1][2], &src_v[1][2], &kernel[6])}
                    ${compute_vec(dst_v[1][3], &src_v[1][3], &kernel[6])}

                    GI_FLOAT32_t tmp0, tmp1;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][1]), tmp1)}
                    GiStoreFloat32(output + 1 * 4, tmp1);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][2]), tmp0)}
                    GiStoreFloat32(output + 2 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][3]), tmp1)}
                    GiStoreFloat32(output + 3 * 4, tmp1);

                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat32(output + OW * 4 + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][1]), tmp1)}
                    GiStoreFloat32(output + OW * 4 + 1 * 4, tmp1);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][2]), tmp0)}
                    GiStoreFloat32(output + OW * 4 + 2 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][3]), tmp1)}
                    GiStoreFloat32(output + OW * 4 + 3 * 4, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1], &src_v[1][0], &kernel[0])}
                    ${load_vec_x(3, src_v[0], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[0][0], &kernel[3])}
                    ${load_vec_x(3, src_v[1], input + 3 * IW * 4)}
                    ${compute_vec(dst_v[1], &src_v[1][0], &kernel[6])}
                    
                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4 * OW, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp0)}
                    GiStoreFloat32(output + 1 * 4 * OW, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 3 < ow_end; ow += 4) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[4] = {bias_fix, bias_fix, bias_fix, bias_fix};

                    GI_FLOAT32_FIXLEN_t src_v[2][6];
                    ${load_vec_x(6, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], &kernel[0])}
                    ${compute_vec(dst_v[2], &src_v[0][2], &kernel[0])}
                    ${compute_vec(dst_v[3], &src_v[0][3], &kernel[0])}
                    ${load_vec_x(6, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1], &src_v[1][1], &kernel[3])}
                    ${compute_vec(dst_v[2], &src_v[1][2], &kernel[3])}
                    ${compute_vec(dst_v[3], &src_v[1][3], &kernel[3])}
                    ${load_vec_x(6, src_v[0], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[0][1], &kernel[6])}
                    ${compute_vec(dst_v[2], &src_v[0][2], &kernel[6])}
                    ${compute_vec(dst_v[3], &src_v[0][3], &kernel[6])}

                    GI_FLOAT32_t tmp0, tmp1;

                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp1)}
                    GiStoreFloat32(output + 1 * 4, tmp1);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[2]), tmp0)}
                    GiStoreFloat32(output + 2 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[3]), tmp1)}
                    GiStoreFloat32(output + 3 * 4, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[3][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[2], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[2][0], &kernel[6])}

                    GI_FLOAT32_t tmp;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp)}
                    GiStoreFloat32(output, tmp);
                }
            }
        }
        input_data += ICB * IHW * 4;
        output_data += ICB * OHW * 4;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";

    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_init = [&]() -> std::string {
        return nonline_gen->GenIntrinsicInitFloat();
    };
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        auto input = str[0];
        auto output = str[1];
        return nonline_gen->GenIntrinsicFloat(input, output);
    };
    auto gen_bias_load = [&]() {
        if (with_bias) {
            return "GI_FLOAT32_t bias = GiLoadFloat32(bias_data + g * 4);\n";
        } else {
            return "GI_FLOAT32_t bias = GiBroadcastFloat32(0.f);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "float* bias_data = inputs[2]->ptr;\n";
        } else {
            return "";
        }
    };

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("kernel", kernel)
                    .add("pad_h", pad_h)
                    .add("pad_w", pad_w)
                    .add("stride", 1)
                    .add("nonline_mode", nonline_mode)
                    .add("compute_vec", compute_vec<3>)
                    .add("load_vec_x", load_vec_x)
                    .add("load_vec_x", load_vec_x)
                    .add("nonline_gen_func", nonline_gen_func)
                    .add("nonline_gen_init", nonline_gen_init)
                    .add("gen_bias_ptr", gen_bias_ptr)
                    .add("gen_bias_load", gen_bias_load)
                    .render(writer.str());
    return ss.str();
}

std::string ChannelWiseFloatMk4::GenBodyMk4K5S1(TContext* ctx) const {
    int kernel = ctx->getAttrUInt("kernel_h");
    int pad_h = ctx->getAttrUInt("pad_h");
    int pad_w = ctx->getAttrUInt("pad_w");
    bool with_bias = is_bias(ctx);
    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::stringstream writer;
    writer << " {\n";
    writer << R"(
    size_t N = inputs[0]->layout.dims[0];
    size_t ICB = inputs[0]->layout.dims[1];
    size_t IH = inputs[0]->layout.dims[2];
    size_t IW = inputs[0]->layout.dims[3];
    size_t IHW = IH * IW;

    size_t OCB = outputs[0]->layout.dims[1];
    size_t OH = outputs[0]->layout.dims[2];
    size_t OW = outputs[0]->layout.dims[3];
    size_t OHW = OH * OW;

    TINYNN_ASSERT(ICB == OCB);

    float* input_data = inputs[0]->ptr;
    float* output_data = outputs[0]->ptr;
    float* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n < N; n++) {
        for (int g = 0; g < ICB; g++) {
            float* src = input_data + g * IHW * 4;
            float* dst = output_data + g * OHW * 4;
            const float* filter = weight_data + g * 25 * 4;
            ${nonline_gen_init()}

            ${gen_bias_load()}
            GI_FLOAT32_FIXLEN_t bias_fix = GiFloat32Type2FixLenType(bias);

            size_t oh_start = ${pad_h};
            size_t ow_start = ${pad_w};
            size_t oh_end = OH - ${pad_h};
            size_t ow_end = OW - ${pad_w};
            PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode}(src, NULL, dst, 1, IH, IW, OH, OW, filter, &bias);
            size_t oh = oh_start;
            for (; oh + 1 < oh_end; oh += 2) {
                size_t ih = oh - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2][2] = {bias_fix, bias_fix, bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][6];
                    GI_FLOAT32_FIXLEN_t kernel[2][5];
                    //! line 0
                    ${load_vec_x(6, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], kernel[0])}
                    //! line 1
                    ${load_vec_x(6,  src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[1][1], kernel[1])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], kernel[0])}
                    //! line 2
                    ${load_vec_x(6,  src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], kernel[0])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[1][1], &src_v[0][1], kernel[1])}
                    //! line 3
                    ${load_vec_x(6,  src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[1][1], kernel[1])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], kernel[0])}
                    //! line 4
                    ${load_vec_x(6,  src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], kernel[0])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[1][1], &src_v[0][1], kernel[1])}

                    ${load_vec_x(6, src_v[1], input + 5 * IW * 4)};
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])};
                    ${compute_vec(dst_v[1][1], &src_v[1][1], kernel[0])};

                    GI_FLOAT32_t tmp0, tmp1;

                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][1]), tmp1)}
                    GiStoreFloat32(output + 1 * 4, tmp1);

                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat32(output + OW * 4 + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][1]), tmp1)}
                    GiStoreFloat32(output + OW * 4 + 1 * 4, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][5];
                    GI_FLOAT32_FIXLEN_t kernel[2][5];
                    //! line 0
                    ${load_vec_x(5, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5, src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}
                    //! line 2
                    ${load_vec_x(5, src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[1])}
                    //! line 3
                    ${load_vec_x(5, src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}
                    //! line 4
                    ${load_vec_x(5, src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[1])}

                    ${load_vec_x(5, src_v[1], input + 5 * IW * 4)}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}

                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4 * OW, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp0)}
                    GiStoreFloat32(output + 1 * 4 * OW, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};

                    GI_FLOAT32_FIXLEN_t src_v[2][6];
                    GI_FLOAT32_FIXLEN_t kernel[2][5];
                    //! line 0
                    ${load_vec_x(6, src_v[0], input + 0 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], kernel[0])}
                    //! line 1
                    ${load_vec_x(6,  src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][1], kernel[1])}
                    //! line 2
                    ${load_vec_x(6,  src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], kernel[0])}
                    //! line 3
                    ${load_vec_x(6,  src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][1], kernel[1])}
                    //! line 4
                    ${load_vec_x(6,  src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], kernel[0])}

                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp0)}
                    GiStoreFloat32(output + 1 * 4, tmp0);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][5];
                    GI_FLOAT32_FIXLEN_t kernel[2][5];

                    //! line 0
                    ${load_vec_x(5, src_v[0], input + 0 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5,  src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 2
                    ${load_vec_x(5,  src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 3
                    ${load_vec_x(5,  src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 4
                    ${load_vec_x(5,  src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                }
            }
        }
        input_data += ICB * IHW * 4;
        output_data += ICB * OHW * 4;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";

    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_init = [&]() -> std::string {
        return nonline_gen->GenIntrinsicInitFloat();
    };
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        auto input = str[0];
        auto output = str[1];
        return nonline_gen->GenIntrinsicFloat(input, output);
    };
    auto gen_bias_load = [&]() {
        if (with_bias) {
            return "GI_FLOAT32_t bias = GiLoadFloat32(bias_data + g * 4);\n";
        } else {
            return "GI_FLOAT32_t bias = GiBroadcastFloat32(0.f);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "float* bias_data = inputs[2]->ptr;\n";
        } else {
            return "";
        }
    };

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("kernel", kernel)
                    .add("pad_h", pad_h)
                    .add("pad_w", pad_w)
                    .add("stride", 1)
                    .add("nonline_mode", nonline_mode)
                    .add("compute_vec", compute_vec<5>)
                    .add("load_vec_x", load_vec_x)
                    .add("load_vec_x", load_vec_x)
                    .add("nonline_gen_func", nonline_gen_func)
                    .add("nonline_gen_init", nonline_gen_init)
                    .add("gen_bias_ptr", gen_bias_ptr)
                    .add("gen_bias_load", gen_bias_load)
                    .render(writer.str());
    return ss.str();
}

std::string ChannelWiseFloatMk4::GenBodyMk4K3S2(TContext* ctx) const {
    int kernel = ctx->getAttrUInt("kernel_h");
    int pad_h = ctx->getAttrUInt("pad_h");
    int pad_w = ctx->getAttrUInt("pad_w");
    bool with_bias = is_bias(ctx);
    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::stringstream writer;
    writer << " {\n";
    writer << R"(
    size_t IH = inputs[0]->layout.dims[2];
    size_t IW = inputs[0]->layout.dims[3];
    size_t IHW = IH * IW;
    size_t ICB = inputs[0]->layout.dims[1];
    size_t N = inputs[0]->layout.dims[0];

    size_t OH = outputs[0]->layout.dims[2];
    size_t OW = outputs[0]->layout.dims[3];
    size_t OCB = outputs[0]->layout.dims[1];
    size_t OHW = OH * OW;

    TINYNN_ASSERT(ICB == OCB);

    float* input_data = inputs[0]->ptr;
    float* output_data = outputs[0]->ptr;
    float* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n< N; n++) {
        for (int g = 0; g < ICB; g++) {
            float* filter = weight_data + g * 9 * 4;
            float* src = input_data + g * IHW * 4;
            float* dst = output_data + g * OHW * 4;
            ${nonline_gen_init()}

            GI_FLOAT32_FIXLEN_t kernel[9];
            for (int i = 0; i< 9; i++)
                kernel[i] =  GiFloat32Type2FixLenType(GiLoadFloat32(filter + i * 4));
            ${gen_bias_load()}
            GI_FLOAT32_FIXLEN_t bias_fix = GiFloat32Type2FixLenType(bias);

            size_t oh_start = (${pad_h} + 1) / 2;
            size_t ow_start = (${pad_w} + 1) / 2;
            size_t oh_end = (IH + ${pad_h} - 3) / 2 + 1;
            size_t ow_end = (IW + ${pad_w} - 3) / 2 + 1;

            PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode}(src, NULL, dst, 2, IH, IW, OH, OW, kernel, &bias);
            size_t oh = oh_start;
            for (; oh + 1 < oh_end; oh += 2) {
                size_t ih = oh * 2 - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow * 2 - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2][2] = {bias_fix, bias_fix, bias_fix, bias_fix};

                    GI_FLOAT32_FIXLEN_t src_v[2][5];
                    ${load_vec_x(5, src_v[0], input)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], &kernel[0])}
                    ${load_vec_x(5, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[0][1], &src_v[1][2], &kernel[3])}
                    ${load_vec_x(5, src_v[0], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], &kernel[6])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], &kernel[0])}
                    ${load_vec_x(5, src_v[1], input + 3 * IW * 4)}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1][1], &src_v[1][2], &kernel[3])}
                    ${load_vec_x(5, src_v[0], input + 4 * IW * 4)}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], &kernel[6])}

                    GI_FLOAT32_t tmp0, tmp1;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][1]), tmp1)}
                    GiStoreFloat32(output + 1 * 4, tmp1);

                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat32(output + OW * 4 + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][1]), tmp1)}
                    GiStoreFloat32(output + OW * 4 + 1 * 4, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow * 2 - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[0], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + 3 * IW * 4)}
                    ${compute_vec(dst_v[1], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[0], input + 4 * IW * 4)}
                    ${compute_vec(dst_v[1], &src_v[0][0], &kernel[6])}

                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp0)}
                    GiStoreFloat32(output + OW * 4, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh * 2 - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow * 2 - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[3][5];
                    ${load_vec_x(5, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][2], &kernel[0])}
                    ${load_vec_x(5, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1], &src_v[1][2], &kernel[3])}
                    ${load_vec_x(5, src_v[2], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[2][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[2][2], &kernel[6])}

                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)};
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp0)};
                    GiStoreFloat32(output + 1 * 4, tmp0);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow * 2 - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[3][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[2], input + 2 * IW * 4)}
                    ${compute_vec(dst_v[0], &src_v[2][0], &kernel[6])}

                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output, tmp0);
                }
            }
        }
        input_data += ICB * IHW * 4;
        output_data += ICB * OHW * 4;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";
    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_init = [&]() -> std::string {
        return nonline_gen->GenIntrinsicInitFloat();
    };
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        auto input = str[0];
        auto output = str[1];
        return nonline_gen->GenIntrinsicFloat(input, output);
    };
    auto gen_bias_load = [&]() {
        if (with_bias) {
            return "GI_FLOAT32_t bias = GiLoadFloat32(bias_data + g * 4);\n";
        } else {
            return "GI_FLOAT32_t bias = GiBroadcastFloat32(0.f);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "float* bias_data = inputs[2]->ptr;\n";
        } else {
            return "";
        }
    };
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("kernel", kernel)
                    .add("pad_h", pad_h)
                    .add("pad_w", pad_w)
                    .add("stride", 2)
                    .add("nonline_mode", nonline_mode)
                    .add("compute_vec", compute_vec<3>)
                    .add("load_vec_x", load_vec_x)
                    .add("load_vec_x", load_vec_x)
                    .add("nonline_gen_func", nonline_gen_func)
                    .add("nonline_gen_init", nonline_gen_init)
                    .add("gen_bias_ptr", gen_bias_ptr)
                    .add("gen_bias_load", gen_bias_load)
                    .render(writer.str());
    return ss.str();
}

std::string ChannelWiseFloatMk4::GenBodyMk4K5S2(TContext* ctx) const {
    int kernel = ctx->getAttrUInt("kernel_h");
    int pad_h = ctx->getAttrUInt("pad_h");
    int pad_w = ctx->getAttrUInt("pad_w");
    bool with_bias = is_bias(ctx);
    int stride = 2;

    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::stringstream writer;
    writer << " {\n";
    writer << R"(
    size_t N = inputs[0]->layout.dims[0];
    size_t ICB = inputs[0]->layout.dims[1];
    size_t IH = inputs[0]->layout.dims[2];
    size_t IW = inputs[0]->layout.dims[3];
    size_t IHW = IH * IW;

    size_t OCB = outputs[0]->layout.dims[1];
    size_t OH = outputs[0]->layout.dims[2];
    size_t OW = outputs[0]->layout.dims[3];
    size_t OHW = OH * OW;
    const size_t stride = 2;
    const size_t flt_size = 5;

    TINYNN_ASSERT(ICB == OCB);

    float* input_data = inputs[0]->ptr;
    float* output_data = outputs[0]->ptr;
    float* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n < N; n++) {
        for (int g = 0; g < ICB; g++) {
            float* src = input_data + g * IHW * 4;
            float* dst = output_data + g * OHW * 4;
            const float* filter = weight_data + g * 25 * 4;
            ${nonline_gen_init()}

            ${gen_bias_load()}
            GI_FLOAT32_FIXLEN_t bias_fix = GiFloat32Type2FixLenType(bias);
            size_t oh_start = (${pad_h} + stride - 1) / stride;
            size_t ow_start = (${pad_w} + stride - 1) / stride;
            size_t oh_end = (IH + ${pad_h} - flt_size) / stride + 1;
            size_t ow_end = (IW + ${pad_w} - flt_size) / stride + 1;

            PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode}(src, NULL, dst, 2, IH, IW, OH, OW, filter, &bias);
            size_t oh = oh_start;
            for (; oh + 1 < oh_end; oh += 2) {
                size_t ih = oh * stride - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow * stride - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2][2] = {bias_fix, bias_fix, bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][7];
                    GI_FLOAT32_FIXLEN_t kernel[3][5];
                    //! line 0
                    ${load_vec_x(7, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], kernel[0])}
                    //! line 1
                    ${load_vec_x(7,  src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[1][2], kernel[1])}
                    //! line 2
                    ${load_vec_x(7,  src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[2], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[2])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], kernel[2])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], kernel[0])}
                    //! line 3
                    ${load_vec_x(7,  src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[1][2], kernel[0])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1][1], &src_v[1][2], kernel[1])}
                    //! line 4
                    ${load_vec_x(7,  src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], kernel[1])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[2])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], kernel[2])}
                    //! line 5
                    ${load_vec_x(7,  src_v[1], input + 5 * IW * 4)}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][2], kernel[0])}
                    //! line 6
                    ${load_vec_x(7, src_v[0], input + 6 * IW * 4)};
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[1])};
                    ${compute_vec(dst_v[1][1], &src_v[0][2], kernel[1])};
                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0][1]), tmp0)}
                    GiStoreFloat32(output + 1 * 4, tmp0);

                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat32(output + OW * 4 + 0 * 4, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1][1]), tmp0)}
                    GiStoreFloat32(output + OW * 4 + 1 * 4, tmp0);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow * stride - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][5];
                    GI_FLOAT32_FIXLEN_t kernel[3][5];
                    //! line 0
                    ${load_vec_x(5, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5, src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 2
                    ${load_vec_x(5, src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[2], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[2])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[0])}
                    //! line 3
                    ${load_vec_x(5, src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[1])}
                    //! line 4
                    ${load_vec_x(5, src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[2])}
                    //! line 5
                    ${load_vec_x(5, src_v[1], input + 5 * IW * 4)}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}
                    //! line 6
                    ${load_vec_x(5, src_v[0], input + 6 * IW * 4)}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[1])}
                    GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4 * OW, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[1]), tmp0)}
                    GiStoreFloat32(output + 1 * 4 * OW, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh * stride - ${pad_h};
                size_t ow = ow_start;

                for (; ow < ow_end; ow++) {
                    size_t iw = ow * stride - ${pad_w};
                    const float* input = src + ih * IW * 4 + iw * 4;
                    float* output = dst + oh * OW * 4 + ow * 4;
                    GI_FLOAT32_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT32_FIXLEN_t src_v[2][5];
                    GI_FLOAT32_FIXLEN_t kernel[2][5];

                    //! line 0
                    ${load_vec_x(5, src_v[0], input + 0 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5,  src_v[1], input + 1 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 2
                    ${load_vec_x(5,  src_v[0], input + 2 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 3
                    ${load_vec_x(5,  src_v[1], input + 3 * IW * 4)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 4
                    ${load_vec_x(5,  src_v[0], input + 4 * IW * 4)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 4)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                     GI_FLOAT32_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat32Type(dst_v[0]), tmp0)}
                    GiStoreFloat32(output + 0 * 4, tmp0);
                }
            }
        }
        input_data += ICB * IHW * 4;
        output_data += ICB * OHW * 4;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";

    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_init = [&]() -> std::string {
        return nonline_gen->GenIntrinsicInitFloat();
    };
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        auto input = str[0];
        auto output = str[1];
        return nonline_gen->GenIntrinsicFloat(input, output);
    };
    auto gen_bias_load = [&]() {
        if (with_bias) {
            return "GI_FLOAT32_t bias = GiLoadFloat32(bias_data + g * 4);\n";
        } else {
            return "GI_FLOAT32_t bias = GiBroadcastFloat32(0.f);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "float* bias_data = inputs[2]->ptr;\n";
        } else {
            return "";
        }
    };

    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("kernel", kernel)
                    .add("pad_h", pad_h)
                    .add("pad_w", pad_w)
                    .add("stride", stride)
                    .add("nonline_mode", nonline_mode)
                    .add("compute_vec", compute_vec<5>)
                    .add("load_vec_x", load_vec_x)
                    .add("load_vec_x", load_vec_x)
                    .add("nonline_gen_func", nonline_gen_func)
                    .add("nonline_gen_init", nonline_gen_init)
                    .add("gen_bias_ptr", gen_bias_ptr)
                    .add("gen_bias_load", gen_bias_load)
                    .render(writer.str());
    return ss.str();
}
// vim: syntax=cpp.doxygen
