#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ConvKernel/ConvKernel.h"
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
                                    const gi_float16_t* src, const gi_float16_t* bias,
                                    gi_float16_t* dst, const size_t stride,
                                    const size_t IH, const size_t IW,
                                    const size_t OH, const size_t OW,
                                    const gi_float16_t* kernel,
                                    const GI_FLOAT16_t* init))";
        if (m_flt_GI) {
            signature = R"(
        static void PaddingComputeK${kernel}P${pad_h}x${pad_w}S${stride}Bias${nonline_mode} (
                                    const gi_float16_t* src, const gi_float16_t* bias,
                                    gi_float16_t* dst, const size_t stride,
                                    const size_t IH, const size_t IW,
                                    const size_t OH, const size_t OW,
                                    const GI_FLOAT16_FIXLEN_t* kernel,
                                    const GI_FLOAT16_t* init))";
        }
        return signature;
    }
    std::string GenComputeCodeGeneral(
            std::shared_ptr<ActivationGenIntrinsicBase> nonline_gen) {
        //! the compute code
        auto compute = [&]() -> std::string {
            std::string flt_read =
                    R"(GiLoadFloat16(kernel + (kh * ${kernel} + kw) * 8))";
            if (m_flt_GI) {
                flt_read = R"(GiFixLenType2GiFloat16Type(kernel[kh * ${kernel} + kw]))";
            }
            std::string body = R"(
            int iw = ow * ${stride} - ${pad_w};
            GI_FLOAT16_t result = *init;
            for (int kh = 0; kh < ${kernel}; kh++) {
                if (kh + ih < 0 || kh + ih >= (int)(IH))
                    continue;
                for (int kw = 0; kw < ${kernel}; kw++) {
                    if (kw + iw < 0 || kw + iw >= (int)(IW))
                        continue;
                    const gi_float16_t* sptr = src + (kh + ih) * IW * 8 + (kw + iw) * 8;
                    result = GiMlaqFloat16(result, ${flt_read},
                                      GiLoadFloat16(sptr));
                }
            }
            gi_float16_t* output = dst + oh * OW * 8 + ow * 8;)";
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
                GiStoreFloat16(output, result);
            }
        }
        for (size_t oh = oh_start; oh < oh_end; oh++) {
            int ih = oh * stride - ${pad_h};
            for (size_t ow = 0; ow < ow_start; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);
            }
            for (size_t ow = ow_end; ow < OW; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);

            }
        }
        for (size_t oh = oh_end; oh < OH; oh++) {
            int ih = oh * stride - ${pad_h};
            for (size_t ow = 0; ow < OW; ow++) {
                ${compute()};
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);

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
            GI_FLOAT16_t result = *init;
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(src));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[5]), GiLoadFloat16(src + 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[7]), GiLoadFloat16(src + IW * 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[8]), GiLoadFloat16(src + IW * 8 + 8));
            gi_float16_t* output = dst;
            ${nonline_gen(result, result)}
            GiStoreFloat16(output, result);
        }
        // line one mid
        for (size_t ow = ow_start; ow < ow_end; ow++) {
            int iw = ow * ${stride} - ${pad_w};
            GI_FLOAT16_t result = *init;
            const gi_float16_t* sptr = src + iw * 8;
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[3]), GiLoadFloat16(sptr));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[5]), GiLoadFloat16(sptr + 16));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[6]), GiLoadFloat16(sptr + IW * 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[7]), GiLoadFloat16(sptr + IW * 8 + 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[8]), GiLoadFloat16(sptr + IW * 8 + 16));
            gi_float16_t* output = dst + ow * 8;
            ${nonline_gen(result, result)}
            GiStoreFloat16(output, result);
        }
        // line one right
        if (OW != ow_end) {
            GI_FLOAT16_t result = *init;
            const gi_float16_t* sptr = src + (ow_end * ${stride} - ${pad_w}) * 8;
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[3]), GiLoadFloat16(sptr));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[6]), GiLoadFloat16(sptr + IW * 8));
            result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[7]), GiLoadFloat16(sptr + IW * 8 + 8));
            gi_float16_t* output = dst + ow_end * 8;
            ${nonline_gen(result, result)}
            GiStoreFloat16(output, result);
        }
        // mid line
        for (size_t oh = oh_start; oh < oh_end; oh++) {
            int ih = oh * stride - ${pad_h};
            // left
            {
                GI_FLOAT16_t result = *init;
                const gi_float16_t* sptr = src + ih * IW * 8;
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[1]), GiLoadFloat16(sptr));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[2]), GiLoadFloat16(sptr + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[5]), GiLoadFloat16(sptr + IW * 8 + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[7]), GiLoadFloat16(sptr + 2 * IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[8]),
                                   GiLoadFloat16(sptr + 2 * IW * 8 + 8));
                gi_float16_t* output = dst + oh * OW * 8;
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);
            }
            // right
            if (OW != ow_end) {
                GI_FLOAT16_t result = *init;
                const gi_float16_t* sptr = src + ih * IW * 8 + (ow_end * ${stride} - ${pad_w}) * 8;
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[0]), GiLoadFloat16(sptr));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[1]), GiLoadFloat16(sptr + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[3]), GiLoadFloat16(sptr + IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + IW * 8 + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[6]), GiLoadFloat16(sptr + 2 * IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[7]),
                                   GiLoadFloat16(sptr + 2 * IW * 8 + 8));
                gi_float16_t* output = dst + oh * OW * 8 + ow_end * 8;
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);
            }
        }
        // last line left
        if (OH != oh_end) {
            size_t oh = OH - 1;
            {
                GI_FLOAT16_t result = *init;
                const gi_float16_t* sptr = src + (oh_end * ${stride} - ${pad_h}) * IW * 8;
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[1]), GiLoadFloat16(sptr));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[2]), GiLoadFloat16(sptr + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[5]), GiLoadFloat16(sptr + IW * 8 + 8));
                gi_float16_t* output = dst + oh_end * OW * 8;
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);
            }
            // last line mid
            for (size_t ow = ow_start; ow < ow_end; ow++) {
                int iw = ow * ${stride} - ${pad_w};
                GI_FLOAT16_t result = *init;
                const gi_float16_t* sptr = src + (oh_end * ${stride} - ${pad_h}) * IW * 8 + iw * 8;
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[0]), GiLoadFloat16(sptr));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[1]), GiLoadFloat16(sptr + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[2]), GiLoadFloat16(sptr + 16));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[3]), GiLoadFloat16(sptr + IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + IW * 8 + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[5]), GiLoadFloat16(sptr + IW * 8 + 16));
                gi_float16_t* output = dst + oh_end * OW * 8 + ow * 8;
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);
            }
            // last line right
            if (OW != ow_end) {
                GI_FLOAT16_t result = *init;
                const gi_float16_t* sptr = src + (oh_end * ${stride} - ${pad_h}) * IW * 8 +
                                    (ow_end * ${stride} - ${pad_w}) * 8;
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[0]), GiLoadFloat16(sptr));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[1]), GiLoadFloat16(sptr + 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[3]), GiLoadFloat16(sptr + IW * 8));
                result = GiMlaqFloat16(result, GiFixLenType2GiFloat16Type(kernel[4]), GiLoadFloat16(sptr + IW * 8 + 8));
                gi_float16_t* output = dst + oh_end * OW * 8 + ow_end * 8;
                ${nonline_gen(result, result)}
                GiStoreFloat16(output, result);
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
        auto nonline_gen = create_activation_gener_instrinsic(m_nonline_mode, "f16");
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
           << " = GiFloat16Type2FixLenType(GiMlaqFloat16(GiFixLenType2GiFloat16Type("
           << dst << "), GiFixLenType2GiFloat16Type((" << src1 << ")[" << i
           << "]), GiFixLenType2GiFloat16Type((" << src2 << ")[" << i << "])));\n";
    }
    return ss.str();
}

static std::string load_vec_x(std::vector<std::string> str) {
    int times = std::stoi(str[0]);
    std::string dst = str[1];
    std::string src = str[2];
    std::stringstream ss;
    for (int i = 0; i < times; i++) {
        ss << "(" << dst << ")[" << i << "] = GiFloat16Type2FixLenType(GiLoadFloat16(("
           << src << ")+ 8 *" << i << "));\n";
    }
    return ss.str();
}

}  // namespace

bool ChannelWiseFloat16Mk8::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            (ctx->getAttrUInt("kernel_w") == 3 || ctx->getAttrUInt("kernel_w") == 5) &&
            //! stride_h == stride_w and stride == 1 or stride == 2
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            (ctx->getAttrUInt("stride_h") == 1 || ctx->getAttrUInt("stride_h") == 2) &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;

    bool param_mode_ok = ctx->getAttrStr("sparse") == "GROUP" &&
                         ctx->getAttrStr("format") == "NCHW88" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f16" &&
                   ctx->getAttrOprand("operand:1").dtype == "f16" &&
                   ctx->getAttrOprand("operand:2").dtype == "f16";
    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 8;
    bool channel_wise_ok = ctx->getAttrOprand("operand:1").shape.size() == 6 &&
                           ctx->getAttrOprand("operand:1").shape[5] == 8 &&
                           ctx->getAttrOprand("operand:1").shape[1] == 1 &&
                           ctx->getAttrOprand("operand:1").shape[2] == 1;
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           channel_wise_ok;
}

std::string ChannelWiseFloat16Mk8::GetKernelBody(TContext* ctx) const {
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
    writer << "#include\"gi_float16.h\"\n";
    writer << "#include<string.h>\n";
    writer << border_compute.GenWholeFunction();
    writer << "\n\n";

    writer << GenCommonRet() << " " << GetKernelSignature(ctx);

    if (3 == kern_size && 1 == stride) {
        writer << GenBodyMk8K3S1(ctx);
    } else if (3 == kern_size && 2 == stride) {
        writer << GenBodyMk8K3S2(ctx);
    } else if (5 == kern_size && 1 == stride) {
        writer << GenBodyMk8K5S1(ctx);
    } else if (5 == kern_size && 2 == stride) {
        writer << GenBodyMk8K5S2(ctx);
    } else {
        CC_ABORT << "unsupported stride in mk8 channel wise kernel.\n";
    }
    return writer.str();
}

std::string ChannelWiseFloat16Mk8::GenBodyMk8K3S1(TContext* ctx) const {
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

    gi_float16_t* input_data = inputs[0]->ptr;
    gi_float16_t* output_data = outputs[0]->ptr;
    gi_float16_t* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n < N; n++) {
        for (int g = 0; g < ICB; g++) {
            gi_float16_t* filter = weight_data + g * 9 * 8;
            gi_float16_t* src = input_data + g * IHW * 8;
            gi_float16_t* dst = output_data + g * OHW * 8;
            ${nonline_gen_init()}

            GI_FLOAT16_FIXLEN_t kernel[9];
            for (int i = 0; i< 9; i++)
                kernel[i] = GiFloat16Type2FixLenType(GiLoadFloat16(filter + i * 8));
            ${gen_bias_load()}
            GI_FLOAT16_FIXLEN_t bias_fix = GiFloat16Type2FixLenType(bias);

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
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2][4] = {bias_fix, bias_fix, bias_fix, bias_fix, 
                                               bias_fix, bias_fix, bias_fix,bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][6];

                    ${load_vec_x(6, src_v[0], input)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], &kernel[0])}
                    ${compute_vec(dst_v[0][2], &src_v[0][2], &kernel[0])}
                    ${compute_vec(dst_v[0][3], &src_v[0][3], &kernel[0])}
                    ${load_vec_x(6, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[0][1], &src_v[1][1], &kernel[3])}
                    ${compute_vec(dst_v[0][2], &src_v[1][2], &kernel[3])}
                    ${compute_vec(dst_v[0][3], &src_v[1][3], &kernel[3])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], &kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], &kernel[0])}
                    ${compute_vec(dst_v[1][2], &src_v[1][2], &kernel[0])}
                    ${compute_vec(dst_v[1][3], &src_v[1][3], &kernel[0])}
                    ${load_vec_x(6, src_v[0], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], &kernel[6])}
                    ${compute_vec(dst_v[0][2], &src_v[0][2], &kernel[6])}
                    ${compute_vec(dst_v[0][3], &src_v[0][3], &kernel[6])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], &kernel[3])}
                    ${compute_vec(dst_v[1][1], &src_v[0][1], &kernel[3])}
                    ${compute_vec(dst_v[1][2], &src_v[0][2], &kernel[3])}
                    ${compute_vec(dst_v[1][3], &src_v[0][3], &kernel[3])}
                    ${load_vec_x(6, src_v[1], input + 3 * IW * 8)}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], &kernel[6])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], &kernel[6])}
                    ${compute_vec(dst_v[1][2], &src_v[1][2], &kernel[6])}
                    ${compute_vec(dst_v[1][3], &src_v[1][3], &kernel[6])}

                    GI_FLOAT16_t tmp0, tmp1;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][1]), tmp1)}
                    GiStoreFloat16(output + 1 * 8, tmp1);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][2]), tmp0)}
                    GiStoreFloat16(output + 2 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][3]), tmp1)}
                    GiStoreFloat16(output + 3 * 8, tmp1);

                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat16(output + OW * 8 + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][1]), tmp1)}
                    GiStoreFloat16(output + OW * 8 + 1 * 8, tmp1);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][2]), tmp0)}
                    GiStoreFloat16(output + OW * 8 + 2 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][3]), tmp1)}
                    GiStoreFloat16(output + OW * 8 + 3 * 8, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1], &src_v[1][0], &kernel[0])}
                    ${load_vec_x(3, src_v[0], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[0][0], &kernel[3])}
                    ${load_vec_x(3, src_v[1], input + 3 * IW * 8)}
                    ${compute_vec(dst_v[1], &src_v[1][0], &kernel[6])}
                    
                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8 * OW, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp0)}
                    GiStoreFloat16(output + 1 * 8 * OW, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 3 < ow_end; ow += 4) {
                    size_t iw = ow - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[4] = {bias_fix, bias_fix, bias_fix, bias_fix};

                    GI_FLOAT16_FIXLEN_t src_v[2][6];
                    ${load_vec_x(6, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], &kernel[0])}
                    ${compute_vec(dst_v[2], &src_v[0][2], &kernel[0])}
                    ${compute_vec(dst_v[3], &src_v[0][3], &kernel[0])}
                    ${load_vec_x(6, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1], &src_v[1][1], &kernel[3])}
                    ${compute_vec(dst_v[2], &src_v[1][2], &kernel[3])}
                    ${compute_vec(dst_v[3], &src_v[1][3], &kernel[3])}
                    ${load_vec_x(6, src_v[0], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[0][1], &kernel[6])}
                    ${compute_vec(dst_v[2], &src_v[0][2], &kernel[6])}
                    ${compute_vec(dst_v[3], &src_v[0][3], &kernel[6])}

                    GI_FLOAT16_t tmp0, tmp1;

                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp1)}
                    GiStoreFloat16(output + 1 * 8, tmp1);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[2]), tmp0)}
                    GiStoreFloat16(output + 2 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[3]), tmp1)}
                    GiStoreFloat16(output + 3 * 8, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[3][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[2], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[2][0], &kernel[6])}

                    GI_FLOAT16_t tmp;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp)}
                    GiStoreFloat16(output, tmp);
                }
            }
        }
        input_data += ICB * IHW * 8;
        output_data += ICB * OHW * 8;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";

    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode, "f16");
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
            return "GI_FLOAT16_t bias = GiLoadFloat16(bias_data + g * 8);\n";
        } else {
            return "GI_FLOAT16_t bias = GiBroadcastFloat16(0.0);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "gi_float16_t* bias_data = inputs[2]->ptr;\n";
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

std::string ChannelWiseFloat16Mk8::GenBodyMk8K5S1(TContext* ctx) const {
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

    gi_float16_t* input_data = inputs[0]->ptr;
    gi_float16_t* output_data = outputs[0]->ptr;
    gi_float16_t* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n < N; n++) {
        for (int g = 0; g < ICB; g++) {
            gi_float16_t* src = input_data + g * IHW * 8;
            gi_float16_t* dst = output_data + g * OHW * 8;
            const gi_float16_t* filter = weight_data + g * 25 * 8;
            ${nonline_gen_init()}

            ${gen_bias_load()}
            GI_FLOAT16_FIXLEN_t bias_fix = GiFloat16Type2FixLenType(bias);

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
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2][2] = {bias_fix, bias_fix, bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][6];
                    GI_FLOAT16_FIXLEN_t kernel[2][5];
                    //! line 0
                    ${load_vec_x(6, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], kernel[0])}
                    //! line 1
                    ${load_vec_x(6,  src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[1][1], kernel[1])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], kernel[0])}
                    //! line 2
                    ${load_vec_x(6,  src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], kernel[0])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[1][1], &src_v[0][1], kernel[1])}
                    //! line 3
                    ${load_vec_x(6,  src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[1][1], kernel[1])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][1], kernel[0])}
                    //! line 4
                    ${load_vec_x(6,  src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][1], kernel[0])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[1][1], &src_v[0][1], kernel[1])}

                    ${load_vec_x(6, src_v[1], input + 5 * IW * 8)};
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])};
                    ${compute_vec(dst_v[1][1], &src_v[1][1], kernel[0])};

                    GI_FLOAT16_t tmp0, tmp1;

                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][1]), tmp1)}
                    GiStoreFloat16(output + 1 * 8, tmp1);

                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat16(output + OW * 8 + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][1]), tmp1)}
                    GiStoreFloat16(output + OW * 8 + 1 * 8, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][5];
                    GI_FLOAT16_FIXLEN_t kernel[2][5];
                    //! line 0
                    ${load_vec_x(5, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5, src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}
                    //! line 2
                    ${load_vec_x(5, src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[1])}
                    //! line 3
                    ${load_vec_x(5, src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}
                    //! line 4
                    ${load_vec_x(5, src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[1])}

                    ${load_vec_x(5, src_v[1], input + 5 * IW * 8)}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}

                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8 * OW, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp0)}
                    GiStoreFloat16(output + 1 * 8 * OW, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};

                    GI_FLOAT16_FIXLEN_t src_v[2][6];
                    GI_FLOAT16_FIXLEN_t kernel[2][5];
                    //! line 0
                    ${load_vec_x(6, src_v[0], input + 0 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], kernel[0])}
                    //! line 1
                    ${load_vec_x(6,  src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][1], kernel[1])}
                    //! line 2
                    ${load_vec_x(6,  src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], kernel[0])}
                    //! line 3
                    ${load_vec_x(6,  src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[1][1], kernel[1])}
                    //! line 4
                    ${load_vec_x(6,  src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][1], kernel[0])}

                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp0)}
                    GiStoreFloat16(output + 1 * 8, tmp0);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][5];
                    GI_FLOAT16_FIXLEN_t kernel[2][5];

                    //! line 0
                    ${load_vec_x(5, src_v[0], input + 0 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5,  src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 2
                    ${load_vec_x(5,  src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 3
                    ${load_vec_x(5,  src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 4
                    ${load_vec_x(5,  src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                }
            }
        }
        input_data += ICB * IHW * 8;
        output_data += ICB * OHW * 8;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";

    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode, "f16");
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
            return "GI_FLOAT16_t bias = GiLoadFloat16(bias_data + g * 8);\n";
        } else {
            return "GI_FLOAT16_t bias = GiBroadcastFloat16(0.0);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "gi_float16_t* bias_data = inputs[2]->ptr;\n";
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

std::string ChannelWiseFloat16Mk8::GenBodyMk8K3S2(TContext* ctx) const {
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

    gi_float16_t* input_data = inputs[0]->ptr;
    gi_float16_t* output_data = outputs[0]->ptr;
    gi_float16_t* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n< N; n++) {
        for (int g = 0; g < ICB; g++) {
            gi_float16_t* filter = weight_data + g * 9 * 8;
            gi_float16_t* src = input_data + g * IHW * 8;
            gi_float16_t* dst = output_data + g * OHW * 8;
            ${nonline_gen_init()}

            GI_FLOAT16_FIXLEN_t kernel[9];
            for (int i = 0; i< 9; i++)
                kernel[i] =  GiFloat16Type2FixLenType(GiLoadFloat16(filter + i * 8));
            ${gen_bias_load()}
            GI_FLOAT16_FIXLEN_t bias_fix = GiFloat16Type2FixLenType(bias);

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
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2][2] = {bias_fix, bias_fix, bias_fix, bias_fix};

                    GI_FLOAT16_FIXLEN_t src_v[2][5];
                    ${load_vec_x(5, src_v[0], input)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], &kernel[0])}
                    ${load_vec_x(5, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[0][1], &src_v[1][2], &kernel[3])}
                    ${load_vec_x(5, src_v[0], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], &kernel[6])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], &kernel[0])}
                    ${load_vec_x(5, src_v[1], input + 3 * IW * 8)}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1][1], &src_v[1][2], &kernel[3])}
                    ${load_vec_x(5, src_v[0], input + 4 * IW * 8)}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], &kernel[6])}

                    GI_FLOAT16_t tmp0, tmp1;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][1]), tmp1)}
                    GiStoreFloat16(output + 1 * 8, tmp1);

                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat16(output + OW * 8 + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][1]), tmp1)}
                    GiStoreFloat16(output + OW * 8 + 1 * 8, tmp1);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow * 2 - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[0], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + 3 * IW * 8)}
                    ${compute_vec(dst_v[1], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[0], input + 4 * IW * 8)}
                    ${compute_vec(dst_v[1], &src_v[0][0], &kernel[6])}

                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp0)}
                    GiStoreFloat16(output + OW * 8, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh * 2 - ${pad_h};
                size_t ow = ow_start;
                for (; ow + 1 < ow_end; ow += 2) {
                    size_t iw = ow * 2 - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[3][5];
                    ${load_vec_x(5, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[0][2], &kernel[0])}
                    ${load_vec_x(5, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${compute_vec(dst_v[1], &src_v[1][2], &kernel[3])}
                    ${load_vec_x(5, src_v[2], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[2][0], &kernel[6])}
                    ${compute_vec(dst_v[1], &src_v[2][2], &kernel[6])}

                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)};
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp0)};
                    GiStoreFloat16(output + 1 * 8, tmp0);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow * 2 - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[3][3];
                    ${load_vec_x(3, src_v[0], input)}
                    ${compute_vec(dst_v[0], &src_v[0][0], &kernel[0])}
                    ${load_vec_x(3, src_v[1], input + IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], &kernel[3])}
                    ${load_vec_x(3, src_v[2], input + 2 * IW * 8)}
                    ${compute_vec(dst_v[0], &src_v[2][0], &kernel[6])}

                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output, tmp0);
                }
            }
        }
        input_data += ICB * IHW * 8;
        output_data += ICB * OHW * 8;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";
    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode, "f16");
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
            return "GI_FLOAT16_t bias = GiLoadFloat16(bias_data + g * 8);\n";
        } else {
            return "GI_FLOAT16_t bias = GiBroadcastFloat16(0.0);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "gi_float16_t* bias_data = inputs[2]->ptr;\n";
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

std::string ChannelWiseFloat16Mk8::GenBodyMk8K5S2(TContext* ctx) const {
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

    gi_float16_t* input_data = inputs[0]->ptr;
    gi_float16_t* output_data = outputs[0]->ptr;
    gi_float16_t* weight_data = inputs[1]->ptr;
    ${gen_bias_ptr()}

    for(int n = 0; n < N; n++) {
        for (int g = 0; g < ICB; g++) {
            gi_float16_t* src = input_data + g * IHW * 8;
            gi_float16_t* dst = output_data + g * OHW * 8;
            const gi_float16_t* filter = weight_data + g * 25 * 8;
            ${nonline_gen_init()}

            ${gen_bias_load()}
            GI_FLOAT16_FIXLEN_t bias_fix = GiFloat16Type2FixLenType(bias);
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
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2][2] = {bias_fix, bias_fix, bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][7];
                    GI_FLOAT16_FIXLEN_t kernel[3][5];
                    //! line 0
                    ${load_vec_x(7, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], kernel[0])}
                    //! line 1
                    ${load_vec_x(7,  src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[1][2], kernel[1])}
                    //! line 2
                    ${load_vec_x(7,  src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[2], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[2])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], kernel[2])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], kernel[0])}
                    //! line 3
                    ${load_vec_x(7,  src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[0][1], &src_v[1][2], kernel[0])}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[1])}
                    ${compute_vec(dst_v[1][1], &src_v[1][2], kernel[1])}
                    //! line 4
                    ${load_vec_x(7,  src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0][0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[0][1], &src_v[0][2], kernel[1])}
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[2])}
                    ${compute_vec(dst_v[1][1], &src_v[0][2], kernel[2])}
                    //! line 5
                    ${load_vec_x(7,  src_v[1], input + 5 * IW * 8)}
                    ${compute_vec(dst_v[1][0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1][1], &src_v[1][2], kernel[0])}
                    //! line 6
                    ${load_vec_x(7, src_v[0], input + 6 * IW * 8)};
                    ${compute_vec(dst_v[1][0], &src_v[0][0], kernel[1])};
                    ${compute_vec(dst_v[1][1], &src_v[0][2], kernel[1])};
                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0][1]), tmp0)}
                    GiStoreFloat16(output + 1 * 8, tmp0);

                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][0]), tmp0)}
                    GiStoreFloat16(output + OW * 8 + 0 * 8, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1][1]), tmp0)}
                    GiStoreFloat16(output + OW * 8 + 1 * 8, tmp0);
                }
                for (; ow < ow_end; ow++) {
                    size_t iw = ow * stride - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[2] = {bias_fix, bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][5];
                    GI_FLOAT16_FIXLEN_t kernel[3][5];
                    //! line 0
                    ${load_vec_x(5, src_v[0], input)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5, src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 2
                    ${load_vec_x(5, src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[2], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[2])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[0])}
                    //! line 3
                    ${load_vec_x(5, src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[0])}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[1])}
                    //! line 4
                    ${load_vec_x(5, src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[1])}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[2])}
                    //! line 5
                    ${load_vec_x(5, src_v[1], input + 5 * IW * 8)}
                    ${compute_vec(dst_v[1], &src_v[1][0], kernel[0])}
                    //! line 6
                    ${load_vec_x(5, src_v[0], input + 6 * IW * 8)}
                    ${compute_vec(dst_v[1], &src_v[0][0], kernel[1])}
                    GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8 * OW, tmp0);
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[1]), tmp0)}
                    GiStoreFloat16(output + 1 * 8 * OW, tmp0);
                }
            }
            for (; oh < oh_end; oh++) {
                size_t ih = oh * stride - ${pad_h};
                size_t ow = ow_start;

                for (; ow < ow_end; ow++) {
                    size_t iw = ow * stride - ${pad_w};
                    const gi_float16_t* input = src + ih * IW * 8 + iw * 8;
                    gi_float16_t* output = dst + oh * OW * 8 + ow * 8;
                    GI_FLOAT16_FIXLEN_t dst_v[1] = {bias_fix};
                    GI_FLOAT16_FIXLEN_t src_v[2][5];
                    GI_FLOAT16_FIXLEN_t kernel[2][5];

                    //! line 0
                    ${load_vec_x(5, src_v[0], input + 0 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 0 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 1
                    ${load_vec_x(5,  src_v[1], input + 1 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 1 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 2
                    ${load_vec_x(5,  src_v[0], input + 2 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 2 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                    //! line 3
                    ${load_vec_x(5,  src_v[1], input + 3 * IW * 8)}
                    ${load_vec_x(5, kernel[1], filter + 3 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[1][0], kernel[1])}
                    //! line 4
                    ${load_vec_x(5,  src_v[0], input + 4 * IW * 8)}
                    ${load_vec_x(5, kernel[0], filter + 4 * 5 * 8)}
                    ${compute_vec(dst_v[0], &src_v[0][0], kernel[0])}
                     GI_FLOAT16_t tmp0;
                    ${nonline_gen_func(GiFixLenType2GiFloat16Type(dst_v[0]), tmp0)}
                    GiStoreFloat16(output + 0 * 8, tmp0);
                }
            }
        }
        input_data += ICB * IHW * 8;
        output_data += ICB * OHW * 8;
    }
    return TinyNN_SUCCESS;)";
    writer << "\n}";

    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode, "f16");
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
            return "GI_FLOAT16_t bias = GiLoadFloat16(bias_data + g * 8);\n";
        } else {
            return "GI_FLOAT16_t bias = GiBroadcastFloat16(0.0);\n";
        }
    };
    auto gen_bias_ptr = [&]() {
        if (with_bias) {
            return "gi_float16_t* bias_data = inputs[2]->ptr;\n";
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
