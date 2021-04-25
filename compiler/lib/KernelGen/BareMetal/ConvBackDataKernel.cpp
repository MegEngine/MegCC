/**
 * \file
 * compiler/lib/KernelGen/BareMetal/ConvBackDataKernel.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <math.h>
#include <sstream>
#include <string>

#include "Activation.h"
#include "ConvBackDataKernel.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;
namespace {
std::string gen_filter_stride(std::string format_str, std::string sparse) {
    std::stringstream ss;
    if (format_str == "NCHW") {
        ss << R"(const int filter_stride[5] = {ocpg * icpg * fh * fw, icpg * fh * fw,
                                       fh * fw, fw, 1};)";
    } else {
        CC_ASSERT(format_str == "NCHW44") << "format not support\n";
        if (sparse == "DENSE") {
            ss << R"(const int filter_stride[7] = {ocpg * icpg * fh * fw, icpg * fh * fw * 4,
                                       fh * fw * 16, fw * 16, 16, 4, 1};)";
        } else {
            CC_ASSERT(sparse == "GROUP") << "spare must be GOURP or DENSE\n";
            ss << R"(const int filter_stride[4] = {fh * fw * 4, fw * 4, 4, 1};)";
        }
    }
    return ss.str();
}
std::string gen_inline_addr(std::string format_str, std::string sparse) {
    std::stringstream ss;
    ss << GenFormatIter::gen_inline_format_iter_body(format_str);
    ss << R"(static inline size_t get_filter_addr_)" << format_str << "_"
       << sparse;
    ss << R"((const int group, const int ocpg,
                                     const int icpg, const int fh, const int fw,
                                     const int* stride) {)";
    if (format_str == "NCHW") {
        ss << R"(return (size_t)group * stride[0] + ocpg * stride[1] + icpg * stride[2] +
           fh * stride[3] + fw * stride[4];)";
    } else {
        CC_ASSERT(format_str == "NCHW44") << "format not support\n";
        if (sparse == "DENSE") {
            ss << R"(return (size_t)group * stride[0] + ocpg / 4 * stride[1] + icpg / 4 * stride[2] +
           fh * stride[3] + fw * stride[4] + (icpg % 4) * stride[5] + (ocpg % 4) * stride[6];)";
        } else {
            CC_ASSERT(sparse == "GROUP") << "spare must be GOURP or DENSE\n";
            ss << R"(return (size_t)group / 4 * stride[0] + fh * stride[1] + fw * stride[2] + (group % 4) * stride[3];)";
        }
    }
    ss << "}\n";
    return ss.str();
}

std::string get_format(TContext* ctx) {
    auto format_str = ctx->getAttrStr("format");
    return format_str;
}

std::string get_src_foramt(const std::string& filter_format) {
    return filter_format;
}

std::string get_dst_foramt(const std::string& filter_format) {
    return filter_format;
}

}  // namespace

bool ConvBackDataGeneral::IsAvailable(TContext* ctx) const {
    bool param_mode_ok = (ctx->getAttrStr("format") == "NCHW" ||
                          ctx->getAttrStr("format") == "NCHW44") &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool type_float_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                         ((ctx->getAttrOprand("operand:0").dtype == "f32" &&
                           ctx->getAttrOprand("operand:1").dtype == "f32" &&
                           ctx->getAttrOprand("operand:2").dtype == "f32"));

    return param_mode_ok && (type_float_ok);
}

std::string ConvBackDataGeneral::GetKernelSymbol(TContext* ctx) const {
    std::stringstream extra_ss;
    extra_ss << "_" << SymbolHelper::gen_io_str(ctx);
    if (ctx->haveAttr("nonlineMode") &&
        ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        extra_ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    std::string name_temp =
            "kernel_convBackData_${kernel_h}x${kernel_w}_${format}_${sparse}_p$"
            "{pad_h}x${pad_w}_s${stride_h}x${stride_w}_d${dilate_h}x${dilate_w}"
            "${extra}";
    return StringTemplate::StringTemplateArgs(ctx)
            .add_ctx_int("kernel_h")
            .add_ctx_int("kernel_w")
            .add("format", get_format(ctx))
            .add_ctx_str("sparse")
            .add_ctx_int("pad_h")
            .add_ctx_int("pad_w")
            .add_ctx_int("stride_h")
            .add_ctx_int("stride_w")
            .add_ctx_int("dilate_h")
            .add_ctx_int("dilate_w")
            .add("extra", extra_ss.str())
            .render(name_temp);
}

std::string ConvBackDataGeneral::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    std::string noline_mode = context->haveAttr("nonlineMode")
                                      ? context->getAttrStr("nonlineMode")
                                      : "IDENTITY";
    auto sparse_str = context->getAttrStr("sparse");
    auto filter_format_str = get_format(context);
    auto src_format_str = get_src_foramt(filter_format_str);
    auto dst_format_str = get_dst_foramt(filter_format_str);
    auto dst_dtype = context->getAttrOprand("operand:1").dtype;
    auto flt_dtype = context->getAttrOprand("operand:0").dtype;
    int src_idx = context->getAttrInt("nr_operands") - 1;
    std::string src_dtype =
            context->getAttrOprand("operand:" + std::to_string(src_idx)).dtype;

    auto src_specifier = Utils::cvt_dtype_specifier(src_dtype);
    auto flt_specifier = Utils::cvt_dtype_specifier(flt_dtype);
    auto dst_specifier = Utils::cvt_dtype_specifier(dst_dtype);
    std::string acc_specifier = "float";
    if (src_specifier == "int8_t" && flt_specifier == "int8_t") {
        acc_specifier = "int";
    }

    uint32_t spatial_start = 2;
    uint32_t channel_pos = 1;
    uint32_t batch_pos = 0;
    uint32_t ocpg_ratio = 1;
    uint32_t icpg_ratio = 1;

    std::string group_str = "1";
    if (filter_format_str == "NCHW") {
        if (sparse_str == "GROUP") {
            group_str = "filter_weight->layout.dims[0]";
        }
    } else if (filter_format_str == "NCHW44") {
        ocpg_ratio = 4;
        icpg_ratio = 4;
        if (sparse_str == "GROUP") {
            group_str = "filter_weight->layout.dims[0] * 4";
        }
    } else if (filter_format_str == "NCHW_NCHW44") {
        ocpg_ratio = 4;
        icpg_ratio = 1;
        CC_ASSERT(sparse_str == "DENSE");
    } else {
        CC_ABORT << "not support filter_format_str " << filter_format_str;
    }

    ss << R"(
#include <stdbool.h>
)";
    ss << GenActivation::gen_func_call_with_typecvt_dep(
                  noline_mode, acc_specifier, dst_specifier)
       << "\n";
    ss << gen_inline_addr(filter_format_str, sparse_str);
    ss << GenCommonRet() << " " << GetKernelSignature(context) << "{\n";
    ss << "const uint32_t spatial_start = " << spatial_start << ";\n";
    ss << "const uint32_t channel_pos = " << channel_pos << ";\n";
    ss << "const uint32_t batch_pos = " << batch_pos << ";\n";
    ss << "const uint32_t ocpg_ratio = " << ocpg_ratio << ";\n";
    ss << "const uint32_t icpg_ratio = " << icpg_ratio << ";\n";
    ss << "const uint32_t ph = " << context->getAttrUInt("pad_h") << ";\n";
    ss << "const uint32_t pw = " << context->getAttrUInt("pad_w") << ";\n";
    ss << "const uint32_t sh = " << context->getAttrUInt("stride_h") << ";\n";
    ss << "const uint32_t sw = " << context->getAttrUInt("stride_w") << ";\n";
    ss << "const uint32_t dh = " << context->getAttrUInt("dilate_h") << ";\n";
    ss << "const uint32_t dw = " << context->getAttrUInt("dilate_w") << ";\n";
    ss << "const uint32_t fh = " << context->getAttrUInt("kernel_h") << ";\n";
    ss << "const uint32_t fw = " << context->getAttrUInt("kernel_w") << ";\n";

    std::string body_template = R"(
       
    const Tensor* dst_tensor = inputs[1];
    TINYNN_ASSERT(dst_tensor);

    const Tensor* filter_weight = inputs[0]; 
    TINYNN_ASSERT(filter_weight);

    Tensor* src_tensor = outputs[0];
    TINYNN_ASSERT(src_tensor);

    ${src_specifier}* src_ptr = src_tensor->ptr;
    TINYNN_ASSERT(src_ptr);

    const ${flt_specifier}* flt_ptr = filter_weight->ptr;
    TINYNN_ASSERT(flt_ptr);

    const ${dst_specifier}* dst_ptr = dst_tensor->ptr;
    TINYNN_ASSERT(dst_ptr);

    const Layout src_layout = src_tensor->layout;
    const Layout dst_layout = dst_tensor->layout;
    const float scale = src_tensor->dtype.param.scale;
    const float flt_scale = filter_weight->dtype.param.scale;
    const float dst_scale = dst_tensor->dtype.param.scale;
    TINYNN_ASSERT(batch_pos < src_layout.nr_dim);
    const uint32_t batch = src_layout.dims[batch_pos];
    const uint32_t ih = src_layout.dims[spatial_start];
    const uint32_t iw = src_layout.dims[spatial_start + 1];
    const uint32_t oh = dst_layout.dims[spatial_start];
    const uint32_t ow = dst_layout.dims[spatial_start + 1];
    const uint32_t group = ${group};
    const uint32_t ocpg = dst_layout.dims[channel_pos] * ocpg_ratio / group;
    const uint32_t icpg = src_layout.dims[channel_pos] * icpg_ratio / group;
    TINYNN_ASSERT_MSG(ocpg * group == dst_layout.dims[channel_pos] * ocpg_ratio, "%d * %d == %d * %d", ocpg, group, dst_layout.dims[channel_pos], ocpg_ratio);
    TINYNN_ASSERT_MSG(icpg * group == src_layout.dims[channel_pos] * icpg_ratio, "%d * %d == %d * %d", icpg, group, src_layout.dims[channel_pos], icpg_ratio);
    ${filter_stride}
    //! init grad data to zero
    for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (uint32_t group_idx = 0; group_idx < group; ++group_idx) {
            for (uint32_t icpg_idx = 0; icpg_idx < icpg; ++icpg_idx) {
                uint32_t ic_idx = group_idx * icpg + icpg_idx;
                for (uint32_t ih_idx = 0; ih_idx < ih; ++ih_idx) {
                    for (uint32_t iw_idx = 0; iw_idx < iw; ++iw_idx) {
                        src_ptr[${src_layout_iter_symbol}(batch_idx, ic_idx, ih_idx, iw_idx, src_layout.stride, false)] = 0;
                    }
                }
            }
        }
    }

    for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (uint32_t group_idx = 0; group_idx < group; ++group_idx) {
            for (uint32_t ocpg_idx = 0; ocpg_idx < ocpg; ++ocpg_idx) {
                uint32_t oc_idx = group_idx * ocpg + ocpg_idx;
                for (uint32_t oh_idx = 0; oh_idx < oh; ++oh_idx) {
                    for (uint32_t ow_idx = 0; ow_idx < ow; ++ow_idx) {
                        ${acc_specifier} dval = dst_ptr[${dst_layout_iter_symbol}(batch_idx, oc_idx, oh_idx,
                                                ow_idx, dst_layout.stride,
                                                true)];
                        for (uint32_t fh_idx = 0; fh_idx < fh; ++fh_idx) {
                            for (uint32_t fw_idx = 0; fw_idx < fw; ++fw_idx) {
                                uint32_t ih_idx =
                                        oh_idx * sh + fh_idx * dh - ph;
                                uint32_t iw_idx =
                                        ow_idx * sw + fw_idx * dw - pw;
                                if (ih_idx < ih && iw_idx < iw) {
                                    for (uint32_t icpg_idx = 0; icpg_idx < icpg;
                                         ++icpg_idx) {
                                        uint32_t ic_idx =
                                                group_idx * icpg + icpg_idx;
                                        ${acc_specifier}* sval = &src_ptr[${src_layout_iter_symbol}(
                                                batch_idx, ic_idx, ih_idx,
                                                iw_idx, src_layout.stride,
                                                false)];
                                        ${acc_specifier} fval = flt_ptr[${filter_iter_symbol}(
                                                group_idx, ocpg_idx, icpg_idx,
                                                fh_idx, fw_idx, filter_stride)];
                                        *sval += dval * fval;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return TinyNN_SUCCESS;
})";
    ss << StringTemplate::StringTemplateArgs()
                    .add("group", group_str)
                    .add("filter_stride",
                         gen_filter_stride(filter_format_str, sparse_str))
                    .add("src_layout_iter_symbol",
                         GenFormatIter::gen_inline_format_iter_symbol(
                                 src_format_str))
                    .add("dst_layout_iter_symbol",
                         GenFormatIter::gen_inline_format_iter_symbol(
                                 dst_format_str))
                    .add("filter_iter_symbol", "get_filter_addr_" +
                                                       filter_format_str + "_" +
                                                       sparse_str)
                    .add("act_func", GenActivation::gen_func_call_with_typecvt(
                                             noline_mode, "dval", acc_specifier,
                                             dst_specifier, "scale",
                                             "flt_scale", "dst_scale"))
                    .add("src_specifier", src_specifier)
                    .add("flt_specifier", flt_specifier)
                    .add("dst_specifier", dst_specifier)
                    .add("acc_specifier", acc_specifier)
                    .render(body_template);
    return ss.str();
}

// vim: syntax=cpp.doxygen
