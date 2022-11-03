/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Pooling.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "FormatHelper.h"
#include "Pooling.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool PoolingKernel::IsAvailable(TContext* context) const {
    bool mode_ok = context->getAttrStr("format") == "NCHW";
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto dst_dtype = context->getAttrOprand("operand:1").dtype;
    bool dtype_ok =
            (src_dtype == dst_dtype) && (Utils::is_float_dtype(src_dtype) ||
                                         Utils::is_quant_dtype(src_dtype));
    if (Utils::is_quant_dtype(src_dtype)) {
        CC_ASSERT(context->getAttrOprand("operand:0").scale ==
                  context->getAttrOprand("operand:1").scale)
                << "quant pooling only support same scale\n";
    }
    return mode_ok && dtype_ok;
}
//! kernel gen
std::string PoolingImpl::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_pooling";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << context->getAttrStr("format");
    ss << "_p" << context->getAttrInt("pad_h") << "x"
       << context->getAttrInt("pad_w");
    ss << "_s" << context->getAttrInt("stride_h") << "x"
       << context->getAttrInt("stride_w");
    ss << "_w" << context->getAttrInt("window_h") << "x"
       << context->getAttrInt("window_w");
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

namespace {

std::string get_acc_dtype_specifier(const std::string& src_dtype,
                                    const std::string& mode) {
    if (Utils::is_float_dtype(src_dtype)) {
        return "float";
    } else if (Utils::is_quant_dtype(src_dtype, 8)) {
        if (mode == "MAX") {
            return "int8_t";
        } else {
            return "int";
        }
    } else {
        CC_ASSERT("not support dtype and mode ")
                << src_dtype << "," << mode << "\n";
    }
    return "InvalidType";
}

struct Pooler {
    Pooler(std::string mode, uint32_t windows_cnt, std::string acc_specifier)
            : m_mode(mode),
              m_windows_cnt(windows_cnt),
              m_acc_specifier(acc_specifier){};

    std::string gen_dep() {
        if (m_mode == "MAX") {
            return StringTemplate::StringTemplateArgs()
                    .add("acc_specifier", m_acc_specifier)
                    .render(R"(
static inline ${acc_specifier} max(${acc_specifier} a, ${acc_specifier} b){
    return a > b? a:b;
})");
        } else {
            return "";
        }
    }
    std::string gen_init_str() {
        if (m_mode == "MAX") {
            if (m_acc_specifier == "float") {
                return "float res = - __FLT_MAX__;";
            } else if (m_acc_specifier == "int8_t") {
                return "int8_t res = -128;";
            } else {
                CC_ABORT << "not support specifier " << m_acc_specifier << "\n";
            }
        } else if (m_mode == "AVERAGE") {
            return StringTemplate::StringTemplateArgs()
                    .add("acc_specifier", m_acc_specifier)
                    .render("${acc_specifier} sum = 0;");
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return StringTemplate::StringTemplateArgs()
                    .add("acc_specifier", m_acc_specifier)
                    .render(R"(
                      ${acc_specifier} sum = 0;
                      int count = 0;
                    )");
        }
        return "";
    }

    std::string gen_feed_str(std::string val) const {
        if (m_mode == "MAX") {
            return "res = max(res, " + val + ");";
        } else if (m_mode == "AVERAGE") {
            return "sum += " + val + ";";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return "sum += " + val + ";\n" + "count ++;";
        }
        return "";
    }

    std::string gen_final_str() {
        if (m_mode == "MAX") {
            return "";
        } else if (m_mode == "AVERAGE") {
            if (m_acc_specifier == "int") {
                return "sum = roundf((float)sum / " +
                       std::to_string(m_windows_cnt) + ");";
            } else {
                return "sum /= " + std::to_string(m_windows_cnt) + ";";
            }
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            if (m_acc_specifier == "int") {
                return "sum = roundf((float)sum / count);";
            } else {
                return "sum /= count;";
            }
        }
        return "";
    }

    std::string get_result_str() {
        if (m_mode == "MAX") {
            return "res";
        } else {
            return "sum";
        }
        return "";
    }

    std::string m_mode;
    uint32_t m_windows_cnt;
    std::string m_acc_specifier;
};

}  // namespace

std::string PoolingKernel::GetKernelBody(TContext* context) const {
    auto format_str = context->getAttrStr("format");
    auto mode_str = context->getAttrStr("mode");
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto dst_dtype = context->getAttrOprand("operand:1").dtype;
    auto src_specifier = Utils::cvt_dtype_specifier(src_dtype);
    auto dst_specifier = Utils::cvt_dtype_specifier(dst_dtype);
    auto acc_specifier = get_acc_dtype_specifier(src_dtype, mode_str);

    std::stringstream ss;
    const uint32_t window_h = context->getAttrInt("window_h");
    const uint32_t window_w = context->getAttrInt("window_w");
    Pooler pooler(mode_str, window_h * window_w, acc_specifier);
    ss << R"(
#include <stdbool.h>
)";
    ss << pooler.gen_dep();
    ss << GenFormatIter::gen_inline_format_iter_body(format_str);
    auto format_iter_symbol =
            GenFormatIter::gen_inline_format_iter_symbol(format_str);
    ss << GenCommonRet() << " " << GetKernelSignature(context) << "{\n";

    std::string body_temp = R"(
    const int oc_ratio = ${oc_ratio};
    const int ic_ratio = ${ic_ratio};
    const int batch_pos = ${batch_pos};
    const int spatial_start = ${spatial_start};
    const int dst_oc_idx = ${dst_oc_idx};
    const int src_ic_idx = ${src_ic_idx};

    const uint32_t window_h = ${window_h};
    const uint32_t window_w = ${window_w};
    const uint32_t ph = ${pad_h};
    const uint32_t pw = ${pad_w};
    const uint32_t sh = ${stride_h};
    const uint32_t sw = ${stride_w};
    ${src_specifier}* input_data = (${src_specifier}*)inputs[0]->ptr;
    TINYNN_ASSERT(input_data);
    ${dst_specifier}* output_data = (${dst_specifier}*)outputs[0]->ptr;
    TINYNN_ASSERT(output_data);
    const Tensor* src_tensor = inputs[0];
    TINYNN_ASSERT(src_tensor);
    const Tensor* dst_tensor = outputs[0];
    TINYNN_ASSERT(dst_tensor);
    Layout src_layout = inputs[0]->layout;
    const Layout dst_layout = dst_tensor->layout;
    const uint32_t batch = src_layout.dims[batch_pos];
    const uint32_t ih = src_layout.dims[spatial_start];
    const uint32_t iw = src_layout.dims[spatial_start + 1];
    const uint32_t oc = dst_layout.dims[dst_oc_idx] * oc_ratio;
    const uint32_t ic = src_layout.dims[src_ic_idx] * ic_ratio;
    const uint32_t oh = dst_layout.dims[spatial_start];
    const uint32_t ow = dst_layout.dims[spatial_start + 1];
    for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (uint32_t oc_idx = 0; oc_idx < oc; ++oc_idx) {
            for (uint32_t oh_idx = 0; oh_idx < oh; ++oh_idx) {
                for (uint32_t ow_idx = 0; ow_idx < ow; ++ow_idx) {
                    ${pool_init_str}
                    for (uint32_t fh_idx = 0; fh_idx < window_h; ++fh_idx) {
                        for (uint32_t fw_idx = 0; fw_idx < window_w; ++fw_idx) {
                            const uint32_t ih_idx = -ph + oh_idx * sh + fh_idx;
                            const uint32_t iw_idx = -pw + ow_idx * sw + fw_idx;
                            if (ih_idx < ih && iw_idx < iw) {
                                ${acc_specifier} in_elem = input_data[${format_iter_symbol}(
                                        batch_idx, oc_idx, ih_idx, iw_idx,
                                        src_layout.stride, false)];
                                ${gen_feed_str(in_elem)}
                            }
                        }
                    }
                    ${gen_final_str}
                    output_data[${format_iter_symbol}(batch_idx, oc_idx, oh_idx,
                                                     ow_idx, dst_layout.stride,
                                                     true)] = ${get_result_str};
                }
            }
        }
    }
    return TinyNN_SUCCESS;
})";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add("oc_ratio", 1)
                    .add("ic_ratio", 1)
                    .add("batch_pos", 0)
                    .add("spatial_start", 2)
                    .add("dst_oc_idx", 1)
                    .add("src_ic_idx", 1)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .add_ctx_int("window_h")
                    .add_ctx_int("window_w")
                    .add("pool_init_str", pooler.gen_init_str())
                    .add("format_iter_symbol", format_iter_symbol)
                    .add("gen_feed_str", StringTemplate::object_bind(
                                                 &Pooler::gen_feed_str, pooler))
                    .add("gen_final_str", pooler.gen_final_str())
                    .add("get_result_str", pooler.get_result_str())
                    .add("src_specifier", src_specifier)
                    .add("dst_specifier", dst_specifier)
                    .add("acc_specifier", acc_specifier)
                    .render(body_temp);

    return ss.str();
}

// vim: syntax=cpp.doxygen
