/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/PoolingKernel/Fp16PoolingNchw88.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <sstream>
#include "../GIMathHelper.h"
#include "Pooling.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool PoolingNchw88Fp16::IsAvailable(TContext* context) const {
    bool format_ok = context->getAttrStr("format") == "NCHW88";
    auto mode_str = context->getAttrStr("mode");
    bool mode_ok = mode_str == "MAX" || mode_str == "AVERAGE" ||
                   mode_str == "AVERAGE_COUNT_EXCLUDE_PADDING";
    bool dtype_ok = context->getAttrOprand("operand:0").dtype == "f16";
    return format_ok && mode_ok && dtype_ok;
}

std::string PoolingNchw88Fp16::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "GI_";
    ss << PoolingImpl::GetKernelSymbol(context);
    ss << "_" << SymbolHelper::gen_io_str(context) << "_fallback";
    return ss.str();
}

namespace {
struct Pooler {
    Pooler(std::string mode, uint32_t windows_cnt)
            : m_mode(mode), m_windows_cnt(windows_cnt){};
    std::string gen_init_str() {
        if (m_mode == "MAX") {
            return "GI_FLOAT16_t res = GiBroadcastFloat16(-65504);";
        } else if (m_mode == "AVERAGE") {
            return "GI_FLOAT16_t sum = GiBroadcastFloat16(0.0);";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return R"(GI_FLOAT16_t sum = GiBroadcastFloat16(0.f);
                      int count = 0;
            )";
        }
    }
    std::string gen_feed_str(std::string val) const {
        if (m_mode == "MAX") {
            return "res = GiMaximumFloat16(res, " + val + ");";
        } else if (m_mode == "AVERAGE") {
            return "sum = GiAddFloat16(sum, " + val + ");";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return "sum = GiAddFloat16(sum, " + val + ");\n" + "++count;";
        }
    }
    std::string gen_final_str() {
        if (m_mode == "MAX") {
            return "";
        } else if (m_mode == "AVERAGE") {
            return "GI_FLOAT32_t win_cnt = GiCastToFloat32(GiBroadcastInt32(" +
                   std::to_string(m_windows_cnt) + "));\n" +
                   "sum =  GiDivideFloat16(sum, GiCastFloat32ToFloat16(win_cnt, "
                   "win_cnt));";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return "GI_FLOAT32_t win_cnt = GiCastToFloat32(GiBroadcastInt32(count));\n "
                   "sum =  GiDivideFloat16(sum, GiCastFloat32ToFloat16(win_cnt, "
                   "win_cnt));";
        }
    }
    std::string get_result_str() {
        if (m_mode == "MAX") {
            return "res";
        } else {
            return "sum";
        }
    }

    std::string m_mode;
    uint32_t m_windows_cnt;
};

}  // namespace

std::string PoolingNchw88Fp16::GetKernelBody(TContext* context) const {
    auto format_str = context->getAttrStr("format");
    auto mode_str = context->getAttrStr("mode");
    std::stringstream ss;
    const uint32_t window_h = context->getAttrInt("window_h");
    const uint32_t window_w = context->getAttrInt("window_w");
    Pooler pooler(mode_str, window_h * window_w);
    ss << R"(
#include <stdbool.h>
#include "gi_float.h"
#include "gi_float16.h"
)";
    GIMathHelper gi_math;
    ss << gi_math.GiDivideFloat16() << "\n";
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
    ${dtype_specifier}* input_data = (${dtype_specifier}*)inputs[0]->ptr;
    TINYNN_ASSERT(input_data);
    ${dtype_specifier}* output_data = (${dtype_specifier}*)outputs[0]->ptr;
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
    const uint32_t ic = oc;
    const uint32_t oh = dst_layout.dims[spatial_start];
    const uint32_t ow = dst_layout.dims[spatial_start + 1];
    for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        for (uint32_t oc_idx = 0; oc_idx < oc; oc_idx += 8) {
            for (uint32_t oh_idx = 0; oh_idx < oh; ++oh_idx) {
                for (uint32_t ow_idx = 0; ow_idx < ow; ++ow_idx) {
                    ${pool_init_str}
                    for (uint32_t fh_idx = 0; fh_idx < window_h; ++fh_idx) {
                        const uint32_t ih_idx = -ph + oh_idx * sh + fh_idx;
                        if (ih_idx >= ih)
                            continue;
                        for (uint32_t fw_idx = 0; fw_idx < window_w; ++fw_idx) {                            
                            const uint32_t iw_idx = -pw + ow_idx * sw + fw_idx;
                            if (iw_idx < iw) {
                                GI_FLOAT16_t in_elem = GiLoadFloat16(input_data + batch_idx * ic * ih * iw + oc_idx * ih * iw + (ih_idx * iw + iw_idx) * 8);
                                ${gen_feed_str(in_elem)}
                            }
                        }
                    }
                    ${gen_final_str}
                    GiStoreFloat16(output_data + batch_idx * oc * oh * ow + oc_idx * oh * ow + (oh_idx * ow + ow_idx) * 8, ${get_result_str});
                }
            }
        }
    }
    return TinyNN_SUCCESS;
})";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add("oc_ratio", 8)
                    .add("ic_ratio", 8)
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
                    .add("gen_feed_str",
                         StringTemplate::object_bind(&Pooler::gen_feed_str, pooler))
                    .add("gen_final_str", pooler.gen_final_str())
                    .add("get_result_str", pooler.get_result_str())
                    .add("dtype_specifier",
                         Utils::cvt_dtype_specifier(
                                 context->getAttrOprand("operand:0").dtype))
                    .render(body_temp);

    return ss.str();
}

// vim: syntax=cpp.doxygen
