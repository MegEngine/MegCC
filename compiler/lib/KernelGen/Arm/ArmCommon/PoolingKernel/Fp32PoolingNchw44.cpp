/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Fp32PoolingNchw44.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "Arm/ArmCommon/NeonIntrinCompat.h"
#include "Arm/ArmCommon/Pooling.h"
#include "Utils/StringTemplate.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool PoolingNchw44Fp32::IsAvailable(TContext* context) const {
    bool mode_ok = context->getAttrStr("format") == "NCHW44";
    bool dtype_ok = context->getAttrOprand("operand:0").dtype == "f32";
    return mode_ok && dtype_ok;
}

namespace {
struct Pooler {
    Pooler(std::string mode, uint32_t windows_cnt)
            : m_mode(mode), m_windows_cnt(windows_cnt){};
    std::string gen_dep() {
        if (m_mode == "MAX") {
            return R"(
static inline float32x4_t max(float32x4_t a, float32x4_t b){
    return vmaxq_f32(a, b);
})";
        } else {
            return "";
        }
    }
    std::string gen_init_str() {
        if (m_mode == "MAX") {
            return "float32x4_t res = vdupq_n_f32(-__FLT_MAX__);";
        } else if (m_mode == "AVERAGE") {
            return "float32x4_t sum = vdupq_n_f32(0.f);";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return R"(float32x4_t sum = vdupq_n_f32(0.f);
                      int count = 0;
            )";
        }
    }
    std::string gen_feed_str(std::string val) const {
        if (m_mode == "MAX") {
            return "res = max(res, " + val + ");";
        } else if (m_mode == "AVERAGE") {
            return "sum = vaddq_f32(sum, " + val + ");";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return "sum = vaddq_f32(sum, " + val + ");\n" + "++count;";
        }
    }
    std::string gen_final_str() {
        if (m_mode == "MAX") {
            return "";
        } else if (m_mode == "AVERAGE") {
            return "sum =  vdivq_f32(sum, vcvtq_f32_s32(vdupq_n_s32(" +
                   std::to_string(m_windows_cnt) + ")));";
        } else {
            CC_ASSERT(m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING")
                    << "not support pooling mode\n";
            return "sum =  vdivq_f32(sum, vcvtq_f32_s32(vdupq_n_s32(count)));";
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

std::string PoolingNchw44Fp32::GetKernelBody(TContext* context) const {
    auto format_str = context->getAttrStr("format");
    auto mode_str = context->getAttrStr("mode");
    std::stringstream ss;
    const uint32_t window_h = context->getAttrInt("window_h");
    const uint32_t window_w = context->getAttrInt("window_w");
    Pooler pooler(mode_str, window_h * window_w);
    ss << R"(
#include <arm_neon.h>
#include <stdbool.h>
)";
    ss << gen_neon_intrin_compat();
    ss << pooler.gen_dep();
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
    float* input_data = (float*)inputs[0]->ptr;
    TINYNN_ASSERT(input_data);
    float* output_data = (float*)outputs[0]->ptr;
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
        for (uint32_t oc_idx = 0; oc_idx < oc; oc_idx += 4) {
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
                                float32x4_t in_elem = vld1q_f32(input_data + batch_idx * ic * ih * iw + oc_idx * ih * iw + (ih_idx * iw + iw_idx) * 4);
                                ${gen_feed_str(in_elem)}
                            }
                        }
                    }
                    ${gen_final_str}
                    vst1q_f32(output_data + batch_idx * oc * oh * ow + oc_idx * oh * ow + (oh_idx * ow + ow_idx) * 4, ${get_result_str});
                }
            }
        }
    }
    return TinyNN_SUCCESS;
})";
    ss << StringTemplate::StringTemplateArgs(context)
                    .add("oc_ratio", 4)
                    .add("ic_ratio", 4)
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
                    .add("gen_feed_str", StringTemplate::object_bind(
                                                 &Pooler::gen_feed_str, pooler))
                    .add("gen_final_str", pooler.gen_final_str())
                    .add("get_result_str", pooler.get_result_str())
                    .render(body_temp);

    return ss.str();
}

// vim: syntax=cpp.doxygen
