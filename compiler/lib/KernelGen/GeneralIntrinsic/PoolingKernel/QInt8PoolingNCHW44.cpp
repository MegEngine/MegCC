/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/PoolingKernel/QInt8PoolingNCHW44.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <stdint.h>
#include <sstream>

#include "Pooling.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool PoolingNchw44QInt8::IsAvailable(TContext* context) const {
    bool format_ok = context->getAttrStr("format") == "NCHW44";
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto dst_dtype = context->getAttrOprand("operand:1").dtype;
    bool dtype_ok =
            (src_dtype == dst_dtype) && Utils::is_quant_dtype(src_dtype, 8);
    if (Utils::is_quant_dtype(src_dtype, 8)) {
        CC_ASSERT(context->getAttrOprand("operand:0").scale ==
                  context->getAttrOprand("operand:1").scale)
                << "quant pooling only support same scale\n";
    }
    return format_ok && dtype_ok;
}

std::string PoolingNchw44QInt8::GetKernelSymbol(TContext* context) const {
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
            return R"(
                GI_INT32_t ans;
                GI_INT8_t max = GiBroadcastInt8(INT8_MIN);
            )";
        } else if (m_mode == "AVERAGE" ||
                   m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING") {
            std::string str = R"(
                GI_INT32_t ans;
                GI_INT32_t sum0 = GiBroadcastInt32(0);
            )";
            if (m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING") {
                str += "uint32_t count = 0;\n";
            };
            return str;
        } else {
            CC_ASSERT(false) << "not support pooling mode\n";
        }
        return "";
    }

    std::string gen_feed_str() {
        if (m_mode == "MAX") {
            return R"(
                int32_t* in_ptr = (int32_t*)input_ptr;
                GI_INT32_t in_elem =  GiBroadcastInt32(*(in_ptr+ih_idx*iw+iw_idx));
                max =GiMaximumInt8(max,GiReinterInt32ToInt8(in_elem));
            )";
        } else if (m_mode == "AVERAGE" ||
                   m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING") {
            std::string str = R"(
                int32_t* in_ptr = (int32_t*)input_ptr;
                GI_INT32_t in_elem = GiBroadcastInt32(*(in_ptr+ih_idx*iw+iw_idx)); 
                GI_INT8_t src0 = GiReinterInt32ToInt8(in_elem);
                GI_INT32_t src1 = GiMoveLowLongInt16(GiMoveLowLongInt8(src0));
                sum0 = GiAddInt32(sum0, src1);
            )";
            if (m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING") {
                str += "++count;\n";
            }
            return str;
        } else {
            CC_ASSERT(false) << "not support pooling mode\n";
        }
        return "";
    }

    std::string gen_final_str() {
        if (m_mode == "MAX") {
            return R"(
                 ans = GiReinterpretInt8AsInt32(max);
            )";
        } else if (m_mode == "AVERAGE" ||
                   m_mode == "AVERAGE_COUNT_EXCLUDE_PADDING") {
            std::string str = "";
            if (m_mode == "AVERAGE") {
                str += "uint32_t window_count = " +
                       std::to_string(m_windows_cnt) + ";\n";
            } else {
                str += R"(
                    uint32_t window_count = count;
                )";
            }
            str += R"(
                float count_div = 1.0/window_count;
                GI_INT8_t ans0 = GiCvtFromFloat32ToInt8(GiMultiplyScalerFloat32(GiCastToFloat32(sum0), count_div));
                ans = GiReinterpretInt8AsInt32(ans0);
            )";
            return str;

        } else {
            CC_ASSERT(false) << "not support pooling mode\n";
        }
        return "";
    }

    std::string m_mode;
    uint32_t m_windows_cnt;
};
}  // namespace

std::string PoolingNchw44QInt8::GetKernelBody(TContext* context) const {
    auto format_str = context->getAttrStr("format");
    auto mode_str = context->getAttrStr("mode");
    std::stringstream ss;
    const uint32_t window_h = context->getAttrInt("window_h");
    const uint32_t window_w = context->getAttrInt("window_w");
    Pooler pooler(mode_str, window_h * window_w);
    ss << R"(
        #include <stdbool.h>
        #include <stdint.h>
        #include "gi_float.h"
        #include "gi_int.h"
    )";
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
    int8_t* input_data = (int8_t*)inputs[0]->ptr;
    TINYNN_ASSERT(input_data);
    int8_t* workspace_ptr = (int8_t*)workspace->ptr;
    TINYNN_ASSERT(workspace_ptr);
    int8_t* output_data = (int8_t*)outputs[0]->ptr;
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

    for(uint32_t n_idx=0; n_idx < batch; ++n_idx){
        for(uint32_t c_idx = 0; c_idx < oc; c_idx+=4){
            int8_t* input_ptr =input_data+n_idx*ic*ih*iw + c_idx*ih*iw;
            int8_t* output_ptr = output_data+n_idx*oc*oh*ow + c_idx*oh*ow;
            for(uint32_t oh_idx = 0;oh_idx < oh; ++oh_idx){
                for(uint32_t ow_idx = 0;ow_idx < ow; ++ow_idx){
                    ${gen_init_str}
                    for(uint32_t fh_idx=0; fh_idx < window_h; ++fh_idx){
                        const uint32_t ih_idx = oh_idx*sh + fh_idx-ph;
                        if(ih_idx>=ih)
                            continue;
                        for(uint32_t fw_idx = 0;fw_idx < window_w; ++fw_idx){
                            const uint32_t iw_idx = ow_idx*sw + fw_idx-pw;
                            if(iw_idx<iw){
                                ${gen_feed_str}
                            }
                        }
                    }

                    ${gen_final_str}

                    int32_t* out_ptr = (int32_t*)output_ptr;
                    GiStoreLane0Int32(out_ptr+oh_idx*ow+ow_idx, ans);
                     
                }
            }
        }
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs(context)
                    .add("mode_str", mode_str)
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
                    .add("gen_init_str", pooler.gen_init_str())
                    .add("gen_feed_str", pooler.gen_feed_str())
                    .add("gen_final_str", pooler.gen_final_str())
                    .render(body_temp);

    return ss.str();
}
