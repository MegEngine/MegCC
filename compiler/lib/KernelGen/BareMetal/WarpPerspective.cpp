/**
 * \file
 * compiler/lib/KernelGen/BareMetal/WarpPerspective.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "WarpPerspective.h"
#include "../Utils/StringTemplate.h"
#include "../Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool WarpPerspectiveKernel::IsAvailable(TContext* ctx) const {
    auto nr_operands = ctx->getAttrInt("nr_operands");
    auto src_layout = ctx->getAttrOprand("operand:0");
    auto mat_layout = ctx->getAttrOprand("operand:1");
    auto dst_layout = ctx->getAttrOprand("operand:2");
    if (nr_operands == 4) {
        dst_layout = ctx->getAttrOprand("operand:3");
    } else {
        CC_ASSERT(nr_operands == 3);
    }
    bool dtype_valid = (src_layout.dtype == "f32" || src_layout.dtype == "ui8") &&
                       mat_layout.dtype == "f32" &&
                       dst_layout.dtype == src_layout.dtype;
    bool shape_valid =
            (nr_operands == 3 && src_layout.shape[0] == mat_layout.shape[0] &&
             src_layout.shape[0] == dst_layout.shape[0]) ||
            (nr_operands == 4);
    return dtype_valid && shape_valid;
}
//! kernel gen
std::string WarpPerspectiveKernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    auto border_val_str = std::to_string(ctx->getAttrFloat("border_val"));
    bool is_dynamic = Utils::is_any_op_dynamic(ctx);
    auto bmode = ctx->getAttrStr("bmode");
    border_val_str[border_val_str.find('.')] = '_';
    ss << "kernel_warpperspective";
    ss << "_" << ctx->getAttrStr("format");
    ss << "_" << ctx->getAttrStr("imode");
    ss << "_" << ctx->getAttrStr("bmode");
    ss << "_" << ctx->getAttrOprand("operand:0").dtype;
    if (bmode != "CONSTANT") {
        ss << "_" << border_val_str;
    }
    if (is_dynamic) {
        ss << "_DYNAMIC";
    }
    return ss.str();
}
namespace {

std::string gen_get_real_coord(const std::string& bmode) {
    std::string body_temp = R"(
            static inline int get_real_coord(int p, int len){
                if ((unsigned)p >= (unsigned)len){
                    ${core_temp}
                }
                return p;
            }
        )";
    std::string core_temp;
    if (bmode == "REFLECT") {
        core_temp = R"(
            if (len == 1)
                return 0;
            do {
                if (p < 0)
                    p = -p - 1;
                else
                    p = len - 1 - (p - len);
            } while ((unsigned)p >= (unsigned)len);
        )";
    } else if (bmode == "REFLECT_101") {
        core_temp = R"(
            if (len == 1)
                return 0;
            do {
                if (p < 0)
                    p = -p - 1 + 1;
                else
                    p = len - 1 - (p - len) - 1;
            } while ((unsigned)p >= (unsigned)len);
        )";
    } else if (bmode == "REPLICATE") {
        core_temp = R"(
            p = p < 0 ? 0 : len - 1;
        )";
    } else if (bmode == "CONSTANT") {
        core_temp = R"(
            p = -1;
        )";
    } else if (bmode == "WRAP") {
        core_temp = R"(
            if (p < 0)
                p -= ((p - len + 1) / len) * len;
            
            while (p >= len) {
                p -= len;
            }
        )";
    } else {
        CC_ABORT << "no support bmode " << bmode << "\n";
    }
    return StringTemplate::StringTemplateArgs()
            .add("core_temp", core_temp)
            .render(body_temp);
}

std::string gen_visit(
        const std::string& bmode, float border_val, const std::string& dtype_c_str) {
    std::string temp_body;
    if (bmode != "CONSTANT") {
        temp_body = R"(
            static inline float visit_src(const ${dtype_c_str}* sptr,int c, int h, int w, size_t sstrd[3]){
                return sptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w];
            }
        )";
    } else {
        temp_body = R"(
            static inline float visit_src(const ${dtype_c_str}* sptr,int c, int h, int w, size_t sstrd[3]){
                if (h != -1 && w != -1){
                    return sptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w];
                }else{
                    return ${border_val};
                }
            }
        )";
    }
    return StringTemplate::StringTemplateArgs()
            .add("border_val", std::to_string(border_val))
            .add("dtype_c_str", dtype_c_str)
            .render(temp_body);
}

}  // namespace

std::string WarpPerspectiveKernel::GetKernelBody(TContext* ctx) const {
    auto format = ctx->getAttrStr("format");
    auto imode = ctx->getAttrStr("imode");
    auto bmode = ctx->getAttrStr("bmode");
    auto input = ctx->getAttrOprand("operand:0");
    float border_val = ctx->getAttrFloat("border_val");
    auto dtype_str = ctx->getAttrOprand("operand:0").dtype;
    auto dst_dtype = Utils::get_last_operand(ctx).dtype;
    std::string src_ctype = Utils::cvt_dtype_specifier(dtype_str);
    std::string dst_ctype = Utils::cvt_dtype_specifier(dst_dtype);
    uint32_t spatial_start = 2;
    uint32_t batch_pos = 0;
    uint32_t channel_pos = 1;
    bool is_dynamic = Utils::is_any_op_dynamic(ctx);
    uint32_t mid_idx = is_dynamic ? 3 : 2;

    std::stringstream ss;
    ss << R"(
        #include <math.h>
        #define rep(i, n) for (int i = 0; i < (n); ++i)
    )";
    ss << gen_get_real_coord(bmode);
    ss << gen_visit(bmode, border_val, src_ctype);
    std::string visit_dst_temp = R"(
        static inline void visit_dst(${dst_ctype}* dptr,int c, int h, int w, size_t sstrd[3], float val){
            dptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w] = ${round_func}(val);
        }
    )";
    ss << StringTemplate::StringTemplateArgs()
                    .add("dst_ctype", dst_ctype)
                    .add("round_func", Utils::is_float_dtype(dst_dtype) ? "" : "roundf")
                    .render(visit_dst_temp);
    std::string stride_str = R"(
        size_t sstrd[3] = {ih * iw, iw, 1};
        size_t dstrd[3] = {oh * ow, ow, 1};
    )";
    if (format == "NHWC") {
        spatial_start = 1;
        channel_pos = 3;
        stride_str = R"(
        size_t sstrd[3] = {1, iw * ic, ic};
        size_t dstrd[3] = {1, ow * ic, ic};
    )";
    } else {
        CC_ASSERT(format == "NCHW");
    }
    ss << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    ss << "const uint32_t spatial_start = " << spatial_start << ";\n";
    ss << "const uint32_t batch_pos = " << batch_pos << ";\n";
    ss << "const uint32_t channel_pos = " << channel_pos << ";\n";
    std::string temp_body = R"(
        const Tensor* src_tensor = inputs[0];
        TINYNN_ASSERT(src_tensor);
        const Tensor* weight_tensor = inputs[1];
        TINYNN_ASSERT(weight_tensor);
        const Tensor* dst_tensor = outputs[0];
        TINYNN_ASSERT(dst_tensor);

        const ${dtype_c_str}* src_ptr = src_tensor->ptr;
        TINYNN_ASSERT(src_ptr);
        const float* weight_ptr = weight_tensor->ptr;
        TINYNN_ASSERT(weight_ptr);
        ${output_dtype_c_str}* dst_ptr = dst_tensor->ptr;
        TINYNN_ASSERT(dst_ptr);
        const int* mid_ptr = NULL;
        if (nr_input > ${mid_idx}){
            mid_ptr = inputs[${mid_idx}]->ptr;
            TINYNN_ASSERT(mid_ptr);
        }

        const Layout src_layout = src_tensor->layout;
        const Layout dst_layout = dst_tensor->layout;

        const int batch = dst_layout.dims[batch_pos];
        const int ic = src_layout.dims[channel_pos];
        const int ih = src_layout.dims[spatial_start];
        const int iw = src_layout.dims[spatial_start + 1];
        
        const int oh = dst_layout.dims[spatial_start];
        const int ow = dst_layout.dims[spatial_start + 1];
        
        ${stride_str}

        const size_t in_batch_stride = (size_t)ic * ih * iw;
        const size_t out_batch_stride = (size_t)ic * oh * ow;

        rep(batch_idx, batch){
            const float* mptr = weight_ptr + batch_idx * 3 * 3;
            const ${dtype_c_str}* batch_src_ptr = src_ptr + batch_idx * in_batch_stride;
            if (nr_input > ${mid_idx}){
                batch_src_ptr = src_ptr + mid_ptr[batch_idx] * in_batch_stride;
            }
            rep(oh_idx, oh)
            rep(ow_idx, ow){
                float numeratorw = mptr[0] * ow_idx + mptr[1] * oh_idx + mptr[2];
                float numeratorh = mptr[3] * ow_idx + mptr[4] * oh_idx + mptr[5];
                float denominator = mptr[6] * ow_idx + mptr[7] * oh_idx + mptr[8];
                float alphaw = numeratorw / denominator;
                float alphah = numeratorh / denominator;
                int iw0 = get_real_coord(floor(alphaw) + 0, iw);
                int iw1 = get_real_coord(floor(alphaw) + 1, iw);
                int ih0 = get_real_coord(floor(alphah) + 0, ih);
                int ih1 = get_real_coord(floor(alphah) + 1, ih);

                alphaw -= floor(alphaw);
                alphah -= floor(alphah);
                float alphaw_p = 1.0f - alphaw;
                float alphah_p = 1.0f - alphah;
                rep(ic_idx, ic){
                    float val = visit_src(batch_src_ptr, ic_idx, ih0, iw0, sstrd) * alphaw_p * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih0, iw1, sstrd) * alphaw * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw0, sstrd) * alphaw_p * alphah +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw1, sstrd) * alphaw * alphah;
                    visit_dst(dst_ptr, ic_idx, oh_idx, ow_idx, dstrd, val);
                }
            }
            dst_ptr += out_batch_stride;
        }
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("stride_str", stride_str)
                    .add("dtype_c_str", src_ctype)
                    .add("output_dtype_c_str", dst_ctype)
                    .add("mid_idx", mid_idx)
                    .render(temp_body);
    return ss.str();
}

// vim: syntax=cpp.doxygen
