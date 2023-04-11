#include "WarpAffine.h"
#include "../Utils/StringTemplate.h"
#include "../Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool WarpAffineKernel::IsAvailable(TContext* ctx) const {
    auto nr_operands = ctx->getAttrInt("nr_operands");
    auto src_layout = ctx->getAttrOprand("operand:0");
    auto mat_layout = ctx->getAttrOprand("operand:1");
    auto dst_layout = ctx->getAttrOprand("operand:2");

    CC_ASSERT(nr_operands == 3);

    bool dtype_valid = ((src_layout.dtype == "f32" && dst_layout.dtype == "f32") ||
                        (src_layout.dtype == "ui8" && dst_layout.dtype == "ui8")) &&
                       mat_layout.dtype == "f32";
    return dtype_valid;
}
//! kernel gen
std::string WarpAffineKernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    auto border_val_str = std::to_string(ctx->getAttrFloat("border_val"));
    border_val_str[border_val_str.find('.')] = '_';
    ss << "kernel_warpaffine";
    ss << "_" << ctx->getAttrStr("format");
    ss << "_" << ctx->getAttrStr("imode");
    ss << "_" << ctx->getAttrStr("border_mode");
    ss << "_" << ctx->getAttrOprand("operand:0").dtype;
    ss << "_" << border_val_str;
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
        const std::string& bmode, float border_val, const std::string& dtype_c_str,
        const std::string& dtype) {
    std::stringstream temp_body;
    if (bmode != "CONSTANT") {
        temp_body << R"(
            static inline float visit_src(const ${dtype_c_str}* sptr,int c, int h, int w, size_t sstrd[3], ${dtype_c_str} bval){
                return sptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w];
            }
        )";
    } else {
        temp_body << R"(
            static inline float visit_src(const ${dtype_c_str}* sptr,int c, int h, int w, size_t sstrd[3], ${dtype_c_str} bval){
                if (h != -1 && w != -1){
                    return sptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w];
                }else{
                    return bval;
                }
            }
        )";
    }
    if (dtype == "ui8") {
        temp_body << R"(
            static inline uint8_t saturate_cast_ui8(int x){
                return (uint8_t)((unsigned)x <= UCHAR_MAX ? x
                                                        : x > 0 ? UCHAR_MAX : 0);
            }
            static inline uint8_t output_cvt(float val){
                int ival = (int)roundf(val);
                return saturate_cast_ui8(val);
            }
        )";
    } else {
        CC_ASSERT(dtype == "f32");
        temp_body << R"(
            static inline ${dtype_c_str} output_cvt(float val){
                return val;
            }
        )";
    }
    temp_body << R"(
        static inline void visit_dst(${dtype_c_str}* dptr,int c, int h, int w, size_t sstrd[3], float val){
            dptr[sstrd[0] * c + sstrd[1] * h + sstrd[2] * w] = output_cvt(val);
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("border_val", std::to_string(border_val))
            .add("dtype_c_str", dtype_c_str)
            .render(temp_body.str());
}

std::string to_lower_case(std::string data) {
    for (auto& c : data) {
        c = tolower(c);
    }
    return data;
}
}  // namespace
std::string WarpAffineKernel::GetKernelBody(TContext* ctx) const {
    auto format = ctx->getAttrStr("format");
    auto imode = ctx->getAttrStr("imode");
    auto bmode = ctx->getAttrStr("border_mode");
    auto input = ctx->getAttrOprand("operand:0");
    float border_val = ctx->getAttrFloat("border_val");
    auto dtype_str = ctx->getAttrOprand("operand:0").dtype;
    std::string dtype_c_str = Utils::cvt_dtype_specifier(dtype_str);
    uint32_t spatial_start = 2;
    uint32_t batch_pos = 0;
    uint32_t channel_pos = 1;
    std::stringstream ss;
    ss << R"(
        #include <limits.h>
        #include <math.h>
        #define rep(i, n) for (int i = 0; i < (n); ++i)
    )";
    ss << gen_get_real_coord(bmode);
    ss << gen_visit(bmode, border_val, dtype_c_str, dtype_str);

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
        ${dtype_c_str}* dst_ptr = dst_tensor->ptr;
        TINYNN_ASSERT(dst_ptr);
        const int* mid_ptr = NULL;
        if (nr_input > 2){
            mid_ptr = inputs[2]->ptr;
            TINYNN_ASSERT(mid_ptr);
        }
        const ${dtype_c_str} bval = ${border_val};
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
            const float* mptr = weight_ptr + batch_idx * 2 * 3;
            const ${dtype_c_str}* batch_src_ptr = src_ptr + batch_idx * in_batch_stride;
            if (nr_input > 2){
                batch_src_ptr = src_ptr + mid_ptr[batch_idx] * in_batch_stride;
            }
            rep(oh_idx, oh)
            rep(ow_idx, ow){
                float alphaw = mptr[0] * ow_idx + mptr[1] * oh_idx + mptr[2];
                float alphah = mptr[3] * ow_idx + mptr[4] * oh_idx + mptr[5];

                int iw0 = get_real_coord(floorf(alphaw) + 0, iw);
                int iw1 = get_real_coord(floorf(alphaw) + 1, iw);
                int ih0 = get_real_coord(floorf(alphah) + 0, ih);
                int ih1 = get_real_coord(floorf(alphah) + 1, ih);

                alphaw -= floorf(alphaw);
                alphah -= floorf(alphah);
                float alphaw_p = 1.0f - alphaw;
                float alphah_p = 1.0f - alphah;
                rep(ic_idx, ic){
                    float val = visit_src(batch_src_ptr, ic_idx, ih0, iw0, sstrd, bval) * alphaw_p * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih0, iw1, sstrd, bval) * alphaw * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw0, sstrd, bval) * alphaw_p * alphah +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw1, sstrd, bval) * alphaw * alphah;
                    visit_dst(dst_ptr, ic_idx, oh_idx, ow_idx, dstrd, val);
                }
            }
            dst_ptr += out_batch_stride;
        }
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("stride_str", stride_str)
                    .add("dtype_c_str", dtype_c_str)
                    .add("border_val", std::to_string(border_val))
                    .render(temp_body);
    return ss.str();
}

bool WarpAffineKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    bool mode_ok = context->getAttrStr("format") == "NHWC" &&
                   context->getAttrStr("imode") == "LINEAR";
    return dtype_ok && mode_ok;
}

std::string WarpAffineKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto bmode = context->getAttrStr("border_mode");
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_warp_affine_" << to_lower_case(bmode) << "_linear_" << src_dtype;
    return ss.str();
}

std::string WarpAffineKernel::GetCVKernelSignature(TContext* context) const {
    auto bmode = context->getAttrStr("border_mode");
    bool is_const_bmode = bmode == "CONSTANT";
    std::string const_bmode_arg = is_const_bmode ? ", uint8_t const_board_val" : "";
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, const double* trans" +
           const_bmode_arg + ")";
}

std::string WarpAffineKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    auto bmode = context->getAttrStr("border_mode");
    bool is_const_bmode = bmode == "CONSTANT";
    auto input = context->getAttrOprand("operand:0");
    float border_val = context->getAttrFloat("border_val");
    auto dtype_str = context->getAttrOprand("operand:0").dtype;
    std::string dtype_c_str = Utils::cvt_dtype_specifier(dtype_str);
    std::string border_val_str = is_const_bmode ? "const_board_val" : "0";
    std::stringstream ss;
    ss << R"(
        #include <limits.h>
        #include <math.h>
        #include "tinycv_c.h"
        #define rep(i, n) for (int i = 0; i < (n); ++i)
    )";
    ss << gen_get_real_coord(bmode);
    ss << gen_visit(bmode, border_val, dtype_c_str, dtype_str);
    std::string body_temp = R"(
        void ${kernel_sig}{
            uint8_t * batch_src_ptr = src->data;
            uint8_t * dst_ptr = dst->data;
            const int ih = src->rows;
            const int iw = src->cols;
            
            const int ic = src->channels;
            const int oh = dst->rows;
            const int ow = dst->cols;
            float mptr[6];
            for(int i = 0; i < 6; ++i){
                mptr[i] = trans[i];
            }
            size_t sstrd[3] = {1, iw * ic, ic};
            size_t dstrd[3] = {1, ow * ic, ic};
            const uint8_t bval = ${border_val};

            rep(oh_idx, oh)
            rep(ow_idx, ow){
                float alphaw = mptr[0] * ow_idx + mptr[1] * oh_idx + mptr[2];
                float alphah = mptr[3] * ow_idx + mptr[4] * oh_idx + mptr[5];

                int iw0 = get_real_coord(floorf(alphaw) + 0, iw);
                int iw1 = get_real_coord(floorf(alphaw) + 1, iw);
                int ih0 = get_real_coord(floorf(alphah) + 0, ih);
                int ih1 = get_real_coord(floorf(alphah) + 1, ih);

                alphaw -= floorf(alphaw);
                alphah -= floorf(alphah);
                float alphaw_p = 1.0f - alphaw;
                float alphah_p = 1.0f - alphah;
                rep(ic_idx, ic){
                    float val = visit_src(batch_src_ptr, ic_idx, ih0, iw0, sstrd, bval) * alphaw_p * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih0, iw1, sstrd, bval) * alphaw * alphah_p +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw0, sstrd, bval) * alphaw_p * alphah +
                                visit_src(batch_src_ptr, ic_idx, ih1, iw1, sstrd, bval) * alphaw * alphah;
                    visit_dst(dst_ptr, ic_idx, oh_idx, ow_idx, dstrd, val);
                }
            }
            
        }
    )";

    ss << StringTemplate::StringTemplateArgs()
                    .add("kernel_sig", kernel_sig)
                    .add("border_val", border_val_str)
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen
