#include <float.h>
#include <sstream>

#include "Common/Resize.h"
#include "FormatHelper.h"
#include "Resize.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool ResizeKernel::IsAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "f32";
    bool mode_ok = context->getAttrStr("imode") == "LINEAR";

    bool format_ok = context->getAttrStr("format") == "NCHW" ||
                     context->getAttrStr("format") == "NCHW44";
    return dtype_ok && mode_ok && format_ok;
}
//! kernel gen
std::string ResizeKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto fmt = context->getAttrStr("format");
    auto imode = context->getAttrStr("imode");
    ss << "kernel_resize_linear_" << fmt << "_" << imode << "_" << src_dtype;
    return ss.str();
}

std::string ResizeKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    auto imode = context->getAttrStr("imode");
    auto fmt = context->getAttrStr("format");
    auto specifier = Utils::cvt_dtype_specifier(src_dtype);
    ss << R"(
        #include <stdalign.h>
    )";
    auto coord_str = ResizeHelper::GenCoordHelper(imode, specifier);
    auto gen_layout_dims = ResizeHelper::GenLayoutDims(fmt);
    auto get_offset = ResizeHelper::GenGetOffset(fmt);
    ss << StringTemplate::StringTemplateArgs()
                    .add("coord_helper_str", coord_str)
                    .add("get_offset", get_offset)
                    .render(R"(
        static inline float output_converter(float x){
            return x;
        }
        ${coord_helper_str}
        ${get_offset}
        #define rep(i, n) for (int i = 0; i < (n); ++i)
    )");
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
        const Tensor* src_tensor = inputs[0];
        const Tensor* dst_tensor = outputs[0];
        ${specifier}* sptr = (${specifier}*)(src_tensor->ptr);
        ${specifier}* dptr = (${specifier}*)(dst_tensor->ptr);
        TINYNN_ASSERT(sptr);
        TINYNN_ASSERT(dptr);
        
        const Layout src_layout = src_tensor->layout;
        const Layout dst_layout = dst_tensor->layout;
        ${gen_layout_dims}
        float scale_h = (float)(OH) / IH;
        float scale_w = (float)(OW) / IW;
        
        ${normal_impl}

        return TinyNN_SUCCESS;
    })";
    auto normal_impl = ResizeHelper::GenNormImpl(fmt);
    ss << StringTemplate::StringTemplateArgs()
                    .add("specifier", specifier)
                    .add("normal_impl", normal_impl)
                    .add("gen_layout_dims", gen_layout_dims)
                    .render(body_temp);
    return ss.str();
}

bool ResizeKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "ui8";
    bool mode_ok = context->getAttrStr("imode") == "LINEAR" &&
                   context->getAttrStr("format") == "NHWC";
    return dtype_ok && mode_ok;
}

//! kernel gen
std::string ResizeKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_resize_linear_" << src_dtype;
    return ss.str();
}

std::string ResizeKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) + "(const TinyMat* src, const TinyMat* dst)";
}

std::string ResizeKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include <math.h>
        #include <string.h>
        #include "tinycv_c.h"
        #define rep(i, n) for (int i = 0; i < (n); ++i)
        static inline uint8_t output_converter(float x){
            x = fmin(255.0f, fmax(0.0f, x));
            return (uint8_t) roundf(x);
        }
        static inline void get_nearest_linear_coord(float scale, int size, int idx, float* ah0, int* ih0, float* ah1, int* ih1){
            if (size == 1) {
                *ah0 = 1.f;
                *ih0 = 0;
                *ah1 = 0.f;
                *ih1 = 0;
            }

            float alpha = (idx + 0.5f) / scale - 0.5f;
            int origin_idx = (int)(floorf(alpha));
            alpha -= origin_idx;

            if (origin_idx < 0) {
                origin_idx = 0;
                alpha = 0;
            } else if (origin_idx + 1 >= size) {
                origin_idx = size - 2;
                alpha = 1;
            }

            *ah0 = 1 - alpha;
            *ih0 = origin_idx;
            *ah1 = alpha;
            *ih1 = origin_idx + 1;
        }
        void ${kernel_sig}{
            uint8_t * sptr = src->data;
            uint8_t * dptr = dst->data;
            int IH = src->rows;
            int IW = src->cols;
            int C = src->channels;
            int OH = dst->rows;
            int OW = dst->cols;

            float scale_h = (float)(OH) / IH;
            float scale_w = (float)(OW) / IW;

            rep(oh, OH) rep(ow, OW) {
                int ih0, ih1, iw0, iw1;
                float ah0, ah1, aw0, aw1;

                get_nearest_linear_coord(scale_h, IH, oh, &ah0, &ih0, &ah1, &ih1);
                
                get_nearest_linear_coord(scale_w, IW, ow, &aw0, &iw0, &aw1, &iw1);

                rep(c, C) {
                    dptr[(oh * OW + ow) * C + c] = output_converter(
                            sptr[(ih0 * IW + iw0) * C + c] * ah0 * aw0 +
                            sptr[(ih0 * IW + iw1) * C + c] * ah0 * aw1 +
                            sptr[(ih1 * IW + iw0) * C + c] * ah1 * aw0 +
                            sptr[(ih1 * IW + iw1) * C + c] * ah1 * aw1);
                }
            }
            
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
