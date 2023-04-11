#include <float.h>
#include <sstream>

#include "FormatHelper.h"
#include "Rotate.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool RotateKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "ui8";
    return dtype_ok;
}

//! kernel gen
std::string RotateKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_rotate_" << src_dtype;
    return ss.str();
}

std::string RotateKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, bool clockwise)";
}

std::string RotateKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include "tinycv_c.h"
        #define rep(i, n) for (int i = 0; i < (n); ++i)
        
        void ${kernel_sig}{
            uint8_t * sptr = src->data;
            uint8_t * dptr = dst->data;
            int IH = src->rows;
            int IW = src->cols;
            int C = src->channels;
            int OH = dst->rows;
            int OW = dst->cols;

            rep(ih, IH) rep(iw, IW) {
                int ow = clockwise ? IH - ih - 1 : ih;
                int oh = clockwise ? iw : IW - iw - 1;
                rep(c, C) {
                    dptr[oh * OW * C + ow * C + c] = sptr[ih * IW * C + iw * C + c];
                }
            }
            
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
