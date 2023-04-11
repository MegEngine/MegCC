#include <float.h>
#include <sstream>

#include "Flip.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool FlipKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    return dtype_ok;
}

//! kernel gen
std::string FlipKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_flip_" << src_dtype;
    return ss.str();
}

std::string FlipKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, bool vertical, bool "
           "horizontal)";
}

std::string FlipKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include <string.h>
        #include "tinycv_c.h"
        #define rep(i, n) for (int i = 0; i < (n); ++i)
        void ${kernel_sig}{
            uint8_t * src_base_ptr = src->data;
            uint8_t * dst_base_ptr = dst->data;
            size_t rows = src->rows;
            size_t cols = src->cols;
            size_t ch = src->channels;
            size_t src_step = cols * ch;
            size_t dst_step = cols * ch;
            for (size_t sr = 0; sr < rows; ++sr) {
                const uint8_t *sptr = src_base_ptr + sr * src_step;
                size_t dr = (vertical ? rows - sr - 1 : sr);
                uint8_t *dptr = dst_base_ptr + dr * dst_step;
                if (!horizontal) {
                    memcpy(dptr, sptr, sizeof(uint8_t) * cols * ch);
                } else {
                    size_t sc = 0;
                    size_t dc = cols * ch;
                    for (; sc + 8 * ch <= cols * ch; sc += 8 * ch, dc -= 8 * ch) {
                        rep(c, ch) dptr[dc - 1 * ch + c] = sptr[sc + 0 * ch + c];
                        rep(c, ch) dptr[dc - 2 * ch + c] = sptr[sc + 1 * ch + c];
                        rep(c, ch) dptr[dc - 3 * ch + c] = sptr[sc + 2 * ch + c];
                        rep(c, ch) dptr[dc - 4 * ch + c] = sptr[sc + 3 * ch + c];
                        rep(c, ch) dptr[dc - 5 * ch + c] = sptr[sc + 4 * ch + c];
                        rep(c, ch) dptr[dc - 6 * ch + c] = sptr[sc + 5 * ch + c];
                        rep(c, ch) dptr[dc - 7 * ch + c] = sptr[sc + 6 * ch + c];
                        rep(c, ch) dptr[dc - 8 * ch + c] = sptr[sc + 7 * ch + c];
                    }
                    for (; sc < cols * ch; sc += ch, dc -= ch) {
                        rep(c, ch) dptr[dc - ch + c] = sptr[sc + c];
                    }
                }
            }
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
