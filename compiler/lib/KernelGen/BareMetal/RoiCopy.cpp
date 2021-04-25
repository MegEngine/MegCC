
/**
 * \file
 * compiler/lib/KernelGen/BareMetal/RoiCopy.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "FormatHelper.h"
#include "RoiCopy.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool RoiCopyKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = src_dtype == "ui8";
    return dtype_ok;
}

//! kernel gen
std::string RoiCopyKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_roi_copy_" << src_dtype;
    return ss.str();
}

std::string RoiCopyKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, size_t row_from, size_t "
           "row_to, size_t col_from, size_t col_to)";
}

std::string RoiCopyKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include <string.h>
        #include "tinycv_c.h"

        void ${kernel_sig}{
            uint8_t * sptr = src->data;
            uint8_t * dptr = dst->data;
            int IH = src->rows;
            int IW = src->cols;

            int C = src->channels;
            int OH = dst->rows;
            int OW = dst->cols;
            
            for(int row_id = row_from; row_id < row_to; ++row_id){
                memcpy(dptr, sptr + row_id * IW * C + col_from * C, OW * C * sizeof(uint8_t));
                dptr += OW * C;
            }
        }
    )";

    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
