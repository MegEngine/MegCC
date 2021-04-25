
/**
 * \file
 * compiler/lib/KernelGen/BareMetal/CVTranspose.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "CVTranspose.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool CvTransposeKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    return dtype_ok;
}

//! kernel gen
std::string CvTransposeKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    ss << "tinycv_transpose_" << src_dtype;
    return ss.str();
}

std::string CvTransposeKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst)";
}

std::string CvTransposeKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string body_temp = R"(
        #include <string.h>
        #include "tinycv_c.h"
        void ${kernel_sig}{
            uint8_t * src_base_ptr = src->data;
            uint8_t * dst_base_ptr = dst->data;
            int src_rows = src->rows;
            int src_cols = src->cols;
            int src_chans = src->channels;
            for(int row_id = 0; row_id < src_rows; ++row_id)
                for(int col_id = 0; col_id < src_cols; ++col_id){
                    uint8_t * src_ptr = src_base_ptr + row_id * src_cols * src_chans + col_id * src_chans;
                    uint8_t * dst_ptr = dst_base_ptr + col_id * src_rows * src_chans + row_id * src_chans;
                    for(int channel_id = 0 ; channel_id < src_chans; ++channel_id){
                        dst_ptr[channel_id] = src_ptr[channel_id];
                    }
                }
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
