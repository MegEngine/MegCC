/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/CVTranspose.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "../Arm64/Transpose.h"
#include "../Armv7/Transpose.h"
#include "CVTranspose.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

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
        #include <arm_neon.h>
        #include <string.h>
        #include "tinycv_c.h"


#if defined(__aarch64__)
${a64_fast_transpose_impl}
#else
${a32_fast_transpose_impl}
#endif

        void ${kernel_sig}{
            uint8_t * src_base_ptr = src->data;
            uint8_t * dst_base_ptr = dst->data;
            int src_cols = src->cols;
            int src_rows = src->rows;
            int src_chans = src->channels;
            fast_transpose_impl_8(src_base_ptr, dst_base_ptr, src_cols, src_rows, src_chans, src_cols * src_chans, src_rows * src_chans);
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("kernel_sig", kernel_sig)
            .add("a64_fast_transpose_impl", Arm64::gen_transpose(1))
            .add("a32_fast_transpose_impl", Armv7::gen_transpose(1))
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
