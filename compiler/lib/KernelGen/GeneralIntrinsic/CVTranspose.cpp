/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/CVTranspose.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <sstream>

#include "CVTranspose.h"
#include "Transpose.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

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
    return GetCVKernelSymbol(context) + "(const TinyMat* src, const TinyMat* dst)";
}
std::vector<KernelObj> CvTransposeKernel::GetDependInternalSymbol(
        TContext* context) const {
    GeneralIntrinsic::CommonTransposeKernel common_tran;
    auto dtype_str = context->getAttrOprand("operand:0").dtype;
    std::shared_ptr<CodeGenContext> tran_ctx = std::make_shared<CodeGenContext>();
    int32_t type_size = Utils::get_dtype_size(dtype_str);
    tran_ctx->setAttr("type_size", CCAttr(type_size));
    auto ctx = std::static_pointer_cast<TContext>(tran_ctx).get();
    return {
            {common_tran.GetKernelSymbol(ctx), common_tran.GetKernelBody(ctx),
             common_tran.GetBodyGuardBegin(ctx), common_tran.GetBodyGuardEnd(ctx),
             common_tran.GetDependInternalSymbol(ctx)}};
}
std::string CvTransposeKernel::GetCVKernelBody(TContext* context) const {
    GeneralIntrinsic::CommonTransposeKernel common_tran;
    std::shared_ptr<TContext> tran_ctx = std::make_shared<CodeGenContext>();
    auto kernel_sig = GetCVKernelSignature(context);
    auto dtype_str = context->getAttrOprand("operand:0").dtype;
    int32_t type_size = Utils::get_dtype_size(dtype_str);
    tran_ctx->setAttr("type_size", CCAttr(type_size));
    std::string body_temp = R"(
        #include <string.h>
        #include "gi_int.h"
        #include "tinycv_c.h"

        ${gi_fast_transpose_impl}

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
            .add("gi_fast_transpose_impl",
                 common_tran.GetKernelSignature(tran_ctx.get()))
            .render(body_temp);
}

// vim: syntax=cpp.doxygen
