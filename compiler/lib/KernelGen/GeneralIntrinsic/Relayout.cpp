/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/Relayout.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>

#include "Common/Relayout.h"
#include "Relayout.h"
#include "Transpose.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
std::vector<KernelObj> RelayoutKernel::GetDependInternalSymbol(
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

bool RelayoutKernel::IsAvailable(TContext* context) const {
    auto dtype_str = context->getAttrOprand("operand:0").dtype;
    int32_t type_size = Utils::get_dtype_size(dtype_str);
    bool ok_dtype = (context->getAttrOprand("operand:0").dtype ==
                     context->getAttrOprand("operand:1").dtype) &&
                    (type_size == 1 || type_size == 2 || type_size == 4);
    std::vector<size_t> shape_in = context->getAttrOprand("operand:0").shape;
    std::vector<size_t> shape_out = context->getAttrOprand("operand:0").shape;
    bool ok_shape = shape_in.size() == shape_out.size();
    for (size_t i = 0; ok_shape && i < shape_in.size(); ++i) {
        ok_shape = ok_shape && shape_in.at(i) == shape_out.at(i);
    }
    return ok_dtype && ok_shape;
}

//! kernel gen
std::string RelayoutKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "GI_kernel_relayout_"
       << SymbolHelper::gen_valid_dtype(context->getAttrOprand("operand:0").dtype);
    return ss.str();
}

std::string RelayoutKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    auto src_dtype_str = context->getAttrOprand("operand:0").dtype;
    std::string specifier = Utils::cvt_dtype_specifier(src_dtype_str);
    ss << R"(
        #include <stdbool.h>
        #include <string.h>
    )";
    if (src_dtype_str == "f16") {
        ss << R"(
        #include "gi_float16.h"

    )";
    } else {
        ss << R"( 
        #include "gi_float.h"
        #include "gi_int.h"

    )";
    }
    GeneralIntrinsic::CommonTransposeKernel common_tran;
    std::shared_ptr<TContext> tran_ctx = std::make_shared<CodeGenContext>();
    int32_t type_size = Utils::get_dtype_size(src_dtype_str);
    tran_ctx->setAttr("type_size", CCAttr(type_size));
    ss << RelayoutHelper::GetLayoutHelper();
    ss << RelayoutHelper::GetTransposeModule(
            specifier, type_size, common_tran.GetKernelSignature(tran_ctx.get()));
    ss << RelayoutHelper::GetNonconMemcpyModule(specifier);

    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    const Tensor* src_tensor = inputs[0];
    const Tensor* dst_tensor = outputs[0];
    TINYNN_ASSERT(src_tensor->dtype.type_enum == dst_tensor->dtype.type_enum);

    ${specifier}* src_data = (${specifier}*)(src_tensor->ptr);
    ${specifier}* dst_data = (${specifier}*)(dst_tensor->ptr);
    TINYNN_ASSERT(src_data);
    TINYNN_ASSERT(dst_data);

    Layout src_layout = src_tensor->layout;
    Layout dst_layout = dst_tensor->layout;
    src_layout = collapse_contiguous(src_layout);
    dst_layout = collapse_contiguous(dst_layout);
    size_t nr_elem = 1;
    for (int i = 0; i < src_layout.nr_dim; ++i) {
        nr_elem *= src_layout.dims[i];
    }
    bool src_contig = is_contiguous(src_layout);
    bool dst_contig = is_contiguous(dst_layout);

    if (src_contig && dst_contig){ 
        memcpy(dst_data, src_data, nr_elem * sizeof(${specifier}));
    } else {
        //! compress layout to retain no contig shape
        //! try transpose opt
        ${do_transpose}
        //! try partial copy
        if(src_contig && copy_check(dst_layout)){
            memcpy_cont2nocont(dst_data, src_data, src_layout, dst_layout, postive_memcpy);
            return TinyNN_SUCCESS;
        }
        if(dst_contig && copy_check(src_layout)){
            memcpy_cont2nocont(src_data, dst_data, dst_layout, src_layout, reverse_memcpy);
            return TinyNN_SUCCESS;
        }

        //! fallback to naive
        NoconIter src_iter = init_iter(src_layout);
        NoconIter dst_iter = init_iter(dst_layout);
        for (size_t i = 0; i < nr_elem; ++i){
            dst_data[dst_iter.offset] = src_data[src_iter.offset];
            inc_iter(src_layout, &src_iter, src_layout.nr_dim-1);
            inc_iter(dst_layout, &dst_iter, dst_layout.nr_dim-1);
        }
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("specifier", specifier)
                    .add("do_transpose", RelayoutHelper::GetTransposeCall())
                    .render(body_temp);
    return ss.str();
}