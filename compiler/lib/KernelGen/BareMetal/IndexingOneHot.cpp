/**
 * \file
 * compiler/lib/KernelGen/BareMetal/IndexingOneHot.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "IndexingOneHot.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

bool IndexingOneHotKernel::IsAvailable(TContext* context) const {
    int nr_operand = context->getAttrInt("nr_operands");
    bool ok_operand = nr_operand == 3;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool ok_dtype = Utils::is_int_dtype(src_dtype, 32) ||
                    Utils::is_float_dtype(src_dtype, 32);
    return ok_dtype && ok_operand;
}
//! kernel gen
std::string IndexingOneHotKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_indexingonehot_" << context->getAttrInt("axis") << "_"
       << context->getAttrOprand("operand:0").dtype;
    return ss.str();
}

std::string IndexingOneHotKernel::GetKernelBody(TContext* context) const {
    std::stringstream axis_init_ss;
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    int axis = context->getAttrInt("axis");
    // clang-format off
    writer << StringTemplate::StringTemplateArgs(context).add("axis", axis).render(
     R"(
    float* src = (float*)inputs[0]->ptr;
    int* idx = (int*)inputs[1]->ptr;
    float* dst = (float*)outputs[0]->ptr;

    const Tensor* src_tensor = inputs[0];
    const Tensor* dst_tensor = outputs[0];
    const Layout src_layout = src_tensor->layout;
    const Layout dst_layout = dst_tensor->layout;

    int axis = ${axis};
    int batch = 1;
    int elems = 1;
    for (int i = 0; i < axis; ++i){
        batch *= src_layout.dims[i];
    }
    for (int i = axis + 1; i < src_layout.nr_dim; ++i){
        elems *= src_layout.dims[i];
    }
    int batch_stride = src_layout.dims[axis] * elems;
    for (int bid = 0; bid < batch; ++bid){
        float* src_ptr = src + bid * batch_stride;
        for(int id = 0; id < elems; ++id){
            *dst = src_ptr[*idx * elems + id];
            ++dst;
            ++idx;
        }
    }

    return TinyNN_SUCCESS;
    })"
    );
    // clang-format on
    return writer.str();
}

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen