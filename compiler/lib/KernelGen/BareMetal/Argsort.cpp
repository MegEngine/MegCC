/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Argsort.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Argsort.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

bool ArgSortKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32";
    return ok_dtype;
}
//! kernel gen
std::string ArgSortKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_argsort_" << context->getAttrOprand("operand:0").dtype << "_"
       << context->getAttrStr("order");
    return ss.str();
}

std::string ArgSortKernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    bool ascend = context->getAttrStr("order") == "ASCENDING";
    std::string compare_sign = ascend ? "<" : ">";
    writer << "#include <string.h>\n";
    writer << StringTemplate::StringTemplateArgs(context)
                      .add("compare_sign", compare_sign)
                      .render(R"(
      static inline void swap(float* val, int a, int b){
        float temp = val[a];
        val[a] = val[b];
        val[b] = temp;
      }
      static inline void swap_int(int* val, int a, int b){
        float temp = val[a];
        val[a] = val[b];
        val[b] = temp;
      }
      static void q_sort(float* val, int* idx, int left, int right){
        if (left >= right - 1)
          return;
        int select_idx = right - 1;
        float select_val = val[select_idx];
        int last = left - 1;
        for(int i = left; i < right; ++i){
          if(val[i] ${compare_sign}= select_val){
            ++last;
            swap(val, i, last);
            swap_int(idx, i, last);
          }
        }
        q_sort(val, idx, 0, last);
        q_sort(val, idx, last + 1, right);
      }
    )");
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    // clang-format off
    writer << R"(
    const float* src_data = (float*)inputs[0]->ptr;
    Layout src_layout = inputs[0]->layout;
    float* val_data = (float*)outputs[0]->ptr;
    int* idx_data = (int*)outputs[1]->ptr;
    
    int batch = src_layout.dims[0];
    int vec_len = src_layout.dims[1];
    memcpy(val_data, src_data, sizeof(float) * batch * vec_len);

    for(int batch_id = 0; batch_id < batch; ++batch_id){
      float* out_data = val_data + batch_id * vec_len;
      int* out_idx = idx_data + batch_id * vec_len;
      for(int i = 0; i < vec_len; ++i){
        out_idx[i] = i;
      }
      q_sort(out_data, out_idx, 0, vec_len);
    }

    return TinyNN_SUCCESS;

    })";
    // clang-format on
    return writer.str();
}

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen
