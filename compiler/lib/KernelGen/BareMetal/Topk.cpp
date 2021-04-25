/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Topk.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Topk.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

bool TopkKernel::IsAvailable(TContext* ctx) const {
    bool ok_dtype = ctx->getAttrOprand("operand:0").dtype == "f32";
    return ok_dtype;
}

namespace {
void get_mode_bool(bool& with_index, bool& only_kth, TContext* ctx) {
    with_index = "KTH_ONLY" != ctx->getAttrStr("mode");
    only_kth = "KTH_ONLY" == ctx->getAttrStr("mode");
}
}  // namespace

//! kernel gen
std::string TopkKernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    int k = ctx->getAttrInt("k");
    std::string flag = k > 0 ? "p" : "n";
    ss << "kernel_topk_" << ctx->getAttrOprand("operand:0").dtype << "_"
       << ctx->getAttrStr("mode") << "_" << flag;
    bool with_index;
    bool only_kth;
    get_mode_bool(with_index, only_kth, ctx);
    if (only_kth) {
        ss << "_" << std::abs(k);
    }
    return ss.str();
}

std::string TopkKernel::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    bool with_index;
    bool only_kth;
    get_mode_bool(with_index, only_kth, ctx);
    std::string index_workspace = with_index ? " n * sizeof(int) " : "0";
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;        
        const uint32_t n = in_layout.dims[1];
        *workspace = n * sizeof(float) + ${index_workspace};
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add("index_workspace", index_workspace)
                    .render(workspace_temp);
    return ss.str();
}

std::string TopkKernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto mode = ctx->getAttrStr("mode");
    int k = ctx->getAttrInt("k");
    std::string compare_sign = k > 0 ? "<" : ">";
    bool with_index;
    bool only_kth;
    get_mode_bool(with_index, only_kth, ctx);
    std::string declear_index;
    std::string init_index;
    std::string init_k =
            only_kth ? "int k = " + std::to_string(std::abs(k)) + ";\n" : "";
    std::string call_sort = "q_sort_val(val_workspace, 0, vec_len);\n";
    std::string write_back_val =
            R"(
              float* out_data = val_data + batch_id * k_len;
              memcpy(out_data, val_workspace, k_len * sizeof(float));
            )";
    std::string write_back_index;
    if (with_index) {
        declear_index = R"(
                int* idx_data = (int*)outputs[1]->ptr;
                int* idx_workspace = workspace->ptr + vec_len * sizeof(float);
                )";
        init_index = R"(
                for(int i = 0; i < vec_len; ++i){
                  idx_workspace[i] = i;
                }
        )";
        call_sort = "q_sort(val_workspace, idx_workspace, 0, vec_len);\n";
        write_back_index = R"(
          int* out_idx = idx_data + batch_id * k_len;
          memcpy(out_idx, idx_workspace, k_len * sizeof(int));
      )";
    }
    if (only_kth) {
        write_back_val = R"(
          float* out_data = val_data + batch_id;
          *out_data = val_workspace[k - 1];
        )";
    }

    writer << "#include <string.h>\n";
    writer << StringTemplate::StringTemplateArgs(ctx)
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
      static void q_sort_val(float* val, int left, int right){
        if (left >= right - 1)
          return;
        int select_idx = right - 1;
        float select_val = val[select_idx];
        int last = left - 1;
        for(int i = left; i < right; ++i){
          if(val[i] ${compare_sign}= select_val){
            ++last;
            swap(val, i, last);
          }
        }
        q_sort_val(val, 0, last);
        q_sort_val(val, last + 1, right);
      }
    )");
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(ctx) << "{\n";
    // clang-format off
    std::string body_temp = R"(
    const float* src_data = (float*)inputs[0]->ptr;
    Layout src_layout = inputs[0]->layout;
    Layout dst_layout = outputs[0]->layout;
    float* val_data = (float*)outputs[0]->ptr;
    
    int batch = src_layout.dims[0];
    int vec_len = src_layout.dims[1];
    int k_len = dst_layout.dims[1];
    ${init_k}

    float* val_workspace = workspace->ptr;
    ${declear_index}

    for(int batch_id = 0; batch_id < batch; ++batch_id){
      memcpy(val_workspace, src_data + batch_id * vec_len, sizeof(float) * vec_len);

      ${init_index}
      ${call_sort}

      ${write_back_val}
      ${write_back_index}
    }

    return TinyNN_SUCCESS;

    })";
    // clang-format on
    writer << StringTemplate::StringTemplateArgs()
                      .add("declear_index", declear_index)
                      .add("init_index", init_index)
                      .add("call_sort", call_sort)
                      .add("write_back_index", write_back_index)
                      .add("write_back_val", write_back_val)
                      .add("init_k", init_k)
                      .render(body_temp);
    return writer.str();
}

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen
