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
    bool ok_mode = ctx->getAttrStr("mode") == "KTH_ONLY" ||
                   ctx->getAttrStr("mode") == "VALUE_IDX_SORTED" ||
                   ctx->getAttrStr("mode") == "VALUE_IDX_NOSORT";
    return ok_dtype && ok_mode;
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
    std::string init_k;
    std::string call_sort;
    std::string write_back_val;
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
        call_sort =
                "kth_element_sorted(val_workspace, idx_workspace, val_data + batch_id "
                "* k_len, idx_data + batch_id * k_len, vec_len, k_len);\n";
        if (mode == "VALUE_IDX_NOSORT") {
            call_sort =
                    "kth_element_no_sort(val_workspace, idx_workspace, 0, vec_len - 1, "
                    "k_len - 1);\n";
            write_back_index = R"(
                    int* out_idx = idx_data + batch_id * k_len;
                    memcpy(out_idx, idx_workspace, k_len * sizeof(int));
                )";
            write_back_val = R"(
                    float* out_data = val_data + batch_id * k_len;
                    memcpy(out_data, val_workspace, k_len * sizeof(float));
                  )";
        }
    }
    if (only_kth) {
        init_k = "int k = " + std::to_string(std::abs(k)) + ";\n";
        call_sort = "kth_element_no_sort(val_workspace, 0, vec_len - 1, k - 1);\n";
        write_back_val = R"(
          float* out_data = val_data + batch_id;
          *out_data = val_workspace[k - 1];
        )";
    }

    writer << "#include <string.h>\n";
    if (mode != "VALUE_IDX_SORTED")
        writer << "#include <stdlib.h>\n";

    writer << R"(
      static inline void swap(float* val, int a, int b){
        float temp = val[a];
        val[a] = val[b];
        val[b] = temp;
      }
    )";

    if (with_index) {
        writer << R"(
        static inline void swap_int(int* val, int a, int b) {
          int temp = val[a];
          val[a] = val[b];
          val[b] = temp;
        }
      )";
    }

    if (mode == "KTH_ONLY") {
        writer << StringTemplate::StringTemplateArgs(ctx)
                          .add("compare_sign", compare_sign)
                          .render(R"(
        static inline int partition(float* val, int left, int right) {
          float x = val[right];
          int i = left - 1;
          for (int j = left; j < right; ++j) {
              if (val[j] ${compare_sign} x) {
                  ++i;
                  swap(val, i, j);
              }
          }
          ++i;
          swap(val, i, right);
          return i;
        }
        static inline int randomized_partition(float* val, int left, int right) {
          int rdm_idx = left + rand() % (right - left + 1);
          swap(val, rdm_idx, right);
          return partition(val, left, right);
        }
        static inline void kth_element_no_sort(float* val, int left, int right, const int k) {
          if (left >= right)
              return;
          int i = randomized_partition(val, left, right);
          if (i == k) {
              return;
          } else if (i > k) {
              kth_element_no_sort(val, left, i - 1, k);
          } else {
              kth_element_no_sort(val, i + 1, right, k);
          }
        }
      )");
    }

    if (mode == "VALUE_IDX_NOSORT") {
        writer << StringTemplate::StringTemplateArgs(ctx)
                          .add("compare_sign", compare_sign)
                          .render(R"(
          static inline int partition(float* val, int* idx, int left, int right) {
            float x = val[right];
            int i = left - 1;
            for (int j = left; j < right; ++j) {
                if (val[j] ${compare_sign} x) {
                    ++i;
                    swap(val, i, j);
                    swap_int(idx, i, j);
                }
            }
            ++i;
            swap(val, i, right);
            swap_int(idx, i, right);
            return i;
          }
          static inline int randomized_partition(float* val, int* idx, int left, int right) {
            int rdm_idx = left + rand() % (right - left + 1);
            swap(val, rdm_idx, right);
            swap_int(idx, rdm_idx, right);
            return partition(val, idx, left, right);
          }
          static inline void kth_element_no_sort(
                float* val, int* idx, int left, int right, const int k) {
            if (left >= right)
                return;
            int i = randomized_partition(val, idx, left, right);
            if (i == k) {
                return;
            } else if (i > k) {
                kth_element_no_sort(val, idx, left, i - 1, k);
            } else {
                kth_element_no_sort(val, idx, i + 1, right, k);
            }
          }
        )");
    }

    if (mode == "VALUE_IDX_SORTED") {
        writer << StringTemplate::StringTemplateArgs(ctx)
                          .add("compare_sign", compare_sign)
                          .render(R"(
          typedef struct Heap {
              int size;
              float* val;
              int* idx;
          } Heap;
          static inline void shift_down(Heap * heap, int idx) {
            int left = (idx << 1) + 1;
            if (left >= heap->size)
                return;
            int right = left + 1;
            int candidate = left;
            if (right < heap->size)
                candidate = heap->val[left] ${compare_sign} heap->val[right] ? right : left;
            if (heap->val[idx] ${compare_sign} heap->val[candidate]) {
                swap(heap->val, idx, candidate);
                swap_int(heap->idx, idx, candidate);
                shift_down(heap, candidate);
            }
          }
          static inline void shift_up(Heap * heap, int idx) {
            int dad = (idx - 1) >> 1;
            if (dad < 0)
                return;
            if (heap->val[dad] ${compare_sign} heap->val[idx]) {
                swap(heap->val, idx, dad);
                swap_int(heap->idx, idx, dad);
                shift_up(heap, dad);
            }
          }
          static inline void insert(Heap * heap, float val, int idx) {
            heap->val[heap->size] = val;
            heap->idx[heap->size] = idx;
            shift_up(heap, heap->size);
            ++heap->size;
          }
          static inline void pop(Heap * heap) {
            --heap->size;
            swap(heap->val, 0, heap->size);
            swap_int(heap->idx, 0, heap->size);
            shift_down(heap, 0);
          }
          static inline float top(Heap * heap) { return heap->val[0]; }
          static inline void kth_element_sorted(
                  const float* val, const int* idx, float* dst_val, int* dst_idx,
                  const int vec_len, const int k) {
            Heap heap;
            heap.size = 0;
            heap.val = dst_val;
            heap.idx = dst_idx;
            int i = 0;
            for (; i < k; ++i) {
                insert(&heap, val[i], idx[i]);
            }
            for (; i < vec_len; ++i) {
                if (val[i] ${compare_sign} top(&heap)) {
                    pop(&heap);
                    insert(&heap, val[i], idx[i]);
                }
            }
            for (i = 0; i < k; ++i) {
                pop(&heap);
            }
          }
        )");
    }
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
