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
        int temp = val[a];
        val[a] = val[b];
        val[b] = temp;
      }
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
      static inline void pop(Heap * heap) {
        --heap->size;
        swap(heap->val, 0, heap->size);
        swap_int(heap->idx, 0, heap->size);
        shift_down(heap, 0);
      }
      static inline void build_heap_from_array(Heap* heap){
        int last_dad = (heap->size - 2) >> 1;
        for(; last_dad >= 0; --last_dad)
          shift_down(heap, last_dad);
      }
      static inline void sort(
              float* dst_val, int* dst_idx, const int vec_len) {
        Heap heap;
        heap.size = vec_len;
        heap.val = dst_val;
        heap.idx = dst_idx;
        build_heap_from_array(&heap);
        int i;
        for (i = 0; i < vec_len; ++i) {
            pop(&heap);
        }
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
      sort(out_data, out_idx, vec_len);
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
