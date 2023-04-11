#pragma once
#include <stddef.h>
#include <sstream>
#include <string>
#include "megbrain/common.h"
#include "megcore.h"
#include "megdnn/handle.h"
#include "megdnn/tensor_iter.h"
#define megcore_check(expr)                                                      \
    do {                                                                         \
        megcoreStatus_t _err = (expr);                                           \
        if (_err != megcoreSuccess) {                                            \
            fprintf(stderr, "mgb failed : line=%d %s:%d\n", (int)_err, __FILE__, \
                    __LINE__);                                                   \
            abort();                                                             \
        }                                                                        \
    } while (0)
namespace megdnn {
namespace test {

void* megdnn_malloc(megdnn::Handle* handle, size_t size_in_bytes);
void megdnn_free(megdnn::Handle* handle, void* ptr);

template <typename Ctype>
std::string tensor_str(megdnn::TensorND& tensor) {
    std::stringstream ss;
    ss << tensor.layout.to_string();
    auto it0 = megdnn::tensor_iter_valonly<Ctype>(tensor).begin();
    for (size_t i = 0; i < tensor.layout.total_nr_elems(); i++) {
        Ctype val = *it0;
        ++it0;
        ss << val << ", ";
    }
    ss << "\n";
    return ss.str();
}

template <typename Ctype>
void print_tensor(megdnn::TensorND& tensor) {
    printf("%s\n", tensor_str<Ctype>(tensor).c_str());
}

class WorkspaceBundle {
public:
    WorkspaceBundle(
            void* ptr, SmallVector<size_t> sizes_in_bytes, size_t align_in_bytes = 512);
    /**
     * \returns raw workspace ptr.
     *
     * Note that ptr() is different than get(0), in that
     * the result of ptr() is possibly not aligned.
     */
    void* ptr() const;
    /**
     * \returns the i-th workspace ptr (aligned)
     */
    void* get(size_t i) const;
    /**
     * \returns total size taking into account paddings to solve alignment
     * issue.
     */
    size_t total_size_in_bytes() const;
    size_t get_size(size_t i) const;
    size_t nr_workspace() const;
    void set(void* ptr);

    Workspace get_workspace(size_t i) const {
        return {static_cast<dt_byte*>(get(i)), get_size(i)};
    }

private:
    void* m_ptr;
    SmallVector<size_t> m_sizes;
    SmallVector<size_t> m_aligned_sizes;
    size_t m_align_in_bytes;
};

#define MEGDNN_MARK_USED_VAR(v) static_cast<void>(v)

std::shared_ptr<TensorNDArray> dnn_alloc_tensors(
        megdnn::Handle* handle, const TensorLayoutArray& layouts, const size_t offset);

void dnn_copy_tensors(const TensorNDArray& dest, const TensorNDArray& src);
}  // namespace test
}  // namespace megdnn
