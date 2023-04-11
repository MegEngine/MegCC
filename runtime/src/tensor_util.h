#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H
#include <stdbool.h>
#include "data_struct.h"
#include "utils.h"

static inline bool force_layout_contiguous(Layout* layout) {
    int stride = 1;
    for (int i = layout->nr_dim - 1; i >= 0; --i) {
        layout->stride[i] = stride;
        stride *= layout->dims[i];
    }
    return stride != 0;
}

typedef struct {
    int inc_dims[MAX_DIM];
    int reset_stride[MAX_DIM];
    int offset;
} NoconIter;

static inline NoconIter init_iter(const Layout layout) {
    NoconIter iter;
    for (int i = layout.nr_dim - 1; i >= 0; --i) {
        iter.inc_dims[i] = 0;
        iter.reset_stride[i] = (layout.dims[i] - 1) * layout.stride[i];
    }
    iter.offset = 0;
    return iter;
}

static inline void inc_iter(const Layout layout, NoconIter* iter) {
    for (int i = layout.nr_dim - 1; i >= 0; --i) {
        iter->inc_dims[i] += 1;
        if (iter->inc_dims[i] >= layout.dims[i]) {
            iter->inc_dims[i] = 0;
            iter->offset = iter->offset - iter->reset_stride[i];
        } else {
            iter->offset = iter->offset + layout.stride[i];
            break;
        }
    }
}

static inline bool is_contiguous(Layout layout) {
    int stride = 1;
    for (int i = layout.nr_dim - 1; i >= 0; --i) {
        if (layout.stride[i] < 0 || layout.stride[i] != stride) {
            return 0;
        } else {
            stride *= layout.dims[i];
        }
    }
    return stride != 0;
}

static inline void broadcast_layout(Layout* layout_in, const Layout layout_dst) {
    uint32_t diff_dim = layout_dst.nr_dim - layout_in->nr_dim;
    //! move old shape to end
    for (uint32_t j = 0; j < layout_in->nr_dim; j++) {
        if (layout_in->dims[j] == layout_dst.dims[diff_dim + j]) {
            layout_in->dims[diff_dim + j] = layout_in->dims[j];
            layout_in->stride[diff_dim + j] = layout_in->stride[j];
        } else {
            layout_in->dims[j] = layout_dst.dims[j];
            layout_in->stride[diff_dim + j] = 0;
        }
    }
    //! add new axis to high
    for (uint32_t j = 0; j < diff_dim; j++) {
        layout_in->dims[j] = layout_dst.dims[j];
        layout_in->stride[j] = 0;
    }
}

static inline int32_t get_tensor_value(const Tensor* tensor, int index) {
    if (tensor->dtype.type_enum == TinyNN_FLOAT) {
        return *((float*)(tensor->ptr) + index);
    } else if (tensor->dtype.type_enum == TinyNN_INT) {
        return *((int32_t*)(tensor->ptr) + index);
    } else if (tensor->dtype.type_enum == TinyNN_INT8) {
        return *((int8_t*)(tensor->ptr) + index);
    } else {
        TINYNN_ASSERT_MSG(0, "unsupport dtype.\n");
    }
    return 0;
}
#endif
