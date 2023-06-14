#ifndef __IO_TENSOR_H_
#define __IO_TENSOR_H_

#include "data_struct.h"

/**
 * @brief Create a dummy head.
 *
 * @return ComboIOTensor*
 */
static inline ComboIOTensor* create_combo_io_tensor() {
    ComboIOTensor* res = (ComboIOTensor*)tinynn_malloc(sizeof(ComboIOTensor));
    memset(res, 0, sizeof(ComboIOTensor));
    return res;
}

static inline ComboIOTensor* find_combo_io_tensor_by_name(
        const CombineModel* cb_model, const char* io_name) {
    //! dummy head
    ComboIOTensor* curr = cb_model->combo_iotensor;
    while (curr->next) {
        ComboIOTensor* next = curr->next;
        TINYNN_ASSERT(next->tensors && next->tensors[0]);
        if (!strcmp(io_name, next->tensors[0]->name))
            break;
        curr = next;
    }
    //! curr->next == NULL means match io_name failed, create new one and set it as
    //! curr->next. Otherwise, match successed.
    if (!curr->next) {
        ComboIOTensor* next = (ComboIOTensor*)tinynn_malloc(sizeof(ComboIOTensor));
        next->model = (CombineModel*)cb_model;
        next->next = NULL;
        next->tensors =
                (Tensor**)tinynn_malloc(sizeof(Tensor*) * cb_model->nr_device_model);
        memset(next->tensors, 0, sizeof(Tensor*) * cb_model->nr_device_model);
        curr->next = next;
    }
    return curr->next;
}

static inline void destroy_combo_io_tensor(ComboIOTensor* dummy_head) {
    while (dummy_head) {
        ComboIOTensor* next = dummy_head->next;
        FREE(dummy_head->tensors);
        dummy_head->tensors = NULL;
        FREE(dummy_head);
        dummy_head = next;
    }
}

static inline Tensor* get_active_tensor(ComboIOTensor* tensor) {
    return tensor->tensors[tensor->model->active_device_model_idx];
}

#endif