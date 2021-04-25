/**
 * \file
 * compiler/test/kernel/common/src/workspace_wrapper.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "test/kernel/common/dnn_helper.h"
#include "test/kernel/common/workspace_wrapper.h"
namespace megdnn {
namespace test {

WorkspaceWrapper::WorkspaceWrapper() : WorkspaceWrapper(nullptr, 0) {}

WorkspaceWrapper::WorkspaceWrapper(Handle* handle, size_t size_in_bytes)
        : m_handle(handle) {
    m_workspace.size = size_in_bytes;
    if (m_workspace.size > 0) {
        m_workspace.raw_ptr =
                static_cast<dt_byte*>(megdnn_malloc(handle, size_in_bytes));
    } else {
        m_workspace.raw_ptr = nullptr;
    }
}

void WorkspaceWrapper::update(size_t size_in_bytes) {
    mgb_assert(this->valid());
    if (size_in_bytes > m_workspace.size) {
        // free workspace
        if (m_workspace.size > 0) {
            megdnn_free(m_handle, m_workspace.raw_ptr);
            m_workspace.raw_ptr = nullptr;
        }
        // alloc new workspace
        m_workspace.size = size_in_bytes;
        if (m_workspace.size > 0) {
            m_workspace.raw_ptr = static_cast<dt_byte*>(
                    megdnn_malloc(m_handle, size_in_bytes));
        } else {
            m_workspace.raw_ptr = nullptr;
        }
    }
}
void WorkspaceWrapper::clear() {
    if (m_workspace.size > 0) {
        megdnn_free(m_handle, m_workspace.raw_ptr);
        m_workspace.raw_ptr = nullptr;
    }
}

WorkspaceWrapper::~WorkspaceWrapper() {
    clear();
}

WorkspaceWrapper& WorkspaceWrapper::operator=(const WorkspaceWrapper& wrapper) {
    //! must clean old workspace before renew workspace
    clear();
    m_handle = wrapper.m_handle;
    m_workspace = wrapper.m_workspace;
    return *this;
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
