#include "test/kernel/common/dnn_helper.h"
#include <numeric>
#include "test/kernel/common/memory_manager.h"
using namespace megdnn;

void* megdnn::test::megdnn_malloc(Handle* handle, size_t size_in_bytes) {
    auto mm = MemoryManagerHolder::instance()->get(handle);
    return mm->malloc(size_in_bytes);
}

void megdnn::test::megdnn_free(Handle* handle, void* ptr) {
    auto mm = MemoryManagerHolder::instance()->get(handle);
    mm->free(ptr);
}

megdnn::test::WorkspaceBundle::WorkspaceBundle(
        void* ptr, SmallVector<size_t> sizes_in_bytes, size_t align_in_bytes)
        : m_ptr(ptr),
          m_sizes(std::move(sizes_in_bytes)),
          m_align_in_bytes(align_in_bytes) {
    m_aligned_sizes.reserve(m_sizes.size());
    for (auto size : m_sizes) {
        auto aligned_size = size;
        if (size % m_align_in_bytes != 0) {
            aligned_size += m_align_in_bytes - size % m_align_in_bytes;
        }
        m_aligned_sizes.push_back(aligned_size);
    }
}

void* megdnn::test::WorkspaceBundle::ptr() const {
    return m_ptr;
}

void* megdnn::test::WorkspaceBundle::get(size_t i) const {
    auto addr = reinterpret_cast<uintptr_t>(m_ptr);
    if (addr % m_align_in_bytes != 0)
        addr += m_align_in_bytes - addr % m_align_in_bytes;
    for (size_t j = 0; j < i; ++j) {
        addr += m_aligned_sizes[j];
    }
    return reinterpret_cast<void*>(addr);
}

size_t megdnn::test::WorkspaceBundle::nr_workspace() const {
    return m_sizes.size();
}

size_t megdnn::test::WorkspaceBundle::get_size(size_t i) const {
    return m_sizes[i];
}

void megdnn::test::WorkspaceBundle::set(void* ptr) {
    m_ptr = ptr;
}

size_t megdnn::test::WorkspaceBundle::total_size_in_bytes() const {
    //! return 0 if the WorkspaceBundle is empty
    size_t size = std::accumulate(
            m_aligned_sizes.begin(), m_aligned_sizes.end(), static_cast<size_t>(0));
    return size ? size + m_align_in_bytes : size;
}

std::shared_ptr<TensorNDArray> megdnn::test::dnn_alloc_tensors(
        megdnn::Handle* handle, const TensorLayoutArray& layouts, const size_t offset) {
    auto deleter = [handle, offset](TensorNDArray* ptr) {
        for (auto&& i : *ptr) {
            auto pdata = static_cast<dt_byte*>(i.raw_ptr()) + i.layout.span().low_byte -
                         offset;
            megdnn_free(handle, pdata);
        }
        delete ptr;
    };
    std::shared_ptr<TensorNDArray> ret{new TensorNDArray, deleter};
    for (size_t i = 0; i < layouts.size(); ++i) {
        auto span = layouts[i].span();
        ret->emplace_back(
                static_cast<dt_byte*>(
                        megdnn_malloc(handle, span.dist_byte() + offset)) -
                        span.low_byte + offset,
                layouts[i]);
    }
    return ret;
}

void megdnn::test::dnn_copy_tensors(
        const TensorNDArray& dest, const TensorNDArray& src) {
    mgb_assert(dest.size() == src.size());
    for (size_t i = 0; i < dest.size(); ++i) {
        mgb_assert(dest[i].layout.total_nr_elems() == src[i].layout.total_nr_elems());
        mgb_assert(dest[i].layout.dtype == src[i].layout.dtype);
        memcpy(static_cast<int8_t*>(dest[i].raw_ptr()) + dest[i].layout.span().low_byte,
               static_cast<int8_t*>(src[i].raw_ptr()) + src[i].layout.span().low_byte,
               dest[i].layout.span().dist_byte());
    }
}