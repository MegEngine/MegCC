#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace mlir {
namespace Kernel {
namespace migrate {
//! convert from a ptr to another type that has may_alias attr; use raw_cast if
//! possible
template <typename T, typename U>
T* __attribute__((__may_alias__)) aliased_ptr(U* src) {
    return reinterpret_cast<T*>(src);
}

/*!
 * \brief raw memory storage for incomplete type Obj; Obj only needs to be
 *      complete in ctor and dtor
 */
template <class Obj, size_t SIZE, size_t ALIGN>
class alignas(ALIGN) IncompleteObjStorage {
    uint8_t m_mem[SIZE];

public:
    IncompleteObjStorage() {
        static_assert(
                sizeof(Obj) <= SIZE && !(ALIGN % alignof(Obj)),
                "SIZE and ALIGN do not match Obj");
        new (m_mem) Obj;
    }

    IncompleteObjStorage(const IncompleteObjStorage& rhs) {
        new (m_mem) Obj(rhs.get());
    }
    IncompleteObjStorage(IncompleteObjStorage&& rhs) noexcept {
        new (m_mem) Obj(std::move(rhs.get()));
    }

    IncompleteObjStorage& operator=(const IncompleteObjStorage& rhs) {
        get() = rhs.get();
        return *this;
    }

    IncompleteObjStorage& operator=(IncompleteObjStorage&& rhs) noexcept {
        get() = std::move(rhs.get());
        return *this;
    }

    ~IncompleteObjStorage() noexcept { get().~Obj(); }

    Obj& get() { return *aliased_ptr<Obj>(m_mem); }

    const Obj& get() const { return const_cast<IncompleteObjStorage*>(this)->get(); }
};

//! use size and align of another object
template <class Obj, class Mock>
using IncompleteObjStorageMock = IncompleteObjStorage<Obj, sizeof(Mock), alignof(Mock)>;

/*!
 * \brief update dest if val is greater than it
 */
template <typename T>
inline bool update_max(T& dest, const T& val) {
    if (dest < val) {
        dest = val;
        return true;
    }
    return false;
}

/*!
 * \brief update dest if val is less than it
 */
template <typename T>
inline bool update_min(T& dest, const T& val) {
    if (val < dest) {
        dest = val;
        return true;
    }
    return false;
}

/*!
 * \brief align *val* to be multiples of *align*
 * \param align required alignment, which must be power of 2
 */
template <typename T>
static inline T get_aligned_power2(T val, T align) {
    auto d = val & (align - 1);
    val += (align - d) & (align - 1);
    return val;
}
}  // namespace migrate
}  // namespace Kernel
}  // namespace mlir

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
