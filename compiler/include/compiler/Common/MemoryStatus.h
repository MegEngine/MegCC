#pragma once

#include <cstdint>

namespace megcc {
struct MemoryStatus {
    MemoryStatus(uint64_t _ = 0) : status(_) {}

    bool isPrealloc() { return kindof(Kind::Prealloc); }

    bool isStaticAssignment() { return kindof(Kind::StaticAssignment); }

    bool isDynamicAllocation() { return kindof(Kind::DynamicAllocation); }

    bool isForwarded() { return status & forwardFlag; }

    MemoryStatus& setPrealloc() {
        setKind(Kind::Prealloc);
        return *this;
    }

    MemoryStatus& setStaticAssignment() {
        setKind(Kind::StaticAssignment);
        return *this;
    }

    MemoryStatus& setDynamicAllocation() {
        setKind(Kind::DynamicAllocation);
        return *this;
    }

    MemoryStatus& setForwarded() {
        status |= forwardFlag;
        return *this;
    }

    explicit operator uint64_t() const { return status; }

    bool operator==(const MemoryStatus& other) { return status == other.status; }

private:
    /* layout of memory status
                   1bit      2bit
    +----------+-----------+------+
    | reserved | forwarded | kind |
    +----------+-----------+------+
    */

    static constexpr uint64_t kindMask = (1 << 1) | (1 << 0);
    static constexpr uint64_t forwardFlag = 1 << 2;

    enum class Kind : uint8_t {
        Unknown = 0,
        Prealloc = 1,
        StaticAssignment = 2,
        DynamicAllocation = 3
    };

    bool kindof(Kind kind) { return static_cast<Kind>(status & kindMask) == kind; }

    void setKind(Kind kind) {
        status = (status & (~kindMask)) | static_cast<uint64_t>(kind);
    }

    uint64_t status = 0;
};

}  // namespace megcc

// vim: syntax=cpp.doxygen
