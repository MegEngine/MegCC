#pragma once

#include <type_traits>

#include "megbrain/opr/param_defs.h"
#include "megdnn/opr_param_defs.h"

// to include bitwise operations on some enumerate class which type is
// BitCombinedEnum
#include "megbrain/graph/operator_node.h"

namespace mgb {
namespace reflection {
namespace detail {
template <typename Enum>
struct EnumTrait : public std::false_type {};

template <typename Enum>
struct BitCombinedEnumTrait : public std::false_type {};

#include "megbrain/enum_reflection.h.inl"

}  // namespace detail

template <typename Enum, std::enable_if_t<detail::EnumTrait<Enum>::value, int> = 0>
std::string nameOfEnumValue(Enum value) {
    return detail::EnumTrait<Enum>::nameof(value);
}

template <
        typename Enum,
        std::enable_if_t<detail::BitCombinedEnumTrait<Enum>::value, int> = 0>
std::vector<std::string> nameOfEnumValue(Enum value) {
    return detail::BitCombinedEnumTrait<Enum>::nameof(value);
}
}  // namespace reflection
}  // namespace mgb

// vim: syntax=cpp.doxygen
