#pragma once

#include "Utils/Utils.h"
namespace megcc {
namespace KernelGen {
namespace ArmCommon {

class ArmSimdHelper {
    Utils::DtypeEnum m_dtype_enum;
    using Enum = Utils::DtypeEnum;

public:
    ArmSimdHelper(std::string dtype) : m_dtype_enum(Utils::get_dtype_enum(dtype)){};
    std::string get_dtype_sym() {
        switch (m_dtype_enum) {
            case Enum::float32:
                return "f32";
            case Enum::int32:
            case Enum::qsi32:
                return "s32";
            case Enum::qsi8:
            case Enum::int8:
                return "s8";
            default:
                CC_ABORT << "not support dtype enum " << m_dtype_enum << "\n";
        }
        return "";
    }
    std::string get_ld1q_symbol() { return "vld1q_" + get_dtype_sym(); }
    std::string get_st1q_symbol() { return "vst1q_" + get_dtype_sym(); }
    std::string get_dupq_n_symbol() { return "vdupq_n_" + get_dtype_sym(); }
    int get_nr_elem_q() {
        switch (m_dtype_enum) {
            case Enum::int32:
            case Enum::float32:
                return 4;
            case Enum::int8:
                return 16;
            default:
                CC_ABORT << "not support dtype enum " << m_dtype_enum << "\n";
        }
        return 0;
    }
    std::string get_specifier_q_symbol() {
        switch (m_dtype_enum) {
            case Enum::float32:
                return "float32x4_t";
            case Enum::int32:
            case Enum::qsi32:
                return "int32x4_t";
            case Enum::int8:
            case Enum::qsi8:
                return "int8x16_t";
            default:
                CC_ABORT << "not support dtype enum " << m_dtype_enum << "\n";
        }
        return "";
    }
};

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc