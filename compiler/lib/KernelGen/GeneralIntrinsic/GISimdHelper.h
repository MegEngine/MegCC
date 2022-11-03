/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/GISimdHelper.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "Utils/Utils.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class GISimdHelper {
    Utils::DtypeEnum m_dtype_enum;
    using Enum = Utils::DtypeEnum;

public:
    GISimdHelper(std::string dtype)
            : m_dtype_enum(Utils::get_dtype_enum(dtype)){};
    std::string get_dtype_sym() {
        switch (m_dtype_enum) {
            case Enum::float32:
                return "Float32";
            case Enum::int32:
            case Enum::qsi32:
                return "Int32";
            case Enum::qsi8:
            case Enum::int8:
                return "Int8";
            default:
                CC_ABORT << "not support dtype enum " << m_dtype_enum << "\n";
        }
        return "";
    }
    std::string get_ld1q_symbol() { return "GiLoad" + get_dtype_sym(); }
    std::string get_st1q_symbol() { return "GiStore" + get_dtype_sym(); }
    std::string get_dupq_n_symbol() { return "GiBroadcast" + get_dtype_sym(); }
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
                return "GI_FLOAT32_t";
            case Enum::int32:
            case Enum::qsi32:
                return "GI_INT32_t";
            case Enum::int8:
            case Enum::qsi8:
                return "GI_INT8_t";
            default:
                CC_ABORT << "not support dtype enum " << m_dtype_enum << "\n";
        }
        return "";
    }
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc