/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Activation.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "compiler/Common/Logger.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

class GenActivation {
public:
    static std::string gen_func_dep(std::string mode);
    static std::string gen_func_call_with_typecvt_dep(
            std::string mode, std::string src_specifier, std::string dst_specifier);
    static std::string gen_func_call(std::string mode, std::string args);
    static std::string gen_func_call_with_typecvt(
            std::string mode, std::string args, std::string src_specifier,
            std::string dst_specifier, std::string scale_name,
            std::string flt_scale_name, std::string div_scale_name);
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
