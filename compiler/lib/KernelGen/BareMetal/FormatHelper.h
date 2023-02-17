/**
 * \file
 * compiler/lib/KernelGen/BareMetal/FormatHelper.h
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

class GenFormatIter {
public:
    static std::string gen_inline_format_iter_symbol(std::string format_str);
    static std::string gen_inline_format_iter_body(std::string format_str);
};

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
