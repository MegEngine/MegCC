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
