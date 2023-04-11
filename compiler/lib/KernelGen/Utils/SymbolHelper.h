#pragma once
#include <sstream>
#include <string>
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
class SymbolHelper {
public:
    static std::string gen_valid_dtype(const std::string& dtype_str) {
        auto sp_idx = dtype_str.find_first_of("<");
        if (sp_idx != std::string::npos) {
            return dtype_str.substr(0, sp_idx);
        }
        return dtype_str;
    }
    static std::string gen_io_str(TContext* context) {
        std::stringstream ss;
        for (int i = 0; i < context->getAttrInt("nr_operands"); ++i) {
            auto operand_name = "operand:" + std::to_string(i);
            auto operand = context->getAttrOprand(operand_name);
            ss << gen_valid_dtype(operand.dtype);
        }
        return ss.str();
    }
};
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
