/**
 * \file
 * compiler/lib/KernelGen/BareMetal/Reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Reduce.h"
#include "../Utils/StringTemplate.h"
#include "../Utils/SymbolHelper.h"
#include "../Utils/Utils.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

namespace {
std::string gen_helper(std::string mode, std::string dtype) {
    Utils::DtypeHelper dtype_helper(dtype);
    std::stringstream writer;
    if (mode == "MAX") {
        writer << dtype_helper.inline_max_func();
    } else if (mode == "MIN") {
        writer << dtype_helper.inline_min_func();
    } else if (
            mode == "SUM" || mode == "MEAN" || mode == "PRODUCT" || mode == "SUM_SQR") {
    } else {
        CC_ABORT << "unknown reduce mode " << mode.c_str() << "\n";
    }
    return writer.str();
}
std::string gen_init(std::string mode, std::string dtype) {
    Utils::DtypeHelper dtype_helper(dtype);
    std::stringstream writer;
    if (mode == "MAX") {
        writer << dtype_helper.min();
    } else if (mode == "MIN") {
        writer << dtype_helper.max();
    } else if (mode == "SUM" || mode == "MEAN" || mode == "SUM_SQR") {
        writer << dtype_helper.zero();
    } else if (mode == "PRODUCT") {
        writer << dtype_helper.one();
    } else {
        CC_ABORT << "unknown reduce mode " << mode.c_str() << "\n";
    }
    return writer.str();
}
std::string gen_apply(std::string mode, std::string acc, std::string val) {
    std::stringstream writer;
    if (mode == "MAX") {
        writer << "max(" << acc << ", " << val << ")";
    } else if (mode == "MIN") {
        writer << "min(" << acc << ", " << val << ")";
    } else if (mode == "SUM" || mode == "MEAN") {
        writer << acc << " + " << val;
    } else if (mode == "PRODUCT") {
        writer << acc << " * " << val;
    } else if (mode == "SUM_SQR") {
        writer << acc << " + ( " << val << " * " << val << ")";
    } else {
        CC_ABORT << "unknown reduce mode " << mode.c_str() << "\n";
    }
    return writer.str();
}
std::string gen_write(std::string mode, std::string acc, std::string size) {
    std::stringstream writer;
    if (mode == "MAX" || mode == "SUM" || mode == "MIN" || mode == "PRODUCT" ||
        mode == "SUM_SQR") {
        writer << acc;
    } else if (mode == "MEAN") {
        writer << acc << " / (" << size << ")";
    } else {
        CC_ABORT << "unknown reduce mode " << mode.c_str() << "\n";
    }
    return writer.str();
}
}  // namespace

bool ReduceKernel::IsAvailable(TContext* context) const {
    auto data_type = context->getAttrStr("data_type");
    bool ok_data_type = data_type == "DEFAULT";
    auto dtype = context->getAttrOprand("operand:0").dtype;
    bool ok_dtype = Utils::is_float_dtype(dtype, 32) || Utils::is_int_dtype(dtype, 32);
    return ok_data_type && ok_dtype;
}

//! kernel gen
std::string ReduceKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_reduce";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << context->getAttrStr("data_type");
    ss << "_a" << context->getAttrInt("axis");
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string ReduceKernel::GetKernelBody(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    size_t axis = context->getAttrInt("axis");
    auto input = context->getAttrOprand("operand:0");
    std::stringstream writer;
    auto dtype = context->getAttrOprand("operand:0").dtype;
    auto specifier = Utils::cvt_dtype_specifier(dtype);
    writer << gen_helper(mode, dtype);
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << "const size_t axis = " << axis << ";\n";
    writer << StringTemplate::StringTemplateArgs()
                      .add("specifier", specifier)
                      .add("gen_init", gen_init(mode, dtype))
                      .add("gen_apply",
                           gen_apply(mode, "acc", "input_data[i * B * C + j * C + k]"))
                      .add("gen_write", gen_write(mode, "acc", "B"))
                      .render(R"(
        ${specifier}* input_data = (${specifier}*)inputs[0]->ptr;
        ${specifier}* output_data = (${specifier}*)outputs[0]->ptr;

        Layout in_layout = inputs[0]->layout;
        int A = 1, B, C = 1;
        for (int i = 0; i < axis; ++ i)
            A *= in_layout.dims[i];
        B = in_layout.dims[axis];
        for (int i = axis + 1; i < in_layout.nr_dim; ++ i)
            C *= in_layout.dims[i];

        for (int i = 0; i < A; ++ i) {
            for (int k = 0; k < C; ++ k) {
                ${specifier} acc = ${gen_init};
                for (int j = 0; j < B; ++ j) {
                    acc = ${gen_apply};
                }
                
                output_data[i * C + k] = ${gen_write};
            }
        }
        return TinyNN_SUCCESS;
    })");
    return writer.str();
}

// vim: syntax=cpp.doxygen
