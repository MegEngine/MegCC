/**
 * \file
 * compiler/lib/KernelGen/BareMetal/ElemwiseMultiType.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>

#include "Common/ElemwiseCommon.h"
#include "ElemwiseMultiType.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

namespace {

std::string gen_dep(std::string mode) {
    return R"(
        static inline int8_t fp32_to_int8(float src){
                int res = roundf(src);
                res = res > 127? 127:res;
                res = res < -128? -128:res;
                return (int8_t)(res);
        }
    )";
}
std::string gen_unary(std::string mode) {
    if (mode == "QRELU") {
        return "int8_t out_val = fp32_to_int8(((scale_0 * val_0) > 0?(scale_0 "
               "* "
               "val_0 ):0) * scale_div)";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary(std::string mode) {
    if (mode == "QADD") {
        return "int8_t out_val = fp32_to_int8((scale_0 * val_0 + scale_1 * "
               "val_1) * scale_div);";
    } else if (mode == "QFUSE_ADD_RELU") {
        return R"(
        float val0 = scale_0 * val_0;
        float val1 = scale_1 * val_1;     
        int8_t out_val  = fp32_to_int8( ((val0 + val1) > 0? (val0 + val1):0) * scale_div);
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

}  // namespace

bool ElemwiseMultiTypeKernel::IsAvailable(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    auto nr_operands = context->getAttrInt("nr_operands");
    bool nr_operands_ok = nr_operands == 2 || nr_operands == 3;
    bool mode_ok_unary = nr_operands == 2 && mode == "QRELU";
    bool mode_ok_binary =
            nr_operands == 3 && (mode == "QADD" || mode == "QFUSE_ADD_RELU");
    return nr_operands_ok && (mode_ok_unary || mode_ok_binary);
}

std::string ElemwiseMultiTypeKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_elementwise_multitype";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string ElemwiseMultiTypeKernel::GetKernelBody(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    std::stringstream writer;
    writer << "#include <math.h> \n";
    writer << gen_dep(mode);
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);
    if (context->getAttrInt("nr_operands") == 2) {
        auto op0 = context->getAttrOprand("operand:0");
        auto dst = context->getAttrOprand("operand:1");
        CC_ASSERT(Utils::is_quant_dtype(op0.dtype, 8) &&
                  Utils::is_quant_dtype(dst.dtype, 8));
        auto op0_specifier = Utils::cvt_dtype_specifier(op0.dtype);
        auto dst_specifier = Utils::cvt_dtype_specifier(dst.dtype);
        std::string binary_str = R"({
                ${op0_specifier}* input_0 = (${op0_specifier}*)inputs[0]->ptr;
                float scale_0 = inputs[0]->dtype.param.scale;
                TINYNN_ASSERT(input_0);
                ${dst_specifier}* output_data = (${dst_specifier}*)outputs[0]->ptr;
                float scale_dst = outputs[0]->dtype.param.scale;
                TINYNN_ASSERT(output_data);
                float scale_div = 1.f / scale_dst;

                Layout in_layout = inputs[0]->layout;
                size_t nr_elem = 1;
                for (int i = 0; i < in_layout.nr_dim; ++i) {
                    nr_elem *= in_layout.dims[i];
                }
                for(size_t i = 0; i < nr_elem; ++i){
                    ${op0_specifier} val_0 = input_0[i];
                    ${act};
                    output_data[i] = out_val;
                }
                return TinyNN_SUCCESS;
                }
            )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("op0_specifier", op0_specifier)
                          .add("dst_specifier", dst_specifier)
                          .add("act", gen_unary(mode))
                          .render(binary_str);
    } else if (context->getAttrInt("nr_operands") == 3) {
        auto op0 = context->getAttrOprand("operand:0");
        auto op1 = context->getAttrOprand("operand:1");
        auto dst = context->getAttrOprand("operand:2");
        CC_ASSERT(Utils::is_quant_dtype(op0.dtype, 8) &&
                  Utils::is_quant_dtype(op1.dtype, 8) &&
                  Utils::is_quant_dtype(dst.dtype, 8));
        CC_ASSERT(op0.shape == op1.shape) << "no support broadcast\n";
        auto op0_specifier = Utils::cvt_dtype_specifier(op0.dtype);
        auto op1_specifier = Utils::cvt_dtype_specifier(op1.dtype);
        auto dst_specifier = Utils::cvt_dtype_specifier(dst.dtype);
        std::string binary_str = R"({
                ${op0_specifier}* input_0 = (${op0_specifier}*)inputs[0]->ptr;
                float scale_0 = inputs[0]->dtype.param.scale;
                TINYNN_ASSERT(input_0);
                ${op1_specifier}* input_1 = (${op1_specifier}*)inputs[1]->ptr;
                float scale_1 = inputs[1]->dtype.param.scale;
                TINYNN_ASSERT(input_1);
                ${dst_specifier}* output_data = (${dst_specifier}*)outputs[0]->ptr;
                float scale_dst = outputs[0]->dtype.param.scale;
                TINYNN_ASSERT(output_data);
                float scale_div = 1.f / scale_dst;

                Layout in_layout = inputs[0]->layout;
                size_t nr_elem = 1;
                for (int i = 0; i < in_layout.nr_dim; ++i) {
                    nr_elem *= in_layout.dims[i];
                }
                for(size_t i = 0; i < nr_elem; ++i){
                    ${op0_specifier} val_0 = input_0[i];
                    ${op1_specifier} val_1 = input_1[i];
                    ${act};
                    output_data[i] = out_val;
                }
                return TinyNN_SUCCESS;
                }
            )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("op0_specifier", op0_specifier)
                          .add("op1_specifier", op1_specifier)
                          .add("dst_specifier", dst_specifier)
                          .add("act", gen_binary(mode))
                          .render(binary_str);
    } else {
        CC_ABORT << "not support operands size "
                 << context->getAttrInt("nr_operands") << "\n";
    }
    return writer.str();
}

// vim: syntax=cpp.doxygen
