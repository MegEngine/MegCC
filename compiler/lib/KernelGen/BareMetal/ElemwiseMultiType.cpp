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
    } else if (mode == "QMUL") {
        return R"(
            int8_t out_val = fp32_to_int8(val_0 * val_1 * scale_mul);
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
    bool dtype_ok_unary =
            nr_operands == 2 &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:0").dtype) &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:1").dtype, 8);
    bool mode_ok_binary =
            nr_operands == 3 &&
            (mode == "QADD" || mode == "QFUSE_ADD_RELU" || mode == "QMUL");
    bool dtype_ok_binary =
            nr_operands == 3 &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:0").dtype) &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:1").dtype) &&
            Utils::is_quant_dtype(context->getAttrOprand("operand:2").dtype, 8);
    const auto& op0_shape = context->getAttrOprand("operand:0").shape;
    const auto& op1_shape = context->getAttrOprand("operand:1").shape;
    size_t op1_nr_elem = 1;
    for (auto dim : op1_shape) {
        op1_nr_elem *= dim;
    }
    //! broadcast mode 0: op0 shape: (a, b, c, d, ...), op1 shape: (1, b, 1, 1, ...)
    //! broadcast mode 1: op0 shape: (a, b, c, d, ...), op1_nr_elem = 1
    bool shape_ok_binary =
            nr_operands == 3 &&
            ((op0_shape == op1_shape) ||
             (op0_shape.size() == op1_shape.size() && op0_shape.size() > 2 &&
              op0_shape[1] == op1_shape[1] && op1_nr_elem == op1_shape[1]) ||
             (op1_nr_elem == 1));
    return nr_operands_ok && ((mode_ok_unary && dtype_ok_unary) ||
                              (mode_ok_binary && dtype_ok_binary && shape_ok_binary));
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
        auto op0_specifier = Utils::cvt_dtype_specifier(op0.dtype);
        auto dst_specifier = Utils::cvt_dtype_specifier(dst.dtype);
        std::string unary_str = R"({
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
                          .render(unary_str);
    } else if (context->getAttrInt("nr_operands") == 3) {
        auto op0 = context->getAttrOprand("operand:0");
        auto op1 = context->getAttrOprand("operand:1");
        auto dst = context->getAttrOprand("operand:2");
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
                float scale_mul = scale_0 * scale_1 * scale_div;

                Layout in_layout0 = inputs[0]->layout;
                size_t nr_elem0 = 1;
                for (int i = 0; i < in_layout0.nr_dim; ++i) {
                    nr_elem0 *= in_layout0.dims[i];
                }
                Layout in_layout1 = inputs[1]->layout;
                size_t nr_elem1 = 1;
                for (int i = 0; i < in_layout1.nr_dim; ++i) {
                    nr_elem1 *= in_layout1.dims[i];
                }
                if (nr_elem0 == nr_elem1) {
                    for(size_t i = 0; i < nr_elem0; ++i){
                        ${op0_specifier} val_0 = input_0[i];
                        ${op1_specifier} val_1 = input_1[i];
                        ${act};
                        output_data[i] = out_val;
                    }
                } else if (nr_elem1 == 1) {
                    ${op1_specifier} val_1 = input_1[0];
                    for(size_t i = 0; i < nr_elem0; ++i){
                        ${op0_specifier} val_0 = input_0[i];
                        ${act};
                        output_data[i] = out_val;
                    }
                } else {
                    TINYNN_ASSERT(nr_elem0 > nr_elem1);
                    for (int i = 0; i < in_layout0.dims[0]; ++i) {
                        for (int j = 0; j < in_layout0.dims[1]; ++j) {
                            ${op1_specifier} val_1 = input_1[j];
                            for (int k = 0; k < in_layout0.stride[1]; ++k) {
                                int idx = i * in_layout0.stride[0] + j * in_layout0.stride[1] + k;
                                ${op0_specifier} val_0 = input_0[idx];
                                ${act};
                                output_data[idx] = out_val;
                            }
                        }
                    }
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
        CC_ABORT << "not support operands size " << context->getAttrInt("nr_operands")
                 << "\n";
    }
    return writer.str();
}

// vim: syntax=cpp.doxygen
