/**
 * \file
 * compiler/lib/KernelGen/BareMetal/FusedElemwiseKernel.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>

#include "Common/ElemwiseCommon.h"
#include "FusedElemwiseKernel.h"
#include "FormatHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

namespace {

std::string gen_dep_func() {
    std::string res;
    res = R"(
            static inline float clip(float val, float min, float max){
                if(val < min){
                    val = min;
                }
                if(val > max){
                    val = max;
                }
                return val;
            }
        )";
    return res;
}

std::string gen_unary(std::string mode) {
    if (mode == "RELU") {
        return "${specifier} ${out} = ${in} > 0 ? ${in}:0;";
    } else if (mode == "EXP") {
        return "${specifier} ${out} = expf(${in});";
    } else if (mode == "SIGMOID") {
        return "${specifier} ${out} =  1 / (1 + expf(-${in}))";
    } else if (mode == "NEGATE") {
        return "${specifier} ${out} = -${in}";
    } else if (mode == "ROUND") {
        return "${specifier} ${out} = roundf(${in})";
    } else if (mode == "H_SWISH") {
        return "${specifier} ${out} = ${in} * clip(${in} + 3, 0, 6) / 6";
    } else if (mode == "ABS") {
        return "${specifier} ${out} = ${in} > 0? ${in}:-${in}";
    } else if (mode == "LOG") {
        return "${specifier} ${out} = logf(${in})";
    } else if (mode == "SILU") {
        return "${specifier} ${out} = ${in} / (1 + expf(-${in}))";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary(std::string mode) {
    if (mode == "ADD") {
        return " ${specifier} ${out} = ${val1} + ${val2}";
    } else if (mode == "SUB") {
        return "${specifier} ${out} = ${val1} - ${val2}";
    } else if (mode == "MUL") {
        return "${specifier} ${out} = ${val1} * ${val2}";
    } else if (mode == "TRUE_DIV") {
        return "${specifier} ${out} =  ${val1} / ${val2}";
    } else if (mode == "FUSE_ADD_RELU") {
        return "${specifier} ${out} =(${val1} + ${val2}) > 0? (${val1} + ${val2}):0";
    } else if (mode == "FUSE_ADD_SIGMOID") {
        return "${specifier} ${out} =1.f/(1.f+ expf(-(${val1} + ${val2})))";
    } else if (mode == "FUSE_ADD_TANH") {
        return "${specifier} ${out} =tanh(${val1} + ${val2})";
    } else if (mode == "MAX") {
        return "${specifier} ${out} =(${val1} > ${val2} ? ${val1} : ${val2})";
    } else if (mode == "MIN") {
        return "${specifier} ${out} =(${val1} < ${val2} ? ${val1} : ${val2})";
    } else if (mode == "LT") {
        return "${specifier} ${out} =(${val1} < ${val2})";
    } else if (mode == "LEQ") {
        return "${specifier} ${out} =(${val1} <= ${val2})";
    } else if (mode == "EQ") {
        return "${specifier} ${out} =(${val1} == ${val2})";
    } else if (mode == "FLOOR_DIV") {
        return "${specifier} ${out} =floorf(${val1} / ${val2})";
    } else if (mode == "MOD") {
        //! WARNING: this is just for integer float please use fmod(x,y) in C
        //! and fmodf(x,y) in c++
        return "${specifier} ${out} = ${val1} % ${val2}";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_tenary(std::string mode) {
    if (mode == "FUSE_MUL_ADD3") {
        return " ${specifier} ${out} = ${val1} * ${val2} + ${val3}";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_quater(std::string mode) {
    if (mode == "FUSE_MUL_ADD4") {
        return " ${specifier} ${out} = ${val1} * ${val2} + ${val3} * ${val4}";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_op(std::vector<std::string> mode, const std::string& specifier){
    size_t nr_str = mode.size();
    CC_ASSERT(nr_str >= 3);
    if (nr_str == 3) {
        auto Op = gen_unary(mode[1]);
        return StringTemplate::StringTemplateArgs()
                .add("specifier", specifier)
                .add("in", mode[0])
                .add("out", mode[2])
                .render(Op);
    } else if (nr_str == 4) {
        auto Op = gen_binary(mode[2]);
        return StringTemplate::StringTemplateArgs()
                .add("specifier", specifier)
                .add("val1", mode[0])
                .add("val2", mode[1])
                .add("out", mode[3])
                .render(Op);
    } else if (nr_str == 5) {
        auto Op = gen_tenary(mode[3]);
        return StringTemplate::StringTemplateArgs()
                .add("specifier", specifier)
                .add("val1", mode[0])
                .add("val2", mode[1])
                .add("val3", mode[2])
                .add("out", mode[4])
                .render(Op);
    } else if (nr_str == 6) {
        auto Op = gen_quater(mode[4]);
        return StringTemplate::StringTemplateArgs()
                .add("specifier", specifier)
                .add("val1", mode[0])
                .add("val2", mode[1])
                .add("val3", mode[2])
                .add("val4", mode[3])
                .add("out", mode[5])
                .render(Op);
    } else {
        CC_ABORT<<"Not support mode in FusedElemwise\n";
    }
    return "";
}

template <TensorType>
std::string gen_one_get_data(size_t id, const std::string& specifier);

template <>
std::string gen_one_get_data<SCALAR>(size_t id, const std::string& specifier) {
    std::string get_data = R"(
       static inline ${specifier} get_input${id}(size_t elem_id, ${specifier}* ptr,
                                    Layout* dst, size_t* stride) {
           return *ptr;
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", specifier)
            .add("id", std::to_string(id))
            .render(get_data);
}

template <>
std::string gen_one_get_data<VECTOR>(size_t id, const std::string& specifier) {
    std::string get_data = R"(
       static inline ${specifier} get_input${id}(size_t elem_id, ${specifier}* ptr, 
                                    Layout* dst, size_t* stride) {
           return ptr[elem_id];
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", specifier)
            .add("id", std::to_string(id))
            .render(get_data);
}

template <>
std::string gen_one_get_data<BCAST101>(size_t id, const std::string& specifier) {
    std::string get_data = R"(
       static inline ${specifier} get_input${id}(size_t elem_id, ${specifier}* ptr, 
                                    Layout* dst_layout, size_t* stride) {
            size_t channel_id = (elem_id / (dst_layout->stride[1])) % dst_layout->dims[1];
           return ptr[channel_id];
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", specifier)
            .add("id", std::to_string(id))
            .render(get_data);
}

template <>
std::string gen_one_get_data<BCAST101x4>(size_t id,
                                         const std::string& specifier) {
    std::string get_data = R"(
       static inline ${specifier} get_input${id}(size_t elem_id, ${specifier}* ptr,
                                    Layout* dst_layout, size_t* stride) {
            size_t channel_id = (elem_id / (dst_layout->stride[1])) % dst_layout->dims[1];
            size_t pack_c_id = elem_id % 4;
           return ptr[channel_id * 4 + pack_c_id];
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", specifier)
            .add("id", std::to_string(id))
            .render(get_data);
}

template <>
std::string gen_one_get_data<UNKNOWN_TENSOR_TYPE>(
        size_t id, const std::string& specifier) {
    std::string get_data = R"(
       static inline ${specifier} get_input${id}(size_t elem_id, ${specifier}* ptr,
                                    const Layout* dst_layout, size_t* stride) {
            size_t offset = 0;
            for(int i=< dst_layout->nr_dim -1; i>=0; i--){
                size_t shape = elem_id % dst_layout->dims[i];
                elem_id = elem_id - shape;
                elem_id /= dst_layout->dims[i];
                offset += shape* stride[i];
            }
            return ptr[offset];
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("specifier", specifier)
            .add("id", std::to_string(id))
            .render(get_data);
}

std::string gen_get_data_func(std::vector<TensorType> tensor_types,
                              const std::string& specifier) {
    size_t nr_operands = tensor_types.size();
    std::string functions;
    for (size_t i = 0; i < nr_operands; i++) {
        auto tensor_type = tensor_types[i];
        switch (tensor_type) {
            case TensorType::SCALAR:
                functions += gen_one_get_data<SCALAR>(i, specifier);
                break;
            case TensorType::VECTOR:
                functions += gen_one_get_data<VECTOR>(i, specifier);
                break;
            case TensorType::BCAST101:
                functions += gen_one_get_data<BCAST101>(i, specifier);
                break;
            case TensorType::BCAST101x4:
                functions += gen_one_get_data<BCAST101x4>(i, specifier);
                break;
            case TensorType::UNKNOWN_TENSOR_TYPE:
                functions += gen_one_get_data<UNKNOWN_TENSOR_TYPE>(i, specifier);
                break;
            default:
                CC_ABORT << "Not support tensor type in fused elemwise\n";
        }
    }
    return functions;
}

}  // namespace

bool FusedElmwiseKernel::IsAvailable(TContext* context) const {
    auto mode_size = context->getAttrInt("modes:size");
    bool mode_ok = true;
    for (int i = 0; i < mode_size; i++) {
        auto operand = context->getAttrOprand("operand:" + std::to_string(i));
        auto modes = Utils::split_string(
                context->getAttrStr("modes:" + std::to_string(i)), ',');
        size_t modes_size = modes.size();
        auto mode = modes[modes_size - 2];
        bool mode_ok_unary =
                mode == "RELU" || mode == "SIGMOID" || mode == "EXP" ||
                mode == "NEGATE" || mode == "ROUND" || mode == "ABS" ||
                mode == "H_SWISH" || mode == "LOG" || mode == "SILU";
        bool mode_ok_binary = mode == "ADD" || mode == "SUB" || mode == "MUL" ||
                              mode == "MAX" || mode == "MIN" || mode == "LEQ" ||
                              mode == "LT" || mode == "FLOOR_DIV" ||
                              mode == "EQ" || mode == "TRUE_DIV" ||
                              mode == "FUSE_ADD_RELU" ||
                              mode == "FUSE_ADD_SIGMOID" ||
                              mode == "FUSE_ADD_TANH" || mode == "MOD";
        bool mode_ok_other = mode == "FUSE_MUL_ADD3" || mode == "FUSE_MUL_ADD4";
        mode_ok = mode_ok_unary || mode_ok_binary || mode_ok_other;
    }
    return mode_ok;
}

std::string FusedElmwiseKernel::GetKernelSymbol(TContext* context) const {
    size_t nr_operands = context->getAttrInt("nr_operands");
    auto dst_operand = context->getAttrOprand("operand:" +
                                              std::to_string(nr_operands - 1));
    std::stringstream ss;
    ss << "kernel_fused_elementwise";
    auto mode_size = context->getAttrInt("modes:size");
    for (int i = 0; i < mode_size; i++) {
        auto modes = Utils::split_string(
                context->getAttrStr("modes:" + std::to_string(i)), ',');
        size_t modes_size = modes.size();
        auto mode = modes[modes_size - 2];
        ss << "_" << mode ;
    }
    for (size_t i = 0; i < nr_operands; i++) {
        auto operand = context->getAttrOprand("operand:" + std::to_string(i));
        auto tensor_type = GetOperandTensorType(dst_operand, operand);
        ss << "_tensortype" << tensor_type ;
    }
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string FusedElmwiseKernel::GetKernelBody(TContext* context) const {
    size_t nr_operands = context->getAttrInt("nr_operands");
    auto dst_operand = context->getAttrOprand("operand:" +
                                              std::to_string(nr_operands - 1));
    auto op0 = context->getAttrOprand("operand:0");
    auto src_dtype = op0.dtype;
    auto specifier = Utils::cvt_dtype_specifier(src_dtype);

    std::vector<TensorType> tensor_types;
    for (size_t i = 0; i < nr_operands - 1; i++) {
        auto operand = context->getAttrOprand("operand:" + std::to_string(i));
        auto tensor_type = GetOperandTensorType(dst_operand, operand);
        tensor_types.push_back(tensor_type);
    }

    std::vector<std::vector<std::string>> op_modes;
    auto mode_size = context->getAttrInt("modes:size");
    for (int i = 0; i < mode_size; i++) {
        auto modes = Utils::split_string(
                context->getAttrStr("modes:" + std::to_string(i)), ',');
        op_modes.push_back(modes);
    }

    std::stringstream writer;
    writer << "#include <math.h> \n";
    writer << "#include <stdbool.h> \n";
    writer << gen_dep_func();
    writer << gen_get_data_func(tensor_types, specifier);
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);

    std::string body;
    //! get the output ptr
    std::string out_memory =R"(
        ${specifier}* D_ptr = (${specifier}*)outputs[0]->ptr;
        TINYNN_ASSERT(D_ptr);
        Layout D_layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (size_t i = 0; i < D_layout.nr_dim; ++i) {
            nr_elem *= D_layout.dims[i];
        }
        )";
    body += StringTemplate::StringTemplateArgs()
                    .add("specifier", specifier)
                    .render(out_memory);
    //! get input memory ptr
    for (size_t id = 0; id < nr_operands - 1; id++) {
        std::string input_str;
        input_str += R"(
        ${specifier}* I${id}_ptr = (${specifier}*)inputs[${id}]->ptr;
        TINYNN_ASSERT(I${id}_ptr);
        Layout I${id}_layout = inputs[0]->layout;
        )";
        if (tensor_types[id] == TensorType::UNKNOWN_TENSOR_TYPE) {
            input_str += R"(
        // get the broadcast location in the given input layout
        size_t I${id}_stride[MAX_DIM];
        for (int i =0; i<D_layout.nr_dim; ++i) {
            bool non_broadcast = i<I${id}_layout.nr_dim && I${id}_layout.dims[i]==D_layout.dims[i];
            size_t stride = i<I${id}_layout.nr_dim ? $I${id}_layout.stride[i]:1;
            ${id}_stride[i] = non_broadcast ? stride: 0;
        })";
        }
        body += StringTemplate::StringTemplateArgs()
                        .add("specifier", specifier)
                        .add("id", std::to_string(id))
                        .render(input_str);
    }

    std::string compute_body = R"(
        for(size_t i = 0; i < nr_elem; i ++){
    )";
    for (size_t id = 0; id < nr_operands - 1; id++) {
        std::string input_data;
        if (tensor_types[id] != TensorType::UNKNOWN_TENSOR_TYPE) {
            input_data += R"(
            ${specifier} I${id} = get_input${id}(i, I${id}_ptr, &D_layout, NULL);
            )";
        } else {
            input_data += R"(
            ${specifier} I${id} = get_input${id}(i, I${id}_ptr, &D_layout, I${id}_stride);
            )";
        }
        compute_body += StringTemplate::StringTemplateArgs()
                        .add("specifier", specifier)
                        .add("id", std::to_string(id))
                        .render(input_data);
    }

    for (size_t id = 0; id < op_modes.size(); id++) {
        compute_body += gen_op(op_modes[id], specifier);
        compute_body += ";\n";
    }

    compute_body += R"(
        D_ptr[i] = D;
    )";

    compute_body += "}\n";

    writer << "{\n";
    writer << body;
    writer << compute_body;
    writer << R"(
        return TinyNN_SUCCESS;
    })";

    return writer.str();
}

// vim: syntax=cpp.doxygen
