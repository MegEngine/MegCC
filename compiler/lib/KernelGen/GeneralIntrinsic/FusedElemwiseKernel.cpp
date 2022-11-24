/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/FusedElemwiseKernel.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>

#include "Common/ElemwiseCommon.h"
#include "FusedElemwiseKernel.h"
#include "GIMathHelper.h"
#include "GeneralIntrinsic/GISimdHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

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

std::string gen_unary_simd_unroll(std::string mode) {
    if (mode == "RELU") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMaximum${Gitype}(simd_zero, ${in}_0);
            ${simd_specifier} ${out}_1 = GiMaximum${Gitype}(simd_zero, ${in}_1);
                )";
    } else if (mode == "EXP") {
        return R"(
            ${simd_specifier} ${out}_0 = GiExpPs${Gitype}(${in}_0);
            ${simd_specifier} ${out}_1 = GiExpPs${Gitype}(${in}_1);
        )";
    } else if (mode == "SIGMOID") {
        return R"(
            ${simd_specifier} ${out}_0 = GiSigmoidPs${Gitype}(${in}_0);
            ${simd_specifier} ${out}_1 = GiSigmoidPs${Gitype}(${in}_1);
        )";
    } else if (mode == "NEGATE") {
        return R"(
            ${simd_specifier} ${out}_0 = GiNeg${Gitype}(${in}_0);
            ${simd_specifier} ${out}_1 = GiNeg${Gitype}(${in}_1);
        )";
    } else if (mode == "H_SWISH") {
        return R"(
            ${simd_specifier} ${out}_0 = GiHSwish${Gitype}(${in}_0);
            ${simd_specifier} ${out}_1 = GiHSwish${Gitype}(${in}_1);
        )";
    } else if (mode == "ABS") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAbs${Gitype}(${in}_0);
            ${simd_specifier} ${out}_1 = GiAbs${Gitype}(${in}_1);
        )";
    } else if (mode == "LOG") {
        return R"(
            ${simd_specifier} ${out}_0 = GiLogPs${Gitype}(${in}_0);
            ${simd_specifier} ${out}_1 = GiLogPs${Gitype}(${in}_1);
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_unary_simd(std::string mode) {
    if (mode == "RELU") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMaximum${Gitype}(simd_zero, ${in}_0);
                )";
    } else if (mode == "EXP") {
        return R"(
            ${simd_specifier} ${out}_0 = GiExpPs${Gitype}(${in}_0);
        )";
    } else if (mode == "SIGMOID") {
        return R"(
            ${simd_specifier} ${out}_0 = GiSigmoidPs${Gitype}(${in}_0);
        )";
    } else if (mode == "NEGATE") {
        return R"(
            ${simd_specifier} ${out}_0 = GiNeg${Gitype}(${in}_0);
        )";
    } else if (mode == "H_SWISH") {
        return R"(
            ${simd_specifier} ${out}_0 = GiHSwish${Gitype}(${in}_0);
        )";
    } else if (mode == "ABS") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAbs${Gitype}(${in}_0);
        )";
    } else if (mode == "LOG") {
        return R"(
            ${simd_specifier} ${out}_0 = GiLogPs${Gitype}(${in}_0);
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary_simd_unroll(std::string mode) {
    if (mode == "ADD") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAdd${Gitype}(${val0}_0, ${val1}_0);
            ${simd_specifier} ${out}_1 = GiAdd${Gitype}(${val0}_1, ${val1}_1);
        )";
    } else if (mode == "SUB") {
        return R"(
            ${simd_specifier} ${out}_0 = GiSubtract${Gitype}(${val0}_0, ${val1}_0);
            ${simd_specifier} ${out}_1 = GiSubtract${Gitype}(${val0}_1, ${val1}_1);
        )";
    } else if (mode == "MUL") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMultiply${Gitype}(${val0}_0, ${val1}_0);
            ${simd_specifier} ${out}_1 = GiMultiply${Gitype}(${val0}_1, ${val1}_1);
        )";
    } else if (mode == "MAX") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMaximum${Gitype}(${val0}_0, ${val1}_0);
            ${simd_specifier} ${out}_1 = GiMaximum${Gitype}(${val0}_1, ${val1}_1);
        )";
    } else if (mode == "MIN") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMinimum${Gitype}(${val0}_0, ${val1}_0);
            ${simd_specifier} ${out}_1 = GiMinimum${Gitype}(${val0}_1, ${val1}_1);
        )";
    } else if (mode == "TRUE_DIV") {
        return R"(
            ${simd_specifier} ${out}_0 = GiDivide${Gitype}(${val0}_0, ${val1}_0);
            ${simd_specifier} ${out}_1 = GiDivide${Gitype}(${val0}_1, ${val1}_1);
        )";
    } else if (mode == "FUSE_ADD_RELU") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMaximum${Gitype}(GiAdd${Gitype}(${val0}_0, ${val1}_0), simd_zero);
            ${simd_specifier} ${out}_1 = GiMaximum${Gitype}(GiAdd${Gitype}(${val0}_1, ${val1}_1), simd_zero);
        )";
    } else if (mode == "FUSE_ADD_SIGMOID") {
        return R"(
            ${simd_specifier} ${out}_0 = GiSigmoidPs${Gitype}(GiAdd${Gitype}(${val0}_0, ${val1}_0));
            ${simd_specifier} ${out}_1 = GiSigmoidPs${Gitype}(GiAdd${Gitype}(${val0}_1, ${val1}_1));
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary_simd(std::string mode) {
    if (mode == "ADD") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAdd${Gitype}(${val0}_0, ${val1}_0);
        )";
    } else if (mode == "SUB") {
        return R"(
            ${simd_specifier} ${out}_0 = GiSubtract${Gitype}(${val0}_0, ${val1}_0);
        )";
    } else if (mode == "MUL") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMultiply${Gitype}(${val0}_0, ${val1}_0);
        )";
    } else if (mode == "MAX") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMaximum${Gitype}(${val0}_0, ${val1}_0);
        )";
    } else if (mode == "MIN") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMinimum${Gitype}(${val0}_0, ${val1}_0);
        )";
    } else if (mode == "TRUE_DIV") {
        return R"(
            ${simd_specifier} ${out}_0 = GiDivide${Gitype}(${val0}_0, ${val1}_0);
        )";
    } else if (mode == "FUSE_ADD_RELU") {
        return R"(
            ${simd_specifier} ${out}_0 = GiMaximum${Gitype}(GiAdd${Gitype}(${val0}_0, ${val1}_0), simd_zero);
        )";
    } else if (mode == "FUSE_ADD_SIGMOID") {
        return R"(
            ${simd_specifier} ${out}_0 = GiSigmoidPs${Gitype}(GiAdd${Gitype}(${val0}_0, ${val1}_0));
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary(std::string mode) {
    if (mode == "ADD") {
        return " ${specifier} ${out} = ${val0} + ${val1}";
    } else if (mode == "SUB") {
        return "${specifier} ${out} = ${val0} - ${val1}";
    } else if (mode == "MUL") {
        return "${specifier} ${out} = ${val0} * ${val1}";
    } else if (mode == "MAX") {
        return "${specifier} ${out} =(${val0} > ${val1} ? ${val0} : ${val1})";
    } else if (mode == "MIN") {
        return "${specifier} ${out} =(${val0} < ${val1} ? ${val0} : ${val1})";
    } else if (mode == "TRUE_DIV") {
        return "${specifier} ${out} =  ${val0} / ${val1}";
    } else if (mode == "FUSE_ADD_RELU") {
        return "${specifier} ${out} =(${val0} + ${val1}) > 0? (${val0} + "
               "${val1}):0";
    } else if (mode == "FUSE_ADD_SIGMOID") {
        return "${specifier} ${out} =1.f/(1.f+ expf(-(${val0} + ${val1})))";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_tenary_simd_unroll(std::string mode) {
    if (mode == "FUSE_MUL_ADD3") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAdd${Gitype}(GiMultiply${Gitype}(${val0}_0, ${val1}_0), ${val2}_0);
            ${simd_specifier} ${out}_1 = GiAdd${Gitype}(GiMultiply${Gitype}(${val0}_1, ${val1}_1), ${val2}_1);
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_tenary_simd(std::string mode) {
    if (mode == "FUSE_MUL_ADD3") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAdd${Gitype}(GiMultiply${Gitype}(${val0}_0, ${val1}_0), ${val2}_0);
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_tenary(std::string mode) {
    if (mode == "FUSE_MUL_ADD3") {
        return " ${specifier} ${out} = ${val0} * ${val1} + ${val2}";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_quater_simd_unroll(std::string mode) {
    if (mode == "FUSE_MUL_ADD4") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAdd${Gitype}(GiMultiply${Gitype}(${val0}_0, ${val1}_0), GiMultiply${Gitype}(${val2}_0, ${val3}_0));
            ${simd_specifier} ${out}_1 = GiAdd${Gitype}(GiMultiply${Gitype}(${val0}_1, ${val1}_1), GiMultiply${Gitype}(${val2}_1, ${val3}_1));
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_quater_simd(std::string mode) {
    if (mode == "FUSE_MUL_ADD4") {
        return R"(
            ${simd_specifier} ${out}_0 = GiAdd${Gitype}(GiMultiply${Gitype}(${val0}_0, ${val1}_0), GiMultiply${Gitype}(${val2}_0, ${val3}_0));
        )";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_quater(std::string mode) {
    if (mode == "FUSE_MUL_ADD4") {
        return " ${specifier} ${out} = ${val0} * ${val1} + ${val2} * ${val3}";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_init(std::shared_ptr<GISimdHelper> simd_helper) {
    auto ins = simd_helper->get_dupq_n_symbol();
    std::string init = "${simd_specifier} simd_zero = " + ins + "(0);";
    auto templateS = StringTemplate::StringTemplateArgs();
    return templateS
            .add("simd_specifier", simd_helper->get_specifier_q_symbol())
            .render(init);
}

std::vector<std::string> gen_op(std::vector<std::string> mode,
                                std::shared_ptr<GISimdHelper> simd_helper,
                                const std::string& specifier) {
    size_t nr_str = mode.size();
    CC_ASSERT(nr_str >= 3);
    auto templateS = StringTemplate::StringTemplateArgs();
    templateS.add("specifier", specifier)
            .add("simd_specifier", simd_helper->get_specifier_q_symbol())
            .add("Gitype", simd_helper->get_dtype_sym());
    std::vector<std::string> ret;
    if (nr_str == 3) {
        templateS.add("in", mode[0]).add("out", mode[2]);
        ret.push_back(templateS.render(gen_unary_simd_unroll(mode[1])));
        ret.push_back(templateS.render(gen_unary_simd(mode[1])));
        ret.push_back(templateS.render(gen_unary(mode[1])));
    } else if (nr_str == 4) {
        templateS.add("val0", mode[0]).add("val1", mode[1]).add("out", mode[3]);
        ret.push_back(templateS.render(gen_binary_simd_unroll(mode[2])));
        ret.push_back(templateS.render(gen_binary_simd(mode[2])));
        ret.push_back(templateS.render(gen_binary(mode[2])));
    } else if (nr_str == 5) {
        templateS.add("val0", mode[0])
                .add("val1", mode[1])
                .add("val2", mode[2])
                .add("out", mode[4]);
        ret.push_back(templateS.render(gen_tenary_simd_unroll(mode[3])));
        ret.push_back(templateS.render(gen_tenary_simd(mode[3])));
        ret.push_back(templateS.render(gen_tenary(mode[3])));
    } else if (nr_str == 6) {
        templateS.add("val0", mode[0])
                .add("val1", mode[1])
                .add("val2", mode[2])
                .add("val3", mode[3])
                .add("out", mode[5]);
        ret.push_back(templateS.render(gen_quater_simd_unroll(mode[4])));
        ret.push_back(templateS.render(gen_quater_simd(mode[4])));
        ret.push_back(templateS.render(gen_quater(mode[4])));
    } else {
        CC_ABORT << "Not support mode in FusedElemwise\n";
    }
    return ret;
}

template <TensorType>
std::string gen_one_get_data(size_t id,
                             std::shared_ptr<GISimdHelper> simd_helper,
                             const std::string& specifier);

template <>
std::string gen_one_get_data<SCALAR>(size_t id,
                                     std::shared_ptr<GISimdHelper> simd_helper,
                                     const std::string& specifier) {
    std::string get_data = R"(
       static inline ${simd_specifier} get_input${id}(${specifier}* ptr, size_t c) {
           return ${gi_broadcast}(*ptr);
       }
    )";
    std::string get_naive_data = R"(
       static inline ${specifier} get_naive_input${id}(${specifier}* ptr, size_t c) {
           return *(ptr);
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("simd_specifier", simd_helper->get_specifier_q_symbol())
            .add("specifier", specifier)
            .add("gi_broadcast", simd_helper->get_dupq_n_symbol())
            .add("id", std::to_string(id))
            .render(get_data + get_naive_data);
}

template <>
std::string gen_one_get_data<VECTOR>(size_t id,
                                     std::shared_ptr<GISimdHelper> simd_helper,
                                     const std::string& specifier) {
    std::string get_data = R"(
       static inline ${simd_specifier} get_input${id}(size_t elem_id, ${specifier}* ptr) {
           return ${gi_load}(ptr + elem_id);
       }
    )";
    std::string get_naive_data = R"(
       static inline ${specifier} get_naive_input${id}(size_t elem_id, ${specifier}* ptr) {
           return *(ptr + elem_id);
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("simd_specifier", simd_helper->get_specifier_q_symbol())
            .add("specifier", specifier)
            .add("gi_load", simd_helper->get_ld1q_symbol())
            .add("id", std::to_string(id))
            .render(get_data + get_naive_data);
}

template <>
std::string gen_one_get_data<BCAST101>(
        size_t id, std::shared_ptr<GISimdHelper> simd_helper,
        const std::string& specifier) {
    std::string get_data = R"(
       static inline ${simd_specifier} get_input${id}(${specifier}* ptr, 
                                        size_t channel) {
           return ${gi_broadcast}(*(ptr + channel));
       }
    )";
    std::string get_naive_data = R"(
       static inline ${specifier} get_naive_input${id}(${specifier}* ptr, 
                                        size_t channel) {
           return *(ptr + channel);
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("simd_specifier", simd_helper->get_specifier_q_symbol())
            .add("gi_broadcast", simd_helper->get_dupq_n_symbol())
            .add("specifier", specifier)
            .add("id", std::to_string(id))
            .render(get_data + get_naive_data);
}

template <>
std::string gen_one_get_data<BCAST101x4>(
        size_t id, std::shared_ptr<GISimdHelper> simd_helper,
        const std::string& specifier) {
    std::string get_data = R"(
       static inline ${simd_specifier} get_input${id}(${specifier}* ptr,
                                     size_t channel) {
           return ${gi_load}(ptr + channel * 4);
       }
    )";
    std::string get_naive_data = R"(
       static inline ${specifier} get_naive_input${id}(${specifier}* ptr,
                                     size_t channel) {
           return *(ptr + channel * 4);
       }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("simd_specifier", simd_helper->get_specifier_q_symbol())
            .add("specifier", specifier)
            .add("gi_load", simd_helper->get_ld1q_symbol())
            .add("id", std::to_string(id))
            .render(get_data + get_naive_data);
}

std::string gen_get_data_func(std::vector<TensorType> tensor_types,
                              std::shared_ptr<GISimdHelper> simd_helper,
                              const std::string& specifier) {
    size_t nr_operands = tensor_types.size();
    std::string functions;
    for (size_t i = 0; i < nr_operands; i++) {
        auto tensor_type = tensor_types[i];
        switch (tensor_type) {
            case TensorType::SCALAR:
                functions +=
                        gen_one_get_data<SCALAR>(i, simd_helper, specifier);
                break;
            case TensorType::VECTOR:
                functions +=
                        gen_one_get_data<VECTOR>(i, simd_helper, specifier);
                break;
            case TensorType::BCAST101:
                functions +=
                        gen_one_get_data<BCAST101>(i, simd_helper, specifier);
                break;
            case TensorType::BCAST101x4:
                functions +=
                        gen_one_get_data<BCAST101x4>(i, simd_helper, specifier);
                break;
            default:
                CC_ABORT << "Not support tensor type in fused elemwise\n";
        }
    }
    return functions;
}

}  // namespace

std::string FusedElmwiseKernel::GetKernelBody(TContext* context) const {
    size_t nr_operands = context->getAttrInt("nr_operands");
    auto dst_operand = context->getAttrOprand("operand:" +
                                              std::to_string(nr_operands - 1));
    auto op0 = context->getAttrOprand("operand:0");
    auto src_dtype = op0.dtype;
    auto specifier = Utils::cvt_dtype_specifier(src_dtype);

    auto dtype = Utils::get_dtype_enum(context->getAttrOprand("operand:0").dtype);

    std::vector<TensorType> tensor_types;
    for (size_t i = 0; i < nr_operands - 1; i++) {
        auto operand = context->getAttrOprand("operand:" + std::to_string(i));
        auto tensor_type = GetOperandTensorType(dst_operand, operand);
        CC_ASSERT(tensor_type != TensorType::UNKNOWN_TENSOR_TYPE)
                << "Now not support broadcast type in FusedElemwise Kernel\n";
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
    writer << R"(#include "gi_float.h" )";
    writer << R"(
                #include "gi_int.h" 
                )";

    GIMathHelper gi_math;
    bool exp = false;
    bool sigmoid = false;
    bool hswith = false;
    for (size_t id = 0; id < op_modes.size(); id++) {
        auto modes = op_modes[id];
        size_t nr_str = modes.size();
        auto mode = modes[nr_str - 2];
        if ((mode == "EXP" || mode == "SIGMOID") && !exp) {
            writer << gi_math.GiExpPsFloat32() << "\n";
            exp = true;
        }
        if (("SIGMOID" == mode || "FUSE_ADD_SIGMOID" == mode) && !sigmoid) {
            writer << gi_math.GiSigmoidPsFloat32() << "\n";
            sigmoid = true;
        }
        if ("H_SWISH" == mode && !hswith) {
            writer << gi_math.GiHSwishFloat32() << "\n";
            writer << gen_dep_func() << "\n";
            hswith = true;
        }
    }

    auto gi_simd_type = std::make_shared<GISimdHelper>(src_dtype);
    auto templateS = StringTemplate::StringTemplateArgs();
    templateS.add("specifier", specifier)
            .add("simd_specifier", gi_simd_type->get_specifier_q_symbol());

    writer << gen_get_data_func(tensor_types, gi_simd_type, specifier);
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);

    std::string body;
    //! get the output ptr
    std::string out_memory = R"(
        ${specifier}* D_ptr = (${specifier}*)outputs[0]->ptr;
        TINYNN_ASSERT(D_ptr);
        Layout D_layout = outputs[0]->layout;
        size_t batch = D_layout.dims[0];
        size_t channel = D_layout.nr_dim > 1? D_layout.dims[1]: 1;
        size_t channel_elem = D_layout.nr_dim > 1? D_layout.stride[1]: 1;
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
        )";
        body += templateS.add("id", std::to_string(id)).render(input_str);
    }

    std::string compute_body = R"(
        size_t elem_id=0;
        ${init_simd}
        for(size_t b = 0; b < batch; b++) {
            for(size_t c = 0; c < channel; c++) {
                ${simd_load_channel}
                ${naive_load_channel}
                size_t simd_len = ${simd_len};
                size_t id = 0;
                for(; id + 2*(simd_len) -1 < channel_elem; id+=2*simd_len) {
                    ${simd_unroll_load}
                    ${simd_unroll_compute}
                    ${simd_unroll_store}
                    elem_id += 2 * simd_len;
                }
                for(; id + simd_len -1 < channel_elem; id+=simd_len) {
                    ${simd_load}
                    ${simd_compute}
                    ${simd_store}
                    elem_id += simd_len;
                }
                for(; id < channel_elem; id++){
                    ${naive_load}
                    ${naive_compute}
                    ${naive_store}
                    elem_id += 1;
                }
            }
        }
    )";
    std::string simd_load_channel;
    std::string simd_unroll_load;
    std::string simd_load;
    std::string naive_load;
    std::string naive_load_channel;
    for (size_t id = 0; id < nr_operands - 1; id++) {
        if (tensor_types[id] == TensorType::VECTOR) {
            std::string tmp_simd_unroll = R"(
            ${simd_specifier} I${id}_0 = get_input${id}(elem_id, I${id}_ptr);
            ${simd_specifier} I${id}_1 = get_input${id}(elem_id + 4, I${id}_ptr);
            )";
            simd_unroll_load += templateS.add("id", std::to_string(id))
                                        .render(tmp_simd_unroll);
            std::string tmp_simd = R"(
            ${simd_specifier} I${id}_0 = get_input${id}(elem_id, I${id}_ptr);
            )";
            simd_load +=
                    templateS.add("id", std::to_string(id)).render(tmp_simd);
            std::string tmp_naive = R"(
            ${specifier} I${id} = get_naive_input${id}(elem_id, I${id}_ptr);
            )";
            naive_load +=
                    templateS.add("id", std::to_string(id)).render(tmp_naive);
        } else {
            std::string tmp_simd_unroll = R"(
            ${simd_specifier} I${id}_0 = get_input${id}(I${id}_ptr, c);
            ${simd_specifier} I${id}_1 = get_input${id}(I${id}_ptr, c);
            )";
            simd_load_channel += templateS.add("id", std::to_string(id))
                                         .render(tmp_simd_unroll);
            std::string tmp_naive = R"(
            ${specifier} I${id} = get_naive_input${id}(I${id}_ptr, c);
            )";
            naive_load_channel += templateS.add("id", std::to_string(id))
                                          .render(tmp_naive);
        }
    }
    std::string simd_unroll_compute;
    std::string simd_compute;
    std::string naive_compute;
    for (size_t id = 0; id < op_modes.size(); id++) {
        auto op = gen_op(op_modes[id], gi_simd_type, specifier);
        CC_ASSERT(op.size()==3);
        naive_compute += op[2] + ";\n";
        simd_compute += op[1] + ";\n";
        simd_unroll_compute += op[0] + ";\n";
    }
    std::string simd_init = gen_init(gi_simd_type);
    std::string simd_unroll_store = gi_simd_type->get_st1q_symbol() +
                                    "(D_ptr+ elem_id, D_0);\n" +
                                    gi_simd_type->get_st1q_symbol() +
                                    "(D_ptr+ elem_id + simd_len, D_1);\n";
    std::string simd_store =
            gi_simd_type->get_st1q_symbol() + "(D_ptr+elem_id, D_0);\n";
    std::string naive_store = R"(
        D_ptr[elem_id] = D;
    )";

    writer << "{\n";
    writer << body;
    writer << templateS.add("init_simd", simd_init)
                      .add("simd_load_channel", simd_load_channel)
                      .add("naive_load_channel", naive_load_channel)
                      .add("simd_unroll_load", simd_unroll_load)
                      .add("simd_unroll_compute", simd_unroll_compute)
                      .add("simd_unroll_store", simd_unroll_store)
                      .add("simd_load", simd_load)
                      .add("simd_compute", simd_compute)
                      .add("simd_store", simd_store)
                      .add("naive_load", naive_load)
                      .add("naive_compute", naive_compute)
                      .add("naive_store", naive_store)
                      .add("simd_len", gi_simd_type->get_nr_elem_q())
                      .render(compute_body);
    writer << R"(
        return TinyNN_SUCCESS;
    })";

    return writer.str();
}

bool FusedElmwiseKernel::IsAvailable(TContext* context) const {
    auto mode_size = context->getAttrInt("modes:size");
    bool mode_ok = true;
    for (int i = 0; i < mode_size; i++) {
        auto operand = context->getAttrOprand("operand:" + std::to_string(i));
        auto modes = Utils::split_string(
                context->getAttrStr("modes:" + std::to_string(i)), ',');
        size_t modes_size = modes.size();
        auto mode = modes[modes_size - 2];
        bool mode_ok_unary = mode == "RELU" || mode == "SIGMOID" ||
                             mode == "EXP" || mode == "H_SWISH" ||
                             mode == "NEGATE";
        bool mode_ok_binary = mode == "ADD" || mode == "SUB" || mode == "MUL" ||
                              mode == "MAX" || mode == "MIN" ||
                              mode == "TRUE_DIV" || mode == "FUSE_ADD_RELU" ||
                              mode == "FUSE_ADD_SIGMOID";
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


// vim: syntax=cpp.doxygen
