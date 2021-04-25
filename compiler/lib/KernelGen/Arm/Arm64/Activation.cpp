/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/Activation.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Activation.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;
std::shared_ptr<ActivationGenAsmBase>
megcc::KernelGen::Arm64::create_activation_gener(std::string mode) {
    if (mode == "IDENTITY") {
        return std::make_shared<ActivationGenAsm<NonlineMode::IDENTITY>>();
    } else if (mode == "H_SWISH") {
        return std::make_shared<ActivationGenAsm<NonlineMode::H_SWISH>>();
    } else if (mode == "RELU") {
        return std::make_shared<ActivationGenAsm<NonlineMode::RELU>>();
    } else if (mode == "SIGMOID") {
        //! SIGMOID should impl after matmul
        return std::make_shared<ActivationGenAsm<NonlineMode::IDENTITY>>();
    } else {
        CC_ABORT << "UNsupport NonlineMode\n";
        return nullptr;
    }
}

std::string ActivationGenAsmBase::GenAsmQuantInit(
        const std::vector<std::string> args_reg, const std::string& mode,
        const std::vector<std::string> args_ptr) {
    if (mode == "RELU") {
        CC_ASSERT(args_reg.size() >= 2);
        CC_ASSERT(args_ptr.size() >= 1);
        return StringTemplate::StringTemplateArgs()
                .add("reg_0", args_reg[0])
                .add("reg_scale", args_reg.back())
                .add("arg_ptr_scale", args_ptr.back())
                .render(R"(
            "eor ${reg_0}.16b, ${reg_0}.16b, ${reg_0}.16b\n"
            "ld1r {${reg_scale}.4s}, [%[${arg_ptr_scale}]]\n"
        )");
    } else if (mode == "H_SWISH") {
        CC_ASSERT(args_reg.size() >= 5);
        CC_ASSERT(args_ptr.size() >= 2);
        return StringTemplate::StringTemplateArgs()
                .add("reg_0", args_reg[0])
                .add("reg_3", args_reg[1])
                .add("reg_6", args_reg[2])
                .add("reg_6_inv", args_reg[3])
                .add("reg_scale", args_reg.back())
                .add("arg_ptr_inv", args_ptr[0])
                .add("arg_ptr_scale", args_ptr.back())
                .render(R"(
            "eor ${reg_0}.16b, ${reg_0}.16b, ${reg_0}.16b\n"
            "fmov ${reg_3}.4s, #3.000000000000000000e+00\n"
            "fmov ${reg_6}.4s, #6.000000000000000000e+00\n"
            "ld1r {${reg_6_inv}.4s}, [%[${arg_ptr_inv}]]\n"
            "ld1r {${reg_scale}.4s}, [%[${arg_ptr_scale}]]\n"
        )");
    } else {
        CC_ASSERT(args_reg.size() >= 1);
        CC_ASSERT(args_ptr.size() >= 1);
        CC_ASSERT(mode == "IDENTITY");
        return StringTemplate::StringTemplateArgs()
                .add("reg_scale", args_reg.back())
                .add("arg_ptr_scale", args_ptr.back())
                .render(R"(
            "ld1r {${reg_scale}.4s}, [%[${arg_ptr_scale}]]\n"
        )");
    }
    return "";
}
std::string ActivationGenAsmBase::GenAsmQuantStore(
        std::vector<std::string> int_regs, std::string scale_reg,
        const std::string& output_sym, const int elem_offset,
        const std::string dst_specifier,
        const std::vector<std::string> args_reg, const std::string& mode,
        bool with_store) {
    std::stringstream ss;
    CC_ASSERT(int_regs.size() == 1 || int_regs.size() == 2);
    if (dst_specifier == "int8_t") {
        std::string st_int_reg = int_regs[0];
        if (int_regs.size() == 1) {
            st_int_reg[st_int_reg.find_first_of("v")] = 's';
        } else {
            st_int_reg[st_int_reg.find_first_of("v")] = 'd';
        }
        std::string reg_0 = "None";
        std::string reg_3 = "None";
        std::string reg_6 = "None";
        std::string reg_6_inv = "None";
        std::string reg_t0 = "None";
        std::string reg_t1 = "None";
        if (mode == "RELU") {
            CC_ASSERT(args_reg.size() >= 1);
            reg_0 = args_reg[0];
        } else if (mode == "H_SWISH") {
            CC_ASSERT(int_regs.size() == 2) << "you need to impl int_reg == 1";
            CC_ASSERT(args_reg.size() >= 6);
            reg_0 = args_reg[0];
            reg_3 = args_reg[1];
            reg_6 = args_reg[2];
            reg_6_inv = args_reg[3];
            reg_t0 = args_reg[4];
            reg_t1 = args_reg[5];
        }
        std::stringstream temp_ss;
        temp_ss << R"(
                "scvtf  ${int_reg}.4s,   ${int_reg}.4s\n" )";
        if (int_regs.size() == 2) {
            temp_ss << R"(
                "scvtf  ${int_reg_2}.4s,   ${int_reg_2}.4s\n" )";
        }
        if (mode == "RELU") {
            temp_ss << R"(
                    "fmax  ${int_reg}.4s,   ${int_reg}.4s, ${zero_reg}.4s\n" )";
            if (int_regs.size() == 2) {
                temp_ss << R"(
                    "fmax  ${int_reg_2}.4s,   ${int_reg_2}.4s, ${zero_reg}.4s\n" )";
            }
        } else if (mode == "H_SWISH") {
            CC_ASSERT(int_regs.size() == 2);
            //! PERF: reorder below to improve perf
            temp_ss << R"(
                    "fadd  ${reg_t0}.4s,   ${int_reg}.4s, ${reg_3}.4s\n" 
                    "fadd  ${reg_t1}.4s,   ${int_reg_2}.4s, ${reg_3}.4s\n"
                    "fmax  ${reg_t0}.4s,   ${reg_t0}.4s, ${zero_reg}.4s\n" 
                    "fmax  ${reg_t1}.4s,   ${reg_t1}.4s, ${zero_reg}.4s\n"
                    "fmin  ${reg_t0}.4s,   ${reg_t0}.4s, ${reg_6}.4s\n" 
                    "fmin  ${reg_t1}.4s,   ${reg_t1}.4s, ${reg_6}.4s\n"
                    "fmul  ${int_reg}.4s,   ${reg_t0}.4s, ${int_reg}.4s\n" 
                    "fmul  ${int_reg_2}.4s,   ${reg_t1}.4s, ${int_reg_2}.4s\n"
                    "fmul  ${int_reg}.4s,   ${int_reg}.4s, ${reg_6_inv}.4s\n" 
                    "fmul  ${int_reg_2}.4s,   ${int_reg_2}.4s, ${reg_6_inv}.4s\n"
            )";

        } else {
            CC_ASSERT(mode == "IDENTITY");
        }
        if (int_regs.size() == 1) {
            temp_ss << R"(
                "fmul   ${int_reg}.4s,   ${int_reg}.4s,   ${scale_reg}.4s\n"
                "fcvtas ${int_reg}.4s,   ${int_reg}.4s\n"
                "sqxtn  ${int_reg}.4h,   ${int_reg}.4s\n"
                "sqxtn  ${int_reg}.8b,   ${int_reg}.8h\n"
            )";
            if (with_store) {
                temp_ss << R"(
                    "str    ${st_int_reg},   [%[${output_sym}], #${byte_offset}]\n"
                )";
            }
        } else {
            CC_ASSERT(int_regs.size() == 2);
            temp_ss << R"(
                "fmul   ${int_reg}.4s,   ${int_reg}.4s,   ${scale_reg}.4s\n"
                "fmul   ${int_reg_2}.4s,   ${int_reg_2}.4s,   ${scale_reg}.4s\n"
                "fcvtas ${int_reg}.4s,   ${int_reg}.4s\n"
                "fcvtas ${int_reg_2}.4s,   ${int_reg_2}.4s\n"
                "sqxtn  ${int_reg}.4h,   ${int_reg}.4s\n"
                "sqxtn  ${int_reg_2}.4h,   ${int_reg_2}.4s\n"

                "sqxtn  ${int_reg}.8b,   ${int_reg}.8h\n"
                "sqxtn  ${int_reg_2}.8b,   ${int_reg_2}.8h\n"
            )";
            if (with_store) {
                temp_ss << R"(
                    "ins    ${int_reg}.s[1], ${int_reg_2}.s[0]\n"
                    "str    ${st_int_reg},   [%[${output_sym}], #${byte_offset}]\n"
                )";
            }
        }
        auto gener = StringTemplate::StringTemplateArgs()
                             .add("int_reg", int_regs[0])
                             .add("st_int_reg", st_int_reg)
                             .add("scale_reg", scale_reg)
                             .add("output_sym", output_sym)
                             .add("byte_offset", elem_offset)
                             .add("zero_reg", reg_0)
                             .add("reg_3", reg_3)
                             .add("reg_6", reg_6)
                             .add("reg_6_inv", reg_6_inv)
                             .add("reg_t0", reg_t0)
                             .add("reg_t1", reg_t1);
        if (int_regs.size() == 2) {
            gener.add("int_reg_2", int_regs[1]);
        }
        ss << gener.render(temp_ss.str());
    } else {
        CC_ASSERT(dst_specifier == "int32_t");
        if (!with_store) {
            return "";
        }
        int byte_offset = elem_offset * sizeof(int32_t);
        std::vector<std::string> st_int_regs = int_regs;
        for (auto& st_int_reg : st_int_regs) {
            st_int_reg[st_int_reg.find_first_of("v")] = 'q';
        }
        if (int_regs.size() == 1) {
            ss << StringTemplate::StringTemplateArgs()
                            .add("int_reg", st_int_regs[0])
                            .add("output_sym", output_sym)
                            .add("byte_offset", byte_offset)
                            .render(R"(
            "str    ${int_reg}, [%[${output_sym}], #${byte_offset}]\n"
        )");
        } else {
            CC_ASSERT(int_regs.size() == 2);
            auto byte_offset_2 = byte_offset + 16;
            ss << StringTemplate::StringTemplateArgs()
                            .add("int_reg", st_int_regs[0])
                            .add("int_reg_2", st_int_regs[1])
                            .add("output_sym", output_sym)
                            .add("byte_offset", byte_offset)
                            .add("byte_offset_2", byte_offset_2)
                            .render(R"(
            "str    ${int_reg}, [%[${output_sym}], #${byte_offset}]\n"
            "str    ${int_reg_2}, [%[${output_sym}], #${byte_offset_2}]\n"
        )");
        }
    }
    return ss.str();
}
// vim: syntax=cpp.doxygen
