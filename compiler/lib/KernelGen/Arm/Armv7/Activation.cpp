/**
 * \file
 * compiler/lib/KernelGen/Arm/Armv7/Activation.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Activation.h"

using namespace megcc;
using namespace KernelGen;
using namespace Armv7;

std::shared_ptr<ActivationGenAsmBase>
megcc::KernelGen::Armv7::create_activation_gener(std::string mode) {
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
    return "";
}
std::string ActivationGenAsmBase::GenAsmQuantStore(
        std::vector<std::string> int_regs, std::string scale_reg,
        const std::string& output_sym, const int elem_offset,
        const std::string dst_specifier,
        const std::vector<std::string> args_reg, const std::string& mode,
        bool with_store) {
    return "";
}

// vim: syntax=cpp.doxygen
