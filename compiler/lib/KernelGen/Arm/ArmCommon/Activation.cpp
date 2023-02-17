/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Activation.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Activation.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

std::shared_ptr<ActivationGenIntrinsicBase> megcc::KernelGen::ArmCommon::
        create_activation_gener_instrinsic(std::string mode) {
    if (mode == "IDENTITY") {
        return std::make_shared<ActivationGenIntrinsic<NonlineMode::IDENTITY>>();
    } else if (mode == "RELU") {
        return std::make_shared<ActivationGenIntrinsic<NonlineMode::RELU>>();
    } else if (mode == "H_SWISH") {
        return std::make_shared<ActivationGenIntrinsic<NonlineMode::H_SWISH>>();
    } else {
        CC_ABORT << "UNsupport NonlineMode\n";
        return nullptr;
    }
}

// vim: syntax=cpp.doxygen
