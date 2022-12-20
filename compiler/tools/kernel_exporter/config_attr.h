/**
 * \file
 * compiler/tools/kernel_exporter/config_attr.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

//#include <data_struct.h>

#include "compiler/KernelGen/KernelGen.h"
#include "megbrain/common.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"

namespace {
using ConvParam = megdnn::ConvolutionForward::Param;
using ConvBiasParam = megdnn::ConvBiasForward::Param;
using KPT = megcc::KernelGen::KernelPack::KernType;
using KA = megcc::KernelGen::Arch;

}  // namespace

namespace megcc {
namespace exporter {

std::vector<megcc::CodeGenContext> config_attr(KPT k_type, std::string k_name,
                                               bool use_default_attr);

}  // namespace exporter
}  // namespace megcc
