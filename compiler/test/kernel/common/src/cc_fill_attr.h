/**
 * \file
 * compiler/test/kernel/common/src/cc_fill_attr.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include <data_struct.h>

#include "compiler/KernelGen/KernelGen.h"
#include "megbrain/common.h"
#include "megcc_test_config.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn.h"

namespace {
using ConvParam = megdnn::ConvolutionForward::Param;
using ConvBiasParam = megdnn::ConvBiasForward::Param;

}  // namespace

namespace megcc {
namespace test {

using KernelGenRet =
        std::pair<std::vector<const ::megcc::KernelGen::KernelFunc*>,
                  const ::megcc::KernelGen::DeduceFunc*>;
using TensorNDArray = megdnn::SmallVector<megdnn::TensorND>;
template <typename Opr>
KernelGenRet opr_fill_attr(
        std::unordered_map<std::string, CCAttr>& attr_map, Opr* opr,
        const TensorNDArray& tensors, KernelGen::Arch arch,
        const std::unordered_map<std::string, CCAttr>& proxy_attr);

}  // namespace test
}  // namespace megcc