/**
 * \file compiler/include/compiler/Conversion/MGBToKernel/MGBToKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

class TypeConverter;

void populateMGBToKernelConversionPatterns(TypeConverter& typeConverter,
                                           RewritePatternSet& patterns);

std::unique_ptr<OperationPass<ModuleOp>> createMGBToKernelPass();

#define GEN_PASS_REGISTRATION
#include "compiler/Conversion/MGBToKernel/Passes.h.inc"

}  // namespace mlir

// vim: syntax=cpp.doxygen
