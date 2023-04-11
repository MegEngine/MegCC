#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

class TypeConverter;

void populateMGBToKernelConversionPatterns(
        TypeConverter& typeConverter, RewritePatternSet& patterns);

std::unique_ptr<OperationPass<ModuleOp>> createMGBToKernelPass();

#define GEN_PASS_REGISTRATION
#include "compiler/Conversion/MGBToKernel/Passes.h.inc"

}  // namespace mlir

// vim: syntax=cpp.doxygen
