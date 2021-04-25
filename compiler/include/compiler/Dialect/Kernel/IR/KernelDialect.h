/**
 * \file compiler/include/compiler/Dialect/Kernel/IR/KernelDialect.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace OpTrait {
namespace Kernel {
// trait to mark abstract kernel
template <typename ConcreteType>
class AbstractKernelTrait
        : OpTrait::TraitBase<ConcreteType, AbstractKernelTrait> {};

}  // namespace Kernel
}  // namespace OpTrait

namespace Kernel {
class KernelDialect : public Dialect {
public:
    explicit KernelDialect(MLIRContext* context);
    static StringRef getDialectNamespace() { return "Kernel"; }
};

bool isContiguous(MemRefType memref);

}  // namespace Kernel
}  // namespace mlir

#include "compiler/Dialect/Kernel/IR/KernelInterfaces.h.inc"

#define GET_OP_CLASSES
#include "compiler/Dialect/Kernel/IR/KernelDialect.h.inc"

// vim: syntax=cpp.doxygen
