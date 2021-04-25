/**
 * \file compiler/lib/CodeGen/GlobalCtx.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#pragma once
#include "mlir/IR/BuiltinOps.h"
namespace megcc {
namespace codegen {

mlir::MLIRContext* getGlobalCTX();
}
}  // namespace megcc