/**
 * \file compiler/include/compiler/Dialect/Kernel/IR/KernelBase.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef KERNEL_BASE
#define KERNEL_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def Kernel_Dialect : Dialect {
    let name = "Kernel";
    let cppNamespace = "::mlir::Kernel";
}

class KernelBase<string mnemonic, list<Trait> traits=[]>:
        Op<Kernel_Dialect, mnemonic, traits>;

#endif // KERNEL_BASE
