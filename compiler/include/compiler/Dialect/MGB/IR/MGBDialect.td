/**
 * \file compiler/include/compiler/Dialect/MGB/IR/MGBDialect.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_DIALECT
#define MGB_DIALECT

include "megbrain/IR/ops.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ParamStorage: Op<Mgb_Dialect, "ParamStorage", [Symbol]> {
  let arguments = (ins
    StrAttr:$sym_name,
    ElementsAttr:$value,
    TypeAttr:$type,
    I32Attr:$user_count
  );
}

def ParamProvider: Op<Mgb_Dialect, "ParamProvider",
      [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoSideEffect]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyType);
  let builders = [
    OpBuilder<(ins "ParamStorage":$storage),
    [{ build($_builder, $_state, storage.type(), storage.sym_name()); }]>];
}

#endif // MGB_DIALECT
