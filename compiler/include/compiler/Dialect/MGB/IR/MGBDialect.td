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
