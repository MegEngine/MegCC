/**
 * \file compiler/include/compiler/Dialect/Kernel/IR/KernelDialect.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef KERNEL_DIALECT
#define KERNEL_DIALECT

include "KernelBase.td"
include "AbstractKernels.td"
include "MemFwdOp.td"

def MemPlan: KernelBase<"MemPlan", [ViewLikeOpInterface, NoSideEffect]> {
    let arguments = (ins
        AnyMemRef:$global_storage
    );
    let results = (outs
        AnyMemRef
    );

    let extraClassDeclaration = [{
        Value getViewSource() { return global_storage(); }
    }];
}

//! TODO: add some trait for auto insert memory deallocate when execute BufferDeallocationPass
def DynamicAlloc: KernelBase<"DynamicAlloc"> {
    let results = (outs 
        Res<AnyMemRef, "", [MemAlloc<DefaultResource>]>:$memref
    );
}

def WeightStorage: KernelBase<"WeightStorage", [Symbol]> {
  let arguments = (ins
    StrAttr:$sym_name,
    ElementsAttr:$value,
    TypeAttr:$type,
    I32Attr:$user_count
  );
}

def GetWeight: KernelBase<"GetWeight",
      [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoSideEffect]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyMemRef);
}

def RawCodeKernelDef: KernelBase<"KernelDef", [
        IsolatedFromAbove,
        Symbol
    ]> {
    let arguments = (ins
        StrAttr:$sym_name, // referenced by KernelCall
        StrAttr:$signature,
        StrAttr:$body,

        StrAttr:$init_sym_name,
        StrAttr:$init_signature,
        StrAttr:$init_body,

        StrAttr:$guard_begin, // macro begin to guard arm64/armv7 kernel
        StrAttr:$guard_end, // macro end

        StrAttr:$deduce_sym_name,
        StrAttr:$deduce_sig,
        StrAttr:$deduce_body,

        BoolAttr:$internal_call
    );
}

def KernelCall: KernelBase<"KernelCall", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        AttrSizedOperandSegments
    ]> {
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$operands,
        Arg<Variadic<AnyMemRef>, "", [MemWrite]>:$results,
        Arg<Optional<AnyMemRef>, "", [MemWrite]>:$workspace,
        DictionaryAttr:$attrMap,
        BoolAttr:$dynamic_shape
    );
}

def ExternOpr: KernelBase<"ExternOpr", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        AttrSizedOperandSegments
    ]> { 
    let arguments = (ins
    Arg<Variadic<AnyMemRef>, "", [MemRead]>:$operands,
    Arg<Variadic<AnyMemRef>, "", [MemWrite]>:$results,
    StrAttr:$name,
    StrAttr:$data,
    UI32Attr:$data_len
  );
}

class InstructBase<string mnemonic, dag extraArgs=(ins), list<Trait> traits=[]>:
    KernelBase<mnemonic, !listconcat(traits, [
        NoSideEffect
    ])> {

    let arguments = !con(
        (ins
        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$operands,
        Arg<AnyMemRef, "", [MemWrite]>:$result),
        extraArgs
    );
}

def SubtensorIns: InstructBase<"SubtensorIns", (ins ArrayAttr:$descs,
        ArrayAttr:$flags)>;

def SetSubtensorIns: InstructBase<"SetSubtensorIns", (ins ArrayAttr:$descs,
        ArrayAttr:$flags)>;

def DimshuffleIns: InstructBase<"DimshuffleIns", (ins ArrayAttr:$pattern)>;

def ConcatIns: InstructBase<"ConcatIns", (ins I32Attr:$axis)>;

def ReshapeIns: InstructBase<"ReshapeIns">;

def BroadcastIns: InstructBase<"BroadcastIns">;

def GetVarShapeIns: InstructBase<"GetVarShapeIns">;

def ArithmeticIns: InstructBase<"ArithmeticIns", (ins StrAttr:$mode) >;

def WarpPerspectiveIns: InstructBase<"WarpPerspectiveIns", (ins
        StrAttr:$imode,
        StrAttr:$bmode,
        StrAttr:$format,
        F32Attr:$border_val,
        ArrayAttr:$mat_idx)>;

def IndexingMultiAxisVecIns: InstructBase<"IndexingMultiAxisVecIns", (ins
        ArrayAttr:$axis)>;

def TypeCvtIns: InstructBase<"TypeCvtIns", (ins
        StrAttr:$i_dtype,
        StrAttr:$o_dtype,

        DefaultValuedAttr<F32Attr, "1.0">:$i_scale,
        DefaultValuedAttr<F32Attr, "1.0">:$o_scale,
        DefaultValuedAttr<UI8Attr, "0">:$i_zero,
        DefaultValuedAttr<UI8Attr, "0">:$o_zero
)>;

#endif // KERNEL_DIALECT
