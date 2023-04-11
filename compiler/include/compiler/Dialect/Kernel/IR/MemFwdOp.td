#ifndef VIEW_LIKE_OP
#define VIEW_LIKE_OP

include "KernelBase.td"
include "KernelInterfaces.td"

class MemFwdBase<string mnemonic, dag extraArgs=(ins), list<Trait> traits=[]>:
    KernelBase<mnemonic, !listconcat(traits, [
        NoSideEffect,
        ViewLikeOpInterface,
        DeclareOpInterfaceMethods<MemFwdInterface>
    ])> {

    let arguments = !con(
        (ins AnyMemRef:$input,
             DefaultValuedAttr<BoolAttr, "false">:$determined),
        extraArgs
    );

    let results = (outs
        AnyMemRef
    );

    let extraClassDeclaration = [{
        Value getViewSource() { return input(); }
    }];
}

def Reshape: MemFwdBase<"Reshape">;

def Dimshuffle: MemFwdBase<"Dimshuffle", (ins I32ArrayAttr:$pattern)>;

def Subtensor: MemFwdBase<"Subtensor", (ins ArrayAttr:$descs,
        ArrayAttr:$flags)>;

#endif // VIEW_LIKE_OP
