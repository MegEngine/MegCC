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
