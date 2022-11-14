/**
 * \file compiler/include/compiler/Dialect/Kernel/IR/AbstractKernels.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef ABSTRACT_KERNELS
#define ABSTRACT_KERNELS

include "KernelBase.td"
include "KernelInterfaces.td"

def AbstractKernelTrait: NativeOpTrait<"Kernel::AbstractKernelTrait">;

class AbstractKernelBase<string mnemonic, list<Trait> traits=[]>:
    KernelBase<mnemonic, !listconcat(traits, [
            AbstractKernelTrait,
            LayoutConstraintInterface])>;

class GenericUnaryKernel<string mnemonic, list<Trait> traits=[]>:
        AbstractKernelBase<mnemonic, traits> {
    let arguments = (ins
        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

class GenericBinaryKernel<string mnemonic, list<Trait> traits=[]>:
        AbstractKernelBase<mnemonic, traits> {
    let arguments = (ins
        Arg<AnyMemRef, "", [MemRead]>:$lhs,
        Arg<AnyMemRef, "", [MemRead]>:$rhs,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

class GenericTernaryKernel<string mnemonic, list<Trait> traits=[]>:
        AbstractKernelBase<mnemonic, traits> {
    let arguments = (ins
        Arg<AnyMemRef, "", [MemRead]>:$x,
        Arg<AnyMemRef, "", [MemRead]>:$y,
        Arg<AnyMemRef, "", [MemRead]>:$z,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

class GenericQuaterKernel<string mnemonic, list<Trait> traits=[]>:
        AbstractKernelBase<mnemonic, traits> {
    let arguments = (ins
        Arg<AnyMemRef, "", [MemRead]>:$x,
        Arg<AnyMemRef, "", [MemRead]>:$y,
        Arg<AnyMemRef, "", [MemRead]>:$z,
        Arg<AnyMemRef, "", [MemRead]>:$w,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

class UnaryElemwiseKernel<string mode>: GenericUnaryKernel<mode, [ElemwiseKernelInterface]>;
class BinaryElemwiseKernel<string mode>: GenericBinaryKernel<mode, [ElemwiseKernelInterface]>;
class TernaryElemwiseKernel<string mode>: GenericTernaryKernel<mode, [ElemwiseKernelInterface]>;
class QuaterElemwiseKernel<string mode>: GenericQuaterKernel<mode, [ElemwiseKernelInterface]>;

def ReluKernel: UnaryElemwiseKernel<"RELU">;
def SigmoidKernel: UnaryElemwiseKernel<"SIGMOID">;
def LogKernel: UnaryElemwiseKernel<"LOG">;
def AbsKernel: UnaryElemwiseKernel<"ABS">;
def HswishKernel: UnaryElemwiseKernel<"H_SWISH">;
def ExpKernel: UnaryElemwiseKernel<"EXP">;
def NegateKernel: UnaryElemwiseKernel<"NEGATE">;
def RoundKernel: UnaryElemwiseKernel<"ROUND">;
def SILUKernel: UnaryElemwiseKernel<"SILU">;

def AddKernel: BinaryElemwiseKernel<"ADD">;
def SubKernel: BinaryElemwiseKernel<"SUB">;
def MulKernel: BinaryElemwiseKernel<"MUL">;
def TrueDivKernel: BinaryElemwiseKernel<"TRUE_DIV">;
def MaxKernel: BinaryElemwiseKernel<"MAX">;
def MinKernel: BinaryElemwiseKernel<"MIN">;
def LeqKernel: BinaryElemwiseKernel<"LEQ">;
def LtKernel: BinaryElemwiseKernel<"LT">;
def FloorDivKernel: BinaryElemwiseKernel<"FLOOR_DIV">;
def ModKernel: BinaryElemwiseKernel<"MOD">;
def EqualKernel: BinaryElemwiseKernel<"EQ">;

def FuseAddReluKernel: BinaryElemwiseKernel<"FUSE_ADD_RELU">;
def FuseAddTanhKernel: BinaryElemwiseKernel<"FUSE_ADD_TANH">;
def FuseAddSigmoidKernel: BinaryElemwiseKernel<"FUSE_ADD_SIGMOID">;

def FuseMulAdd3Kernel: TernaryElemwiseKernel<"FUSE_MUL_ADD3">;

def FuseMulAdd4Kernel: QuaterElemwiseKernel<"FUSE_MUL_ADD4">;

def FusedElemwiseKernel:AbstractKernelBase<"FusedElemwise"> {
   let arguments = (ins
        ArrayAttr:$modes,
        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def Conv2DKernel: AbstractKernelBase<"Conv2d", [AttrSizedOperandSegments]> {
    let arguments = (ins
        StrAttr:$mode,
        UI32Attr:$stride_h,
        UI32Attr:$stride_w,
        UI32Attr:$pad_h,
        UI32Attr:$pad_w,
        UI32Attr:$dilate_h,
        UI32Attr:$dilate_w,
        StrAttr:$sparse,
        StrAttr:$format,
        StrAttr:$compute_mode,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemRead]>:$weight,
        Arg<Optional<AnyMemRef>, "", [MemRead]>:$bias,
        Arg<Optional<AnyMemRef>, "", [MemRead]>:$z,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def ConvBackDataKernel: AbstractKernelBase<"ConvBackData"> {
    let arguments = (ins
        StrAttr:$mode,
        UI32Attr:$stride_h,
        UI32Attr:$stride_w,
        UI32Attr:$pad_h,
        UI32Attr:$pad_w,
        UI32Attr:$dilate_h,
        UI32Attr:$dilate_w,
        StrAttr:$sparse,
        StrAttr:$format,
        StrAttr:$compute_mode,

        Arg<AnyMemRef, "", [MemRead]>:$filter,
        Arg<AnyMemRef, "", [MemRead]>:$diff,
        Arg<AnyMemRef, "", [MemWrite]>:$grad
    );
}

def Pooling2DKernel: AbstractKernelBase<"Pool2d"> {
    let arguments = (ins
        StrAttr:$mode,
        UI32Attr:$pad_h,
        UI32Attr:$pad_w,
        UI32Attr:$stride_h,
        UI32Attr:$stride_w,
        UI32Attr:$window_h,
        UI32Attr:$window_w,
        StrAttr:$format,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def ResizeKernel: AbstractKernelBase<"Resize"> {
    let arguments = (ins
        StrAttr:$imode,
        StrAttr:$format,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def IndexingMultiAxisVecKernel: AbstractKernelBase<"IndexingMultiAxisVec"> {
    let arguments = (ins
        ArrayAttr:$axis,

        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$inputs,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}
def IndexingOneHotKernel: AbstractKernelBase<"IndexingOneHot"> {
    let arguments = (ins
        I32Attr:$axis,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemRead]>:$index,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def ArgmaxKernel: AbstractKernelBase<"Argmax"> {
    let arguments = (ins
        I32Attr:$axis,
        
        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def ArgsortKernel: AbstractKernelBase<"ArgSort"> {
    let arguments = (ins
        StrAttr:$order,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<Variadic<AnyMemRef>, "", [MemWrite]>:$outputs    
    );
}

def TopkKernel: AbstractKernelBase<"Topk"> {
    let arguments = (ins
        StrAttr:$mode,
        I32Attr:$k,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<Variadic<AnyMemRef>, "", [MemWrite]>:$outputs    
    );
}

def MatrixInvKernel: AbstractKernelBase<"MatrixInv"> {
    let arguments = (ins
        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}
def ConcatKernel: AbstractKernelBase<"Concat"> {
    let arguments = (ins
        I32Attr:$axis,
        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}


def WarpAffineKernel: AbstractKernelBase<"WarpAffine"> {
    let arguments = (ins
        StrAttr:$imode,
        StrAttr:$border_mode,
        StrAttr:$format,
        F32Attr:$border_val,

        Arg<AnyMemRef, "", [MemRead]>:$src,
        Arg<AnyMemRef, "", [MemRead]>:$mat,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def PowCKernel: AbstractKernelBase<"PowC"> {
    let arguments = (ins
        F32Attr:$exp,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

class MatrixMulKernelBase<string mnemonic, list<Trait> traits=[]>:
        AbstractKernelBase<mnemonic, traits> {
    let arguments = (ins
        BoolAttr:$transposeA,
        BoolAttr:$transposeB,
        StrAttr:$compute_mode,
        StrAttr:$format,

        Arg<AnyMemRef, "", [MemRead]>:$A,
        Arg<AnyMemRef, "", [MemRead]>:$B,
        Arg<AnyMemRef, "", [MemWrite]>:$C
    );
}

def MatrixMulKernel: MatrixMulKernelBase<"Matmul">;

def BatchedMatrixMulKernel: MatrixMulKernelBase<"BatchedMatmul">;

def ReduceKernel: AbstractKernelBase<"Reduce"> {
    let arguments = (ins
        StrAttr:$mode,
        I32Attr:$axis,
        StrAttr:$data_type,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def RelayoutKernel: GenericUnaryKernel<"Relayout"> {
    let extraClassDeclaration = [{
        bool checkInputLayout(MemRefType memref, size_t input);
    }];
}

def WarpPerspectiveKernel: AbstractKernelBase<"WarpPerspective"> {
    let arguments = (ins
        StrAttr:$imode,
        StrAttr:$bmode,
        StrAttr:$format,
        F32Attr:$border_val,
        ArrayAttr:$mat_idx,

        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$inputs,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def TypeCvtKernel: AbstractKernelBase<"TypeCvt"> {
    let arguments = (ins
        StrAttr:$i_dtype,
        StrAttr:$o_dtype,

        DefaultValuedAttr<F32Attr, "1.0">:$i_scale,
        DefaultValuedAttr<F32Attr, "1.0">:$o_scale,
        DefaultValuedAttr<UI8Attr, "0">:$i_zero,
        DefaultValuedAttr<UI8Attr, "0">:$o_zero,

        Arg<AnyMemRef, "", [MemRead]>:$input,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

def ElemwiseMultiType: AbstractKernelBase<"ElemwiseMultiType"> {
    let arguments = (ins
        StrAttr:$mode,

        Arg<Variadic<AnyMemRef>, "", [MemRead]>:$inputs,
        Arg<AnyMemRef, "", [MemWrite]>:$output
    );
}

#endif // ABSTRACT_KERNELS
