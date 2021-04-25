/**
 * \file compiler/include/megbrain/IR/ops.td
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifndef MGB_OPS
#define MGB_OPS

include "megbrain/IR/base.td"
include "megbrain/IR/param_defs.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def Elemwise : MgbHashableOp<"Elemwise", [ElemwiseParam], [NoSideEffect]> {
  let inputs = (ins Variadic<AnyType>:$input);
  let results = (outs AnyType);
  let nameFunction = [{
    return to_string($_self.mode);
  }];
}

def Reduce: MgbHashableOp<"Reduce", [ReduceParam]> {
  let inputs = (ins Variadic<AnyType>:$input);
  let results = (outs AnyType);
}

def Softmax: MgbHashableOp<"Softmax", [], [NoSideEffect]> {
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
  let extraArguments = (ins
    MgbI32Attr:$axis
  );
}

def TypeCvt: MgbHashableOp<"TypeCvt", [], [NoSideEffect]> {
  let inputs = (ins AnyType:$inputs);
  let extraArguments = (ins
    TypeAttr:$i_dtype,
    TypeAttr:$o_dtype,

    MgbDefaultValuedAttr<MgbF32Attr, "1.0">:$i_scale,
    MgbDefaultValuedAttr<MgbF32Attr, "1.0">:$o_scale,
    MgbDefaultValuedAttr<MgbUI8Attr, "0">:$i_zero,
    MgbDefaultValuedAttr<MgbUI8Attr, "0">:$o_zero
  );
  let results = (outs AnyType);
}

def MatrixInverse: MgbHashableOp<"MatrixInverse", [EmptyParam], [NoSideEffect]>{
  let inputs = (ins
    AnyType:$mat
  );
  let results = (outs AnyType);
}

def MatrixMul: MgbHashableOp<"MatrixMul", [MatrixMulParam, ExecutionPolicyParamBase<"policy">]> {
  let inputs = (ins
    AnyType:$a,
    AnyType:$b
  );
  let results = (outs AnyType);
}

def BatchedMatrixMul: MgbHashableOp<"BatchedMatmul", [MatrixMulParam, ExecutionPolicyParamBase<"policy">]> {
  let inputs = (ins
    AnyType:$a,
    AnyType:$b
  );
  let results = (outs AnyType);
}

def Dot: MgbHashableOp<"Dot", [EmptyParam]>;

def SVD: MgbHashableOp<"SVD", [SVDParam]>;

def Convolution : MgbHashableOp<"Convolution", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]> {
  let inputs = (ins
    AnyType:$input,
    AnyType:$weights
  );
  let results = (outs AnyType);
}

def ConvolutionBackwardData: MgbHashableOp<"ConvolutionBackwardData", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]> {
  let inputs = (ins
    AnyType:$input,
    AnyType:$weights
  );
  let results = (outs AnyType);
}

def Convolution3D: MgbHashableOp<"Convolution3D", [Convolution3DParam, ExecutionPolicyParamBase<"policy">]>;

def DeformableConv : MgbHashableOp<"DeformableConv", [ConvolutionParam, ExecutionPolicyParamBase<"policy">]>;

def GroupLocal: MgbHashableOp<"GroupLocal", [ConvolutionParam]>;

def Pooling: MgbHashableOp<"Pooling", [PoolingParam]> {
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
}

def AdaptivePooling : MgbHashableOp<"AdaptivePooling", [AdaptivePoolingParam]>{
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
}

def ROIPooling: MgbHashableOp<"ROIPooling", [ROIPoolingParam]>;

def DeformablePSROIPooling : MgbHashableOp<"DeformablePSROIPooling", [DeformablePSROIPoolingParam]>;

def ConvBias : MgbHashableOp<"ConvBias", [ConvBiasParam, ExecutionPolicyParamBase<"policy">], [NoSideEffect]> {
  let inputs = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType);
  let extraArguments = (ins
    MgbDefaultValuedAttr<MgbDTypeAttr, "::megdnn::dtype::Float32()">:$dtype
  );
}

def BatchConvBias : MgbHashableOp<"BatchConvBias", [BatchConvBiasParam, ExecutionPolicyParamBase<"policy">]> {
  let extraArguments = (ins
    MgbDTypeAttr:$dtype
  );
}

def BatchNorm : MgbHashableOp<"BatchNorm", [BNParam]>;

def ROIAlign: MgbHashableOp<"ROIAlign", [ROIAlignParam]>;

def WarpPerspective: MgbHashableOp<"WarpPerspective", [WarpPerspectiveParam]> {
  let inputs = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType);
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$mat_idx
  );
}

def WarpAffine: MgbHashableOp<"WarpAffine", [WarpAffineParam], [NoSideEffect]>{
  let inputs = (ins
    AnyType:$input,
    AnyType:$trans
  );
  let results = (outs AnyType);
}

def Remap: MgbHashableOp<"Remap", [RemapParam]>;

def Resize: MgbHashableOp<"Resize", [ResizeParam], [NoSideEffect]>{
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
}

def IndexingOneHot: MgbHashableOp<"IndexingOneHot", [AxisParam], [NoSideEffect]>{
  let inputs = (ins 
      AnyType:$input,
      AnyType:$index
  );
  let results = (outs AnyType);
}

def IndexingSetOneHot: MgbHashableOp<"IndexingSetOneHot", [AxisParam]>;

def Copy: MgbHashableOp<"Copy"> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Argsort: MgbHashableOp<"Argsort", [ArgsortParam], [NoSideEffect]>{
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType:$argsort_value, AnyType:$argsort_index);
}

def Argmax : MgbHashableOp<"Argmax", [AxisParam], [NoSideEffect]>{
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
}

def Argmin : MgbHashableOp<"Argmin", [AxisParam]>;

def CondTake : MgbHashableOp<"CondTake">;

def TopK: MgbHashableOp<"TopK", [TopKParam], [NoSideEffect]>{
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType:$topk_value, AnyType:$topk_index);
  let extraArguments = (ins
    MgbI32Attr:$k
  );
}

def NvOf: MgbHashableOp<"NvOf", [NvOfParam]>;

def UniformRNG: MgbHashableOp<"UniformRNG", [UniformRNGParam]> {
  let hashFunction = [{return mgb::hash($_self.dyn_typeinfo());}];
  let cmpFunction = [{return true;}];
}

def GaussianRNG: MgbHashableOp<"GaussianRNG", [GaussianRNGParam]> {
  let hashFunction = [{
    return mgb::hash_pair_combine(
      mgb::hash($_self.dyn_typeinfo()),
      mgb::hash_pair_combine(mgb::hash($_self.mean), mgb::hash($_self.std)));
  }];
  let cmpFunction = [{return $0.mean == $1.mean && $0.std == $1.std;}];
}

def Linspace: MgbHashableOp<"Linspace", [LinspaceParam]> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Eye: MgbHashableOp<"Eye", [EyeParam]> {
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def GetVarShape : MgbHashableOp<"GetVarShape", [OptionalAxisV1Param]>{
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
}

def Concat: MgbHashableOp<"Concat", [AxisParam]> {
  let inputs = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType);
  let extraArguments = (ins
    MgbCompNodeAttr:$comp_node
  );
}

def Broadcast : MgbHashableOp<"Broadcast", [EmptyParam]>{
  let inputs = (ins
    AnyType:$inp,
    AnyType:$tshp
  );
  let results = (outs AnyType);
}

def Identity: MgbHashableOp<"Identity">;

def CollectiveComm : MgbHashableOp<"CollectiveComm", [CollectiveCommParam]> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbUI32Attr:$nr_devices,
    MgbUI32Attr:$rank,
    MgbBoolAttr:$is_root,
    MgbBoolAttr:$local_grad,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbDTypeAttr:$dtype,
    MgbStringAttr:$backend,
    MgbStringAttr:$comp_node
  );
}

def RemoteSend : MgbHashableOp<"RemoteSend"> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbUI32Attr:$rank_to,
    MgbStringAttr:$backend
  );
}

def RemoteRecv : MgbHashableOp<"RemoteRecv"> {
  let extraArguments = (ins
    MgbStringAttr:$key,
    MgbStringAttr:$addr,
    MgbUI32Attr:$port,
    MgbUI32Attr:$rank_from,
    MgbCompNodeAttr:$cn,
    MgbTensorShapeAttr:$shape,
    MgbDTypeAttr:$dtype,
    MgbStringAttr:$backend
  );
}

def NMSKeep : MgbHashableOp<"NMSKeep"> {
  let extraArguments = (ins
    MgbF32Attr:$iou_thresh,
    MgbUI32Attr:$max_output
  );
}

def ParamPackSplit : MgbHashableOp<"ParamPackSplit"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$offsets,
    MgbArrayAttr<MgbArrayAttr<MgbSizeTAddr>>:$shapes
  );
}

def ParamPackConcat : MgbHashableOp<"ParamPackConcat"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$offsets
  );
}

def Dimshuffle: MgbHashableOp<"Dimshuffle"> {
  let inputs = (ins AnyType:$input);
  let extraArguments = (ins MgbArrayAttr<MgbI32Attr>:$pattern);
  let results = (outs AnyType);
}

def Reshape: MgbHashableOp<"Reshape", [OptionalAxisV1Param]> {
  let inputs = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType);
}

// TODO: merge Add/Remove Axis into AxisAddRemove as megbrain?
def AddAxis: MgbHashableOp<"AddAxis"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$axis
  );
}
def RemoveAxis: MgbHashableOp<"RemoveAxis"> {
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$axis
  );
}

class FancyIndexingBase<string name>: MgbHashableOp<name> {
  let extraArguments = (ins
    MgbArrayAttr<MgbTupleAttr<
        [MgbI32Attr, MgbI32Attr, MgbI32Attr, MgbI32Attr, MgbI32Attr]>>:$descs,
    MgbArrayAttr<MgbTupleAttr<
        [MgbI32Attr, MgbI32Attr, MgbI32Attr, MgbI32Attr, MgbI32Attr]>>:$flags
  );
}

def Subtensor: FancyIndexingBase<"Subtensor"> {
  let inputs = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType);
}

def SetSubtensor: FancyIndexingBase<"SetSubtensor"> {
  let inputs = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType);
}

def IndexingMultiAxisVec: MgbHashableOp<"IndexingMultiAxisVec"> {
  let inputs = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType);
  let extraArguments = (ins
    MgbArrayAttr<MgbI32Attr>:$axis
  );
}

def IncrSubtensor: FancyIndexingBase<"IncrSubtensor">;
def IndexingSetMultiAxisVec: FancyIndexingBase<"IndexingSetMultiAxisVec">;
def IndexingIncrMultiAxisVec: FancyIndexingBase<"IndexingIncrMultiAxisVec">;
def MeshIndexing: FancyIndexingBase<"MeshIndexing">;
def IncrMeshIndexing: FancyIndexingBase<"IncrMeshIndexing">;
def SetMeshIndexing: FancyIndexingBase<"SetMeshIndexing">;
def BatchedMeshIndexing: FancyIndexingBase<"BatchedMeshIndexing">;
def BatchedIncrMeshIndexing: FancyIndexingBase<"BatchedIncrMeshIndexing">;
def BatchedSetMeshIndexing: FancyIndexingBase<"BatchedSetMeshIndexing">;

def FakeQuant: MgbHashableOp<"FakeQuant", [FakeQuantParam]>;
def AssertEqual: MgbHashableOp<"AssertEqual",[AssertEqualParam]>;
def TQT: MgbHashableOp<"TQT", [TQTParam]>;

def ElemwiseMultiType: MgbHashableOp<"ElemwiseMultiType", [ElemwiseMultiTypeParam], [NoSideEffect]> {
  let inputs = (ins Variadic<AnyType>:$input);
  let results = (outs AnyType);
  let nameFunction = [{
    return to_string($_self.mode);
  }];
}

def InplaceAdd: MgbHashableOp<"InplaceAdd", [EmptyParam]>;

def TensorRTRuntime: MgbHashableOp<"TensorRTRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size
  );
}

def AtlasRuntime: MgbHashableOp<"AtlasRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size
  );
}

def CambriconRuntime: MgbHashableOp<"CambriconRuntime"> {
  let extraArguments = (ins
    MgbStringAttr:$buf,
    MgbSizeTAddr:$buf_size,
    MgbStringAttr:$symbol,
    MgbBoolAttr:$tensor_dim_mutable
  );
}

def CvtColor: MgbHashableOp<"CvtColor", [CvtColorParam]>;

def PowC: MgbHashableOp<"PowC", [PowCParam]> {
  let inputs = (ins AnyType:$input);
  let results = (outs AnyType);
}

#endif // MGB_OPS
