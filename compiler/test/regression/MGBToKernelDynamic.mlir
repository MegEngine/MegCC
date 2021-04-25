// RUN: megcc-opt --MGB-to-Kernel --finalizing-bufferize %s | FileCheck %s


// CHECK-LABEL: func @subtensor
func @subtensor(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.SubtensorIns
  %0 = "MGB.Subtensor"(%arg0) {descs = [[0 : i32, 0 : i32, -1 : i32, 1 : i32, 2 : i32]], flags = [[0 : i32, 0 : i32, -1 : i32, 1 : i32, 0 : i32]]} : (tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.SUB
  %1 = "MGB.Elemwise"(%arg1, %0) {mode = 24} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: return
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: func @set_subtensor
func @set_subtensor(%arg0: tensor<1x3x224x224xi8>, %arg1: tensor<?xi8>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>, %arg5: tensor<?xf32>) -> tensor<1x3x224x224xi8> {
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: Kernel.SetSubtensorIns
  %0 = "MGB.SetSubtensor"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {descs = [[0 : i32, 0 : i32, -1 : i32, 1 : i32, 2 : i32]], flags = [[0 : i32, 0 : i32, -1 : i32, 1 : i32, 0 : i32]]} : (tensor<1x3x224x224xi8>, tensor<?xi8>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<1x3x224x224xi8>
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: Kernel.SUB
  %1 = "MGB.Elemwise"(%arg0, %0) {mode = 24} : (tensor<1x3x224x224xi8>, tensor<1x3x224x224xi8>) -> tensor<1x3x224x224xi8>
  // CHECK-NEXT: return
  return %1 : tensor<1x3x224x224xi8>
}
// CHECK-LABEL: func @dimshuffle
func @dimshuffle(%arg0: tensor<?xi8>, %arg1: tensor<?xi8>) -> tensor<?xi8> {
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.DimshuffleIns
  %0 = "MGB.Dimshuffle"(%arg0) {pattern = [0 : i32, 3 : i32, 1 : i32, 2 : i32]} : (tensor<?xi8>) -> tensor<?xi8>
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.SUB
  %1 = "MGB.Elemwise"(%arg1, %0) {mode = 24} : (tensor<?xi8>, tensor<?xi8>) -> tensor<?xi8>
  // CHECK-NEXT: return
  return %1 : tensor<?xi8>
}

// CHECK-LABEL: func @concat
func @concat(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.Concat
  %0 = "MGB.Concat"(%arg0, %arg1, %arg2) {axis = 0 : i32, comp_node = "cpu:default"} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.SUB
  %1 = "MGB.Elemwise"(%arg3, %0) {mode = 24} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: return
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: func @getvarshape
func @getvarshape(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.GetVarShapeIns
  %0 = "MGB.GetVarShape"(%arg0) {axis = 7 : i32} : (tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.SUB
  %1 = "MGB.Elemwise"(%arg1, %0) {mode = 24} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-NEXT: return
  return %1 : tensor<?xi32>
}

// CHECK-LABEL: func @broadcast
func @broadcast(%arg0: tensor<1x3x3xf32>, %arg1: tensor<?xi32>) -> tensor<?xf32> {
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.BroadcastIns
  %0 = "MGB.Broadcast"(%arg0, %arg1) : (tensor<1x3x3xf32>, tensor<?xi32>) -> tensor<?xf32>
  // CHECK-NEXT: Kernel.DynamicAlloc
  // CHECK-NEXT: Kernel.SUB
  %1 = "MGB.Elemwise"(%arg0, %0) {mode = 24} : (tensor<1x3x3xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK-NEXT: return
  return %1 : tensor<?xf32>
}
