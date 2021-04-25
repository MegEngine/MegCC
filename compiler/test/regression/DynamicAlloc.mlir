// RUN: megcc-opt --MGB-to-Kernel --canonicalize %s | FileCheck %s

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
