// RUN: megcc-opt --memory-forwarding %s -split-input-file | FileCheck %s

// -----

// normal memory forwarding, all layouts are contiguous in the graph
// CHECK-LABEL: func @memory_forward_0
func @memory_forward_0(%arg0: memref<2x3x4xf32>, %arg1: memref<2x3x2x2xf32>) -> memref<2x3x4xf32> {
  // CHECK-NEXT: %[[Reshaped:.+]] = "Kernel.Reshape"
  //    CHECK-DAG: determined = true
  %0 = "Kernel.Reshape"(%arg1) : (memref<2x3x2x2xf32>) -> memref<2x3x4xf32>
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.SUB"(%arg0, %[[Reshaped]], %[[Output]])
  %1 = memref.alloc() : memref<2x3x4xf32>
  "Kernel.SUB"(%arg0, %0, %1) : (memref<2x3x4xf32>, memref<2x3x4xf32>, memref<2x3x4xf32>) -> ()
  return %1 : memref<2x3x4xf32>
}

// -----

// Dimshuffle should be a relayout since elementwise requires input layout is contiguous
// CHECK-LABEL: func @memory_forward_1
func @memory_forward_1(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>) -> memref<2x3x4xf32> {
  // CHECK-NEXT: %[[NonCont:.+]] = "Kernel.Dimshuffle"
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: %[[Cont:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.Relayout"(%[[NonCont]], %[[Cont]])
  %0 = "Kernel.Dimshuffle"(%arg1) {pattern = [0 : i32, 2 : i32, 1 : i32]} : (memref<2x4x3xf32>) -> memref<2x3x4xf32>
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.SUB"(%arg0, %[[Cont]], %[[Output]])
  %1 = memref.alloc() : memref<2x3x4xf32>
  "Kernel.SUB"(%arg0, %0, %1) : (memref<2x3x4xf32>, memref<2x3x4xf32>, memref<2x3x4xf32>) -> ()
  return %1 : memref<2x3x4xf32>
}

// CHECK-LABEL: func @memory_forward_2
func @memory_forward_2(%arg0: memref<2x3x4x5xf32>) -> memref<2x5x2x6xf32> {
  // CHECK-NEXT: %0 = "Kernel.Dimshuffle"
  //    CHECK-DAG: determined = true
  %0 = "Kernel.Dimshuffle"(%arg0) {pattern = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} : (memref<2x3x4x5xf32>) -> memref<2x5x4x3xf32>

  // memory forward failure, so allocate a new contiguous buffer with dst shape and copy src to it
  // CHECK-NEXT: %1 = memref.alloc
  // CHECK-NEXT: %2 = "Kernel.Reshape"(%1)
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: "Kernel.Relayout"(%0, %2)
  %1 = "Kernel.Reshape"(%0) : (memref<2x5x4x3xf32>) -> memref<10x12xf32>

  // CHECK-NEXT: %3 = "Kernel.Reshape"(%1)
  //    CHECK-DAG: determined = true
  // return %3
  %2 = "Kernel.Reshape"(%1) : (memref<10x12xf32>) -> memref<2x5x2x6xf32>
  return %2 : memref<2x5x2x6xf32>
}

// CHECK-LABEL: func @memory_forward_3
func @memory_forward_3(%arg0: memref<6x10x8xf32>, %arg1: memref<6x4x8xf32>, %arg2: memref<1x10x8xf32>) -> (memref<6x4x8xf32>, memref<1x10x8xf32>) {
  // CHECK-NEXT: %0 = "Kernel.Subtensor"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: %1 = memref.alloc
  // CHECK-NEXT: "Kernel.Relayout"(%0, %1)
  // CHECK-NEXT: %2 = memref.alloc
  // CHECK-NEXT: "Kernel.ADD"(%1, %arg1, %2)

  // CHECK-NEXT: %3 = "Kernel.Subtensor"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: %4 = memref.alloc
  // CHECK-NEXT: "Kernel.ADD"(%3, %arg2, %4)
  // CHECK-NEXT: return %2, %4

  %0 = "Kernel.Subtensor"(%arg0) {descs = [[1 : i32, 6 : i32, -1 : i32, 1 : i32, -1 : i32]], flags = [[0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]]} : (memref<6x10x8xf32>) -> memref<6x4x8xf32>
  %1 = memref.alloc() : memref<6x4x8xf32>
  "Kernel.ADD"(%0, %arg1, %1) : (memref<6x4x8xf32>, memref<6x4x8xf32>, memref<6x4x8xf32>) -> ()

  %2 = "Kernel.Subtensor"(%arg0) {descs = [[0 : i32, -1 : i32, -1 : i32, -1 : i32, 4 : i32]], flags = [[0 : i32, -1 : i32, -1 : i32, -1 : i32, 0 : i32]]} : (memref<6x10x8xf32>) -> memref<1x10x8xf32>
  %3 = memref.alloc() : memref<1x10x8xf32>
  "Kernel.ADD"(%2, %arg2, %3) : (memref<1x10x8xf32>, memref<1x10x8xf32>, memref<1x10x8xf32>) -> ()

  return %1, %3 : memref<6x4x8xf32>, memref<1x10x8xf32>
}
// -----
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 80 + d1 * 8 - d2 + 220)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 80 + d1 * 8 - d2 + 28)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 6720 + d1 * 2240 + d2 * 280 + d3 * 7 + d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 2240 + d1 * 560 + d2 * 14 + d3 * 2 + 28004)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * -2240 - d1 * 1120 - d2 * 28 + d3 + 53724)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * -6720 + d1 - d2 * 28 - d3 * 1120 - d4 * 2240 - d5 * 1120 + 53724)>
// CHECK-LABEL: func @memory_forward_neg_subtensor
func @memory_forward_neg_subtensor(%arg0: memref<6x10x8xf32>, %arg1: memref<6x4x8xf32>) -> (memref<6x4x8xf32>) {
  // CHECK-NEXT: %0 = "Kernel.Subtensor"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: %1 = memref.alloc
  // CHECK-NEXT: "Kernel.Relayout"(%0, %1)
  //    CHECK-DAG: #[[MAP1]]
  // CHECK-NEXT: %2 = memref.alloc
  // CHECK-NEXT: "Kernel.ADD"(%1, %arg1, %2)
  %0 = "Kernel.Subtensor"(%arg0) {descs = [[1 : i32, 6 : i32, -1 : i32, 1 : i32, -1 : i32], [2 : i32, 0 : i32, -1 : i32, -1 : i32, -1 : i32]], flags = [[0 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32], [0 : i32, -1 : i32, -1 : i32, 0 : i32, -1 : i32]]} : (memref<6x10x8xf32>) -> memref<6x4x8xf32>
  %1 = memref.alloc() : memref<6x4x8xf32>
  "Kernel.ADD"(%0, %arg1, %1) : (memref<6x4x8xf32>, memref<6x4x8xf32>, memref<6x4x8xf32>) -> ()

  return %1 : memref<6x4x8xf32>
}

// CHECK-LABEL: func @memory_forward_neg_step_subtensor
func @memory_forward_neg_step_subtensor(%arg0: memref<6x10x8xf32>, %arg1: memref<6x10x8xf32>) -> (memref<6x10x8xf32>) {
  // CHECK-NEXT: %0 = "Kernel.Subtensor"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: %1 = memref.alloc
  // CHECK-NEXT: "Kernel.Relayout"(%0, %1)
  //    CHECK-DAG: memref<6x10x8xf32>
  //    CHECK-DAG: #[[MAP2]]
  // CHECK-NEXT: %2 = memref.alloc
  // CHECK-NEXT: "Kernel.ADD"(%1, %arg1, %2)
  %0 = "Kernel.Subtensor"(%arg0) {descs = [[2 : i32, 0 : i32, -1 : i32, -1 : i32, -1 : i32]], flags = [[0 : i32, -1 : i32, -1 : i32, 0 : i32, -1 : i32]]} : (memref<6x10x8xf32>) -> memref<6x10x8xf32>
  %1 = memref.alloc() : memref<6x10x8xf32>
  "Kernel.ADD"(%0, %arg1, %1) : (memref<6x10x8xf32>, memref<6x10x8xf32>, memref<6x10x8xf32>) -> ()
  return %1 : memref<6x10x8xf32>
}

// CHECK-LABEL: func @subtensor_dimshuffle_reshape
func @subtensor_dimshuffle_reshape(%arg0: memref<2x3x8x40x7xf32>) -> (memref<15x2xf32>) {
  // CHECK-NEXT: %0 = "Kernel.Reshape"(%arg0)
  //    CHECK-DAG: determined = true
  //    CHECK-DAG: #[[MAP3]]
  // CHECK-NEXT: %1 = "Kernel.Subtensor"(%0)
  //    CHECK-DAG: determined = true
  //    CHECK-DAG: #[[MAP4]]
  // CHECK-NEXT: %2 = "Kernel.Subtensor"(%1)
  //    CHECK-DAG: determined = true
  //    CHECK-DAG: #[[MAP5]]
  // CHECK-NEXT: %3 = "Kernel.Dimshuffle"(%2)
  //    CHECK-DAG: determined = true
  //    CHECK-DAG: #[[MAP6]]
  // CHECK-NEXT: %4 = memref.alloc()
  // CHECK-NEXT: %5 = "Kernel.Reshape"(%4)
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: "Kernel.Relayout"(%3, %5)
  //    CHECK-DAG: #[[MAP6]]
  %0 = "Kernel.Reshape"(%arg0) {axis = 7 : i32} : (memref<2x3x8x40x7xf32>) -> memref<2x3x8x40x7xf32>
  %1 = "Kernel.Subtensor"(%0) {descs = [[4 : i32, 1 : i32, -1 : i32, 2 : i32, -1 : i32], [3 : i32, 0 : i32, -1 : i32, 2 : i32, -1 : i32], [2 : i32, 1 : i32, -1 : i32, 2 : i32, -1 : i32], [1 : i32, 0 : i32, -1 : i32, 1 : i32, -1 : i32], [0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32]], flags = [[0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32], [0 : i32, -1 : i32, 0 : i32, 0 : i32, -1 : i32], [0 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32], [0 : i32, -1 : i32, -1 : i32, 0 : i32, -1 : i32], [0 : i32, -1 : i32, -1 : i32, -1 : i32, 0 : i32]]} : (memref<2x3x8x40x7xf32>) -> memref<3x4x20x3xf32>
  %2 = "Kernel.Subtensor"(%1) {descs = [[3 : i32, -1 : i32, 1 : i32, -1 : i32, -1 : i32], [2 : i32, 0 : i32, -10 : i32, -2 : i32, -1 : i32], [1 : i32, -1 : i32, -1 : i32, -2 : i32, -1 : i32], [0 : i32, 0 : i32, -1 : i32, -1 : i32, -1 : i32]], flags = [[0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32], [0 : i32, -1 : i32, 0 : i32, 0 : i32, -1 : i32], [0 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32], [0 : i32, -1 : i32, -1 : i32, 0 : i32, -1 : i32]]} : (memref<3x4x20x3xf32>) -> memref<3x2x5x1xf32>
  %3 = "Kernel.Dimshuffle"(%2) {pattern = [-1 : i32, 3 : i32, 2 : i32, -1 : i32, 0 : i32, 1 : i32]} : (memref<3x2x5x1xf32>) -> memref<1x1x5x1x3x2xf32>
  %4 = "Kernel.Reshape"(%3) {axis = 7 : i32} : (memref<1x1x5x1x3x2xf32>) -> memref<15x2xf32>
  return %4 : memref<15x2xf32>
}
