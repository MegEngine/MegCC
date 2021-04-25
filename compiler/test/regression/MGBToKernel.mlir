// RUN: megcc-opt --MGB-to-Kernel --finalizing-bufferize %s | FileCheck %s

// CHECK-LABEL: func @elemwise
func @elemwise(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: %[[Output0:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.SUB"(%arg2, %arg1, %[[Output0]])
  %0 = "MGB.Elemwise"(%arg2, %arg1) {mode = 24 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: %[[Output1:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.FUSE_ADD_RELU"(%arg0, %[[Output0]], %[[Output1]])
  %1 = "MGB.Elemwise"(%arg0, %0) {mode = 37 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: %[[Output2:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.FUSE_ADD_SIGMOID"(%arg0, %[[Output1]], %[[Output2]])
  %2 = "MGB.Elemwise"(%arg0, %1) {mode = 38 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return %[[Output2]]
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func @convolution
func @convolution(%arg0: tensor<1x32x112x112xf32>, %arg1: tensor<32x1x1x3x3xf32>) -> tensor<1x32x112x112xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.Conv2d"(%arg0, %arg1, %[[Output]])
  //    CHECK-DAG: pad_h = 1
  //    CHECK-DAG: pad_w = 1
  //    CHECK-DAG: stride_h = 1
  //    CHECK-DAG: stride_w = 1
  //    CHECK-DAG: dilate_h = 1
  //    CHECK-DAG: dilate_w = 1
  //    CHECK-DAG: compute_mode = "DEFAULT"
  //    CHECK-DAG: format = "NCHW"
  //    CHECK-DAG: sparse = "GROUP"
  //    CHECK-DAG: mode = "CROSS_CORRELATION"
  %0 = "MGB.Convolution"(%arg0, %arg1) {compute_mode = 0 : i32, dilate_h = 1 : ui32, dilate_w = 1 : ui32, format = 0 : i32, mode = 0 : i32, pad_h = 1 : ui32, pad_w = 1 : ui32, sparse = 1 : i32, strategy = 1 : i32, stride_h = 1 : ui32, stride_w = 1 : ui32, workspace_limit = 0 : ui64} : (tensor<1x32x112x112xf32>,  tensor<32x1x1x3x3xf32>) -> tensor<1x32x112x112xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<1x32x112x112xf32>
}

// CHECK-LABEL: func @conv_bias
func @conv_bias(%arg0: tensor<1x32x56x56x4xf32>, %arg1: tensor<32x1x1x3x3x4xf32>, %arg2: tensor<1x32x1x1x4xf32>) -> tensor<1x32x56x56x4xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.Conv2d"(%arg0, %arg1, %arg2, %[[Output]])
  //    CHECK-DAG: pad_h = 1
  //    CHECK-DAG: pad_w = 1
  //    CHECK-DAG: stride_h = 1
  //    CHECK-DAG: stride_w = 1
  //    CHECK-DAG: dilate_h = 1
  //    CHECK-DAG: dilate_w = 1
  //    CHECK-DAG: compute_mode = "DEFAULT"
  //    CHECK-DAG: format = "NCHW44"
  //    CHECK-DAG: sparse = "GROUP"
  //    CHECK-DAG: mode = "CROSS_CORRELATION"
  %0 = "MGB.ConvBias"(%arg0, %arg1, %arg2) {compute_mode = 0 : i32, dilate_h = 1 : ui32, dilate_w = 1 : ui32, format = 7 : i32, mode = 0 : i32, nonlineMode = 1 : i32, pad_h = 1 : ui32, pad_w = 1 : ui32, sparse = 1 : i32, strategy = 1 : i32, stride_h = 1 : ui32, stride_w = 1 : ui32, workspace_limit = 18446744073709551615 : ui64} : (tensor<1x32x56x56x4xf32>, tensor<32x1x1x3x3x4xf32>, tensor<1x32x1x1x4xf32>) -> tensor<1x32x56x56x4xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<1x32x56x56x4xf32>
}

// CHECK-LABEL: func @pooling
func @pooling(%arg0: tensor<1x1024x7x7xf32>) -> tensor<1x1024x1x1xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.Pool2d"(%arg0, %[[Output]])
  //    CHECK-DAG: pad_h = 0
  //    CHECK-DAG: pad_w = 0
  //    CHECK-DAG: stride_h = 7
  //    CHECK-DAG: stride_w = 7
  //    CHECK-DAG: window_h = 7
  //    CHECK-DAG: window_w = 7
  //    CHECK-DAG: format = "NCHW"
  //    CHECK-DAG: mode = "AVERAGE"
  %0 = "MGB.Pooling"(%arg0) {format = 0 : i32, mode = 1 : i32, pad_h = 0 : ui32, pad_w = 0 : ui32, stride_h = 7 : ui32, stride_w = 7 : ui32, window_h = 7 : ui32, window_w = 7 : ui32} : (tensor<1x1024x7x7xf32>) -> tensor<1x1024x1x1xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<1x1024x1x1xf32>
}

// CHECK-LABEL: func @matmul
func @matmul(%arg0: tensor<200x1024xf32>, %arg1: tensor<1024x1000xf32>) -> tensor<200x1000xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.Matmul"(%arg0, %arg1, %[[Output]])
  //    CHECK-DAG: transposeA = false
  //    CHECK-DAG: transposeB = false
  //    CHECK-DAG: compute_mode = "DEFAULT"
  //    CHECK-DAG: format = "DEFAULT"
  %0 = "MGB.MatrixMul"(%arg0, %arg1) {compute_mode = 0 : i32, format = 0 : i32, strategy = 1 : i32, transposeA = false, transposeB = false, workspace_limit = 0 : ui64} : (tensor<200x1024xf32>, tensor<1024x1000xf32>) -> tensor<200x1000xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<200x1000xf32>
}

// CHECK-LABEL: func @batched_matmul
func @batched_matmul(%arg0: tensor<200x1024xf32>, %arg1: tensor<1024x1000xf32>) -> tensor<200x1000xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.BatchedMatmul"(%arg0, %arg1, %[[Output]])
  //    CHECK-DAG: transposeA = false
  //    CHECK-DAG: transposeB = false
  //    CHECK-DAG: compute_mode = "DEFAULT"
  //    CHECK-DAG: format = "DEFAULT"
  %0 = "MGB.BatchedMatmul"(%arg0, %arg1) {compute_mode = 0 : i32, format = 0 : i32, strategy = 1 : i32, transposeA = false, transposeB = false, workspace_limit = 0 : ui64} : (tensor<200x1024xf32>, tensor<1024x1000xf32>) -> tensor<200x1000xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<200x1000xf32>
}

// CHECK-LABEL: func @typeCvt
func @typeCvt(%arg0: tensor<1024x1x640x480xi8>) -> tensor<1024x1x640x480xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.TypeCvt"(%arg0, %[[Output]])
  //    CHECK-DAG: i_zero = 10
  //    CHECK-DAG: o_zero = 0
  //    CHECK-DAG: i_scale = 1.000000e+00
  //    CHECK-DAG: o_scale = 1.000000e+00
  //    CHECK-DAG: i_dtype = "ui8"
  //    CHECK-DAG: o_dtype = "f32"
  %0 = "MGB.TypeCvt"(%arg0) {i_dtype = ui8, i_scale = 1.000000e+00 : f32, i_zero = 10 : ui8, o_dtype = f32, o_scale = 1.000000e+00 : f32, o_zero = 0 : ui8} : (tensor<1024x1x640x480xi8>) -> tensor<1024x1x640x480xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<1024x1x640x480xf32>
}

// CHECK-LABEL: func @subtensor
func @subtensor(%arg0: tensor<1024x10x48x96xf32>, %arg1: tensor<1024x5x48x96xf32>) -> tensor<1024x5x48x96xf32> {
  // CHECK-NEXT: %[[SubSrc:.+]] = "Kernel.Subtensor"
  //    CHECk-DAG: determined = false
  %0 = "MGB.Subtensor"(%arg0) {descs = [[1 : i32, 5 : i32, -1 : i32, 1 : i32, -1 : i32]], flags = [[0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]]} : (tensor<1024x10x48x96xf32>) -> tensor<1024x5x48x96xf32>
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.SUB"(%arg1, %[[SubSrc]], %[[Output]])
  %1 = "MGB.Elemwise"(%arg1, %0) {mode = 24} : (tensor<1024x5x48x96xf32>, tensor<1024x5x48x96xf32>) -> tensor<1024x5x48x96xf32>
  // CHECK-NEXT: return %[[Output]]
  return %1 : tensor<1024x5x48x96xf32>
}

// CHECK-LABEL: func @setsubtensor
func @setsubtensor(%arg0: tensor<1024x10x48x96xf32>, %arg1: tensor<1024x5x48x96xf32>) -> tensor<1024x10x48x96xf32> {
  // CHECK-NEXT: %[[SrcUpdate:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.Relayout"(%arg0, %[[SrcUpdate]])
  // CHECK-NEXT: %[[SrcUpdateSub:.+]] = "Kernel.Subtensor"
  //    CHECK-DAG: %[[SrcUpdate]]
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: "Kernel.Relayout"(%arg1, %[[SrcUpdateSub]])
  %0 = "MGB.SetSubtensor"(%arg0, %arg1) {descs = [[1 : i32, 5 : i32, -1 : i32, 1 : i32, -1 : i32]], flags = [[0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]]} : (tensor<1024x10x48x96xf32>, tensor<1024x5x48x96xf32>) -> tensor<1024x10x48x96xf32>
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.SUB"(%arg0, %[[SrcUpdate]], %[[Output]])
  %1 = "MGB.Elemwise"(%arg0, %0) {mode = 24} : (tensor<1024x10x48x96xf32>, tensor<1024x10x48x96xf32>) -> tensor<1024x10x48x96xf32>
  // CHECK-NEXT: return %[[Output]]
  return %1 : tensor<1024x10x48x96xf32>
}

// CHECK-LABEL: func @warpperspective
func @warpperspective(%arg0: tensor<1024x1x640x480xf32>, %arg1: tensor<1024x3x3xf32>) -> tensor<1024x1x48x96xf32> {
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.WarpPerspective"(%arg0, %arg1, %[[Output]])
  //    CHECK-DAG: bmode = "REPLICATE"
  //    CHECK-DAG: border_val = 0.000000e+00
  //    CHECK-DAG: format = "NCHW"
  //    CHECK-DAG: imode = "LINEAR"
  %0 = "MGB.WarpPerspective"(%arg0, %arg1) {bmode = 0 : i32, border_val = 0.000000e+00 : f32, format = 0 : i32, imode = 1 : i32, mat_idx = [48 : i32, 96 : i32]} : (tensor<1024x1x640x480xf32>, tensor<1024x3x3xf32>) -> tensor<1024x1x48x96xf32>
  // CHECK-NEXT: return %[[Output]]
  return %0 : tensor<1024x1x48x96xf32>
}

// CHECK-LABEL: func @concat
func @concat(%arg0: tensor<1024x5x48x96xf32>, %arg1: tensor<1024x5x48x96xf32>, %arg2: tensor<1024x10x48x96xf32>) -> tensor<1024x10x48x96xf32> {
  // CHECK-NEXT: %[[Concat:.+]] = memref.alloc
  // CHECK-NEXT: %[[ComponentA:.+]] = "Kernel.Subtensor"
  //    CHECK-DAG: %[[Concat]]
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: "Kernel.Relayout"(%arg0, %[[ComponentA]])
  // CHECK-NEXT: %[[ComponentB:.+]] = "Kernel.Subtensor"
  //    CHECK-DAG: %[[Concat]]
  //    CHECK-DAG: determined = true
  // CHECK-NEXT: "Kernel.Relayout"(%arg1, %[[ComponentB]])
  %0 = "MGB.Concat"(%arg0, %arg1) {axis = 1 : i32, comp_node = "cpu:default"} : (tensor<1024x5x48x96xf32>, tensor<1024x5x48x96xf32>) -> tensor<1024x10x48x96xf32>
  // CHECK-NEXT: %[[Output:.+]] = memref.alloc
  // CHECK-NEXT: "Kernel.SUB"(%arg2, %[[Concat]], %[[Output]])
  %1 = "MGB.Elemwise"(%arg2, %0) {mode = 24} : (tensor<1024x10x48x96xf32>, tensor<1024x10x48x96xf32>) -> tensor<1024x10x48x96xf32>
  // CHECK-NEXT: return %[[Output]]
  return %1 : tensor<1024x10x48x96xf32>
}
