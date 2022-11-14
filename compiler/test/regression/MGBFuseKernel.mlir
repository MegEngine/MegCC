// RUN: megcc-opt --mgb-fuse-kernel %s | FileCheck %s

// CHECK-LABEL: func @conv_bias
func @conv_bias(%arg0: tensor<1x6x32x32x4xqsi8<1032360002:6.668140e-02>>, %arg1: tensor<32x6x1x1x4x4xqsi8<1009180373:1.018496e-02>>, %arg2: tensor<1x32x1x1x4xqsi32<976357587:6.791476e-04>>) -> tensor<1x32x32x32x4xqsi8<1022875305:3.025373e-02>> {
  // CHECK-NEXT: %0 = "MGB.ConvBias"
  //    CHECK-DAG: nonlineMode = 3
  //    CHECK-DAG: tensor<1x32x32x32x4xqsi8<1022875305:3.025373e-02>>
  // CHECK-NEXT: return
    %0 = "MGB.ConvBias"(%arg0, %arg1, %arg2) {compute_mode = 0 : i32, dilate_h = 1 : ui32, dilate_w = 1 : ui32, dtype = 0 : i32, format = 8 : i32, mode = 0 : i32, nonlineMode = 0 : i32, pad_h = 0 : ui32, pad_w = 0 : ui32, sparse = 0 : i32, strategy = 1 : i32, stride_h = 1 : ui32, stride_w = 1 : ui32, workspace_limit = 18446744073709551615 : ui64} : (tensor<1x6x32x32x4xqsi8<1032360002:6.668140e-02>>, tensor<32x6x1x1x4x4xqsi8<1009180373:1.018496e-02>>, tensor<1x32x1x1x4xqsi32<976357587:6.791476e-04>>) -> tensor<1x32x32x32x4xqsi8<1022938889:3.037216e-02>>
    %1 = "MGB.TypeCvt"(%0) {i_dtype = qsi8<1022938889:3.037216e-02>, i_scale = 1.000000e+00 : f32, i_zero = 0 : ui8, o_dtype = f32, o_scale = 1.000000e+00 : f32, o_zero = 0 : ui8} : (tensor<1x32x32x32x4xqsi8<1022938889:3.037216e-02>>) -> tensor<1x32x32x32x4xf32>
    %2 = "MGB.Elemwise"(%1) {mode = 49 : i32} : (tensor<1x32x32x32x4xf32>) -> tensor<1x32x32x32x4xf32>
    %3 = "MGB.TypeCvt"(%2) {i_dtype = f32, i_scale = 1.000000e+00 : f32, i_zero = 0 : ui8, o_dtype = qsi8<1022875305:3.025373e-02>, o_scale = 1.000000e+00 : f32, o_zero = 0 : ui8} : (tensor<1x32x32x32x4xf32>) -> tensor<1x32x32x32x4xqsi8<1022875305:3.025373e-02>>
  return %3 : tensor<1x32x32x32x4xqsi8<1022875305:3.025373e-02>>
}


// CHECK-LABEL: func @typeCvt
func @typeCvt(%arg0: tensor<1x30xf32>) -> tensor<1x30xqsi8<1042367109:1.574803e-01>> {
  // CHECK-NEXT: %0 = "MGB.TypeCvt"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: tensor<1x30xqsi8<1042367109:1.574803e-01>>
  // CHECK-NEXT: return
  %0 = "MGB.TypeCvt"(%arg0) {i_dtype = ui8, i_scale = 1.000000e+00 : f32, i_zero = 10 : ui8, o_dtype = f32, o_scale = 1.000000e+00 : f32, o_zero = 0 : ui8} : (tensor<1x30xf32>) -> tensor<1x30xqsi8<1050253722:3e-01>>
  %1 = "MGB.TypeCvt"(%0) {i_dtype = ui8, i_scale = 1.000000e+00 : f32, i_zero = 10 : ui8, o_dtype = f32, o_scale = 1.000000e+00 : f32, o_zero = 0 : ui8} : (tensor<1x30xqsi8<1050253722:3e-01>>) -> tensor<1x30xqsi8<1042367109:1.574803e-01>>
  return %1 : tensor<1x30xqsi8<1042367109:1.574803e-01>>
}

// CHECK-LABEL: func @ElemwiseFuse1
func @ElemwiseFuse1(%arg0: tensor<30xf32>, %arg1: tensor<30xf32>, %arg2: tensor<30xf32>) -> tensor<30xf32> {
  // CHECK-NEXT: %0 = "MGB.FusedElemwise"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: %arg1
  //    CHECK-DAG: %arg2
  //    CHECK-DAG: modes = ["I0,I1,ADD,O0", "O0,I2,ADD,O1", "O1,H_SWISH,D"]
  // CHECK-NEXT: return
  %1 = "MGB.Elemwise"(%arg0, %arg1) {mode = 16 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  %2 = "MGB.Elemwise"(%1, %arg2) {mode = 16 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  %3 = "MGB.Elemwise"(%2) {mode = 49 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  return %3 : tensor<30xf32>
}

// CHECK-LABEL: func @ElemwiseFuse2
func @ElemwiseFuse2(%arg0: tensor<30xf32>, %arg1: tensor<?xf32>, %arg2: tensor<30xf32>) -> tensor<30xf32> {
  // CHECK-NEXT: %0 = "MGB.Elemwise"
  %1 = "MGB.Elemwise"(%arg0, %arg1) {mode = 16 : i32} : (tensor<30xf32>, tensor<?xf32>) -> tensor<30xf32>
  // CHECK-NEXT: %1 = "MGB.FusedElemwise"
  //    CHECK-DAG: %arg2
  //    CHECK-DAG: modes = ["I0,I1,ADD,O0", "O0,H_SWISH,D"]
  // CHECK-NEXT: return
  %2 = "MGB.Elemwise"(%1, %arg2) {mode = 16 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  %3 = "MGB.Elemwise"(%2) {mode = 49 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  return %3 : tensor<30xf32>
}

// CHECK-LABEL: func @ElemwiseFuse3
func @ElemwiseFuse3(%arg0: tensor<30xf32>, %arg1: tensor<1xf32>, %arg2: tensor<30xf32>) -> tensor<30xf32> {
  // CHECK-NEXT: %0 = "MGB.FusedElemwise"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: modes = ["I0,RELU,O0", "O0,ABS,D"]
  // CHECK-NEXT: %1 = "MGB.FusedElemwise"
  //    CHECK-DAG: modes = ["I0,H_SWISH,O0", "I0,O0,SUB,D"]
  %1 = "MGB.Elemwise"(%arg0) {mode = 0 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  %2 = "MGB.Elemwise"(%1) {mode = 1 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  %3 = "MGB.Elemwise"(%1, %arg2) {mode = 16 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  %4 = "MGB.Elemwise"(%2) {mode = 49 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  %5 = "MGB.Elemwise"(%2, %4) {mode = 24 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  return %5 : tensor<30xf32>
}

// CHECK-LABEL: func @ElemwiseFuse4
func @ElemwiseFuse4(%arg0: tensor<30xf32>, %arg1: tensor<10xf32>, %arg2: tensor<30xf32>) -> tensor<30xf32> {
  // CHECK-NEXT: %0 = "MGB.FusedElemwise"
  //    CHECK-DAG: %arg0
  //    CHECK-DAG: modes = ["I0,RELU,O0", "O0,ABS,D"]
  // CHECK-NEXT: %1 = "MGB.FusedElemwise"
  //    CHECK-DAG: %arg2
  //    CHECK-DAG: modes = ["I0,I1,ADD,O0", "O0,H_SWISH,O1", "I0,O1,SUB,D"] 
  %0 = "MGB.Elemwise"(%arg0) {mode = 0 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  %1 = "MGB.Elemwise"(%0) {mode = 1 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  %2 = "MGB.Elemwise"(%1, %arg2) {mode = 16 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  %3 = "MGB.Elemwise"(%2) {mode = 49 : i32} : (tensor<30xf32>) -> tensor<30xf32>
  %4 = "MGB.Elemwise"(%1, %3) {mode = 24 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>

  // CHECK-NEXT: %2 = "MGB.TypeCvt"
  %5 = "MGB.TypeCvt"(%4) {i_dtype = f32, i_scale = 1.000000e+00 : f32, i_zero = 10 : ui8, o_dtype = f32, o_scale = 1.000000e+00 : f32, o_zero = 0 : ui8} : (tensor<30xf32>) -> tensor<30xf32>

  // CHECK-NEXT: %3 = "MGB.Elemwise"
  %6 = "MGB.Elemwise"(%5, %4) {mode = 24 : i32} : (tensor<30xf32>, tensor<30xf32>) -> tensor<30xf32>
  // CHECK-NEXT: return
  return %6 : tensor<30xf32>
}
