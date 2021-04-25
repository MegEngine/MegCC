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
