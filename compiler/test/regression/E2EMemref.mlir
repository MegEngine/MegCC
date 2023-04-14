// RUN: megcc-opt --MGB-to-Kernel --finalizing-bufferize --memory-forwarding --static-memory-planning --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @concat_transpose
func @concat_transpose(%arg0: tensor<1x1x2x3xf32> {mgb.func_arg_name = "d0"}, %arg1: tensor<1x4x2x3xf32> {mgb.func_arg_name = "d1"}) -> (tensor<1x2x3x5xf32> {mgb.func_result_name = "dimshuffle(concat[4])[6]"}) {
// CHECK-NEXT: "Kernel.MemPlan"
// CHECK-NEXT: "Kernel.Relayout"
// CHECK-NEXT: "Kernel.MemPlan"
// CHECK-NEXT: "Kernel.Relayout"
%2 = "MGB.Concat"(%arg0, %arg1) {axis = 1 : i32, comp_node = "cpu:default"} : (tensor<1x1x2x3xf32>, tensor<1x4x2x3xf32>) -> tensor<1x5x2x3xf32>
// CHECK-NEXT: "Kernel.MemPlan"
// CHECK-NEXT: "Kernel.MemPlan"
// CHECK-NEXT: "Kernel.Relayout"
%3 = "MGB.Dimshuffle"(%2) {pattern = [0 : i32, 2 : i32, 3 : i32, 1 : i32]} : (tensor<1x5x2x3xf32>) -> tensor<1x2x3x5xf32>
return %3 : tensor<1x2x3x5xf32>
}