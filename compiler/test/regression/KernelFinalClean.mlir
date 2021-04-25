// RUN: megcc-opt --kernel-clean --canonicalize %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0 * 16 + d1)>
// CHECK-LABEL: func @remove_dimshuffle_forward 
func @remove_dimshuffle_forward(%arg0: memref<16x8xf32>) -> memref<16x8xf32, #map> {
    // CHECK-NEXT: "Kernel.Reshape"
    //    CHECK-DAG: %arg0
    //    CHECk-DAG: determined = true
    %1 = "Kernel.Dimshuffle"(%arg0) {determined = true, pattern=[1 : i32, 0 : i32]} : (memref<16x8xf32>) -> memref<8x16xf32, #map>
    %2 = "Kernel.Reshape"(%1) {determined = true} : (memref<8x16xf32, #map>) -> memref<16x8xf32, #map>    
    return %2: memref<16x8xf32, #map>
}
