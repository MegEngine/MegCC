// RUN: megcc-opt --static-memory-planning --canonicalize %s | FileCheck %s

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 16)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1 + 256)>

// check the size of global buffer and memref.alloc was replaced with Kernel.MemPlan correctly
// CHECK-LABEL: func @simple_chain
//    CHECK-DAG: memref<1024xi8> {mgb.func_arg_name = "kGlobalBuffer"}
func @simple_chain(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>, %arg3: memref<128xf32>, %arg4: memref<128xf32>) -> memref<128xf32> {
    // CHECK-NEXT: Kernel.MemPlan
    // CHECK-NEXT: Kernel.ADD
    // CHECK-NEXT: Kernel.MemPlan
    // CHECK-NEXT: Kernel.ADD
    // CHECK-NEXT: Kernel.MemPlan
    // CHECK-NEXT: Kernel.ADD
    // CHECK-NEXT: Kernel.MemPlan
    // CHECK-NEXT: Kernel.ADD
    // CHECK-NEXT: return
    %0 = memref.alloc() : memref<128xf32>
    "Kernel.ADD"(%arg0, %arg1, %0) : (memref<128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %1 = memref.alloc() : memref<128xf32>
    "Kernel.ADD"(%0, %arg2, %1) : (memref<128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %2 = memref.alloc() : memref<128xf32>
    "Kernel.ADD"(%1, %arg3, %2) : (memref<128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %3 = memref.alloc() : memref<128xf32>
    "Kernel.ADD"(%2, %arg4, %3) : (memref<128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    return %3: memref<128xf32>
}

// check buffer was reused when memory forwarding happened
//    only one 128 * sizeof(f32) buffer is needed
// CHECK-LABEL: func @memory_forward
//    CHECK-DAG: memref<512xi8> {mgb.func_arg_name = "kGlobalBuffer"}
#map = affine_map<(d0, d1) -> (d0 + d1 * 16)>
func @memory_forward(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<16x8xf32, #map> {
    // more than one consecutive determined forwarding ops would be folded as one memplan
    // CHECK-NEXT: Kernel.MemPlan
    // CHECK-NEXT: Kernel.ADD
    // CHECK-NEXT: Kernel.MemPlan
    //   CHECK-DAG: #[[MAP0]]
    // CHECK-NEXT: return
    %0 = memref.alloc() : memref<128xf32>
    "Kernel.ADD"(%arg0, %arg1, %0) : (memref<128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %1 = "Kernel.Reshape"(%0) {determined = true} : (memref<128xf32>) -> memref<8x16xf32>
    %2 = "Kernel.Dimshuffle"(%1) {determined = true, pattern=[1 : i32, 0 : i32]} : (memref<8x16xf32>) -> memref<16x8xf32, #map>
    return %2: memref<16x8xf32, #map>
}


// CHECK-LABEL: func @multi_results
//    CHECK-DAG: memref<768xi8> {mgb.func_arg_name = "kGlobalBuffer"}
func @multi_results(%arg0: memref<128xf32>, %arg1: memref<64xf32>, %arg2: memref<8x8xf32>) -> (memref<64xf32>, memref<8x8xf32>) {
    // CHECK-NEXT: "Kernel.MemPlan"
    // CHECK-NEXT: "Kernel.EXP"
    %0 = memref.alloc() : memref<128xf32>
    "Kernel.EXP"(%arg0, %0): (memref<128xf32>, memref<128xf32>) -> ()

    // CHECK-NEXT: "Kernel.MemPlan"
    // CHECK-NEXT: "Kernel.MemPlan"
    // CHECK-NEXT: "Kernel.ADD"
    %1 = "Kernel.Subtensor"(%0) {determined = true, descs=[[0: i32, 0: i32, 64: i32, 1: i32, -1: i32]], flags=[[0: i32, 0: i32, 0: i32, 0: i32, -1: i32]]} : (memref<128xf32>) -> memref<64xf32>
    %2 = memref.alloc() : memref<64xf32>
    "Kernel.ADD"(%1, %arg1, %2) : (memref<64xf32>, memref<64xf32>, memref<64xf32>) -> ()

    // CHECK-NEXT: "Kernel.MemPlan"
    //    CHECK-DAG: #[[MAP1]]
    %3 = "Kernel.Subtensor"(%0) {determined = true, descs=[[0: i32, 64: i32, 128: i32, 1: i32, -1: i32]], flags=[[0: i32, 0: i32, 0: i32, 0: i32, -1: i32]]} : (memref<128xf32>) -> memref<64xf32>
    %4 = "Kernel.Reshape"(%3) {determined = true} : (memref<64xf32>) -> memref<8x8xf32>

    // CHECK-NEXT: return
    return %2, %4 : memref<64xf32>, memref<8x8xf32>
}

#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 12544 + d1 * 196 + d2 + d3 * 28 + d4 * 4)>
// CHECK-LABEL: func @quant_result
//    CHECK-DAG: memref<1x64x4x7x7xqsi8
func @quant_result(%arg0:memref<1x64x7x7x4xqsi8<1042367109:1.574803e-01>>) -> memref<1x64x4x7x7xqsi8<1042367109:1.574803e-01>>{
    // CHECK-NEXT: "Kernel.MemPlan"
    %1 = memref.alloc() : memref<1x64x7x7x4xqsi8<1042367109:1.574803e-01>>
    // CHECK-NEXT: "Kernel.RELU"
    "Kernel.RELU"(%arg0, %1) : (memref<1x64x7x7x4xqsi8<1042367109:1.574803e-01>>, memref<1x64x7x7x4xqsi8<1042367109:1.574803e-01>>) -> ()
    %2 = "Kernel.Dimshuffle"(%1) {determined = true, pattern = [0 : i32, 1 : i32, 4 : i32, 2 : i32, 3 : i32]} : (memref<1x64x7x7x4xqsi8<1042367109:1.574803e-01>>) -> memref<1x64x4x7x7xqsi8<1042367109:1.574803e-01>, #map15>
    %3 = memref.alloc() : memref<1x256x7x7xqsi8<1042367109:1.574803e-01>>
    %4 = "Kernel.Reshape"(%3) {determined = true} : (memref<1x256x7x7xqsi8<1042367109:1.574803e-01>>) -> memref<1x64x4x7x7xqsi8<1042367109:1.574803e-01>>
    // CHECK-NEXT: "Kernel.MemPlan"
    // CHECK-NEXT: "Kernel.MemPlan"
    // CHECK-NEXT: "Kernel.Relayout"
    "Kernel.Relayout"(%2, %4) : (memref<1x64x4x7x7xqsi8<1042367109:1.574803e-01>, #map15>, memref<1x64x4x7x7xqsi8<1042367109:1.574803e-01>>) -> ()
    return %4 : memref<1x64x4x7x7xqsi8<1042367109:1.574803e-01>>
}