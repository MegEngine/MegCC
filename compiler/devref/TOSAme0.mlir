#map0 = affine_map<(d0, d1) -> (d0 * 10 + d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func @my_codegen_elem(%arg0: memref<1x10xf32, #map0>, %arg1: memref<1x10xf32, #map0>) {
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x10xf32, #map0>) outs(%arg1 : memref<1x10xf32, #map0>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %cst = arith.constant 0.000000e+00 : f32
      %0 = arith.maxf %arg2, %cst : f32
      linalg.yield %0 : f32
    }
    return
  }
}