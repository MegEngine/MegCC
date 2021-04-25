# This DIR is used to test codegen standalone
some useful cmd
```bash
mlir-opt ../test/regression/TOSAme0.mlir --tosa-to-linalg --linalg-comprehensive-module-bufferize -convert-linalg-to-loops -lower-affine -convert-scf-to-std  --arith-expand -convert-std-to-llvm='emit-c-wrappers=1' --convert-memref-to-llvm  -convert-arith-to-llvm --reconcile-unrealized-casts > tosame0.mlir 
mlir-translate --mlir-to-llvmir tosame0.mlir > tosame0.bc
clang run_mlir_gen_relu.c tosame0.bc -O0 -g -fsanitize=address -fno-omit-frame-pointer
```

debug llvm use https://github.com/JuliaComputingOSS/llvm-cbe