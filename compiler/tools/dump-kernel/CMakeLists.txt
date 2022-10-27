get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# since it links with imported megengine library, it should use default RPATH settings or
# llvm would rewrite it but missing correct rpath of libmegengine
add_llvm_executable(dump-kernel dump-kernel.cpp NO_INSTALL_RPATH)
llvm_update_compile_flags(dump-kernel)
target_link_libraries(
  dump-kernel
  PRIVATE ${dialect_libs}
          MLIROptLib
          MLIRMGBToKernel
          MLIRMGB
          MLIRMGBTransforms
          MLIRKernel
          MLIRKernelTransforms
          KernelGen
          MLIRLinalg
          MLIRLinalgTransforms
          MLIRTosa
          MLIRTosaTransforms

          MLIRTosaToLinalg
          MLIRLinalgToStandard
          MLIRLinalgToLLVM
          MLIRExecutionEngine
          Common)
mlir_check_all_link_libraries(dump-kernel)
