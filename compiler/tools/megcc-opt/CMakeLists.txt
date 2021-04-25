get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# since it links with imported megbrain library, it should use default RPATH settings or
# llvm would rewrite it but missing correct rpath of libmegbrain
add_llvm_executable(megcc-opt megcc-opt.cpp NO_INSTALL_RPATH)
llvm_update_compile_flags(megcc-opt)
target_link_libraries(
  megcc-opt
  PRIVATE ${dialect_libs}
          MLIROptLib
          MLIRMGBToKernel
          MLIRMGB
          MLIRMGBTransforms
          MLIRKernel
          MLIRKernelTransforms
          KernelGen
          ${TCC_INSTALL_LIB}
          Common)
mlir_check_all_link_libraries(megcc-opt)
