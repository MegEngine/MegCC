get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# since it links with imported megengine library, it should use default RPATH settings
# or llvm would rewrite it but missing correct rpath of libmegengine
add_llvm_executable(mgb-to-tinynn mgb-to-tinynn.cpp NO_INSTALL_RPATH)
llvm_update_compile_flags(mgb-to-tinynn)
target_link_libraries(
  mgb-to-tinynn
  PRIVATE ${dialect_libs}
          MLIRMGBImporter
          MLIRTinyNNExporter
          MLIRMGBToKernel
          MLIRMGB
          MLIRMGBTransforms
          MLIRKernel
          MLIRKernelTransforms
          KernelGen
          Common
          HakoParse)
mlir_check_all_link_libraries(mgb-to-tinynn)
