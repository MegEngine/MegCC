get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(onnx-to-tinynn onnx-to-tinynn.cpp NO_INSTALL_RPATH)
llvm_update_compile_flags(onnx-to-tinynn)
target_link_libraries(
  onnx-to-tinynn
  PRIVATE ${dialect_libs}
          MLIRONNXImporter
          MLIRTinyNNExporter
          MLIRMGBToKernel
          MLIRMGB
          MLIRMGBTransforms
          MLIRKernel
          MLIRKernelTransforms
          KernelGen
          Common
          HakoParse
          ${ONNX_LIBS}
          ${PROTOBUF_LIBS})
mlir_check_all_link_libraries(onnx-to-tinynn)