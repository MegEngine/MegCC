get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(onnx-importer onnx-importer.cpp NO_INSTALL_RPATH)
llvm_update_compile_flags(onnx-importer)
target_link_libraries(onnx-importer PRIVATE ${dialect_libs} MLIRONNXImporter Common
                                            ${ONNX_LIBS} ${PROTOBUF_LIBS})
mlir_check_all_link_libraries(onnx-importer)