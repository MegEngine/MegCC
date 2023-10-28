set(LLVM_OPTIONAL_SOURCES onnx_importer.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DONNX_ML=1 -DONNX_NAMESPACE=onnx")
add_compile_definitions("ONNX_NO_EXCEPTIONS")

add_mlir_translation_library(
  MLIRONNXImporter
  importer.cpp
  DEPENDS
  MLIRMGBIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRMGB
  MLIRStandard)
# detail obj library created in llvm_add_library
target_include_directories(
  obj.MLIRONNXImporter PRIVATE ${MGB_INCLUDE_DIR} ${ONNX_INCLUDE_DIR}
                               ${PROTOBUF_INCLUDE_DIR})
# add onnx-imported
target_link_libraries(MLIRONNXImporter PUBLIC $<BUILD_INTERFACE:onnx_imported>)
# target_compile_options(MLIRONNXImporter PUBLIC -fexceptions)