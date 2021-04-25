file(GLOB_RECURSE SRC *.cpp *.h)
add_mlir_translation_library(
  MLIRTinyNNExporter
  ${SRC}
  DEPENDS
  MLIRKernelIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRKernel
  MLIRStandard
  MLIRKernelTransforms)

# detail obj library created in llvm_add_library
add_dependencies(obj.MLIRTinyNNExporter lite_runtime_schema_fbs)
target_link_libraries(obj.MLIRTinyNNExporter PUBLIC flatbuffers)
target_include_directories(obj.MLIRTinyNNExporter PRIVATE ${LITE_SCHEMA_GEN_DIR})

target_link_libraries(MLIRTinyNNExporter PUBLIC $<BUILD_INTERFACE:KernelGen>)
