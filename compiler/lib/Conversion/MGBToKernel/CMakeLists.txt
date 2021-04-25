add_mlir_conversion_library(
  MLIRMGBToKernel
  MGBToKernel.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion/MGBToKernel
  DEPENDS
  MLIRMGBToKernelPassIncGen
  MGBEnumReflectionGen
  LINK_COMPONENTS
  Core
  LINK_LIBS
  PUBLIC
  MLIRMGB
  MLIRKernel
  MLIRPass
  MLIRStandard
  MLIRTransforms
  MLIRIR
  MLIRStandardOpsTransforms)
# detail obj library created in llvm_add_library
target_include_directories(obj.MLIRMGBToKernel PRIVATE ${MGB_INCLUDE_DIR})
target_link_libraries(MLIRMGBToKernel PUBLIC $<BUILD_INTERFACE:mgb_imported>)
