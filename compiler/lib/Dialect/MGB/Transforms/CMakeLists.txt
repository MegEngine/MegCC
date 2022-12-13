file(GLOB_RECURSE KERNEL_TRANS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_mlir_dialect_library(
  MLIRMGBTransforms
  ${KERNEL_TRANS_SRC}
  DEPENDS
  MLIRMGBTransformsPassIncGen
  MGBEnumReflectionGen
  LINK_LIBS
  PUBLIC
  MLIRMGB
  MLIRPass
  MLIRStandard
  MLIRTransforms)

target_include_directories(obj.MLIRMGBTransforms PRIVATE ${MGB_INCLUDE_DIR})
target_link_libraries(MLIRMGBTransforms PUBLIC $<BUILD_INTERFACE:mgb_imported>)
