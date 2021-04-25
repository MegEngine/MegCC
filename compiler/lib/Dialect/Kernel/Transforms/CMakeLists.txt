file(GLOB_RECURSE KERNEL_TRANS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_mlir_dialect_library(
  MLIRKernelTransforms
  ${KERNEL_TRANS_SRC}
  DEPENDS
  MLIRKernelTransformsPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRKernel
  MLIRPass
  MLIRStandard
  MLIRTransforms)
