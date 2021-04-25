add_mlir_dialect_library(
  MLIRKernel
  KernelDialect.cpp
  DEPENDS
  MLIRKernelIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR)

# FIXME: linking with mgb since we proxy reshape to mgb::Tensor impl detail obj library
# created in llvm_add_library
target_include_directories(obj.MLIRKernel PRIVATE ${MGB_INCLUDE_DIR})
target_link_libraries(MLIRKernel PUBLIC $<BUILD_INTERFACE:mgb_imported>)
