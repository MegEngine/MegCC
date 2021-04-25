add_mlir_dialect_library(
  MLIRMGB
  MGBDialect.cpp
  DEPENDS
  MLIRMGBIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR)
# detail obj library created in llvm_add_library
target_include_directories(obj.MLIRMGB PRIVATE ${MGB_INCLUDE_DIR})
target_link_libraries(MLIRMGB PUBLIC $<BUILD_INTERFACE:mgb_imported>)
