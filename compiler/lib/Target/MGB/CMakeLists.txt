set(LLVM_OPTIONAL_SOURCES importer.cpp exporter.cpp)

add_mlir_translation_library(
  MLIRMGBImporter
  importer.cpp
  DEPENDS
  MLIRMGBIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRMGB
  MLIRStandard)
# detail obj library created in llvm_add_library
target_include_directories(obj.MLIRMGBImporter PRIVATE ${MGB_INCLUDE_DIR})
target_link_libraries(MLIRMGBImporter PUBLIC $<BUILD_INTERFACE:mgb_imported>)
